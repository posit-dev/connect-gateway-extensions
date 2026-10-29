"""Backfill a real Connect's gateway usage log with synthetic, grounded usage.

Connect builds its usage export from JSONL files under
<data dir>/gateway/date=<ingest day>/node=<node>/events.jsonl. This writes such
files for a node of its own, so that Connect's real export route serves them and
the dashboard reads them like any other usage.

The rows are synthetic but grounded. Every user, content item, content owner and
price comes from the live server (CONNECT_SERVER, CONNECT_API_KEY), so each GUID
in a row resolves through the same API the dashboard calls. Calls are shaped like
the mock's (conversations, prompt caching, Workbench sessions, runaway agents);
see demo/fake_connect.py.

    python demo/backfill_connect.py --out /tmp/backfill --integration <guid>
    tar -C /tmp/backfill -cf - . | kubectl -n posit-team exec -i <pod> -c connect -- \
        tar -C /var/lib/rstudio-connect/gateway -xf -

Delete a backfill by removing its node=<node> directories. The node name is the
only thing that marks the rows as synthetic. A seed makes the output repeatable.
"""

import argparse
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent))
import fake_connect as fc  # noqa: E402  (its pick_client and Workbench helpers)

# Models people pick in an IDE or CLI, and the cheap ones content uses. All need
# to be in the server's price catalog. Where the catalog also lists a dated name,
# the response model carries it.
IDE_MODELS = [
    ("claude-sonnet-4-5", 0.45), ("claude-opus-4-5", 0.2), ("claude-opus-4-1", 0.15),
    ("claude-sonnet-4-20250514", 0.1), ("claude-haiku-4-5", 0.1),
]
CONTENT_MODELS = [
    ("claude-haiku-4-5", 0.5), ("claude-haiku-4-5-20251001", 0.2),
    ("claude-sonnet-4-5", 0.2), ("claude-sonnet-4-20250514", 0.1),
]
DATED = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-opus-4-1": "claude-opus-4-1-20250805",
}
# Content that can call a model while someone uses it, not static documents.
LIVE_MODES = {"python-shiny", "shiny", "python-fastapi", "python-streamlit", "jupyter-voila", "api"}
# A content item that can call a model is the dashboard itself or a chat app.
SKIP_TITLES = {"Posit Team AI Usage Dashboard"}
CHAT_WORDS = ("chat", "mcp", "assistant")
# Accounts that act for no one.
SKIP_USERS = {"connect-service-user"}
# The gateway writes a record when a call finishes. A few arrive much later.
LATE_INGEST = 0.005


def get_all(server: str, key: str, path: str) -> list[dict]:
    """Every item of a Connect list route, following its pages."""
    out, page = [], 1
    while True:
        sep = "&" if "?" in path else "?"
        req = Request(f"{server}/__api__/v1/{path}{sep}page_size=500&page_number={page}")
        req.add_header("Authorization", f"Key {key}")
        with urlopen(req, timeout=60) as r:
            body = json.load(r)
        items = body["results"] if isinstance(body, dict) else body
        out += items
        if not isinstance(body, dict) or len(out) >= body.get("total", len(out)) or not items:
            return out
        page += 1


def zulu(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def pick(rnd: random.Random, options: list[tuple[str, float]]) -> str:
    return rnd.choices([o[0] for o in options], [o[1] for o in options])[0]


def build(users: list[dict], content: list[dict], integrations: list[str], rows: int,
          days: int, node: str, seed: int, now: datetime) -> list[dict]:
    rnd = random.Random(seed)
    random.seed(seed)  # fake_connect.pick_client draws from the global generator
    by_guid = {u["guid"]: u for u in users}
    n = len(users)
    # A few heavy users, some light, and a long tail. The real account of whoever
    # runs the server is the heaviest, as it is in real traffic.
    weight = [min(1.0, rnd.lognormvariate(-1.6, 1.0)) for _ in users]
    for i in rnd.sample(range(n), 6):
        weight[i] = rnd.uniform(3, 8)
    cum = [sum(weight[: i + 1]) for i in range(n)]
    active = []
    for i, u in enumerate(users):
        p = 0.85 if weight[i] > 2 else min(0.8, max(0.05, rnd.lognormvariate(-1.8, 0.8)))
        # People who joined partway through the window only appear after that.
        joined = max(0, (now - datetime.fromisoformat(u["created_time"])).days)
        first = min(days - 1, joined)
        days_on = [
            d for d in range(first + 1)
            if (now - timedelta(days=d)).weekday() < 5 or rnd.random() < 0.1
            if rnd.random() < p
        ]
        active.append(days_on or [rnd.randint(0, first)])
        fc.plan_sessions(i, rnd)

    live = [
        c for c in content
        if c["app_mode"] in LIVE_MODES and not c["archived"] and c["owner_guid"] in by_guid
        and (c["title"] or c["name"]) not in SKIP_TITLES
    ]
    cweight = [
        5.0 if any(w in (c["title"] or c["name"]).lower() for w in CHAT_WORDS) else 1.0
        for c in live
    ]
    runaway_content = rnd.randrange(len(live))  # one content item loops far more than the rest
    cweight[runaway_content] *= 2

    out: list[dict] = []
    while len(out) < rows:
        ui = rnd.choices(range(n), cum_weights=cum)[0]
        u = users[ui]
        day = rnd.choice(active[ui])
        t = now - timedelta(days=day, seconds=rnd.randint(0, 86399))
        t = t.replace(hour=int(rnd.gauss(15, 3.5)) % 24)
        from_content = rnd.random() < 0.4
        c = rnd.choices(live, cweight)[0] if from_content else None
        if c and datetime.fromisoformat(c["created_time"]) > t:
            continue  # the content did not exist yet
        runaway = rnd.random() < (0.01 if c is live[runaway_content] else 0.004 if c else 0.01)
        if runaway:
            turns, gap = rnd.randint(80, 320), (2, 12)
        else:
            turns, gap = max(1, int(rnd.expovariate(1 / (3 if c else 7))) + 1), (15, 240)
        model = pick(rnd, CONTENT_MODELS if c else IDE_MODELS)
        integration = rnd.choice(integrations)
        owner = by_guid[c["owner_guid"]] if c else None
        # Most content shows who is using it. Some runs with no viewer, as a
        # scheduled job would.
        viewer = u if c and rnd.random() > 0.15 else None
        user_agent, _ = fc.pick_client(bool(c))
        meta = None if c else json.loads(fc.workbench_metadata(ui, day))
        if not c and rnd.random() < 0.4:
            meta = None  # not every IDE session comes from Workbench
        conv = uuid.uuid4().hex if rnd.random() < 0.9 else None
        job = "".join(rnd.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=16)) if c else None
        context = int(rnd.lognormvariate(8, 1.1))
        cached, touched = 0, None
        for _ in range(turns):
            if touched is not None and (t - touched).total_seconds() > fc.CACHE_TTL:
                cached = 0
            fresh = int(rnd.lognormvariate(6.5, 1.0))
            cache_read = min(cached, context)
            cache_write = context - cache_read
            tin = fresh + cache_write + cache_read
            status = rnd.choices([200, 429, 500], [96, 3, 1])[0]
            tout = 0 if status != 200 else int(rnd.lognormvariate(6.3, 1.0))
            if status != 200:
                tin = cache_read = cache_write = 0
            finish = rnd.choices(
                ["end_turn", "tool_use", "max_tokens"],
                [55, 43, 2] if runaway or not c else [93, 5, 2],
            )[0]
            duration_ms = int(rnd.lognormvariate(7.5, 0.8) * (1 + tin / 200_000))
            ingest = t + timedelta(milliseconds=duration_ms)
            if rnd.random() < LATE_INGEST:
                ingest += timedelta(hours=rnd.uniform(2, 60))
            if ingest <= now:
                out.append({
                    "start_time": zulu(t), "ingest_time": zulu(ingest), "node_name": node,
                    "duration_ms": duration_ms, "provider": "anthropic", "operation": "chat",
                    "integration_guid": integration, "integration_type": "anthropic",
                    "integration_auth_type": "Environment Variables",
                    "content_guid": c["guid"] if c else None,
                    "content_id": int(c["id"]) if c else None,
                    "job_key": job,
                    # The credentials of the call: the owner's for content, else the user's.
                    "authenticated_user_guid": (owner or u)["guid"],
                    "authenticated_username": (owner or u)["username"],
                    "authenticated_email": (owner or u)["email"],
                    "content_owner_guid": owner["guid"] if owner else None,
                    "content_owner_username": owner["username"] if owner else None,
                    "content_owner_email": owner["email"] if owner else None,
                    "viewer_user_guid": viewer["guid"] if viewer else None,
                    "viewer_username": viewer["username"] if viewer else None,
                    "viewer_email": viewer["email"] if viewer else None,
                    "conversation_id": conv,
                    "user_agent": user_agent,
                    "request_metadata": meta,
                    "server_address": "api.anthropic.com", "server_port": 443,
                    "status_code": status, "request_model": model,
                    "response_model": DATED.get(model, model) if status == 200 else "",
                    "response_id": "msg_01" + uuid.UUID(int=rnd.getrandbits(128)).hex[:22],
                    "request_max_tokens": rnd.choice([4096, 8192, 32000, 64000]),
                    "request_temperature": None, "request_top_p": None,
                    "request_stop_sequences": None,
                    "finish_reasons": [finish] if status == 200 else None,
                    "input_tokens": tin, "output_tokens": tout,
                    "cache_creation_input_tokens": cache_write,
                    "cache_read_input_tokens": cache_read,
                })
            cached = context if status == 200 else 0
            context = min(context + fresh + tout, 180_000)
            touched = t + timedelta(milliseconds=duration_ms)
            pause_p = 0 if runaway else (0.05 if c else 0.12)
            pause = rnd.uniform(360, 2700) if rnd.random() < pause_p else rnd.uniform(*gap)
            t += timedelta(seconds=pause, milliseconds=duration_ms)
    return out[:rows]


def write(rows: list[dict], out: Path, node: str) -> int:
    """One events.jsonl per ingest day, as Connect lays its own out."""
    parts = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["ingest_time"]):
        parts[r["ingest_time"][:10]].append(json.dumps(r, separators=(",", ":")))
    for day, lines in parts.items():
        d = out / f"date={day}" / f"node={node}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    return len(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--integration", action="append", required=True,
                    help="an Anthropic integration GUID on the server; repeat for several")
    ap.add_argument("--node", default="connect-backfill")
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()
    server = os.environ["CONNECT_SERVER"].rstrip("/")
    key = os.environ["CONNECT_API_KEY"]
    users = [
        u for u in get_all(server, key, "users")
        if not u["locked"] and u["username"] not in SKIP_USERS
    ]
    content = get_all(server, key, "content")
    rows = build(users, content, a.integration, a.rows, a.days, a.node, a.seed, datetime.now(timezone.utc))
    files = write(rows, a.out, a.node)
    print(f"{len(rows)} rows, {files} files, {len(users)} users, {len(content)} content items")


if __name__ == "__main__":
    main()
