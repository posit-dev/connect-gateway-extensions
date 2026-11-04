"""A mock Connect server: the dashboard's only other backend.

The app has no demo mode. It talks to this server exactly as it talks to a real
Connect, through the same four routes, so anything that differs between the two
is the data, not the code path:

    GET /__api__/v1/system/gateway/usage/export   synthetic Parquet usage
    GET /__api__/v1/system/gateway/pricing        demo/pricing.json, a snapshot
                                                  of a real Connect catalog
    GET /__api__/v1/system/gateway/models         no admin price overrides
    GET /__api__/v1/content                       names for the demo content,
                                                  one untitled, one "deleted"
    GET /__api__/v1/groups                        teams plus cross-cutting groups
    GET /__api__/v1/groups/<guid>/members         (users can be in several)

For the failure tests, POST /__fault__?route=<route>&mode=<mode> makes one route
misbehave until POST /__fault__?reset=1. Routes are export, pricing, overrides,
content, groups and members. Modes: status:<code> (a plain-text refusal),
html:<code> (an HTML error page, as a proxy sends), garbage (200 with a body
that is not Parquet or JSON) and truncate (a body cut off mid-download).

Like Connect, every route wants `Authorization: Key <anything>`, and the export
wants from, to and as_of. Refresh demo/pricing.json from a real server with:

    curl -H "Authorization: Key $CONNECT_API_KEY" \
      $CONNECT_SERVER/__api__/v1/system/gateway/pricing > demo/pricing.json

Run it with `just run-local`, or alone:

    python demo/fake_connect.py            # listens on :3940
"""

import hashlib
import itertools
import json
import os
import random
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import duckdb
import pandas as pd

random.seed(7)
PORT = int(os.environ.get("MOCK_CONNECT_PORT", 3940))
DAYS = 90  # the dashboard loads 90 days, so generate exactly that

# Enough rows that the Overview's 2,000-row table cap is visibly a cap.
ROWS = int(os.environ.get("FAKE_ROWS", 160000))

# "full" (the default) sends everything a current Connect and Workbench send.
# "sparse" is a server whose callers send no User-Agent and no Gateway-Metadata,
# with almost all traffic outside content: so no harness, no Workbench session,
# and an origin only on content rows. That is what a real customer server can
# look like, and it is where empty charts and zero-divisions show up.
PROFILE = os.environ.get("FAKE_PROFILE", "full")
if PROFILE not in ("full", "sparse"):
    raise SystemExit(f"FAKE_PROFILE must be full or sparse, not {PROFILE!r}")

# The most rows one page of a paged list holds, whatever page_size asks for.
# Connect caps page_size too; the tests set a small cap so that reading groups
# and their members has to follow several pages.
PAGE_CAP = int(os.environ.get("MOCK_PAGE_CAP", 500))

_HEAVY = [
    ("ada", "lovelace"), ("alan", "turing"), ("grace", "hopper"),
    ("linus", "torvalds"), ("margaret", "hamilton"), ("dennis", "ritchie"),
    ("barbara", "liskov"), ("ken", "thompson"), ("radia", "perlman"),
    ("donald", "knuth"), ("edsger", "dijkstra"), ("john", "mccarthy"),
]
_FIRST = ["amy", "ben", "cara", "dev", "eli", "fay", "gus", "hana", "ian", "jo",
          "kit", "lena", "max", "nia", "omar", "pia", "quinn", "raj", "sam", "tara"]
_LAST = ["abbott", "bose", "chen", "diaz", "evans", "faro", "gupta", "hale", "ito",
         "jones", "khan", "lopez", "mori", "nash", "ortiz", "park", "ruiz", "shah",
         "tran", "wu"]
_TAIL = [(f, l) for f in _FIRST for l in _LAST]
random.Random(3).shuffle(_TAIL)
# 12 heavy users who are on most days, then a long tail of lighter users, some of
# whom only start partway through the month. That gives the adoption page rolling
# windows that actually differ (DAU < WAU < MAU) and rise over time.
_PEOPLE = _HEAVY + _TAIL[:138]
USERS = [(f"{f}.{l}", f"{f}.{l}@example.com", uuid.uuid4().hex) for f, l in _PEOPLE]
CONTENT = [uuid.uuid4().hex for _ in range(14)]
# Content 0 is a runaway: three times the traffic of any other. Content 1 is a
# heavy one: ordinary traffic, but every request is a very large prompt. That
# gives the Attribution page a content item that is an outlier by spend.
CONTENT_W = [4.0, 1.0] + [1.0] * 12
# Who owns each content item: spread across the heavy users.
CONTENT_OWNER = [i % 12 for i in range(len(CONTENT))]
CONTENT_TOKEN_X = {1: 12}
# Content that never asks for prompt caching: every turn re-sends its history at
# the full input price.
CONTENT_NO_CACHE = {4, 7}
# Anthropic's default prompt cache lives this long after its last use.
CACHE_TTL = 300
_TEAMS = ["data-science", "platform", "analytics", "ml-research", "support-eng"]
_SESSION_KINDS = ["RStudio Pro Session", "Positron Pro Session", "VS Code Session"]
MODELS = [
    ("anthropic", "claude-opus-4-1-20250805", 0.15),
    ("anthropic", "claude-sonnet-4-5-20250929", 0.45),
    ("anthropic", "claude-haiku-4-5-20251001", 0.2),
    ("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", 0.15),
    ("openai", "gpt-5", 0.05),
]


# One user's Workbench sessions across the window: (first day, last day, id, name),
# in "days ago". A real session is long-lived (it is a running IDE), so each lasts
# days to weeks and the same id repeats across all the requests it makes.
_SESSION_PLAN: dict[int, list[tuple[int, int, str, str]]] = {}


def plan_sessions(user_index: int, rnd: random.Random) -> None:
    plan, day, n = [], DAYS - 1, 0
    kind = _SESSION_KINDS[user_index % 3]
    while day >= 0:
        length = rnd.randint(5, 21)
        n += 1
        plan.append((day, max(day - length + 1, 0), f"{rnd.getrandbits(32):08x}", f"{kind} {n}"))
        day -= length
    _SESSION_PLAN[user_index] = plan


def workbench_metadata(user_index: int, days_ago: int) -> str:
    """The JSON a Workbench session attaches, e.g.
    {"cost-center":"ds 42","enforced":"true","team":"data-science","tier":"admin",
     "workbench-project-id":"3f9a1c2e4b7d0","workbench-project-name":"data-science analysis",
     "workbench-project-path":"~/data-science",
     "workbench-session-id":"d1ac60de","workbench-session-name":"RStudio Pro Session 7"}
    """
    sid, name = next(
        (i, nm) for hi, lo, i, nm in _SESSION_PLAN[user_index] if lo <= days_ago <= hi
    )
    team = _TEAMS[user_index % len(_TEAMS)]
    return json.dumps(
        {
            "cost-center": f"{team[:2]} {40 + user_index % 7}",
            "enforced": "true" if user_index % 5 else "false",
            "team": team,
            "tier": "admin" if user_index < 6 else "user",
            "workbench-project-id": hashlib.md5(f"{user_index}/{team}".encode()).hexdigest()[:13],
            "workbench-project-name": f"{team} analysis",
            "workbench-project-path": f"~/{team}",
            "workbench-session-id": sid,
            "workbench-session-name": name,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def make_rows(n: int = ROWS) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    weights = [m[2] for m in MODELS]
    rnd = random.Random(11)
    # Spend weight per user: the heavy dozen dominate, the tail is a long, thin one.
    uw = [8, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1] + [
        min(1.0, rnd.lognormvariate(-2.2, 0.8)) for _ in range(len(USERS) - 12)
    ]
    # The days (as "days ago") each user is active on. Heavy users are on most
    # weekdays; the tail is occasional, and some of it joins part-way through.
    active: list[list[int]] = []
    for i in range(len(USERS)):
        p = 0.92 if i < 12 else min(0.85, max(0.04, rnd.lognormvariate(-1.9, 0.7)))
        first_day = DAYS - 1 if i < 12 or rnd.random() > 0.35 else rnd.randint(2, DAYS - 6)
        days = [
            d for d in range(first_day + 1)
            if (now - timedelta(days=d)).weekday() < 5 or rnd.random() < 0.12
            if rnd.random() < p
        ]
        active.append(days or [rnd.randint(0, first_day)])
    for i in range(len(USERS)):
        plan_sessions(i, rnd)
    cum = list(itertools.accumulate(uw))
    rows = []
    # Rows come in conversations: one user, one model, a run of turns a minute or
    # two apart, with the context growing turn by turn (mostly served from the
    # prompt cache). A few are runaway agent loops: hundreds of fast turns.
    while len(rows) < n:
        ui = random.choices(range(len(USERS)), cum_weights=cum)[0]
        u = USERS[ui]
        day = random.choice(active[ui])
        t = now - timedelta(days=day, seconds=random.randint(0, 86399))
        t = t.replace(hour=int(random.gauss(14, 4)) % 24)
        provider, model, _ = random.choices(MODELS, weights)[0]
        from_content = random.random() < (0.02 if PROFILE == "sparse" else 0.55)
        content_i = random.choices(range(len(CONTENT)), CONTENT_W)[0] if from_content else None
        mult = CONTENT_TOKEN_X.get(content_i, 1) if content_i is not None else 1
        # A content item either acts for its viewer or runs on its owner's
        # credentials; every third one is a service-account app.
        service = content_i is not None and content_i % 3 == 0
        owner = USERS[CONTENT_OWNER[content_i]] if content_i is not None else None
        runaway = random.random() < (0.004 if from_content else 0.012)
        if runaway:
            turns, gap = random.randint(80, 320), (2, 12)
        else:
            mean = 3 if from_content else 7
            turns, gap = max(1, int(random.expovariate(1 / mean)) + 1), (15, 240)
        conv = uuid.uuid4().hex[:12] if random.random() < 0.9 else None
        context = int(random.lognormvariate(8, 1.1)) * mult
        user_agent, origin = pick_client(from_content)
        meta = None if from_content else workbench_metadata(ui, day)
        if PROFILE == "sparse":
            user_agent, meta = None, None
            origin = "connect" if from_content else None
        job = uuid.uuid4().hex[:6] if from_content else None
        caches = content_i not in CONTENT_NO_CACHE
        # How much of the context the provider holds in its prompt cache, and
        # when that was last touched. Nothing is cached before the first turn.
        cached, touched = 0, None
        # People step away; programs do not. A pause longer than the cache's
        # lifetime makes the next turn write the whole history again.
        pause_p = 0 if runaway else (0.05 if from_content else 0.12)
        for _turn in range(turns):
            if touched is not None and (t - touched).total_seconds() > CACHE_TTL:
                cached = 0
            # Fresh input is the new message. The history before it is read from
            # the cache where the cache still holds it, and the rest of the
            # history is written to the cache by this turn.
            fresh = int(random.lognormvariate(6.5, 1.0)) * mult
            if caches:
                cache_read = min(cached, context)
                cache_write = context - cache_read
            else:
                cache_read = cache_write = 0
                fresh += context
            tin = fresh + cache_write + cache_read
            status = random.choices([200, 429, 500], [96, 3, 1])[0]
            tout = 0 if status != 200 else int(random.lognormvariate(6.3, 1.0)) * mult
            finish = (
                "error" if status != 200
                else random.choices(
                    ["end_turn", "tool_use", "max_tokens"],
                    [55, 43, 2] if runaway or not from_content else [93, 5, 2],
                )[0]
            )
            duration_ms = int(random.lognormvariate(7.5, 0.8) * (1 + tin / 200_000))
            rows.append(
                dict(
                    start_time=t,
                    # Connect stamps ingest_time when it writes the record, which is
                    # when the call finished.
                    ingest_time=t + timedelta(milliseconds=duration_ms),
                    node_name="node-1",
                    provider=provider,
                    operation="chat",
                    request_model=model,
                    response_model=model if status == 200 and random.random() > 0.05 else "",
                    integration_guid=uuid.uuid4().hex[:8],
                    integration_type=provider,
                    integration_auth_type="Service Account" if service else "Viewer",
                    duration_ms=duration_ms,
                    status_code=status,
                    finish_reasons=[finish],
                    response_id=uuid.uuid4().hex,
                    server_address=random.choice(
                        ["api.anthropic.com", "bedrock-runtime.us-east-1.amazonaws.com"]
                        if provider != "openai"
                        else ["api.openai.com"]
                    ),
                    server_port=443,
                    input_tokens=tin,
                    output_tokens=tout,
                    cache_creation_input_tokens=cache_write,
                    cache_read_input_tokens=cache_read,
                    # Workbench: the signed-in user. Viewer content: the viewer.
                    # Service-account content: only the owner is known.
                    authenticated_username=None if from_content else u[0],
                    viewer_username=u[0] if from_content and not service else None,
                    content_owner_username=owner[0] if owner else None,
                    viewer_user_guid=u[2] if from_content and not service else None,
                    authenticated_user_guid=None if from_content else u[2],
                    content_owner_guid=owner[2] if owner else None,
                    viewer_email=u[1] if from_content and not service else None,
                    authenticated_email=None if from_content else u[1],
                    content_owner_email=owner[1] if owner else None,
                    content_guid=CONTENT[content_i] if content_i is not None else None,
                    content_id=content_i + 1 if content_i is not None else None,
                    job_key=job,
                    conversation_id=conv,
                    # Only a Workbench call carries request metadata: Posit Workbench
                    # sets the Gateway-Metadata header, Connect content
                    # does not. Connect rows stay NULL, as they do in a real export.
                    request_metadata=meta,
                    # The caller's User-Agent and the origin Connect derives from it.
                    user_agent=user_agent,
                    origin=origin,
                )
            )
            if caches:
                cached = context
            context = min(context + (fresh if caches else fresh - context) + tout, 180_000 * mult)
            touched = t + timedelta(milliseconds=duration_ms)
            pause = random.uniform(360, 2700) if random.random() < pause_p else random.uniform(*gap)
            t += timedelta(seconds=pause, milliseconds=duration_ms)
    return pd.DataFrame(rows)


def pick_client(from_content: bool) -> tuple[str | None, str | None]:
    """A User-Agent and the origin Connect's export would derive from it.

    Connect marks content "connect" whatever the header says. Otherwise only a
    Posit client names its host IDE (posit_workbench_<ide>/ in a Workbench
    session, posit_<ide>/ on a laptop); Claude, Copilot and Codex do not, so
    their requests have no origin.
    """
    r = random.random()
    if from_content:
        return random.choice([
            "r-ellmer/0.3.2", "r-ellmer/0.3.2", "python-httpx/0.27.2",
            "OpenAI/Python 1.54.0", "Anthropic/Python 0.39.0", "Python/3.11 aiohttp/3.9.5", None,
        ]), "connect"
    ide = random.choice(["positron", "rstudio"])
    if r < 0.38:
        return f"PositAssistant/1.2.3 posit_workbench_{ide}/2026.01.0", "workbench"
    if r < 0.46:
        return f"PositAssistant/1.2.3 posit_{ide}/2026.01.0", "local"
    if r < 0.62:
        return "claude-cli/2.0.14 (external, cli)", None
    if r < 0.72:
        return "GitHubCopilotChat/0.22.4 vscode/1.95.0", None
    if r < 0.80:
        return "codex_cli_rs/0.46.0 (Mac OS 15.1.0; arm64)", None
    if r < 0.86:
        return "pi-coding-agent/0.50.0", None
    if r < 0.90:
        return "r-ellmer/0.3.2", None
    if r < 0.95:
        return "Anthropic/Python 0.39.0", None
    if r < 0.97:
        return "curl/8.7.1", None
    return None, None


# The columns of Connect's usage export and their types, in order: the record
# columns, the derived origin, and the date/node partition columns (see
# connect/gateway/usagelog/export.go). tests/test_connect_export.py checks this
# against an export Connect itself built.
EXPORT_COLUMNS = {
    "start_time": "TIMESTAMPTZ", "ingest_time": "TIMESTAMPTZ", "node_name": "VARCHAR",
    "duration_ms": "BIGINT", "provider": "VARCHAR", "operation": "VARCHAR",
    "integration_guid": "VARCHAR", "integration_type": "VARCHAR",
    "integration_auth_type": "VARCHAR", "content_guid": "VARCHAR", "content_id": "BIGINT",
    "job_key": "VARCHAR", "authenticated_user_guid": "VARCHAR",
    "authenticated_username": "VARCHAR", "authenticated_email": "VARCHAR",
    "content_owner_guid": "VARCHAR", "content_owner_username": "VARCHAR",
    "content_owner_email": "VARCHAR", "viewer_user_guid": "VARCHAR",
    "viewer_username": "VARCHAR", "viewer_email": "VARCHAR", "conversation_id": "VARCHAR",
    "user_agent": "VARCHAR", "request_metadata": "JSON", "server_address": "VARCHAR",
    "server_port": "BIGINT", "status_code": "BIGINT", "request_model": "VARCHAR",
    "response_model": "VARCHAR", "response_id": "VARCHAR", "request_max_tokens": "BIGINT",
    "request_temperature": "DOUBLE", "request_top_p": "DOUBLE",
    "request_stop_sequences": "VARCHAR[]", "finish_reasons": "VARCHAR[]",
    "input_tokens": "BIGINT", "output_tokens": "BIGINT",
    "cache_creation_input_tokens": "BIGINT", "cache_read_input_tokens": "BIGINT",
    "origin": "VARCHAR", "date": "VARCHAR", "node": "VARCHAR",
}


def build_parquet() -> bytes:
    """The rows as Connect would export them: its columns, types and order.

    A column the fake does not generate is NULL. The partition columns come
    from where Connect files a record: its ingest day and its node.
    """
    df = make_rows()
    if df.columns.empty:  # FAKE_ROWS=0: DuckDB needs one column to read from
        df = pd.DataFrame({"start_time": pd.Series([], dtype="datetime64[ns, UTC]")})
    derived = {
        "date": "strftime(ingest_time AT TIME ZONE 'UTC', '%Y-%m-%d')",
        "node": "node_name",
    }
    cols = ", ".join(
        f'CAST({derived.get(c, c)} AS {t}) AS "{c}"' if c in df.columns or c in derived
        else f'NULL::{t} AS "{c}"'
        for c, t in EXPORT_COLUMNS.items()
    )
    path = Path(tempfile.mkdtemp()) / "export.parquet"
    duckdb.sql(
        f"COPY (SELECT {cols} FROM df) TO '{path}' (FORMAT parquet, COMPRESSION zstd)"
    )
    return path.read_bytes()


_TITLES = [
    "Sales Forecast", "Churn Explorer", "Support Triage Bot", "Quarterly Review",
    "Model Monitor", "Customer 360", "Docs Assistant", "Pricing Simulator",
    "Ops Dashboard", "Fraud Alerts", "Lead Scoring", "Inventory Planner",
]


# Connect groups. Each user is in their team's group; some are also in
# cross-cutting groups, so memberships overlap and group totals do not add up to
# the whole.
GROUPS = {name: uuid.uuid5(uuid.NAMESPACE_DNS, name).hex for name in
          _TEAMS + ["llm-pilot", "platform-admins", "contractors"]}


def group_members() -> dict[str, list[int]]:
    rnd = random.Random(5)
    out: dict[str, list[int]] = {g: [] for g in GROUPS}
    for i in range(len(USERS)):
        out[_TEAMS[i % len(_TEAMS)]].append(i)
        if i < 30 or rnd.random() < 0.15:
            out["llm-pilot"].append(i)
        if i < 6:
            out["platform-admins"].append(i)
        if i >= 120 and rnd.random() < 0.5:
            out["contractors"].append(i)
    return out


MEMBERS = group_members()


def paged(items: list, query: str) -> bytes:
    """One page of a Connect list: results, current_page and total.

    page_size is capped at PAGE_CAP, as Connect caps it, so a client must use
    total rather than assume a short page is the last one.
    """
    params = parse_qs(query)
    size = min(int(params.get("page_size", ["20"])[0]), PAGE_CAP)
    page = int(params.get("page_number", ["1"])[0])
    return json.dumps({
        "results": items[(page - 1) * size:page * size],
        "current_page": page,
        "total": len(items),
    }).encode()


def groups_list(query: str) -> bytes:
    return paged(
        [{"guid": g, "name": n, "owner_guid": None} for n, g in GROUPS.items()], query
    )


def group_members_json(guid: str, query: str) -> bytes | None:
    name = next((n for n, g in GROUPS.items() if g == guid), None)
    if name is None:
        return None
    users = [
        {"guid": USERS[i][2], "username": USERS[i][0], "email": USERS[i][1]}
        for i in MEMBERS[name]
    ]
    return paged(users, query)


def content_list() -> bytes:
    """What GET /v1/content returns: guid, name and title for each item."""
    items = []
    for i, guid in enumerate(CONTENT[:-1]):  # the last item is "deleted"
        title = _TITLES[i] if i < len(_TITLES) else ""
        items.append({"guid": guid, "name": f"app-{guid[:6]}", "title": title})
    return json.dumps(items).encode()


PRICING = (Path(__file__).parent / "pricing.json").read_bytes()
OVERRIDES = json.dumps({"overrides": []}).encode()
EXPORT = "/__api__/v1/system/gateway/usage/export"


ROUTES = {
    EXPORT: "export",
    "/__api__/v1/system/gateway/pricing": "pricing",
    "/__api__/v1/system/gateway/models": "overrides",
    "/__api__/v1/content": "content",
    "/__api__/v1/groups": "groups",
}


class Handler(BaseHTTPRequestHandler):
    payload = b""
    faults: dict[str, str] = {}

    def _fault(self, mode: str) -> None:
        kind, _, arg = mode.partition(":")
        if kind in ("status", "html"):
            code = int(arg)
            if kind == "status":
                body, ctype = f"injected {code}".encode(), "text/plain"
            else:
                body = (f"<html>\n<head><title>{code} Error</title></head>\n<body>\n"
                        f"<h1>Bad gateway</h1>\n{'<p>proxy said no</p>' * 40}\n"
                        "</body></html>").encode()
                ctype = "text/html"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif kind == "garbage":
            self._send(b"<html>this is not parquet</html>", "text/html")
        elif kind == "truncate":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload[:1000])
            self.close_connection = True
        else:
            self.send_error(500, f"unknown fault {mode}")

    def _control(self, query: str) -> None:
        params = parse_qs(query)
        if "reset" in params:
            self.faults.clear()
        else:
            self.faults[params["route"][0]] = params["mode"][0]
        self._send(b"ok", "text/plain")

    def _authorized(self) -> bool:
        """Connect answers a missing or malformed key with 401; so does this."""
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Key ") and auth[4:].strip():
            return True
        self.send_error(401, "API key required")
        return False

    def _send(self, body: bytes, content_type: str, head: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head: bool) -> None:
        if not self._authorized():
            return
        path, _, query = self.path.partition("?")
        key = ROUTES.get(path) or (
            "members" if path.startswith("/__api__/v1/groups/") and path.endswith("/members")
            else None)
        if key in self.faults:
            self._fault(self.faults[key])
            return
        if path == EXPORT:
            params = parse_qs(query)
            missing = [p for p in ("from", "to", "as_of") if p not in params]
            if missing:
                self.send_error(400, f"missing {', '.join(missing)}")
                return
            self._send(self.payload, "application/octet-stream", head)
        elif path == "/__api__/v1/system/gateway/pricing":
            self._send(PRICING, "application/json", head)
        elif path == "/__api__/v1/system/gateway/models":
            self._send(OVERRIDES, "application/json", head)
        elif path == "/__api__/v1/content":
            self._send(content_list(), "application/json", head)
        elif path == "/__api__/v1/groups":
            self._send(groups_list(query), "application/json", head)
        elif path.startswith("/__api__/v1/groups/") and path.endswith("/members"):
            body = group_members_json(path.split("/")[4], query)
            if body is None:
                self.send_error(404)
            else:
                self._send(body, "application/json", head)
        else:
            self.send_error(404)

    def do_GET(self):  # noqa: N802
        self._route(head=False)

    def do_POST(self):  # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/__fault__":
            self._control(query)
        else:
            self.send_error(404)

    def do_HEAD(self):  # noqa: N802
        self._route(head=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    Handler.payload = build_parquet()
    print(f"mock Connect on http://localhost:{PORT} ({len(Handler.payload)//1024} KB parquet)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
