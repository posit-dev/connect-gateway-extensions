"""The app against a server whose callers send no User-Agent and no Gateway-Metadata.

That is what a real customer server looked like in review: nearly all spend
from people outside content, no harness, no Workbench session, and an origin
only on content rows. Every chart that dropped NULL origins or harnesses came
up empty or short of the headline total, and the session counts read as "0".
These drive the running app (see tests/session.py) against the fake Connect's
"sparse" profile.
"""

from __future__ import annotations

import re

import pytest

from session import Reply, Session, browser_inputs, fetch_page, figure_traces, open_session, run
from test_app_session import PAGE_OUTPUTS, pill_texts, tiles

# Text that means a missing number was formatted as if it were one.
BROKEN = re.compile(r"\$nan|\bnan%|\bNaN\b|\binf\b|\$-?inf|\bNone\b")


def visit(url: str, scenario):
    html = fetch_page(url)

    async def go():
        async with open_session(url) as session:
            first = await session.init(browser_inputs(html))
            return await scenario(session, first, html)

    return run(go())


def dollars(s: str) -> float:
    """$1,234.56 (or <$0.01) as a number."""
    return 0.0 if s.startswith("<") else float(s.replace("$", "").replace(",", ""))


def no_broken_text(reply: Reply) -> None:
    for out, value in reply.values.items():
        html = value.get("html") if isinstance(value, dict) else value
        if isinstance(html, str):
            assert not BROKEN.search(re.sub(r"<[^>]+>", " ", html)), f"{out}: {html[:300]}"


async def first_view(session: Session, first: Reply, html: str) -> Reply:
    return first


def test_every_page_renders_without_errors_or_nan(sparse_app_server):
    reply = visit(sparse_app_server, first_view)
    assert reply.errors == {}
    for page, outputs in PAGE_OUTPUTS.items():
        for output_id in outputs:
            assert reply.values.get(output_id) is not None, f"{page}: {output_id} has no value"
    no_broken_text(reply)


async def by_origin(session: Session, first: Reply, html: str) -> tuple[Reply, dict[str, float]]:
    """The first view, and the Overview's spend by origin."""
    reply = await session.update(group_by="source")
    return first, {t["name"]: sum(t["y"]) for t in figure_traces(reply, "spend_time")}


def test_origin_totals_add_up_to_the_spend_on_the_page(sparse_app_server):
    """Requests with no origin are a bucket of their own, not left out."""
    reply, spend = visit(sparse_app_server, by_origin)
    total = dollars(tiles(reply, "kpis")["Estimated spend"][0])
    assert set(spend) == {"(unidentified)", "connect"}
    # The tile is rounded to the dollar, so the sum is within a dollar of it.
    assert sum(spend.values()) == pytest.approx(total, abs=1)
    assert spend["(unidentified)"] > spend["connect"]


def test_sessions_say_not_reported_rather_than_zero(sparse_app_server):
    reply = visit(sparse_app_server, first_view)
    value, note = tiles(reply, "kpis")["Sessions"]
    assert (value, note) == ("\u2013", "not reported by Workbench")
    assert not any("session" in p for p in pill_texts(reply))


def test_content_average_is_content_spend_over_content_items(sparse_app_server):
    """It divided all of the spend by the content count: $3,769 per item for two items."""
    reply, labels = visit(sparse_app_server, by_origin)
    total = dollars(tiles(reply, "kpis")["Estimated spend"][0])
    items = int(tiles(reply, "kpis")["Content"][0].replace(",", ""))
    avg = dollars(tiles(reply, "kpis")["Content"][1].split(" avg")[0])
    connect = labels["connect"]
    assert avg < total / items
    assert avg == pytest.approx(connect / items, abs=0.01)


def test_harness_charts_keep_requests_with_no_user_agent(sparse_app_server):
    reply = visit(sparse_app_server, first_view)
    names = {t.get("name") for t in figure_traces(reply, "tool_time")}
    assert names == {"(no user agent)"}

    async def by_harness(session: Session, first: Reply, html: str):
        return await session.update(group_by="tool")

    reply = visit(sparse_app_server, by_harness)
    names = {t.get("name") for t in figure_traces(reply, "cache_savings_time")}
    assert names == {"(no user agent)"}


def test_principals_explain_a_kind_with_nobody_in_it(sparse_app_server):
    async def scenario(session: Session, first: Reply, html: str):
        return await session.update(main_tabs="Attribution", pf_kind="session")

    reply = visit(sparse_app_server, scenario)
    assert reply.errors == {}
    assert "No sessions in this view" in reply.values["pf_header"]["html"]
    assert "workbench-session-id" in reply.values["pf_header"]["html"]

