"""The running app, driven as a browser drives it (see tests/session.py).

What these cover is the wiring app.py does between the pages: the filter every
page reads, the bookmark in the URL, the unpriced-model warning, and the CSV
downloads. Each test opens its own session on one `shiny run` server, started
against the fake Connect.
"""

from __future__ import annotations

import csv
import io
import json
import re
from urllib.parse import parse_qs, quote, urlsplit

import pytest

from session import Reply, Session, browser_inputs, fetch_page, open_session, run

# The outputs each page renders on first view. Principals shows nothing until a
# principal is picked, and Conversations' detail nothing until a row is, so they
# are not listed.
PAGE_OUTPUTS = {
    "Overview": ["kpis", "provider_total", "model_time", "origin_total", "origin_share",
                 "provider_share", "model_share"],
    "Adoption": ["adoption_kpis", "adoption_lines", "adoption_table", "tool_time",
                 "usage_heatmap"],
    "Models": ["model_kpis", "models_bars", "models_adoption", "cache_savings_time",
               "cache_rate_time", "latency_time", "errors_time", "models_table"],
    "Attribution": ["attr_kpis", "attr_bars", "attr_scatter", "attr_table"],
    "Conversations": ["conv_kpis", "conv_spend_by", "conv_spend_note", "top_sessions"],
    "Raw Data": ["records_table", "records_title"],
    "Shell": ["nav_rail", "topbar_meta", "topbar_status"],
}


def visit(app_server: str, scenario, url: str = "", **inputs):
    """Open the page at `url` (a query string), run `scenario(session, first_reply, html)`, and return its result."""
    # A bookmark is restored into the page's HTML, and the browser reports those values.
    html = fetch_page(app_server, url)

    async def go():
        async with open_session(app_server) as session:
            first = await session.init({**browser_inputs(html), **inputs}, url)
            return await scenario(session, first, html)

    return run(go())


def tiles(reply: Reply, output: str) -> dict[str, tuple[str, str]]:
    """The stat tiles of a KPI output: label -> (value, note)."""
    html = reply.values[output]["html"]
    found = re.findall(
        r'<div class="tile-label">(.*?)</div>\s*<div class="tile-value[^"]*"[^>]*>(.*?)</div>'
        r'\s*<div class="tile-note">(.*?)</div>',
        html, re.DOTALL,
    )
    clean = lambda s: re.sub(r"<bslib-popover.*?</bslib-popover>|<[^>]+>", "", s).strip()  # noqa: E731
    return {clean(label): (clean(value), clean(note)) for label, value, note in found}


def pill_texts(reply: Reply) -> list[str]:
    return re.findall(r'<span class="pill[^"]*"[^>]*>(.*?)</span>',
                      reply.values["topbar_status"]["html"])


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def bookmark_query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


async def first_view(session: Session, first: Reply, html: str) -> Reply:
    """The scenario that does nothing: the reply to opening the page."""
    return first


def test_every_page_renders_without_errors(app_server):
    reply = visit(app_server, first_view)
    assert reply.errors == {}
    assert reply.notifications() == []
    for page, outputs in PAGE_OUTPUTS.items():
        for output_id in outputs:
            assert reply.values.get(output_id) is not None, f"{page}: {output_id} has no value"


def test_the_records_grid_lists_every_row_newest_first(app_server):
    reply = visit(app_server, first_view)
    grid = reply.values["records_table"]["payload"]
    assert grid["columns"][:3] == ["ts", "origin", "tool"]
    timestamps = [row[0] for row in grid["data"]]
    assert timestamps == sorted(timestamps, reverse=True)
    assert reply.values["records_title"] == f"Gateway records ({len(timestamps):,})"
    assert reply.values["records_capped"] is None  # under the grid's cap


@pytest.mark.parametrize("models", [[], ["claude-sonnet-4-6", "gpt-5"]], ids=["all", "two-models"])
def test_spend_and_tokens_agree_across_pages(app_server, loaded, f, models):
    """The same filter gives the same totals on the Overview, the top bar and every CSV."""
    import filters
    import pricing

    async def scenario(session: Session, first: Reply, html: str):
        reply = first if not models else await session.update(f_models=models)
        return {
            "reply": reply,
            "records": session.download("dl_records"),
            "models": session.download("dl_models"),
        }

    got = visit(app_server, scenario)
    reply: Reply = got["reply"]
    assert reply.errors == {}
    if models:
        f["models"] = models
    where = filters.where(f)
    total = loaded.scalar(f"SELECT SUM({pricing.est_cost()}) FROM gateway_events WHERE {where}")
    requests = loaded.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {where}")
    assert total > 0

    # Overview tile and top bar show the same whole dollars.
    spend, note = tiles(reply, "kpis")["Estimated spend"]
    assert spend == f"${total:,.0f}"
    assert note == f"{requests:,} requests"
    assert f"{spend} est. spend" in pill_texts(reply)
    assert f"{requests:,}" in reply.values["topbar_meta"]["html"]

    # The Raw Data CSV: every request, each cost rounded to 4 places.
    records = csv_rows(got["records"])
    assert len(records) == requests
    assert sum(float(r["est_cost_usd"] or 0) for r in records) == pytest.approx(total, abs=0.1)
    assert sum(int(r["tok_in"] or 0) for r in records) == loaded.scalar(
        f"SELECT SUM(input_tokens) FROM gateway_events WHERE {where}")

    # The Models CSV leaves out requests that name no model, and nothing else.
    named = f"{where} AND model IS NOT NULL AND model <> ''"
    by_model = csv_rows(got["models"])
    assert sum(float(r["est_spend_usd"]) for r in by_model) == pytest.approx(loaded.scalar(
        f"SELECT SUM({pricing.est_cost()}) FROM gateway_events WHERE {named}"))
    assert sum(int(r["requests"]) for r in by_model) == loaded.scalar(
        f"SELECT COUNT(*) FROM gateway_events WHERE {named}")
    assert sum(int(float(r["output_tokens"])) for r in by_model) == sum(
        int(r["tok_out"] or 0) for r in records if r["model"])
    if models:
        assert {r["model"] for r in by_model} == set(models)


def test_csv_downloads_are_registered_and_streamed(app_server):
    async def scenario(session: Session, first: Reply, html: str):
        return first, {k: session.download(k)
                       for k in ("dl_adoption", "dl_models", "dl_attr", "dl_conversations",
                                 "dl_records")}

    reply, files = visit(app_server, scenario)
    assert sorted(k for k in reply.values if k.startswith("dl_")) == [
        "dl_adoption", "dl_attr", "dl_conversations", "dl_models", "dl_pf_conv_groups",
        "dl_pf_records", "dl_pf_rel_0", "dl_pf_rel_1", "dl_records",
    ]
    for name, body in files.items():
        rows = csv_rows(body)
        assert rows, f"{name} is empty"
    assert "est_spend_usd" in files["dl_attr"].splitlines()[0]
    assert files["dl_conversations"].splitlines()[0].startswith("conversation_id")


def test_unpriced_models_are_warned_about_in_the_top_bar(app_server, loaded, f):
    import filters
    import pricing

    unpriced = loaded.q(f"""SELECT model, COUNT(*) requests FROM gateway_events
        WHERE {filters.where(f)} AND model IS NOT NULL AND model <> ''
        GROUP BY 1 HAVING SUM({pricing.est_cost()}) = 0 AND SUM(input_tokens + output_tokens) > 0""")
    assert len(unpriced) == 1, "the fake Connect should leave exactly one model unpriced"
    model, requests = unpriced.model[0], int(unpriced.requests[0])

    async def scenario(session: Session, first: Reply, html: str):
        priced = [m for m in f["all_models"] if m != model][:1]
        return first, await session.update(f_models=priced)

    everything, priced_only = visit(app_server, scenario)
    html = everything.values["topbar_status"]["html"]
    assert "\u26a0 1 unpriced model" in html
    assert f"<code>{model}</code>" in html
    assert f"{requests:,} requests" in html
    # Narrowed to a priced model, there is nothing to warn about.
    assert "unpriced" not in priced_only.values["topbar_status"]["html"]


class TestBookmark:
    """The URL is the app's only saved state; see _BOOKMARKABLE_INPUTS in app.py."""

    # Inputs a page widget reports about itself. Shiny's bookmark would put every
    # one of them in the URL, and a long enough URL is refused with a 414.
    WIDGET_STATE = {
        "records_table_data_view_rows": list(range(2000)),
        "records_table_cell_selection": {"type": "none"},
        "models_table_selected_rows": list(range(500)),
        "model_time_state": {"layout": "x" * 3000},
        "attr_table_data_view_selected": list(range(500)),
    }

    def test_the_url_names_filter_inputs_only_and_stays_short(self, app_server):
        async def scenario(session: Session, first: Reply, html: str):
            return await session.update(main_tabs="Models", **self.WIDGET_STATE)

        reply = visit(app_server, scenario)
        (url,) = reply.query_strings()
        assert set(bookmark_query(url)) == {
            "_inputs_", "preset", "grain", "attr_kind", "pf_kind", "date_range",
            "f_users", "f_contents", "f_sessions", "f_providers", "f_models", "f_meta",
            "dark_mode", "main_tabs",
        }
        assert len(url) < 500  # a server refuses a request line past 8 KB; stay far inside
        assert json.loads(bookmark_query(url)["main_tabs"][0]) == "Models"

    def test_url_state_round_trips(self, app_server):
        picks = ["claude-opus-4-8", "claude-haiku-4-5"]

        async def save(session: Session, first: Reply, html: str):
            reply = await session.update(main_tabs="Models", f_models=picks, preset="7d",
                                         date_range=["2026-10-10", "2026-10-16"])
            return reply

        saved = visit(app_server, save)
        (url,) = saved.query_strings()
        query = bookmark_query(url)
        assert json.loads(query["f_models"][0]) == picks
        assert json.loads(query["preset"][0]) == "7d"

        # A new visitor opens the saved address, sending the page's own defaults.
        restored = visit(app_server, first_view, url="?" + urlsplit(url).query)
        assert restored.errors == {}
        assert restored.values["topbar_meta"]["html"] == saved.values["topbar_meta"]["html"]
        assert pill_texts(restored) == pill_texts(saved)
        assert tiles(restored, "kpis") == tiles(saved, "kpis")
        assert re.search(r'nav-item active"[^>]*>Models</button>', restored.values["nav_rail"]["html"])

    def test_a_bookmark_from_a_renamed_tab_still_opens_a_page(self, app_server):
        query = f"?_inputs_&main_tabs={quote(json.dumps('Audit log'))}"

        reply = visit(app_server, first_view, url=query)
        assert reply.errors == {}
        assert len(re.findall(r'nav-item active"', reply.values["nav_rail"]["html"])) == 1

    def test_a_bookmark_naming_no_tab_opens_the_first_page(self, app_server):
        query = f"?_inputs_&main_tabs={quote(json.dumps('Nonesuch'))}"

        reply = visit(app_server, first_view, url=query)
        assert reply.errors == {}
        assert re.search(r'nav-item active"[^>]*>Overview</button>', reply.values["nav_rail"]["html"])

    def test_clear_all_empties_the_filters_and_the_url(self, app_server):
        async def scenario(session: Session, first: Reply, html: str):
            narrowed = await session.update(f_models=["gpt-5"])
            cleared = await session.update(clear_all=1)
            return narrowed, cleared

        narrowed, cleared = visit(app_server, scenario)
        assert " of " in narrowed.values["topbar_meta"]["html"]
        assert cleared.query_strings()[-1] == "?"
        # The server tells the browser to empty each control; the browser then
        # reports the new values, as it would for a person clearing them.
        updates = [m for msg in cleared.messages for m in msg.get("inputMessages", [])]
        emptied = {m["id"] for m in updates if m["message"].get("value") == []}
        assert {"f_models", "f_providers", "f_meta"} <= emptied
