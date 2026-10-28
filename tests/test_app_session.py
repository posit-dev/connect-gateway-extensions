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

from session import Reply, Session, browser_inputs, fetch_page, figure_traces, open_session, run

# The outputs each page renders on first view. Attribution shows nothing until a
# principal is picked, so it is not listed.
PAGE_OUTPUTS = {
    "Overview": ["kpis", "spend_time", "cache_savings_time",
                 "cache_rate_time", "errors_time"],
    "Adoption": ["adoption_kpis", "adoption_lines", "tool_time", "model_time"],
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


def test_adoption_content_toggle_relabels_the_page(app_server):
    async def scenario(session: Session, first: Reply, html: str):
        return await session.update(adoption_kind="content")

    reply = visit(app_server, scenario)
    assert reply.errors == {}
    assert reply.values["adoption_lines_title"] == "Active content items over time"
    assert "Weekly active content items" in reply.values["adoption_kpis"]["html"]
    for out in ("adoption_lines", "tool_time"):
        assert reply.values.get(out) is not None, out


def test_overview_starts_ungrouped_as_one_series_of_the_whole_spend(app_server):
    reply = visit(app_server, first_view)
    assert reply.errors == {}
    (trace,) = figure_traces(reply, "spend_time")
    assert trace["name"] == "Estimated spend"


def test_the_records_grid_lists_every_row_newest_first(app_server):
    reply = visit(app_server, first_view)
    grid = reply.values["records_table"]["payload"]
    assert grid["columns"][:3] == ["ts", "origin", "tool"]
    timestamps = [row[0] for row in grid["data"]]
    assert timestamps == sorted(timestamps, reverse=True)
    assert reply.values["records_title"] == f"Gateway records ({len(timestamps):,})"
    assert reply.values["records_capped"] is None  # under the grid's cap


@pytest.mark.parametrize("narrow", [False, True], ids=["all", "two-groups"])
def test_spend_and_tokens_agree_across_pages(app_server, loaded, f, narrow):
    """The same filter gives the same totals on the Overview, the top bar and every CSV."""
    import filters
    import pricing

    groups = loaded.q("SELECT DISTINCT group_guid FROM user_groups ORDER BY 1").group_guid
    groups = groups.head(2).tolist() if narrow else []

    async def scenario(session: Session, first: Reply, html: str):
        reply = first if not groups else await session.update(f_groups=groups)
        return {
            "reply": reply,
            "records": session.download("dl_records"),
        }

    got = visit(app_server, scenario)
    reply: Reply = got["reply"]
    assert reply.errors == {}
    f["groups"] = groups
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


def test_refresh_reloads_in_the_background_and_says_so(app_server):
    """The reload is an extended task: the button reports busy, then the new data."""
    async def scenario(session: Session, first: Reply, html: str):
        reply = await session.update(refresh=1)
        # The task finishes in a worker thread, possibly after the app first
        # reports idle; its result comes in a busy/idle round of its own.
        if not reply.notifications():
            more = await session._drain()
            reply.messages += more.messages
        return reply

    # A browser reports the button at 0 on load; the click is the change to 1.
    reply = visit(app_server, scenario, refresh=0)
    assert reply.errors == {}
    notes = reply.notifications()
    assert any("Loaded" in n and "records" in n for n in notes), notes
    assert not any("Refresh failed" in n for n in notes)
    # bslib's task button is told it is busy, then ready again.
    states = [x["message"].get("state") for m in reply.messages
              for x in m.get("inputMessages") or [] if x.get("id") == "refresh"]
    assert states[:1] == ["busy"] and states[-1] == "ready", states


def test_csv_downloads_are_registered_and_streamed(app_server):
    async def scenario(session: Session, first: Reply, html: str):
        return first, {k: session.download(k)
                       for k in ("dl_adoption", "dl_records")}

    reply, files = visit(app_server, scenario)
    assert sorted(k for k in reply.values if k.startswith("dl_")) == [
        "dl_adoption", "dl_pf_records", "dl_pf_rel_0", "dl_pf_rel_1", "dl_records",
    ]
    for name, body in files.items():
        rows = csv_rows(body)
        assert rows, f"{name} is empty"


def test_unpriced_models_are_warned_about_in_the_top_bar(app_server, loaded, f):
    import filters
    import pricing

    unpriced = loaded.q(f"""SELECT model, COUNT(*) requests FROM gateway_events
        WHERE {filters.where(f)} AND model IS NOT NULL AND model <> ''
        GROUP BY 1 HAVING SUM({pricing.est_cost()}) = 0 AND SUM(input_tokens + output_tokens) > 0""")
    assert len(unpriced) == 1, "the fake Connect should leave exactly one model unpriced"
    model, requests = unpriced.model[0], int(unpriced.requests[0])

    # A user who never called the unpriced model.
    priced = loaded.q(f"""SELECT username FROM gateway_events WHERE {filters.where(f)}
        AND username IS NOT NULL GROUP BY 1
        HAVING COUNT(*) FILTER (WHERE model = '{model}') = 0 ORDER BY 1 LIMIT 1""").username

    async def scenario(session: Session, first: Reply, html: str):
        return first, await session.update(f_users=priced.tolist())

    everything, priced_only = visit(app_server, scenario)
    html = everything.values["topbar_status"]["html"]
    assert "\u26a0 1 unpriced model" in html
    assert f"<code>{model}</code>" in html
    assert f"{requests:,} requests" in html
    # Narrowed to someone who only used priced models, there is nothing to warn about.
    assert "unpriced" not in priced_only.values["topbar_status"]["html"]


class TestBookmark:
    """The URL is the app's only saved state; see _BOOKMARKABLE_INPUTS in app.py."""

    # Inputs a page widget reports about itself. Shiny's bookmark would put every
    # one of them in the URL, and a long enough URL is refused with a 414.
    WIDGET_STATE = {
        "records_table_data_view_rows": list(range(2000)),
        "records_table_cell_selection": {"type": "none"},
        "spend_time_state": {"layout": "x" * 3000},
    }

    def test_the_url_names_filter_inputs_only_and_stays_short(self, app_server):
        async def scenario(session: Session, first: Reply, html: str):
            return await session.update(main_tabs="Adoption", **self.WIDGET_STATE)

        reply = visit(app_server, scenario)
        (url,) = reply.query_strings()
        assert set(bookmark_query(url)) == {
            "_inputs_", "preset", "grain", "group_by", "pf_kind", "pf_pick", "date_range",
            "f_users", "f_contents", "f_sessions", "f_groups", "f_meta",
            "main_tabs",
        }
        assert len(url) < 500  # a server refuses a request line past 8 KB; stay far inside
        assert json.loads(bookmark_query(url)["main_tabs"][0]) == "Adoption"

    def test_url_state_round_trips(self, app_server, loaded):
        picks = loaded.q("SELECT DISTINCT group_guid FROM user_groups ORDER BY 1"
                         ).group_guid.head(2).tolist()

        async def save(session: Session, first: Reply, html: str):
            reply = await session.update(main_tabs="Adoption", f_groups=picks, preset="7d",
                                         date_range=["2026-10-10", "2026-10-16"])
            return reply

        saved = visit(app_server, save)
        (url,) = saved.query_strings()
        query = bookmark_query(url)
        assert json.loads(query["f_groups"][0]) == picks
        assert json.loads(query["preset"][0]) == "7d"

        # A new visitor opens the saved address, sending the page's own defaults.
        restored = visit(app_server, first_view, url="?" + urlsplit(url).query)
        assert restored.errors == {}
        assert restored.values["topbar_meta"]["html"] == saved.values["topbar_meta"]["html"]
        assert pill_texts(restored) == pill_texts(saved)
        assert tiles(restored, "kpis") == tiles(saved, "kpis")
        assert re.search(r'nav-item active"[^>]*>Adoption</button>', restored.values["nav_rail"]["html"])

    @pytest.mark.parametrize("old, new", [
        ("Audit log", "Raw Data"), ("Models", "Overview"),
        ("Profiles", "Attribution"), ("Top Spenders", "Attribution"),
        ("Principals", "Attribution"),
    ])
    def test_a_bookmark_from_a_renamed_tab_still_opens_a_page(self, app_server, old, new):
        query = f"?_inputs_&main_tabs={quote(json.dumps(old))}"

        reply = visit(app_server, first_view, url=query)
        assert reply.errors == {}
        # The browser is told to switch to the page that took the old one's place.
        selected = [x["message"].get("value") for m in reply.messages
                    for x in m.get("inputMessages") or [] if x.get("id") == "main_tabs"]
        assert selected == [new]

    def test_a_bookmark_naming_no_tab_opens_the_first_page(self, app_server):
        query = f"?_inputs_&main_tabs={quote(json.dumps('Nonesuch'))}"

        reply = visit(app_server, first_view, url=query)
        assert reply.errors == {}
        assert re.search(r'nav-item active"[^>]*>Overview</button>', reply.values["nav_rail"]["html"])

    def test_clear_all_empties_the_filters_and_the_url(self, app_server, loaded):
        user = loaded.scalar("SELECT MIN(username) FROM gateway_events")

        async def scenario(session: Session, first: Reply, html: str):
            narrowed = await session.update(f_users=[user])
            cleared = await session.update(clear_all=1)
            return narrowed, cleared

        narrowed, cleared = visit(app_server, scenario)
        assert " of " in narrowed.values["topbar_meta"]["html"]
        assert cleared.query_strings()[-1] == "?"
        # The server tells the browser to empty each control; the browser then
        # reports the new values, as it would for a person clearing them.
        updates = [m for msg in cleared.messages for m in msg.get("inputMessages", [])]
        emptied = {m["id"] for m in updates if m["message"].get("value") == []}
        assert {"f_groups", "f_users", "f_meta"} <= emptied
