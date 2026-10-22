"""LLM audit dashboard for the Connect AI Gateway.

Single data source: `gateway_events` — one row per gateway LLM call, metadata
only (see `usage.py`). A call comes from one of two places, and the sidebar
picks which of them to report:

  - Connect    the call ran inside deployed content, so it carries a content
               guid. It groups by content OR by user.
  - Workbench  the call carries no content guid, so it came from a Workbench
               session or a direct API caller. It groups by user.
  - Both       everything, grouped by user, so one person's Connect and
               Workbench usage add up.

The user axis is the principal `usage.py` derives from the three identities the
gateway records: the viewer the content acted for, else the person whose
credentials authenticated the call, else the content owner. `user_kind` names
which of the three a row was charged to.

Layout follows one question per row. Each row asks a question and answers it
on two dimensions: per day, and in total.

Spend is tokens × the price catalog Connect serves (see est_cost). There is
no built-in price list, so the app prices requests the same way whatever
server it reads from.

Run:  .venv/bin/shiny run app:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from starlette.requests import Request

import principals
import theme
import usage
from pages import adoption, attribution, models, overview, raw_data
from pages import conversations as conversations_page
from pages.context import PageContext
from filters import TOKEN_COUNT, dim_sql, sql_str, where
from pricing import est_cost, est_cost_uncached
from table_format import GUID_LABEL_CHARS
from theme import apply, card_usd, color, empty_fig, fmt_num, fmt_usd
from ui_shell import (
    FAVICON, NAV_FIRST, NAV_SECTIONS, _drawer, _shell, brand,
    filter_label, model_choices, nav_rail_ui, plural,
)

PRESETS = {
    # Calendar month to date, not a rolling 30 days.
    "mtd": "MTD",
    "7d": "7D",
    # 30 days is the longest window on offer, and the default.
    "30d": "30D",
    # Not a chip anyone clicks: it is what the preset falls back to when the
    # dates are edited by hand. CSS hides it.
    "custom": "Custom",
}

DEFAULT_PRESET = "30d"


def catalog() -> dict:
    """What the loaded data covers. Re-read after every refresh."""
    return {
        "start": usage.scalar("SELECT MIN(day) FROM gateway_events").date(),
        "end": usage.scalar("SELECT MAX(day) FROM gateway_events").date(),
        "models": usage.q("SELECT DISTINCT model FROM gateway_events ORDER BY 1")[
            "model"
        ].tolist(),
        "providers": usage.q("SELECT DISTINCT provider FROM gateway_events ORDER BY 1")[
            "provider"
        ].tolist(),
        "users": usage.q("""SELECT DISTINCT username FROM gateway_events
                         WHERE username IS NOT NULL ORDER BY 1""")["username"].tolist(),
        # Choice labels for the user picker: the username, and the email beside
        # it when it adds something, so typing either finds the person.
        "user_labels": {
            r.username: (
                f"{r.username} \u00b7 {r.email}"
                if r.email and r.email != r.username
                else r.username
            )
            for r in usage.rows(usage.q("""SELECT username, MAX(user_email) AS email
                                FROM gateway_events WHERE username IS NOT NULL
                                GROUP BY 1 ORDER BY 1"""))
        },
        # Workbench sessions: the id is the value, the name (when there is one)
        # the label, so typing either finds the session.
        "session_labels": {
            r.sid: (f"{r.name} \u00b7 {r.sid}" if r.name and r.name != r.sid else r.sid)
            for r in usage.rows(usage.q("""SELECT workbench_session_id AS sid,
                                      MAX(workbench_session_name) AS name
                               FROM gateway_events
                               WHERE workbench_session_id IS NOT NULL
                               GROUP BY 1 ORDER BY 2, 1"""))
        },
        "contents": usage.q("""SELECT DISTINCT content_guid FROM gateway_events
                            WHERE content_guid IS NOT NULL ORDER BY 1""")[
            "content_guid"
        ].tolist(),
        # The export has no content title; Connect's content list supplies it.
        # Typing a title, an owner or a guid finds the item.
        "content_labels": {
            r.guid: " \u00b7 ".join(
                p for p in (usage.content_name(r.guid), r.owner, r.guid) if p
            )
            for r in usage.rows(usage.q("""SELECT content_guid AS guid,
                                      MAX(content_owner_username) AS owner
                               FROM gateway_events WHERE content_guid IS NOT NULL
                               GROUP BY 1 ORDER BY 1"""))
        },
        "meta": meta_choices(),
        "statuses": usage.q("""SELECT DISTINCT CAST(status_code AS VARCHAR) status
                            FROM gateway_events WHERE status_code IS NOT NULL
                            ORDER BY 1""")["status"].tolist(),
    }


# Request metadata is the caller's own JSON (the Gateway-Metadata header).
# Keys prefixed "workbench-" are written by Workbench itself (session and
# project ids, names and paths), one value per session or project, so they
# make poor filters. Every other key was
# chosen by whoever set the metadata, and those are offered as filters.
WORKBENCH_PREFIX = "workbench-"
# A key with more distinct values than this is an identifier, not a category.
META_MAX_VALUES = 50


def meta_id(key: str, value: str) -> str:
    return json.dumps([key, value])


def meta_choices() -> dict[str, str]:
    """{choice id: label} for every user-defined metadata key/value pair."""
    df = usage.q(
        """SELECT k AS key, json_extract_string(request_metadata, '$."' || k || '"') AS v
           FROM (SELECT request_metadata, unnest(json_keys(request_metadata)) AS k
                 FROM gateway_events WHERE request_metadata IS NOT NULL)
           WHERE NOT starts_with(k, 'workbench-') AND v IS NOT NULL
           GROUP BY 1, 2 ORDER BY 1, 2"""
    )
    out: dict[str, str] = {}
    for key, grp in df.groupby("key", sort=True):
        if len(grp) > META_MAX_VALUES:
            continue
        for v in grp["v"]:
            out[meta_id(str(key), v)] = f"{key}: {v}"
    return out


# The first load runs here, when the app is imported, because the UI below is
# built from what was loaded. usage itself has no import side effects.
usage.load()
CATALOG = catalog()  # seeds the input choices; the server tracks it reactively


def measure_of(f: dict) -> dict:
    """The selected unit of account, as SQL plus its display conventions."""
    if f["measure"] == "tokens":
        return {
            "expr": TOKEN_COUNT,
            "fmt": fmt_num,
            "card_fmt": fmt_num,
            "axis": "tokens / day",
            "total_axis": "tokens",
            "tick": ",.0f",
            "prefix": "",
            "noun": "tokens",
        }
    return {
        "expr": est_cost(),
        "fmt": fmt_usd,
        "card_fmt": card_usd,
        "axis": "est. USD / day",
        "total_axis": "est. USD",
        "tick": ",.2f",
        "prefix": "$",
        "noun": "estimated spend",
    }


def chart_labels(dim: str, values) -> list[str]:
    """Axis and legend labels for one chart category.

    A content guid is 36 characters and crowds a bar axis, so a chart shows
    only its first block. Hover text, tables and CSV exports keep the whole
    guid, because that is what an operator pastes into Connect.

    Two guids that share a prefix would collapse into one axis category and
    silently merge their bars, so the short form is used only when it stays
    unique.
    """
    full = [str(v) for v in values]
    if dim != "content_guid":
        return full
    # A content item's title reads better than any slice of its guid. Items
    # that share a title, or have none, keep a guid fragment so two bars never
    # collapse into one category.
    named = [usage.content_name(v) for v in full]
    labels = [n or v[:GUID_LABEL_CHARS] for n, v in zip(named, full)]
    counts = {lab: labels.count(lab) for lab in set(labels)}
    return [
        f"{lab} ({v[:6]})" if counts[lab] > 1 and n else lab
        for lab, n, v in zip(labels, named, full)
    ]



def fitted_grid(df: pd.DataFrame, scroll_after: int = 12, **kw) -> render.DataGrid:
    """A grid sized to its rows, that scrolls once there are many of them.

    A fixed height leaves a two-row table stranded in a tall blank box, and no
    height at all lets a long one run the page. Switching on the row count
    gives a tight card either way.
    """
    return render.DataGrid(
        df, height=None if len(df) <= scroll_after else "420px", **kw
    )


# How each numeric grid column is written for the eye, keyed by the name the
# query gives it. Money is shown to the dollar because cents are noise next to a
# monthly estimate, and counts get thousands separators.
DISPLAY_FMT = {
    # card_usd: whole dollars, but cents under $1, so 30 cents is not "$0"
    "est_spend_usd": card_usd,
    "tokens": lambda v: f"{v:,.0f}",
    "input_tokens": lambda v: f"{v:,.0f}",
    "output_tokens": lambda v: f"{v:,.0f}",
    "cache_read_tokens": lambda v: f"{v:,.0f}",
    "cache_pct": lambda v: f"{v:,.1f}%",
    "cache_savings_usd": card_usd,
    "error_pct": lambda v: f"{v:,.1f}%",
    "pct_of_total": lambda v: f"{v:,.1f}%",
    "p95_ms": lambda v: f"{v:,.0f}",
    "per_request": lambda v: f"${v:,.4f}",
    "x_median": lambda v: f"{v:,.1f}\u00d7",
    "requests": lambda v: f"{v:,.0f}",
    "weekend_requests": lambda v: f"{v:,.0f}",
    "calls": lambda v: f"{v:,.0f}",
    "lever_savings_usd": fmt_usd,
    "conversations": lambda v: f"{v:,.0f}",
    "cache_hit_pct": lambda v: f"{v:,.0f}%",
    "peak_input": fmt_num,
    "peak_p75": lambda v: f"{v:,.0f}",
}


def formatted(df: pd.DataFrame) -> pd.DataFrame:
    """A display copy of the frame, with numbers rendered as strings.

    Only what is drawn is touched: the CSV exports read the unformatted frames.
    """
    out = df.copy()
    for col, fmt in DISPLAY_FMT.items():
        if col in out.columns:
            # missing() is NaN and inf too: a ratio over zero is blank, not "inf".
            out[col] = [None if theme.missing(v) else fmt(v) for v in out[col]]
    return out



def preset_range(preset: str, start: date, end: date) -> tuple[date, date]:
    """The dates a preset covers, resolved against the loaded window.

    "This month" is the calendar month of the last day of data, not of the
    wall clock: an export that stops part-way through a month would otherwise
    give an empty range on the first of the next one.
    """
    lo, hi = {
        "mtd": (end.replace(day=1), end),
        "7d": (end - timedelta(days=6), end),
        "30d": (end - timedelta(days=29), end),
    }.get(preset, (start, end))
    # Never earlier than the data, so a chip and the date picker agree on it.
    return max(lo, start), hi



def app_ui(request: Request | None):
    return _shell(
        overview.panel(),
        attribution.panel(),
        conversations_page.panel(),
        principals.panel(),
        adoption.panel(),
        models.panel(),
        raw_data.panel(),
        sidebar=_drawer(
            ui.div(
                ui.input_selectize(
                    "f_users",
                    filter_label("Users", "users"),
                    CATALOG["user_labels"],
                    multiple=True,
                    options={"placeholder": "all users \u2014 type a name or email"},
                ),
                ui.input_action_link("clear_users", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            ui.div(
                ui.input_selectize(
                    "f_contents",
                    filter_label("Content", "content"),
                    CATALOG["content_labels"],
                    multiple=True,
                    options={"placeholder": "all content \u2014 type a title, owner or guid"},
                ),
                ui.input_action_link("clear_contents", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            ui.div(
                ui.input_selectize(
                    "f_sessions",
                    filter_label("Session", "session"),
                    CATALOG["session_labels"],
                    multiple=True,
                    options={"placeholder": "all sessions \u2014 type a name or id"},
                ),
                ui.input_action_link("clear_sessions", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            # Provider and Models both start with nothing checked, which shows
            # everything: checking boxes narrows the view, and "clear" gets back
            # to all of it in one click.
            ui.div(
                ui.input_checkbox_group(
                    "f_providers",
                    filter_label("Provider", "provider"),
                    CATALOG["providers"],
                ),
                ui.input_action_link("clear_providers", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            # A checkbox group, like Provider: the model list is short enough to
            # show in full, and seeing every value at once is what tells a reader
            # what the data covers. A search box hid that behind typing.
            ui.div(
                ui.input_checkbox_group(
                    "f_models",
                    filter_label("Models", "models"),
                    model_choices(CATALOG["models"]),
                ),
                ui.input_action_link("clear_models", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            # Metadata the caller set on the request (team, cost-center, ...).
            # Keys Workbench writes itself (workbench-*) are not offered.
            ui.div(
                ui.div(
                    ui.input_checkbox_group(
                        "f_meta", filter_label("Custom Metadata", "meta"), CATALOG["meta"]
                    ),
                    class_="filter-meta",
                ),
                ui.input_action_link("clear_meta", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            # A bookmarked URL is the only state this app keeps between visits,
            # so "clear" has to empty the query string as well as the controls —
            # otherwise a reload would restore what was just cleared.
            ui.input_action_button(
                "clear_all", "Clear all filters", class_="btn-sm clear-all-btn"
            ),
            width=290,
        ),
        id="main_tabs",
        header=ui.head_content(
            ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
            ui.tags.link(
                rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""
            ),
            ui.tags.link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
            ),
            # Relative, not "/favicon.svg": on Connect the app is served under a
            # content path, and a rooted href would look for the favicon at the
            # server root instead. The route is mounted in `App(...)` below.
            ui.tags.link(rel="icon", type="image/svg+xml", href="favicon.svg"),
            ui.include_css(Path(__file__).parent / "www" / "styles.css"),
            ui.include_css(Path(__file__).parent / "www" / "shell.css"),
            # A tiny label that follows the mouse over bars and donut slices.
            # Plotly's own tooltip anchors at the end of a bar and lands on its
            # value, so those charts turn it off (hoverinfo none) and mark
            # themselves meta "nametip".
            ui.tags.script(
                """
                (function () {
                  var tip = document.createElement("div");
                  tip.className = "nametip";
                  // Beside the pointer, centred on it, with the arrow on the
                  // near edge. Past the right edge of the window it flips to
                  // the pointer's left, arrow on its right edge.
                  // A hidden label measures 0x0, so it is placed again once
                  // shown, from the last pointer position.
                  var GAP = 16, mx = 0, my = 0;
                  function place() {
                    var w = tip.offsetWidth, h = tip.offsetHeight;
                    var flip = mx + GAP + w > window.innerWidth - 8;
                    tip.classList.toggle("nametip-flip", flip);
                    tip.style.left = (flip ? mx - GAP - w : mx + GAP) + "px";
                    tip.style.top = (my - h / 2) + "px";
                  }
                  document.addEventListener("mousemove", function (e) {
                    mx = e.clientX; my = e.clientY;
                    place();
                  });
                  function wire(gd) {
                    if (gd._nametip || !gd.on) return;
                    if (!gd.layout || gd.layout.meta !== "nametip") return;
                    gd._nametip = true;
                    gd.on("plotly_hover", function (ev) {
                      var p = ev.points && ev.points[0];
                      var cd = p && (p.customdata !== undefined ? p.customdata
                        : p.data && p.data.customdata && p.data.customdata[p.pointNumber !== undefined ? p.pointNumber : p.i]);
                      var t = p && (typeof cd === "string" ? cd
                        : (p.label !== undefined ? p.label : p.y));
                      if (t === undefined) return;
                      if (!tip.parentNode) document.body.appendChild(tip);
                      tip.textContent = t;
                      tip.style.display = "block";
                      place();
                    });
                    gd.on("plotly_unhover", function () { tip.style.display = "none"; });
                  }
                  new MutationObserver(function () {
                    document.querySelectorAll(".js-plotly-plot").forEach(wire);
                  }).observe(document.documentElement, { childList: true, subtree: true });
                })();
                """
            ),
            # The grid holds only its most recent rows, so its own footer
            # ("of 2000") would read as the total. When the server marks the
            # table as capped, add a "+": "of 2000+". The text node is edited in
            # place, which keeps the grid's own re-renders working.
            ui.tags.script(
                """
                (function () {
                  function patch() {
                    if (!document.getElementById("records-capped")) return;
                    var s = document.querySelector("#records_table .shiny-data-grid-summary");
                    var n = s && s.firstChild;
                    if (n && n.nodeType === 3 && /of [\\d,]+$/.test(n.nodeValue)) n.nodeValue += "+";
                  }
                  new MutationObserver(patch).observe(document.documentElement,
                    { childList: true, subtree: true, characterData: true });
                })();
                """
            ),
        ),
        title=brand(),
        catalog=CATALOG,
        presets=PRESETS,
        default_preset=DEFAULT_PRESET,
        preset_range=preset_range,
        fillable=False,
    )


def server(input, output, session):
    # Integrate bookmarking callbacks
    @session.bookmark.on_bookmarked
    async def _(url: str):
        await session.bookmark.update_query_string(url)

    # A bookmark saved before a page was renamed names a tab that no longer
    # exists; restoring it would select nothing and break the load.
    _LEGACY_TABS = {"Audit log": "Raw Data"}

    @session.bookmark.on_restore
    def _fix_stale_tab(state):
        tab = state.input.get("main_tabs")
        if tab is None:
            return
        tab = _LEGACY_TABS.get(tab, tab)
        if tab not in {n for _, names in NAV_SECTIONS for n in names}:
            tab = NAV_FIRST
        state.input["main_tabs"] = tab

    _BOOKMARKABLE_INPUTS = {
        "dark_mode",
        "preset",
        "date_range",
        "attr_kind",
        "pf_kind",
        "pf_pick",
        "grain",
        "f_users",
        "f_contents",
        "f_sessions",
        "f_providers",
        "f_models",
        "f_meta",
        "main_tabs",
    }

    # Trigger bookmarking
    @reactive.effect
    @reactive.event(input.main_tabs, ignore_init=True)
    async def _():
        session.bookmark.exclude = [
            k for k in dir(input) if k not in _BOOKMARKABLE_INPUTS
        ]
        await session.bookmark()

    @render.ui
    def nav_rail():
        return nav_rail_ui(input.main_tabs() or NAV_FIRST)

    # Not `reactive.event(..., ignore_init=True)`: `nav_go` does not exist until
    # the first click, and an event that skips its first run never reads it, so
    # it would never learn to listen. Reading it bare raises "not yet set",
    # which cancels the run and still subscribes.
    @reactive.effect
    def _nav_go():
        ui.update_navset("main_tabs", selected=input.nav_go())

    # The chips set the dates; editing the dates by hand hands the chip back to
    # "Custom" (or to whichever chip those dates happen to match).
    @reactive.effect
    @reactive.event(input.preset, ignore_init=True)
    def _preset_to_dates():
        if input.preset() == "custom":
            return
        c = cat()
        s, e = preset_range(input.preset(), c["start"], c["end"])
        if tuple(input.date_range() or ()) != (s, e):
            ui.update_date_range("date_range", start=s, end=e)

    @reactive.effect
    @reactive.event(input.date_range, ignore_init=True)
    def _dates_to_preset():
        c = cat()
        got = tuple(input.date_range() or ())
        match = next(
            (
                k
                for k in PRESETS
                if k != "custom" and preset_range(k, c["start"], c["end"]) == got
            ),
            "custom",
        )
        if match != input.preset():
            ui.update_radio_buttons("preset", selected=match)

    @render.ui
    def topbar_meta():
        info = loaded()
        selected = int(usage.q(f"SELECT COUNT(*) n FROM gateway_events WHERE {where(flt())}").n[0])
        # Only say "of" when a filter actually narrows the records.
        records = (
            f"{selected:,} of {info['rows']:,} records"
            if selected != info["rows"]
            else f"{info['rows']:,} records"
        )
        return ui.span(
            f"{records} \u00b7 loaded {info['loaded_at']:%H:%M} UTC",
            class_="topbar-meta",
        )

    @render.ui
    def topbar_status():
        info, c, f = loaded(), cat(), flt()
        r = usage.q(f"""SELECT COUNT(DISTINCT username) users,
                            COUNT(DISTINCT content_guid) contents,
                            COUNT(DISTINCT workbench_session_id) sessions,
                            COALESCE(SUM({est_cost()}),0) est
                     FROM gateway_events WHERE {where(f)}""").iloc[0]
        return ui.div(
            ui.span(f"{int(r.users):,} {plural(r.users, 'user')}", class_="pill"),
            ui.span(
                f"{int(r.contents):,} content {plural(r.contents, 'item')}",
                class_="pill",
            ),
            # Only when the data has sessions at all: "0 sessions" on a server
            # whose Workbench sends no session ids reads as "nobody used one".
            ui.span(
                f"{int(r.sessions):,} {plural(r.sessions, 'session')}",
                class_="pill",
            ) if c["session_labels"] else None,
            ui.span(f"{card_usd(r.est)} est. spend", class_="pill"),
            unpriced_pill(),
            class_="topbar-right",
        )

    # bumped by a refresh; every query depends on it through flt()
    loaded = reactive.value(usage.INFO.copy())

    @reactive.calc
    def cat() -> dict:
        loaded()
        return catalog()

    # The reload runs in a worker thread: it waits on Connect's export for tens of
    # seconds, and on the event loop that froze the session with no sign that
    # anything was happening. The task button reads "Refreshing…" meanwhile, and
    # usage.load holds its own locks, so pages keep querying the old table.
    @ui.bind_task_button(button_id="refresh")
    @reactive.extended_task
    async def reload_task() -> dict:
        return await asyncio.to_thread(usage.load)

    @reactive.effect
    # ignore_init, or opening the dashboard would re-read the export every time
    @reactive.event(input.refresh, ignore_init=True)
    def _refresh():
        reload_task()

    @reactive.effect
    @reactive.event(reload_task.status, ignore_init=True)
    def _refreshed():
        status = reload_task.status()
        if status == "error":
            # the reload rolled back, so the loaded data is still intact
            ui.notification_show(
                f"Refresh failed — showing the data from "
                f"{loaded()['loaded_at']:%H:%M UTC}. {reload_task.error.get()}",
                type="error",
                duration=15,
            )
            return
        if status != "success":
            return
        info = reload_task.value.get()
        c = catalog()
        # A selection that no longer exists in the reloaded data is dropped; an
        # empty result means no narrowing, the same as the initial state.
        ui.update_checkbox_group(
            "f_providers",
            choices=c["providers"],
            selected=[p for p in input.f_providers() or [] if p in c["providers"]],
        )
        ui.update_checkbox_group(
            "f_models",
            choices=model_choices(c["models"]),
            selected=[m for m in input.f_models() or [] if m in c["models"]],
        )
        ui.update_selectize(
            "f_users",
            choices=c["user_labels"],
            selected=[u for u in input.f_users() or [] if u in c["users"]],
        )
        ui.update_selectize(
            "f_sessions",
            choices=c["session_labels"],
            selected=[x for x in input.f_sessions() or [] if x in c["session_labels"]],
        )
        ui.update_selectize(
            "f_contents",
            choices=c["content_labels"],
            selected=[x for x in input.f_contents() or [] if x in c["contents"]],
        )
        ui.update_checkbox_group(
            "f_meta",
            choices=c["meta"],
            selected=[m for m in input.f_meta() or [] if m in c["meta"]],
        )
        # Bounds follow the new window; the chip's dates are re-resolved against it.
        ui.update_date_range("date_range", min=c["start"], max=c["end"])
        if input.preset() != "custom":
            s, e = preset_range(input.preset(), c["start"], c["end"])
            ui.update_date_range("date_range", start=s, end=e)
        loaded.set(info.copy())
        what = (
            f"{info['fetched']:,} new or updated"
            if info.get("mode") == "incremental"
            else "full window re-read"
        )
        ui.notification_show(
            f"Loaded {info['rows']:,} records ({what})", type="message", duration=4
        )

    @reactive.effect
    @reactive.event(input.clear_all)
    async def _clear_all():
        """Everything back to the state a first-time visitor sees, URL included.

        Dark mode is left alone: it is how someone reads the page, not part of
        what the page is showing.
        """
        ui.update_radio_buttons("preset", selected=DEFAULT_PRESET)
        c = cat()
        s, e = preset_range(DEFAULT_PRESET, c["start"], c["end"])
        ui.update_date_range("date_range", start=s, end=e)
        ui.update_selectize("f_users", selected=[])
        ui.update_selectize("f_contents", selected=[])
        ui.update_selectize("f_sessions", selected=[])
        ui.update_checkbox_group("f_providers", selected=[])
        ui.update_checkbox_group("f_models", selected=[])
        ui.update_checkbox_group("f_meta", selected=[])
        await session.bookmark.update_query_string("?")

    @reactive.effect
    @reactive.event(input.clear_models)
    def _clear_models():
        ui.update_checkbox_group("f_models", selected=[])

    @reactive.effect
    @reactive.event(input.clear_providers)
    def _clear_providers():
        ui.update_checkbox_group("f_providers", selected=[])

    @reactive.effect
    @reactive.event(input.clear_meta)
    def _clear_meta():
        ui.update_checkbox_group("f_meta", selected=[])

    @reactive.effect
    @reactive.event(input.clear_users)
    def _clear_users():
        ui.update_selectize("f_users", selected=[])

    @reactive.effect
    @reactive.event(input.clear_sessions)
    def _clear_sessions():
        ui.update_selectize("f_sessions", selected=[])

    @reactive.effect
    @reactive.event(input.clear_contents)
    def _clear_contents():
        ui.update_selectize("f_contents", selected=[])

    @reactive.calc
    def flt() -> dict:
        c = cat()
        dates = input.date_range()
        if dates and dates[0] and dates[1]:
            start, end = dates
        else:
            start, end = preset_range(input.preset(), c["start"], c["end"])
        return {
            "start": start,
            "end": end,
            # Dollars only: the Count toggle was redundant, since tokens sit beside
            # spend in every table and tile that needs them.
            "measure": "spend",
            # Origin is not a filter: Connect and Workbench are always both in view,
            # and the Overview breaks the spend down by origin instead.
            "source": "both",
            # Always grouped by user; there is no control for it.
            "dim": "username",
            "all_providers": c["providers"],
            "all_models": c["models"],
            "all_users": c["users"],
            "all_contents": c["contents"],
            "providers": list(input.f_providers() or []),
            "models": list(input.f_models() or []),
            "users": list(input.f_users() or []),
            "contents": list(input.f_contents() or []),
            "sessions": list(input.f_sessions() or []),
            "meta": list(input.f_meta() or []),
            "mode": input.dark_mode() or "dark",
        }

    def total_bars(
        f, m, dim: str, color_kind: str | None, limit: int = 15, height: int = 320,
        detail: str | None = None,
    ) -> go.Figure:
        """Category totals, biggest last so the bars read bottom-up.

        Origin and harness keep their NULL rows as a named bucket (see
        filters.dim_sql), so the bars add up to the spend on the page.
        `detail` names a second column whose split each bar's hover lists, e.g.
        the providers behind one model.
        """
        k = dim_sql(dim)
        df = usage.q(f"""SELECT {k} AS k, SUM({m["expr"]}) v, COUNT(*) req
                         FROM gateway_events WHERE {where(f)} AND {k} IS NOT NULL
                         GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}""").iloc[::-1]
        df = df[df.v > 0]
        if df.empty:
            return empty_fig(f["mode"], f"No {m['noun']} recorded in this view")
        colors = (
            [color(color_kind, k, f["mode"]) for k in df.k]
            if color_kind
            else [theme.TOKENS[f["mode"]]["series"][0]] * len(df)
        )
        split: dict[str, str] = {}
        if detail:
            d = usage.q(f"""SELECT {k} AS k, {dim_sql(detail)} AS d, SUM({m["expr"]}) v
                            FROM gateway_events WHERE {where(f)} AND {k} IS NOT NULL
                            GROUP BY 1, 2 ORDER BY 3 DESC""")
            for key, grp in d.groupby("k", sort=False):
                split[str(key)] = "".join(
                    f"\n\u2022 {name}: {m['fmt'](v)}" for name, v in zip(grp.d, grp.v)
                )
        # The colours key off the raw value, the axis off the short label, and
        # hover reads the full value out of customdata rather than off the axis.
        fig = go.Figure(
            go.Bar(
                x=df.v,
                y=chart_labels(dim, df.k),
                orientation="h",
                width=0.55,
                marker={"color": colors},
                text=["\u00a0\u00a0" + m["fmt"](v) for v in df.v],
                textposition="outside",
                textfont={"color": theme.TOKENS[f["mode"]]["ink"], "size": 15},
                customdata=[
                    f"{lab}\n{m['fmt'](v)}{split.get(str(key), '')}"
                    for lab, key, v in zip(chart_labels(dim, df.k), df.k, df.v)
                ],
                hoverinfo="none",  # the name follows the mouse; see nametip below
            )
        )
        fig.update_layout(meta="nametip")
        fig.update_xaxes(range=[0, df.v.max() * 1.25], title_text=m["total_axis"])
        apply(fig, f["mode"], height=height, showlegend=False, hover=True)
        return theme.usd_axis(fig) if m["prefix"] == "$" else fig

    # ---- unpriced checks: the topbar and pages/conversations.py share them ---
    def unpriced_pill():
        """Topbar warning for models with no price, or None. One place, every page."""
        df = unpriced_df()
        if df.empty:
            return None
        n, reqs = len(df), int(df.requests.sum())
        shown = df.head(10)
        rows = [
            ui.tags.li(ui.tags.code(r.model), f" \u00b7 {int(r.requests):,} {plural(r.requests, 'request')}")
            for r in usage.rows(shown)
        ]
        if n > len(shown):
            rows.append(ui.tags.li(f"and {n - len(shown)} more"))
        return ui.popover(
            ui.span(
                f"\u26a0 {n:,} unpriced {plural(n, 'model')}",
                class_="pill pill-warn",
                role="button",
                tabindex="0",
            ),
            ui.strong(f"Unpriced {plural(n, 'model')}"),
            ui.p(
                f"No price is available, so {'it counts' if n == 1 else 'they count'} "
                f"as $0 in spend ({reqs:,} {plural(reqs, 'request')}). The "
                "Conversations page leaves them out.",
                class_="mt-2",
            ),
            ui.tags.ul(*rows, class_="mb-0 ps-3"),
            placement="bottom",
        )

    @reactive.calc
    def unpriced_df() -> pd.DataFrame:
        """Models that carried tokens but cost $0 because the catalog has no price."""
        f = flt()
        return usage.q(f"""SELECT model, COUNT(*) requests
                         FROM gateway_events
                         WHERE {where(f)} AND model IS NOT NULL AND model <> ''
                         GROUP BY 1
                         HAVING SUM({est_cost()}) = 0 AND SUM({TOKEN_COUNT}) > 0
                         ORDER BY 2 DESC""")

    def priced_only() -> str:
        """A WHERE fragment that leaves out requests to unpriced models.

        A request with no model (a failed call, usually) is kept: it names no
        model to price, and the NULL test stops NOT IN from dropping it.
        """
        df = unpriced_df()
        if df.empty:
            return ""
        names = ", ".join(sql_str(m) for m in df.model)
        return f" AND (model IS NULL OR model NOT IN ({names}))"

    # ---- exports ------------------------------------------------------------
    def register_export(name: str, df_fn, filename: str):
        def _export():
            yield df_fn().to_csv(index=False)

        # the downloads registry keys off the function name, so name it to match the button id
        _export.__name__ = f"dl_{name}"
        renderer = render.download_button(filename=filename)(_export)
        output(id=f"dl_{name}", suspend_when_hidden=False)(renderer)

    def register_sql_export(name: str, sql_fn, filename: str):
        """A CSV download for a query that can return a very large result.

        DuckDB writes the file and it is streamed out in 1 MB pieces, so the
        whole export is never held in memory.
        """

        def _export():
            path = usage.export_csv(sql_fn())
            try:
                with open(path, "rb") as fh:
                    while chunk := fh.read(1 << 20):
                        yield chunk
            finally:
                path.unlink(missing_ok=True)

        _export.__name__ = f"dl_{name}"
        renderer = render.download_button(filename=filename)(_export)
        output(id=f"dl_{name}", suspend_when_hidden=False)(renderer)

    page_ctx = PageContext(
        flt=flt, where=where, cat=cat, est_cost=est_cost,
        register_export=register_export, register_sql_export=register_sql_export,
        est_cost_uncached=est_cost_uncached, total_bars=total_bars,
        measure_of=measure_of, formatted=formatted, chart_labels=chart_labels,
        fitted_grid=fitted_grid, priced_only=priced_only,
    )
    overview.server(input, page_ctx)
    records_sql = raw_data.server(input, page_ctx)

    open_profile = principals.server(input, page_ctx, records_sql)

    # After Principals: picking an outlier opens it there.
    attribution.server(input, page_ctx, open_profile)
    adoption.server(input, page_ctx)
    models.server(input, page_ctx)
    conversations_page.server(input, page_ctx)


app = App(
    app_ui,
    server,
    bookmark_store="url",
    # One file, not the whole of www/: the stylesheet is inlined into the head and
    # the logo is read at render time, so the favicon is the only asset a browser
    # has to fetch by URL.
    static_assets={"/favicon.svg": FAVICON},
)
