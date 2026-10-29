"""Usage dashboard for the Posit AI Gateway.

Single data source: `gateway_events` — one row per gateway LLM call, metadata
only (see `usage.py`). Every page reports Connect content and Workbench calls
together; the Overview breaks the spend down by origin.

The user axis is the principal `usage.py` derives from the three identities the
gateway records: the viewer the content acted for, else the person whose
credentials authenticated the call, else the content owner. `user_kind` names
which of the three a row was charged to.

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
from shiny import App, reactive, render, ui
from starlette.requests import Request

import access
import principals
import queries
import usage
from pages import adoption, overview, raw_data
from pages.context import PageContext
from filters import Filters
from theme import card_usd
from ui_shell import (
    FAVICON, NAV_FIRST, NAV_SECTIONS, _drawer, _shell, brand,
    filter_label, nav_rail_ui, plural,
)

PRESETS = {
    # Calendar month to date, not a rolling 30 days.
    "mtd": "MTD",
    "7d": "7D",
    # 30 days is the default; 90 days is the longest window on offer.
    "30d": "30D",
    "90d": "90D",
    # Not a chip anyone clicks: it is what the preset falls back to when the
    # dates are edited by hand. CSS hides it.
    "custom": "Custom",
}

DEFAULT_PRESET = "30d"


def catalog() -> dict:
    """What the loaded data covers. Re-read after every refresh."""
    start, end = queries.data_window()
    return {
        "start": start,
        "end": end,
        "users": queries.usernames()["username"].tolist(),
        # Choice labels for the user picker: the username, and the email beside
        # it when it adds something, so typing either finds the person.
        "user_labels": {
            r.username: (
                f"{r.username} \u00b7 {r.email}"
                if r.email and r.email != r.username
                else r.username
            )
            for r in usage.rows(queries.user_emails())
        },
        # Workbench sessions: the id is the value, the name (when there is one)
        # the label, so typing either finds the session.
        "session_labels": {
            r.sid: (f"{r.name} \u00b7 {r.sid}" if r.name and r.name != r.sid else r.sid)
            for r in usage.rows(queries.sessions())
        },
        "contents": queries.content_guids()["content_guid"].tolist(),
        # The export has no content title; Connect's content list supplies it.
        # Typing a title, an owner or a guid finds the item.
        "content_labels": {
            r.guid: " \u00b7 ".join(
                p for p in (usage.content_name(r.guid), r.owner, r.guid) if p
            )
            for r in usage.rows(queries.content_owners())
        },
        # Connect groups: the guid is the value, the name and size the label.
        "group_labels": {
            r.guid: f"{r.name} \u00b7 {r.members:,} members"
            for r in usage.rows(queries.groups())
        },
        "meta": meta_choices(),
        "statuses": queries.status_codes()["status"].tolist(),
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
    df = queries.metadata_pairs()
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


def fitted_grid(df: pd.DataFrame, **kw) -> render.DataGrid:
    """A grid sized to its rows, that scrolls once there are many of them.

    A fixed height leaves a two-row table stranded in a tall blank box, and no
    height at all lets a long one run the page. Switching on the row count
    gives a tight card either way.
    """
    return render.DataGrid(df, height=None if len(df) <= 12 else "420px", **kw)



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
        "90d": (end - timedelta(days=89), end),
    }.get(preset, (start, end))
    # Never earlier than the data, so a chip and the date picker agree on it.
    return max(lo, start), hi



# The drawer's filters: the input is f_<name> and its clear link clear_<name>.
# Each has the function that updates it and the choices it takes from a catalog.
FILTERS = {
    "users": (ui.update_selectize, lambda c: c["user_labels"]),
    "contents": (ui.update_selectize, lambda c: c["content_labels"]),
    "sessions": (ui.update_selectize, lambda c: c["session_labels"]),
    "groups": (ui.update_selectize, lambda c: c["group_labels"]),
    "meta": (ui.update_checkbox_group, lambda c: c["meta"]),
}


def with_clear(name: str, control: ui.Tag) -> ui.Tag:
    """A drawer filter with the link that clears it."""
    return ui.div(
        control,
        ui.input_action_link(f"clear_{name}", "clear", class_="clear-link"),
        class_="filter-with-clear",
    )


def app_ui(request: Request | None):
    reason = access.denial(request.headers) if request else None
    if reason:
        return ui.page_fluid(ui.h4("Access denied"), ui.p(reason))
    return _shell(
        overview.panel(),
        principals.panel(),
        adoption.panel(),
        raw_data.panel(),
        sidebar=_drawer(
            with_clear("users", ui.input_selectize(
                "f_users",
                filter_label("Users", "users"),
                CATALOG["user_labels"],
                multiple=True,
                options={"placeholder": "all users \u2014 type a name or email"},
            )),
            with_clear("contents", ui.input_selectize(
                "f_contents",
                filter_label("Content", "content"),
                CATALOG["content_labels"],
                multiple=True,
                options={"placeholder": "all content \u2014 type a title, owner or guid"},
            )),
            with_clear("sessions", ui.input_selectize(
                "f_sessions",
                filter_label("Session", "session"),
                CATALOG["session_labels"],
                multiple=True,
                options={"placeholder": "all sessions \u2014 type a name or id"},
            )),
            with_clear("groups", ui.input_selectize(
                "f_groups",
                filter_label("Connect groups", "groups"),
                CATALOG["group_labels"],
                multiple=True,
                options={"placeholder": "all groups \u2014 type a group name"},
            )),
            # Metadata the caller set on the request (team, cost-center, ...).
            # Keys Workbench writes itself (workbench-*) are not offered.
            with_clear("meta", ui.div(
                ui.input_checkbox_group(
                    "f_meta", filter_label("Custom Metadata", "meta"), CATALOG["meta"]
                ),
                class_="filter-meta",
            )),
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
            # Dark only: Bootstrap and styles.css read the theme from this
            # attribute, set before the page paints so it never flashes light.
            ui.tags.script("document.documentElement.setAttribute('data-bs-theme', 'dark');"),
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
            # A loading bar over the page until the first views are drawn. The
            # first session is slow; shiny:idle fires once the server has no
            # work left, which is when the page is ready to look at.
            ui.tags.script(
                """
                document.addEventListener("DOMContentLoaded", function () {
                  var o = document.createElement("div");
                  o.className = "first-load";
                  o.innerHTML = '<div class="first-load-bar"></div><div>Loading usage data\u2026</div>';
                  document.body.appendChild(o);
                  function hide() {
                    o.classList.add("done");
                    setTimeout(function () { o.remove(); }, 400);
                  }
                  $(document).one("shiny:idle shiny:disconnected", hide);
                });
                """
            ),
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
    if access.denial(session.http_conn.headers):
        return

    # Integrate bookmarking callbacks
    @session.bookmark.on_bookmarked
    async def _(url: str):
        await session.bookmark.update_query_string(url)

    # A bookmark saved before a page was renamed or folded into another names a
    # tab that no longer exists. The page is already built by the time the
    # bookmark is restored, and the browser falls back to the first tab, so the
    # browser is told which tab to show instead.
    _LEGACY_TABS = {"Audit log": "Raw Data", "Models": "Overview",
                    "Profiles": "Attribution", "Top Spenders": "Attribution",
                    "Principals": "Attribution"}

    @session.bookmark.on_restored
    def _fix_stale_tab(state):
        tab = state.input.get("main_tabs")
        if tab is None or tab in {n for _, names in NAV_SECTIONS for n in names}:
            return
        ui.update_navset("main_tabs", selected=_LEGACY_TABS.get(tab, NAV_FIRST))

    _BOOKMARKABLE_INPUTS = {
        "preset",
        "date_range",
        "pf_kind",
        "pf_pick",
        "grain",
        "group_by",
        "f_users",
        "f_contents",
        "f_sessions",
        "f_groups",
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
        selected = queries.record_count(flt())
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

    # The spend in view, on every page. The counts of users, content and
    # sessions are on the Overview.
    @render.ui
    def topbar_status():
        return ui.div(
            ui.span(f"{card_usd(queries.total_spend(flt()))} est. spend", class_="pill"),
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
        for name, (update, choices_of) in FILTERS.items():
            choices = choices_of(c)
            update(
                f"f_{name}",
                choices=choices,
                selected=[x for x in input[f"f_{name}"]() or [] if x in choices],
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
        """Everything back to the state a first-time visitor sees, URL included."""
        ui.update_radio_buttons("preset", selected=DEFAULT_PRESET)
        c = cat()
        s, e = preset_range(DEFAULT_PRESET, c["start"], c["end"])
        ui.update_date_range("date_range", start=s, end=e)
        for name, (update, _) in FILTERS.items():
            update(f"f_{name}", selected=[])
        await session.bookmark.update_query_string("?")

    def clear_link(name: str, update) -> None:
        @reactive.effect
        @reactive.event(input[f"clear_{name}"])
        def _clear():
            update(f"f_{name}", selected=[])

    for name, (update, _) in FILTERS.items():
        clear_link(name, update)

    @reactive.calc
    def flt() -> Filters:
        c = cat()
        dates = input.date_range()
        if dates and dates[0] and dates[1]:
            # A date input can arrive as text, from a bookmark or a plain client.
            start, end = (date.fromisoformat(d) if isinstance(d, str) else d for d in dates)
        else:
            start, end = preset_range(input.preset(), c["start"], c["end"])
        return Filters(
            start=start,
            end=end,
            all_users=tuple(c["users"]),
            all_contents=tuple(c["contents"]),
            groups=tuple(input.f_groups() or ()),
            users=tuple(input.f_users() or ()),
            contents=tuple(input.f_contents() or ()),
            sessions=tuple(input.f_sessions() or ()),
            meta=tuple(input.f_meta() or ()),
        )

    # ---- unpriced check: the topbar warning ----------------------------------
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
                f"as $0 in spend ({reqs:,} {plural(reqs, 'request')}).",
                class_="mt-2",
            ),
            ui.tags.ul(*rows, class_="mb-0 ps-3"),
            placement="bottom",
        )

    @reactive.calc
    def unpriced_df() -> pd.DataFrame:
        """Models that carried tokens but cost $0 because the catalog has no price."""
        return queries.unpriced_models(flt())

    # ---- exports ------------------------------------------------------------
    def register_export(name: str, df_fn, filename: str):
        def _export():
            yield df_fn().to_csv(index=False)

        # the downloads registry keys off the function name, so name it to match the button id
        _export.__name__ = f"dl_{name}"
        renderer = render.download_button(filename=filename)(_export)
        output(id=f"dl_{name}", suspend_when_hidden=False)(renderer)

    def register_sql_export(name: str, query_fn, filename: str):
        """A CSV download for a query that can return a very large result.

        DuckDB writes the file and it is streamed out in 1 MB pieces, so the
        whole export is never held in memory.
        """

        def _export():
            path = usage.export_csv(query_fn())
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
        flt=flt, cat=cat,
        register_export=register_export, register_sql_export=register_sql_export,
        fitted_grid=fitted_grid,
    )
    overview.server(input, page_ctx)
    raw_data.server(input, page_ctx)

    principals.server(input, page_ctx)
    adoption.server(input, page_ctx)


app = App(
    app_ui,
    server,
    bookmark_store="url",
    # One file, not the whole of www/: the stylesheet is inlined into the head and
    # the logo is read at render time, so the favicon is the only asset a browser
    # has to fetch by URL.
    static_assets={"/favicon.svg": FAVICON},
)
