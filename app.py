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

Spend is tokens × the price catalog Connect serves (see est_cost); a fallback
table below prices the models when the server does not serve the catalog.

Run:  .venv/bin/shiny run app:app --port 8000
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from starlette.requests import Request

import theme
import usage
from theme import apply, card_num, card_usd, color, empty_fig, fmt_num, fmt_usd

# Fallback list-price lookup ($/Mtok: input, output, cache write, cache read).
# est_cost() prefers the live catalog Connect serves (usage.PRICES); this table
# is used only when the server does not serve the pricing endpoint or the fetch
# fails. Anthropic first-party list rates — a model missing from here costs $0
# and silently understates spend, so keep it in step with the models the
# gateway actually sees.
EST_PRICES = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-7": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-6": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-1": (15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-5": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "gpt-5": (1.25, 10.00, 0.0, 0.125),
    "gpt-5-mini": (0.25, 2.00, 0.0, 0.025),
}

# The export writes the provider as "anthropic" or "bedrock"; the price catalog
# names Bedrock "aws.bedrock". Match either so a provider-keyed rate still fires.
_PROVIDER_ALIASES = {"anthropic": ("anthropic",), "aws.bedrock": ("bedrock", "aws.bedrock")}


def _rate_term(i: float, o: float, cw: float, cr: float) -> str:
    """The spend for one row at these four $/Mtok rates, before the /1e6.

    float() coerces each rate, so a string or null price cannot enter the SQL as
    a bare token.
    """
    return (
        f"((input_tokens - cache_read_tokens - cache_creation_tokens) * {float(i)}"
        f" + cache_creation_tokens * {float(cw)} + cache_read_tokens * {float(cr)}"
        f" + output_tokens * {float(o)})"
    )


def _sql_str(s: str) -> str:
    """One value as a SQL string literal, with embedded quotes doubled.

    Provider and model names come from the pricing endpoint, so the CASE text
    must not assume they are quote-free.
    """
    return "'" + str(s).replace("'", "''") + "'"


def _build_est_cost(prices: dict | None) -> str:
    """The est_cost SQL for one price catalog (or the fallback table)."""
    if not prices:
        whens = " ".join(
            f"WHEN model = {_sql_str(m)} THEN {_rate_term(i, o, cw, cr)}"
            for m, (i, o, cw, cr) in EST_PRICES.items()
        )
        return f"CASE {whens} ELSE 0 END / 1e6"

    specific, model_default = [], {}
    for (provider, model), (i, o, cw, cr) in prices.items():
        names = ", ".join(
            _sql_str(a) for a in _PROVIDER_ALIASES.get(provider, (provider,))
        )
        specific.append(
            f"WHEN provider IN ({names}) AND model = {_sql_str(model)}"
            f" THEN {_rate_term(i, o, cw, cr)}"
        )
        if provider == "anthropic":
            model_default[model] = (i, o, cw, cr)
    # Model-only fallback, tried after every provider-keyed branch.
    fallback = [
        f"WHEN model = {_sql_str(m)} THEN {_rate_term(*r)}"
        for m, r in model_default.items()
    ]
    return f"CASE {' '.join(specific + fallback)} ELSE 0 END / 1e6"


# (prices object the SQL was built from, the SQL). The cache holds the object
# itself, not id(prices): a freed catalog can pass its id() to a new object and
# cause a wrong hit, but an `is` test on the object cannot.
_EST_COST_CACHE: tuple = (object(), "")


def est_cost() -> str:
    """Estimated spend per row, as a SQL expression over gateway_events.

    The SQL is rebuilt only when the catalog changes. A refresh replaces
    usage.PRICES with a new object, which misses the cache and forces one
    rebuild; every call in between reuses the string. With a catalog, the price
    comes from (provider, model), and a request whose provider is not priced
    falls back to the model's Anthropic rate, so a provider-name mismatch does
    not read $0. Without a catalog, the fallback table prices by model alone.
    """
    global _EST_COST_CACHE
    prices = usage.PRICES
    cached_prices, cached_sql = _EST_COST_CACHE
    if cached_prices is prices:
        return cached_sql
    sql = _build_est_cost(prices)
    _EST_COST_CACHE = (prices, sql)
    return sql


TOKEN_COUNT = "(input_tokens + output_tokens)"

PRESETS = {
    "all": "All time",
    # Calendar month, not a rolling 30 days: it is the window the spend
    # threshold is judged against, so it needs to be selectable on its own.
    "mtd": "This month",
    "7d": "Last 7 days",
    "30d": "Last 30 days",
    "90d": "Last 90 days",
}

MEASURES = {"spend": "Dollars", "tokens": "Tokens"}

# Where the call ran. Connect content always stamps a content guid; a Workbench
# session and a direct API caller never do. Both is the union, and it groups by
# user so one person's Connect and Workbench usage add up.
SOURCES = {"both": "Both", "connect": "Connect", "workbench": "Workbench"}

# Only Connect can group by content: a Workbench call has no content item, and a
# Both view has to use the axis the two sources share.
GROUP_BY = {"username": "User", "content_guid": "Content"}

GROUP_NOUN = {"username": "user", "content_guid": "content item"}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# The monthly spend limit the Attribution tab judges principals against, in USD.
# It is only a default: the sidebar makes it editable, because the number is a
# policy an organisation sets, not a property of the data. Warnings start at 80%.
DEFAULT_LIMIT_USD = 200

# A limit of zero would divide by zero in the "% of limit" column and make every
# principal infinitely over, so the input floors here.
MIN_LIMIT_USD = 1
RUNAWAY_USD = 25  # a single session costing more than this is flagged


def catalog() -> dict:
    """What the loaded data covers. Re-read after every refresh."""
    return {
        "start": usage.scalar("SELECT MIN(day) FROM gateway_events").date(),
        "end": usage.scalar("SELECT MAX(day) FROM gateway_events").date(),
        "month": usage.scalar("SELECT strftime(MAX(ts), '%Y-%m') FROM gateway_events"),
        "models": usage.q("SELECT DISTINCT model FROM gateway_events ORDER BY 1")[
            "model"
        ].tolist(),
        "providers": usage.q("SELECT DISTINCT provider FROM gateway_events ORDER BY 1")[
            "provider"
        ].tolist(),
        "users": usage.q("""SELECT DISTINCT username FROM gateway_events
                         WHERE username IS NOT NULL ORDER BY 1""")["username"].tolist(),
        "contents": usage.q("""SELECT DISTINCT content_guid FROM gateway_events
                            WHERE content_guid IS NOT NULL ORDER BY 1""")[
            "content_guid"
        ].tolist(),
        "statuses": usage.q("""SELECT DISTINCT CAST(status_code AS VARCHAR) status
                            FROM gateway_events WHERE status_code IS NOT NULL
                            ORDER BY 1""")["status"].tolist(),
    }


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


def plural(n, word: str) -> str:
    return f"{word}s" if n != 1 else word


def price_note(info: dict) -> str:
    """Where the spend estimate's prices came from, for the sidebar.

    `priced_models` is None when Connect served no catalog and est_cost fell
    back to the built-in table, so the reader can tell a live price from a
    stale one.
    """
    n = info.get("priced_models")
    if n is None:
        return "prices: built-in fallback (no live catalog)"
    return f"prices: {n} {plural(n, 'model')} from Connect catalog"


def fmt_limit(v: float) -> str:
    """The monthly limit as money.

    `fmt_usd` shows cents below $100, which is right for a measured amount and
    wrong for a policy number — "$50.00" implies a precision nobody set. Cents
    appear only if the operator typed some.
    """
    return f"${v:,.0f}" if float(v).is_integer() else f"${v:,.2f}"


def fmt_month(month: str) -> str:
    """A `YYYY-MM` key as a reader-facing month, e.g. `2026-09` -> `September 2026`.

    Anything that does not parse is passed through unchanged, so an unexpected
    key still shows up instead of raising in the middle of a render.
    """
    try:
        year, mon = (int(part) for part in str(month).split("-", 1))
        return date(year, mon, 1).strftime("%B %Y")
    except ValueError:
        return str(month)


# How many leading characters of a content guid a chart axis shows.
GUID_LABEL_CHARS = 8


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
    short = [v[:GUID_LABEL_CHARS] for v in full]
    return short if len(set(short)) == len(set(full)) else full


WWW = Path(__file__).parent.resolve() / "www"
LOGO = WWW / "Posit-Logos-2024_horiz-full-color.svg"
# Shiny requires an absolute path for a static asset route, hence the resolve().
FAVICON = WWW / "favicon.svg"


def brand() -> ui.Tag:
    """Navbar brand: the Posit word mark, then what this dashboard is.

    The SVG is inlined rather than linked. It needs no static asset route that
    way, and — the reason that matters — stylesheet rules can reach inside it,
    so the near-black word mark can be recolored for dark mode instead of
    disappearing into the navbar.

    Inlining also puts the file's own `<style>` block in the document's global
    cascade, and the file names its classes `cls-1`..`cls-4`. Plotly injects a
    style block using those same generic names for every widget it renders, and
    because that lands later in the document it repainted the word mark a pale
    lavender a moment after load. So the class names are namespaced on the way
    in — keep the prefix in step with the `.posit-logo-*` rules in styles.css.
    """
    svg = re.sub(r"cls-(\d+)", r"posit-logo-\1", LOGO.read_text().split("?>", 1)[-1])
    return ui.span(
        ui.span(ui.HTML(svg), class_="brand-logo", aria_hidden="true"),
        "Posit Team AI Usage",
        class_="brand",
    )


def model_choices(models: list[str]) -> dict[str, str]:
    """Checkbox labels for the model filter.

    The export carries records whose model name is an empty string — a failed
    call that never reached a model, usually. As a checkbox that would draw a
    blank row, so it is named. The value is left as-is so the filter still
    selects exactly those records.
    """
    return {m: (m if m else "(no model recorded)") for m in models}


def empty_note(msg: str) -> ui.Tag:
    """Why a table is empty. An empty grid alone reads as a broken panel."""
    return ui.div(msg, class_="empty-note")


def help_icon(title: str, *body) -> ui.Tag:
    """A "?" beside a heading that opens an explanation.

    A popover, not a tooltip: the content is a few sentences with markup, and a
    tooltip would vanish the moment the pointer left the icon. `role="button"`
    and `tabindex` are what make the span reachable by keyboard, since a span is
    not focusable on its own.
    """
    return ui.popover(
        ui.span(
            "?",
            class_="help-icon",
            role="button",
            tabindex="0",
            aria_label=title,
        ),
        ui.strong(title),
        *body,
        placement="right",
    )


def filter_chip(label_id: str, *body) -> ui.Tag:
    """A filter chip: a pill above a table that opens the values it filters on.

    The pill's text is an output rather than fixed markup, so the server can name
    the current selection and mark the chip as active. The menu inside is static
    on purpose: re-rendering it would tear down the checkboxes the reader is in
    the middle of clicking, and close the popover under them.
    """
    return ui.popover(
        ui.tags.button(
            ui.output_ui(label_id, inline=True), class_="chip-btn", type="button"
        ),
        *body,
        placement="bottom",
    )


def chip_label(field: str, picked: list[str], labels: dict | None = None) -> ui.Tag:
    """One chip's text: the field on its own, or the field and what it is set to.

    One or two values are named outright, since the name is the useful part. Past
    that the chip would outgrow the bar, so it says how many instead.
    """
    names = [str((labels or {}).get(v, v)) for v in picked]
    if not names:
        return ui.span(field, class_="chip")
    said = ", ".join(names) if len(names) <= 2 else f"{len(names)} selected"
    return ui.span(f"{field}: {said}", class_="chip chip-on")


def chip_menu(*body, note: str | None = None) -> ui.Tag:
    """The body of a chip's popover: the control, and optionally why it behaves so."""
    return ui.div(
        *body,
        ui.div(note, class_="chip-menu-note") if note else None,
        class_="chip-menu",
    )


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
    "est_spend_usd": lambda v: f"${v:,.0f}",
    "pct_of_limit": lambda v: f"{v:,.0f}%",
    "tokens": lambda v: f"{v:,.0f}",
    "requests": lambda v: f"{v:,.0f}",
    "weekend_requests": lambda v: f"{v:,.0f}",
    "calls": lambda v: f"{v:,.0f}",
}


def formatted(df: pd.DataFrame) -> pd.DataFrame:
    """A display copy of the frame, with numbers rendered as strings.

    Only what is drawn is touched: the CSV exports read the unformatted frames,
    and the threshold bands are still decided on the unrounded estimate, so a
    principal at $24.60 can read "$25" and still sit under the limit.
    """
    out = df.copy()
    for col, fmt in DISPLAY_FMT.items():
        if col in out.columns:
            out[col] = [None if pd.isna(v) else fmt(v) for v in out[col]]
    return out


def table_header(title: str, dl_id: str) -> ui.Tag:
    """Card header with a right-aligned CSV export button."""
    return ui.card_header(
        ui.div(
            ui.span(title),
            ui.download_button(dl_id, "Export CSV", class_="btn-sm export-btn"),
            class_="hdr-flex",
        )
    )


def question_row(question, ts_id: str, total_id: str) -> ui.Tag:
    """One question, answered per day on the left and in total on the right."""
    return ui.div(
        ui.h6(question, class_="row-question"),
        ui.layout_columns(
            ui.card(ui.card_header("Per day"), output_widget(ts_id)),
            ui.card(ui.card_header("Total"), output_widget(total_id)),
            col_widths=[7, 5],
        ),
        class_="question-block",
    )


def preset_range(preset: str, start: date, end: date) -> tuple[date, date]:
    """The dates a preset covers, resolved against the loaded window.

    "This month" is the calendar month of the last day of data, not of the
    wall clock: an export that stops part-way through a month would otherwise
    give an empty range on the first of the next one. It matches the month the
    threshold cards report on.
    """
    return {
        "mtd": (end.replace(day=1), end),
        "7d": (end - timedelta(days=6), end),
        "30d": (end - timedelta(days=29), end),
        "90d": (end - timedelta(days=89), end),
    }.get(preset, (start, end))


def app_ui(request: Request):
    return ui.page_navbar(
        ui.nav_panel(
            "Overview",
            ui.output_ui("kpis"),
            question_row("Usage by provider", "provider_ts", "provider_total"),
            question_row("Usage by model", "model_ts", "model_total"),
            question_row(
                ui.output_ui("group_question", inline=True),
                "principal_ts",
                "principal_total",
            ),
        ),
        ui.nav_panel(
            "Attribution",
            ui.output_ui("principal_kpis"),
            ui.div(
                ui.h6(
                    # The limit is a sidebar input, so the question has to be
                    # rendered rather than written into the layout.
                    ui.output_ui("limit_question", inline=True),
                    class_="row-question",
                ),
                ui.card(
                    table_header(
                        "All principals against the monthly limit", "dl_threshold"
                    ),
                    ui.card_body(
                        ui.div(ui.output_ui("th_verdict"), class_="inline-controls"),
                        ui.output_data_frame("th_table"),
                        ui.output_ui("th_note"),
                    ),
                ),
                class_="question-block",
            ),
            # Hidden with Bootstrap's d-none. Shiny suspends an output it is told is
            # hidden, so the queries behind this block do not run while it is off;
            # drop the class to bring the whole question back.
            # See Slack convo for status of this feature:
            # https://positpbc.slack.com/archives/C0BK7JQGS67/p1788700635567709
            ui.div(
                ui.h6("Did any single session run away?", class_="row-question"),
                ui.card(
                    table_header("Session cost against duration", "dl_sessions"),
                    ui.card_body(
                        output_widget("session_scatter"),
                        ui.output_data_frame("top_sessions"),
                        ui.output_ui("session_note"),
                    ),
                ),
                class_="question-block d-none",
            ),
            ui.div(
                ui.h6(
                    "Usage by principal",
                    help_icon(
                        "What is a principal?",
                        ui.p(
                            "The person or account this spend is attributed to. Each row "
                            "below is one principal and the usage charged to them."
                        ),
                        ui.p(
                            "Attribution is not always the same as who typed the prompt. "
                            "A call can arrive with more than one identity attached, so "
                            "the ",
                            ui.strong("via"),
                            " column tells you which one this row is counting, and how "
                            "much weight to give it:",
                        ),
                        ui.tags.dl(
                            ui.tags.dt("via viewer"),
                            ui.tags.dd(
                                "Someone used an app and the app called the model on "
                                "their behalf. Safe to read as their usage."
                            ),
                            ui.tags.dt("via authenticated"),
                            ui.tags.dd(
                                "The person signed in and made the call themselves — a "
                                "Workbench session, for example. Safe to read as their "
                                "usage."
                            ),
                            ui.tags.dt("via owner"),
                            ui.tags.dd(
                                "Content ran on its own credentials, so the only person "
                                "we can name is whoever owns it. Treat this as spend the "
                                "owner is responsible for, not spend they personally "
                                "ran — a scheduled report can bill its owner overnight."
                            ),
                            class_="help-dl",
                        ),
                        ui.p(
                            "A principal shown as a long ID instead of a name is still a "
                            "real account; your server just did not record the name. "
                            "Requests we could not tie to anyone are left out entirely, "
                            "and the note under the table says how many and how much.",
                            class_="mb-0",
                        ),
                    ),
                    class_="row-question",
                ),
                ui.card(
                    table_header("All principals", "dl_principal_table"),
                    ui.card_body(
                        ui.output_data_frame("principal_table"),
                        ui.output_ui("principal_note"),
                    ),
                ),
                class_="question-block",
            ),
        ),
        ui.nav_panel(
            "Timing & egress",
            ui.div(
                ui.h6("Where does the traffic leave to?", class_="row-question"),
                ui.card(
                    table_header("Egress endpoints", "dl_egress"),
                    ui.output_data_frame("egress"),
                ),
                class_="question-block",
            ),
            ui.div(
                ui.h6("When do requests happen?", class_="row-question"),
                ui.card(
                    ui.card_header("Day of week against hour (UTC)"),
                    output_widget("heatmap"),
                ),
                class_="question-block",
            ),
            ui.div(
                ui.h6("Does anything run at the weekend?", class_="row-question"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Weekday against weekend"),
                        ui.card_body(
                            output_widget("weekend_bar"),
                            ui.div(
                                "Weekend derives from the event timestamp (UTC). "
                                "Holidays are NOT flaggable without a company calendar.",
                                class_="chart-footnote",
                            ),
                        ),
                    ),
                    ui.card(
                        table_header("Weekend activity by principal", "dl_weekend"),
                        ui.output_data_frame("weekend_principals"),
                        ui.output_ui("weekend_note"),
                    ),
                    col_widths=[5, 7],
                ),
                class_="question-block",
            ),
        ),
        ui.nav_panel(
            "Audit log",
            ui.div(
                ui.h6("Audit log", class_="row-question"),
                ui.card(
                    table_header("Gateway records", "dl_records"),
                    ui.card_body(
                        # Chips above the table rather than a filter row inside
                        # it: the per-column boxes were only reachable after
                        # scrolling sideways to the column, they held no state a
                        # reader could see at a glance, and the CSV export could
                        # not honour them. These narrow the query itself, so the
                        # export and the count below match what is on screen.
                        #
                        # Status only: the sidebar owns date range, origin and
                        # users, and a second copy of any of them here would
                        # leave a reader working out which one won. The status
                        # code is the one audit field the sidebar does not offer.
                        ui.div(
                            filter_chip(
                                "chip_status_label",
                                chip_menu(
                                    ui.input_checkbox_group(
                                        "a_status", None, CATALOG["statuses"]
                                    ),
                                    note="HTTP status the gateway returned.",
                                ),
                            ),
                            # Shown only when there is something to clear, so the
                            # bar does not carry a dead control.
                            ui.panel_conditional(
                                "(input.a_status || []).length > 0",
                                ui.input_action_link(
                                    "a_clear", "Clear chip", class_="chip-clear"
                                ),
                            ),
                            class_="chip-bar",
                        ),
                        # Both sets of controls are in force at once, and the
                        # sidebar is easy to read past on this tab, where the
                        # chips are the nearer thing.
                        ui.div(
                            "The sidebar's filters — date range, origin, users, "
                            "content, provider and models — apply here too. These "
                            "chips narrow that result further.",
                            class_="chip-bar-note",
                        ),
                        ui.output_data_frame("records_table"),
                        ui.output_ui("records_note"),
                    ),
                ),
                class_="question-block",
            ),
        ),
        ui.nav_spacer(),
        ui.nav_control(ui.input_dark_mode(id="dark_mode", mode="light")),
        sidebar=ui.sidebar(
            ui.input_select("preset", "Date range", PRESETS, selected="all"),
            ui.input_radio_buttons("measure", "Count", MEASURES, selected="spend"),
            ui.input_radio_buttons("source", "Origin", SOURCES, selected="both"),
            # Grouping and the content filter only mean something for Connect: a
            # Workbench call has no content item, and Both has to group on the axis
            # the two sources share.
            ui.panel_conditional(
                "input.source === 'connect'",
                ui.input_radio_buttons(
                    "group_by", "Group by", GROUP_BY, selected="username"
                ),
            ),
            # Not a filter — it changes no row that any tab shows — but it lives
            # with the filters because it is the other number a reader adjusts
            # while reading, and it belongs where they already look for controls.
            ui.div(
                ui.input_numeric(
                    "limit_usd",
                    "Monthly limit (USD)",
                    DEFAULT_LIMIT_USD,
                    min=MIN_LIMIT_USD,
                    step=25,
                ),
                ui.div(
                    "Judges the over-limit table on Attribution.",
                    class_="filter-hint",
                ),
            ),
            ui.div(
                ui.input_selectize(
                    "f_users",
                    "Users",
                    CATALOG["users"],
                    multiple=True,
                    options={"placeholder": "all users"},
                ),
                ui.input_action_link("clear_users", "clear", class_="clear-link"),
                class_="filter-with-clear",
            ),
            ui.panel_conditional(
                "input.source === 'connect'",
                ui.div(
                    ui.input_selectize(
                        "f_contents",
                        "Content",
                        CATALOG["contents"],
                        multiple=True,
                        options={"placeholder": "all content"},
                    ),
                    ui.input_action_link(
                        "clear_contents", "clear", class_="clear-link"
                    ),
                    class_="filter-with-clear",
                ),
            ),
            # Provider and Models both start with nothing checked, which shows
            # everything: checking boxes narrows the view, and "clear" gets back
            # to all of it in one click.
            ui.div(
                ui.input_checkbox_group(
                    "f_providers",
                    "Provider",
                    CATALOG["providers"],
                ),
                ui.input_action_link("clear_providers", "clear", class_="clear-link"),
                ui.div("Nothing checked shows every provider.", class_="filter-hint"),
                class_="filter-with-clear",
            ),
            # A checkbox group, like Provider: the model list is short enough to
            # show in full, and seeing every value at once is what tells a reader
            # what the data covers. A search box hid that behind typing.
            ui.div(
                ui.input_checkbox_group(
                    "f_models",
                    "Models",
                    model_choices(CATALOG["models"]),
                ),
                ui.input_action_link("clear_models", "clear", class_="clear-link"),
                ui.div("Nothing checked shows every model.", class_="filter-hint"),
                class_="filter-with-clear",
            ),
            ui.input_action_button(
                "refresh", "Refresh data", class_="btn-sm refresh-btn"
            ),
            # A bookmarked URL is the only state this app keeps between visits,
            # so "clear" has to empty the query string as well as the controls —
            # otherwise a reload would restore what was just cleared.
            ui.input_action_button(
                "clear_all", "Clear all filters", class_="btn-sm clear-all-btn"
            ),
            ui.output_ui("load_note"),
            ui.div(
                "Single source: Connect AI Gateway event metadata — no prompt or completion "
                "content. A call is charged to the viewer it acted for, else to whoever "
                "authenticated it, else to the content owner. A user shown as a guid means "
                "an administrator turned user enrichment off. Grouping users into business "
                "units needs a directory the gateway does not emit.",
                class_="sidebar-note",
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
                href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap",
            ),
            # Relative, not "/favicon.svg": on Connect the app is served under a
            # content path, and a rooted href would look for the favicon at the
            # server root instead. The route is mounted in `App(...)` below.
            ui.tags.link(rel="icon", type="image/svg+xml", href="favicon.svg"),
            ui.include_css(Path(__file__).parent / "www" / "styles.css"),
        ),
        title=brand(),
        fillable=False,
    )


def server(input, output, session):
    # Integrate bookmarking callbacks
    @session.bookmark.on_bookmarked
    async def _(url: str):
        await session.bookmark.update_query_string(url)

    # Trigger bookmarking
    @reactive.effect
    @reactive.event(input.main_tabs, ignore_init=True)
    async def _():
        await session.bookmark()

    # bumped by a refresh; every query depends on it through flt()
    loaded = reactive.value(usage.INFO.copy())

    @reactive.calc
    def cat() -> dict:
        loaded()
        return catalog()

    @reactive.effect
    # ignore_init, or opening the dashboard would re-read the export every time
    @reactive.event(input.refresh, ignore_init=True)
    def _refresh():
        try:
            info = usage.load()
        except RuntimeError as e:
            # the reload rolled back, so the loaded data is still intact
            ui.notification_show(
                f"Refresh failed — showing the data from "
                f"{loaded()['loaded_at']:%H:%M UTC}. {e}",
                type="error",
                duration=15,
            )
            return
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
            choices=c["users"],
            selected=[u for u in input.f_users() or [] if u in c["users"]],
        )
        ui.update_selectize(
            "f_contents",
            choices=c["contents"],
            selected=[x for x in input.f_contents() or [] if x in c["contents"]],
        )
        # The audit log's status chip is seeded from the same catalog, so it
        # follows the reload too.
        ui.update_checkbox_group(
            "a_status",
            choices=c["statuses"],
            selected=[s for s in input.a_status() or [] if s in c["statuses"]],
        )
        loaded.set(info.copy())
        ui.notification_show(
            f"Loaded {info['rows']:,} records", type="message", duration=4
        )

    @render.ui
    def load_note():
        info, c = loaded(), cat()
        return ui.div(
            f"{info['rows']:,} records · loaded {info['loaded_at']:%H:%M UTC}",
            ui.br(),
            f"window {info['window_from'][:10]} → {info['window_to'][:10]}, "
            f"data {c['start']} → {c['end']}",
            ui.br(),
            # The watermark is what makes the read reproducible, so it belongs
            # beside the row count rather than only in the module doc.
            f"pinned at ingest ≤ {info['as_of'][11:16]} UTC",
            ui.br(),
            price_note(info),
            class_="sidebar-note load-note",
        )

    @reactive.effect
    @reactive.event(input.clear_all)
    async def _clear_all():
        """Everything back to the state a first-time visitor sees, URL included.

        Dark mode is left alone: it is how someone reads the page, not part of
        what the page is showing.
        """
        ui.update_select("preset", selected="all")
        ui.update_radio_buttons("measure", selected="spend")
        ui.update_radio_buttons("source", selected="both")
        ui.update_radio_buttons("group_by", selected="username")
        ui.update_selectize("f_users", selected=[])
        ui.update_selectize("f_contents", selected=[])
        ui.update_checkbox_group("f_providers", selected=[])
        ui.update_checkbox_group("f_models", selected=[])
        ui.update_numeric("limit_usd", value=DEFAULT_LIMIT_USD)
        ui.update_checkbox_group("a_status", selected=[])
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
    @reactive.event(input.clear_users)
    def _clear_users():
        ui.update_selectize("f_users", selected=[])

    @reactive.effect
    @reactive.event(input.clear_contents)
    def _clear_contents():
        ui.update_selectize("f_contents", selected=[])

    @reactive.calc
    def flt() -> dict:
        c = cat()
        start, end = preset_range(input.preset(), c["start"], c["end"])
        source = input.source()
        return {
            "start": start,
            "end": end,
            "month": c["month"],
            "measure": input.measure(),
            "source": source,
            # Only Connect offers the choice; anything else groups by user.
            "dim": input.group_by() if source == "connect" else "username",
            "all_providers": c["providers"],
            "all_models": c["models"],
            "all_users": c["users"],
            "all_contents": c["contents"],
            "providers": list(input.f_providers() or []),
            "models": list(input.f_models() or []),
            "users": list(input.f_users() or []),
            "contents": list(input.f_contents() or []),
            # A box left blank mid-edit reads as None, and a non-positive limit
            # has no meaning, so both fall back rather than reaching the query.
            "limit": max(
                float(input.limit_usd() or DEFAULT_LIMIT_USD), float(MIN_LIMIT_USD)
            ),
            "mode": input.dark_mode() or "light",
        }

    def sql_in(col: str, vals) -> str:
        return f"{col} IN ({','.join(repr(x) for x in vals)})"

    def where(f, extra: str = "") -> str:
        conds = [
            f"ts >= TIMESTAMP '{f['start']} 00:00:00'",
            f"ts <= TIMESTAMP '{f['end']} 23:59:59.999'",
        ]
        if f["source"] != "both":
            conds.append(f"source = '{f['source']}'")
        if f["users"] and set(f["users"]) != set(f["all_users"]):
            conds.append(sql_in("username", f["users"]))
        # The content filter is hidden outside Connect, but a stale selection is
        # still sent, so it is only applied where it is offered.
        if (
            f["source"] == "connect"
            and f["contents"]
            and set(f["contents"]) != set(f["all_contents"])
        ):
            conds.append(sql_in("content_guid", f["contents"]))
        if f["providers"] and set(f["providers"]) != set(f["all_providers"]):
            conds.append(sql_in("provider", f["providers"]))
        if f["models"] and set(f["models"]) != set(f["all_models"]):
            conds.append(sql_in("model", f["models"]))
        return " AND ".join(conds) + extra

    # ---- shared chart shapes ------------------------------------------------
    def daily_lines(f, m, dim: str, order, color_kind: str | None):
        """One line per category, measured per day."""
        # A null key means the record has no value on this axis (no user, or no
        # content). It is not a series — plotting it yields a "nan" line.
        df = usage.q(f"""SELECT day, {dim} AS k, SUM({m["expr"]}) v FROM gateway_events
                         WHERE {where(f)} AND {dim} IS NOT NULL
                         GROUP BY 1,2 ORDER BY 1""")
        if df.empty:
            return empty_fig(f["mode"])
        fig = go.Figure()
        label = dict(zip(order, chart_labels(dim, order)))
        for i, k in enumerate(order):
            d = df[df.k == k]
            if d.empty:
                continue
            c = (
                color(color_kind, k, f["mode"])
                if color_kind
                else theme.TOKENS[f["mode"]]["series"][i % 8]
            )
            fig.add_scatter(
                x=d.day,
                y=d.v,
                name=label[k],
                mode="lines",
                line={"width": 2, "color": c},
                hovertemplate=(
                    f"<b>{m['prefix']}%{{y:{m['tick']}}}</b> {k}<extra></extra>"
                ),
            )
        return apply(fig, f["mode"], time_axis=True, showlegend=True, y_title=m["axis"])

    def total_bars(
        f, m, dim: str, color_kind: str | None, limit: int = 15, height: int = 320
    ):
        """Category totals, biggest last so the bars read bottom-up."""
        df = usage.q(f"""SELECT {dim} AS k, SUM({m["expr"]}) v, COUNT(*) req
                         FROM gateway_events WHERE {where(f)} AND {dim} IS NOT NULL
                         GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}""").iloc[::-1]
        if df.empty:
            return empty_fig(f["mode"])
        colors = (
            [color(color_kind, k, f["mode"]) for k in df.k]
            if color_kind
            else [theme.TOKENS[f["mode"]]["series"][0]] * len(df)
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
                text=[m["fmt"](v) for v in df.v],
                textposition="outside",
                textfont={"color": theme.TOKENS[f["mode"]]["ink"], "size": 11},
                customdata=df[["req", "k"]],
                hovertemplate=(
                    f"<b>{m['prefix']}%{{x:{m['tick']}}}</b> · "
                    "%{customdata[0]:,} requests"
                    "<extra>%{customdata[1]}</extra>"
                ),
            )
        )
        fig.update_xaxes(range=[0, df.v.max() * 1.25], title_text=m["total_axis"])
        return apply(fig, f["mode"], height=height, showlegend=False)

    # ---- overview -----------------------------------------------------------
    @render.ui
    def kpis():
        f = flt()
        r = usage.q(f"""SELECT COUNT(*) requests, COALESCE(SUM({est_cost()}),0) est,
                            COALESCE(SUM({TOKEN_COUNT}),0) tokens,
                            COUNT(DISTINCT username) users,
                            COUNT(DISTINCT content_guid) contents,
                            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) errors,
                            quantile_cont(duration_ms,.95) p95
                     FROM gateway_events WHERE {where(f)}""").iloc[0]
        err = 100 * r.errors / r.requests if r.requests else 0
        headline = (
            (fmt_num(r.tokens), "Tokens", f"{card_usd(r.est)} estimated")
            if f["measure"] == "tokens"
            else (card_usd(r.est), "Estimated spend", f"{fmt_num(r.tokens)} tokens")
        )
        return ui.layout_columns(
            theme.stat_tile(headline[1], headline[0], headline[2]),
            theme.stat_tile(
                "LLM requests",
                card_num(r.requests),
                f"{card_num(r.users)} {plural(r.users, 'user')} · "
                f"{card_num(r.contents)} {plural(r.contents, 'content item')}",
            ),
            theme.stat_tile(
                "Error rate",
                f"{err:.2f}%",
                "of LLM calls",
                status="critical" if err > 2 else "good",
                help=help_icon(
                    "What is the error rate?",
                    ui.p(
                        "The share of LLM calls in this view that the gateway "
                        "recorded an error response from the provider. A failed "
                        "call is one a person or an app asked for and did not get."
                    ),
                    ui.p(
                        "Failures come from either side: a request the provider "
                        "rejected or rate-limited, or trouble at the provider itself. "
                        "Use the audit log to see the individual failures."
                    ),
                    ui.p(
                        "The dot turns red above 2%. It counts requests, not spend, "
                        "and it moves with the filters, a narrow selection of a few "
                        r"calls can read 0% or 50% on very little evidence.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "P95 latency",
                f"{(r.p95 or 0) / 1000:.1f}s",
                "gateway-observed",
                help=help_icon(
                    "What is P95 latency?",
                    ui.p(
                        "How long the slow calls take. 95 out of every 100 LLM calls "
                        "in this view finished faster than this, and 5 took longer."
                    ),
                    ui.p(
                        "It is deliberately not the average. Most calls are quicker "
                        "than this figure, so read it as the wait people notice and "
                        "complain about, not the wait they usually get."
                    ),
                    ui.p(
                        "The clock is the gateway's: time spent calling the model, "
                        "not time your app spends on its own work before or after. A "
                        "few very slow calls are enough to move it, and like every "
                        "number here it is computed only over the current filters.",
                        class_="mb-0",
                    ),
                ),
            ),
            col_widths=[3, 3, 3, 3],
            class_="kpi-row",
        )

    @render_widget
    def provider_ts():
        f = flt()
        return daily_lines(f, measure_of(f), "provider", f["all_providers"], "provider")

    @render_widget
    def provider_total():
        f = flt()
        return total_bars(f, measure_of(f), "provider", "provider", height=260)

    @render_widget
    def model_ts():
        f = flt()
        return daily_lines(f, measure_of(f), "model", f["all_models"], "model")

    @render_widget
    def model_total():
        f = flt()
        return total_bars(f, measure_of(f), "model", "model", height=260)

    def _top_of(f, m, column: str, limit: int = 6):
        """The busiest values of one identity column, for a readable line chart."""
        return usage.q(f"""SELECT {column} AS k FROM gateway_events
                           WHERE {where(f)} AND {column} IS NOT NULL
                           GROUP BY 1 ORDER BY SUM({m["expr"]}) DESC
                           LIMIT {limit}""")["k"].tolist()

    def no_principal(f) -> str:
        """Why the grouping axis is empty, in the terms of the chosen origin."""
        if f["dim"] == "content_guid":
            return "No request in range came from content"
        if f["source"] == "connect":
            return (
                "No Connect request in range names a user, a viewer or an "
                "owner. Group by content instead."
            )
        return "No request in range names a user"

    @render.ui
    def group_question():
        return f"Usage by {GROUP_NOUN[flt()['dim']]}"

    @render_widget
    def principal_ts():
        f, m = flt(), measure_of(flt())
        top = _top_of(f, m, f["dim"])
        if not top:
            return empty_fig(f["mode"], no_principal(f))
        # The chart plots only its own top values, so the sidebar filter for this
        # axis is widened to let them through.
        key = "users" if f["dim"] == "username" else "contents"
        return daily_lines(
            {**f, key: top, f"all_{key}": f[f"all_{key}"] + ["\0"]},
            m,
            f["dim"],
            top,
            None,
        )

    @render_widget
    def principal_total():
        f, m = flt(), measure_of(flt())
        if not _top_of(f, m, f["dim"], 1):
            return empty_fig(f["mode"], no_principal(f))
        return total_bars(f, m, f["dim"], None, height=320)

    # ---- principals ---------------------------------------------------------
    @render.ui
    def principal_kpis():
        f, m = flt(), measure_of(flt())
        r = usage.q(f"""SELECT COALESCE(SUM({m["expr"]}),0) v, COUNT(*) req,
                            COUNT(DISTINCT {f["dim"]}) n
                     FROM gateway_events WHERE {where(f)}""").iloc[0]
        noun = GROUP_NOUN[f["dim"]]
        return ui.layout_columns(
            theme.stat_tile(
                m["noun"].capitalize(), m["card_fmt"](r.v), SOURCES[f["source"]].lower()
            ),
            theme.stat_tile("Requests", card_num(r.req), "in range"),
            theme.stat_tile(
                plural(r.n, noun.capitalize()),
                card_num(r.n),
                f"charged {plural(r.n, noun)} in range",
            ),
            col_widths=[4, 4, 4],
            class_="kpi-row",
        )

    def named(f, extra: str = "") -> str:
        """Rows that name a principal on the current axis.

        A request with no user is not a user called "nan": it is left out of the
        rankings, and the tables say how much it accounts for.
        """
        return where(f, f" AND {f['dim']} IS NOT NULL{extra}")

    def unnamed_note(f) -> ui.Tag | None:
        """How much of the range the grouping axis cannot name."""
        r = usage.q(f"""SELECT COUNT(*) n, ROUND(SUM({est_cost()}),2) v
                        FROM gateway_events
                        WHERE {where(f)} AND {f["dim"]} IS NULL""").iloc[0]
        if not r.n:
            return None
        noun = GROUP_NOUN[f["dim"]]
        if f["dim"] == "content_guid":
            why = "These did not originate in deployed content."
        else:
            why = (
                "A call names no user when nothing authenticated it and no "
                "owner resolved — a Service Account credential, or content "
                "running under its own ephemeral key."
            )
        return ui.div(
            f"{fmt_num(r.n)} {plural(r.n, 'request')} in range name no {noun} "
            f"({fmt_usd(r.v)}). {why}",
            class_="chart-footnote",
        )

    def bands_df() -> pd.DataFrame:
        """Every latest-month principal, ranked against the limit.

        No HAVING gate: a table holding only the principals near the limit is
        empty on a quiet month, which reads as a broken panel rather than as
        the good news it is. `status` is what separates the bands instead, so
        the rows that need attention sort to the top and the rest still show
        what normal usage looks like.
        """
        f = flt()
        limit = f["limit"]
        return usage.q(f"""SELECT {f["dim"]} AS principal,
                               ROUND(SUM({est_cost()}),2) AS est_spend_usd,
                               ROUND(100 * SUM({est_cost()}) / {limit}, 0) AS pct_of_limit,
                               CASE WHEN SUM({est_cost()}) > {limit} THEN '⛔ over'
                                    WHEN SUM({est_cost()}) >= 0.8 * {limit}
                                         THEN '⚠ approaching'
                                    ELSE '✓ under' END AS status,
                               COUNT(*) AS requests
                        FROM gateway_events
                        WHERE {named(f, f" AND month='{f['month']}'")}
                        GROUP BY 1
                        ORDER BY est_spend_usd DESC""")

    @render.ui
    def limit_question():
        return f"Who is over the {fmt_limit(flt()['limit'])} monthly limit?"

    @render.ui
    def th_verdict():
        f, df = flt(), bands_df()
        over = int((df.status == "⛔ over").sum())
        warn = int((df.status == "⚠ approaching").sum())
        noun = plural(len(df), GROUP_NOUN[f["dim"]])
        limit, month = fmt_limit(f["limit"]), fmt_month(f["month"])
        counts = f"{month} · {len(df)} {noun}"
        if over:
            return theme.verdict(
                f"{counts} · {over} over {limit} · {warn} approaching (≥80%)", ok=False
            )
        if warn:
            return theme.verdict(
                f"{counts} · none over {limit} · {warn} approaching (≥80%)", ok=None
            )
        return theme.verdict(f"{counts} · all under 80% of {limit}", ok=True)

    @render.data_frame
    def th_table():
        return fitted_grid(
            formatted(bands_df())
            .rename(
                columns={
                    "est_spend_usd": "estimated spend (usd)",
                    "pct_of_limit": r"% of limit",
                }
            )
            .rename(lambda x: x.replace("_", " "), axis="columns"),
            filters=False,
        )

    @render.ui
    def th_note():
        f = flt()
        if bands_df().empty:
            return empty_note(f"No {GROUP_NOUN[f['dim']]} was named in {f['month']}.")
        return unnamed_note(f)

    @reactive.calc
    def session_ids() -> int:
        """Records carrying a conversation id. Zero means sessions are unknowable."""
        return int(
            usage.scalar(
                f"SELECT COUNT(conversation_id) FROM gateway_events WHERE {where(flt())}"
            )
            or 0
        )

    @reactive.calc
    def sessions_df() -> pd.DataFrame:
        """Every session in range, cheapest included.

        There is no cost floor: gating at $0.25 emptied the panel on a quiet
        range, and the cloud of ordinary sessions is what gives a runaway
        something to stand out against.
        """
        f = flt()
        return usage.q(f"""
            SELECT conversation_id,
                   -- a label, not a key: a null here would reach plotly as NaN
                   COALESCE(ANY_VALUE({f["dim"]}), 'unnamed') AS principal,
                   MIN(ts) AS started,
                   GREATEST(date_diff('second', MIN(ts), MAX(ts)) / 60.0, 0.5) AS duration_min,
                   COUNT(*) AS calls, SUM({est_cost()}) AS est
            FROM gateway_events
            WHERE {where(f)} AND conversation_id IS NOT NULL
            GROUP BY 1
            ORDER BY est DESC""")

    @render_widget
    def session_scatter():
        f, df = flt(), sessions_df()
        if df.empty:
            return empty_fig(
                f["mode"],
                "No session in range"
                if session_ids()
                else "No conversation ids in this data",
            )
        # A log axis has no coordinate for zero, so a session estimated at
        # $0.00 cannot be drawn. It stays in the table, and the footnote counts
        # what the chart is not showing.
        df = df[df.est > 0]
        if df.empty:
            return empty_fig(f["mode"], "Every session in range is estimated at $0.00")
        t = theme.TOKENS[f["mode"]]
        normal, runaway = df[df.est <= RUNAWAY_USD], df[df.est > RUNAWAY_USD]
        hover = (
            "<b>$%{y:,.2f}</b> · %{customdata[0]} · %{customdata[1]} calls · "
            "%{x:,.0f} min<extra></extra>"
        )
        fig = go.Figure()
        fig.add_scatter(
            x=normal.duration_min,
            y=normal.est,
            mode="markers",
            name="session",
            marker={
                "size": 7,
                "color": t["muted"],
                "opacity": 0.45,
                "line": {"color": t["surface"], "width": 1},
            },
            customdata=normal[["principal", "calls"]],
            hovertemplate=hover,
        )
        if not runaway.empty:
            top = runaway.est.max()
            fig.add_scatter(
                x=runaway.duration_min,
                y=runaway.est,
                mode="markers+text",
                name=f"runaway (> ${RUNAWAY_USD})",
                marker={
                    "size": 11,
                    "color": theme.STATUS["critical"],
                    "line": {"color": t["surface"], "width": 2},
                },
                text=[
                    p if r == top else ""
                    for p, r in zip(runaway.principal, runaway.est)
                ],
                textposition="middle left",
                cliponaxis=False,
                textfont={"color": t["ink"], "size": 11},
                customdata=runaway[["principal", "calls"]],
                hovertemplate=hover,
            )
        fig.update_yaxes(
            type="log", title_text="est. USD per session (log)", tickformat="$~s"
        )
        fig.update_xaxes(
            title_text="session duration (minutes)",
            range=[-2, df.duration_min.max() * 1.15],
        )
        return apply(fig, f["mode"], height=300, showlegend=True)

    @render.data_frame
    def top_sessions():
        """Every session, dearest first — not just the top five.

        The five-row cap hid how many sessions there were and gave no way to
        reach the rest; the export was the only path to them.
        """
        cols = [
            "session",
            "principal",
            "started",
            "duration_min",
            "calls",
            "est_spend_usd",
            "usd_per_min",
            "status",
        ]
        df = sessions_df().copy()
        if df.empty:
            # An empty grid still draws its header, so give it the header the
            # filled table has rather than the raw query's column names.
            return fitted_grid(pd.DataFrame(columns=cols))
        df["session"] = df.conversation_id.str[:8]
        df["started"] = df.started.dt.strftime("%Y-%m-%d %H:%M")
        df["usd_per_min"] = (df.est / df.duration_min).round(2)
        df["duration_min"] = df.duration_min.round(0)
        df["est_spend_usd"] = df.est.round(2)
        # The band is read off the unrounded estimate: a session at $25.004 is
        # over the line even though the column shows $25.00.
        df["status"] = ["⛔ runaway" if v > RUNAWAY_USD else "✓ normal" for v in df.est]
        return fitted_grid(formatted(df[cols]), filters=True)

    @render.ui
    def session_note():
        f, df = flt(), sessions_df()
        pct = usage.scalar(f"""SELECT ROUND(100.0 * COUNT(conversation_id) / COUNT(*), 0)
                            FROM gateway_events WHERE {where(f)}""")
        if not session_ids():
            return empty_note(
                "Sessions cannot be grouped here: no record in range carries a conversation id. "
                "IDE and deployed-app chat sessions stamp one; raw API traffic does not."
            )
        zeros = int((df.est <= 0).sum())
        gap = (
            f" {zeros} {plural(zeros, 'session')} priced at exactly $0.00 "
            f"{'is' if zeros == 1 else 'are'} listed but not plotted: "
            f"a log axis has no place for zero."
            if zeros
            else ""
        )
        return ui.div(
            f"A runaway session reads as a lone dot above the cloud. Every session "
            f"in range is listed ({fmt_num(len(df))} in total), always in dollars."
            f"{gap} IDE and deployed-app chat sessions stamp conversation ids; raw "
            f"API traffic does not — {(pct or 0):.0f}% of requests in range carry one.",
            class_="chart-footnote",
        )

    def principal_table_df() -> pd.DataFrame:
        f = flt()
        # "via" is aggregated because one principal can be charged along more
        # than one path: viewer on one call, content owner on the next.
        via = (
            "string_agg(DISTINCT user_kind, ', ') AS via,"
            if f["dim"] == "username"
            else ""
        )
        return usage.q(f"""SELECT {f["dim"]} AS principal,
                             string_agg(DISTINCT source, ', ') AS origin,
                             {via}
                             COUNT(*) AS requests,
                             SUM({TOKEN_COUNT}) AS tokens,
                             ROUND(SUM({est_cost()}),2) AS est_spend_usd,
                             SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END) AS weekend_requests,
                             strftime(MIN(day), '%Y-%m-%d') AS first_seen,
                             strftime(MAX(day), '%Y-%m-%d') AS last_seen
                      FROM gateway_events WHERE {named(f)}
                      GROUP BY 1 ORDER BY est_spend_usd DESC""")

    @render.data_frame
    def principal_table():
        return render.DataGrid(
            formatted(principal_table_df())
            .rename(columns={"est_spend_usd": "estimated spend (usd)"})
            .rename(lambda x: x.replace("_", " "), axis="columns"),
            filters=False,
        )

    @render.ui
    def principal_note():
        return unnamed_note(flt())

    # ---- timing & egress ----------------------------------------------------
    @render_widget
    def heatmap():
        f = flt()
        df = usage.q(f"""SELECT (dayofweek(ts)+6)%7 dow, hour, COUNT(*) n
                      FROM gateway_events WHERE {where(f)} GROUP BY 1,2""")
        if df.empty:
            return empty_fig(f["mode"])
        z = [[0] * 24 for _ in range(7)]
        for _, r in df.iterrows():
            z[int(r.dow)][int(r.hour)] = int(r.n)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=list(range(24)),
                y=WEEKDAYS,
                colorscale=theme.seq_colorscale(f["mode"]),
                xgap=2,
                ygap=2,
                hovertemplate="<b>%{z:,}</b> requests · %{y} %{x}:00 UTC<extra></extra>",
                colorbar={
                    "thickness": 10,
                    "outlinewidth": 0,
                    "tickfont": {"color": theme.TOKENS[f["mode"]]["muted"], "size": 10},
                },
            )
        )
        fig.update_yaxes(autorange="reversed")
        return apply(fig, f["mode"], height=320)

    @render_widget
    def weekend_bar():
        f, m = flt(), measure_of(flt())
        df = usage.q(f"""SELECT CASE WHEN is_weekend THEN 'weekend' ELSE 'weekday' END day_type,
                             SUM({m["expr"]}) v, COUNT(DISTINCT {f["dim"]}) principals
                      FROM gateway_events WHERE {where(f)} GROUP BY 1 ORDER BY 1 DESC""")
        if df.empty:
            return empty_fig(f["mode"])
        fig = go.Figure(
            go.Bar(
                x=df.day_type,
                y=df.v,
                width=0.4,
                marker={"color": theme.TOKENS[f["mode"]]["series"][0]},
                text=[m["fmt"](v) for v in df.v],
                textposition="outside",
                textfont={"color": theme.TOKENS[f["mode"]]["ink"], "size": 12},
                customdata=df.principals,
                hovertemplate=(
                    f"<b>{m['prefix']}%{{y:{m['tick']}}}</b> · "
                    "%{customdata} principals<extra>%{x}</extra>"
                ),
            )
        )
        fig.update_yaxes(range=[0, df.v.max() * 1.25])
        return apply(
            fig, f["mode"], height=260, showlegend=False, y_title=m["total_axis"]
        )

    def weekend_principals_df() -> pd.DataFrame:
        f = flt()
        return usage.q(f"""SELECT {f["dim"]} AS principal,
                             ROUND(SUM({est_cost()}),2) AS est_spend_usd,
                             SUM({TOKEN_COUNT}) AS tokens,
                             COUNT(*) AS requests,
                             strftime(MIN(day), '%Y-%m-%d') AS first,
                             strftime(MAX(day), '%Y-%m-%d') AS last
                      FROM gateway_events WHERE {named(f, " AND is_weekend")}
                      GROUP BY 1 ORDER BY {"tokens" if f["measure"] == "tokens" else "est_spend_usd"} DESC
                      LIMIT 50""")

    @render.data_frame
    def weekend_principals():
        return render.DataGrid(
            formatted(weekend_principals_df())
            .rename(columns={"est_spend_usd": "estimated spend (usd)"})
            .rename(lambda x: x.replace("_", " "), axis="columns")
        )

    @render.ui
    def weekend_note():
        if not weekend_principals_df().empty:
            return None
        return empty_note("No weekend traffic in range.")

    def egress_df() -> pd.DataFrame:
        f = flt()
        return usage.q(f"""SELECT server_address AS endpoint, CAST(server_port AS INT) AS port,
                             provider, COUNT(*) AS requests,
                             strftime(MAX(ts), '%Y-%m-%d %H:%M') AS last_seen
                      FROM gateway_events WHERE {where(f)} GROUP BY 1,2,3 ORDER BY 4 DESC""")

    @render.data_frame
    def egress():
        return render.DataGrid(
            formatted(egress_df()).rename(
                lambda x: x.replace("_", " "), axis="columns"
            ),
        )

    # ---- audit log ----------------------------------------------------------
    # The derived principal answers "by whom", and the three raw identities
    # behind it are kept beside it: audit evidence must show what the gateway
    # recorded, not only what this dashboard concluded from it.
    # `status` is a string, not a number: an HTTP code is a label rather than a
    # quantity, so sorting or ranging over it ("status > 300") would say nothing.
    # The Status chip lists the codes the data actually contains instead.
    RECORD_COLS = """ts, source AS origin, username, user_kind AS via,
                     user_email AS email,
                     authenticated_username, viewer_username,
                     content_owner_username, content_guid, job_key,
                     model, provider, integration_auth_type AS auth_type,
                     CAST(status_code AS VARCHAR) AS status,
                     ROUND(duration_ms) AS ms,
                     input_tokens AS tok_in, output_tokens AS tok_out,
                     cache_read_tokens AS tok_cache_read,
                     ROUND(EST_COST_EXPR,4) AS est_cost_usd,
                     finish_reason, response_id, conversation_id"""

    # The chip names what it is set to, so the bar reads as state rather than as
    # an unlabelled button; `chip_label` handles the wording.
    @render.ui
    def chip_status_label():
        return chip_label("Status", list(input.a_status() or []))

    @reactive.effect
    @reactive.event(input.a_clear)
    def _clear_audit_chips():
        ui.update_checkbox_group("a_status", selected=[])

    @reactive.calc
    def audit_where() -> str:
        """The sidebar's WHERE, narrowed by the chip above the audit table.

        The chip filters the query rather than the rendered grid, so the row
        count in the footnote and the CSV export both agree with what is on
        screen — which is the point of an audit view.
        """
        conds = [where(flt())]
        if statuses := list(input.a_status() or []):
            conds.append(sql_in("CAST(status_code AS VARCHAR)", statuses))
        return " AND ".join(conds)

    def records_sql(where_sql: str, limit: str = "") -> str:
        cols = RECORD_COLS.replace("EST_COST_EXPR", est_cost())
        return f"SELECT {cols} FROM gateway_events WHERE {where_sql} ORDER BY ts DESC{limit}"

    @reactive.calc
    def records_shown() -> pd.DataFrame:
        return usage.q(records_sql(audit_where(), " LIMIT 500"))

    def records_all() -> pd.DataFrame:
        """The export is every filtered row — the table only shows the head."""
        return usage.q(records_sql(audit_where()))

    @render.data_frame
    def records_table():
        # No per-column filter row: the chips above the table do that job, and
        # keeping both would leave a reader guessing which of the two is in force.
        return render.DataGrid(records_shown(), height="440px")

    @render.ui
    def records_note():
        total = (
            usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {audit_where()}")
            or 0
        )
        if not total:
            return empty_note("No records match these filters.")
        return ui.div(
            f"Showing the {min(total, 500):,} most recent of {total:,} matching "
            f"records. Export CSV writes all {total:,}, for audit evidence.",
            class_="chart-footnote",
        )

    # ---- exports ------------------------------------------------------------
    def register_export(name: str, df_fn, filename: str):
        def _export():
            yield df_fn().to_csv(index=False)

        # the downloads registry keys off the function name, so name it to match the button id
        _export.__name__ = f"dl_{name}"
        renderer = render.download_button(filename=filename)(_export)
        output(id=f"dl_{name}", suspend_when_hidden=False)(renderer)

    for _name, _fn, _file in [
        ("threshold", bands_df, "threshold.csv"),
        ("sessions", sessions_df, "sessions.csv"),
        ("principal_table", principal_table_df, "principals.csv"),
        ("weekend", weekend_principals_df, "weekend.csv"),
        ("egress", egress_df, "egress.csv"),
        ("records", records_all, "gateway_records.csv"),
    ]:
        register_export(_name, _fn, _file)


app = App(
    app_ui,
    server,
    bookmark_store="url",
    # One file, not the whole of www/: the stylesheet is inlined into the head and
    # the logo is read at render time, so the favicon is the only asset a browser
    # has to fetch by URL.
    static_assets={"/favicon.svg": FAVICON},
)
