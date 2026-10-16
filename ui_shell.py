"""Navigation rail, filter drawer, and shared page chrome.

No page content or application data is defined here.
"""

import re
from pathlib import Path

from shiny import ui
from shinywidgets import output_widget


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
        "Posit Team Gateway AI Usage",
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


def help_icon(*args) -> ui.Tag:
    """A "?" beside a heading that opens an explanation.

    A popover, not a tooltip: the content is a few sentences with markup, and a
    tooltip would vanish the moment the pointer left the icon. `role="button"`
    and `tabindex` are what make the span reachable by keyboard, since a span is
    not focusable on its own.

    A leading string is the popover's heading; without one the popover is just
    the body.
    """
    title, body = (args[0], args[1:]) if args and isinstance(args[0], str) else ("", args)
    return ui.popover(
        ui.span(
            "?",
            class_="help-icon",
            role="button",
            tabindex="0",
            aria_label=title or "Help",
        ),
        *([ui.strong(title)] if title else []),
        *body,
        placement="right",
    )


FILTER_HELP = {
    "users": "Gateway users.",
    "content": "Connect content items.",
    "session": "Workbench sessions.",
    "provider": "The model provider that served the call.",
    "models": "The model that handled the call.",
    "meta": "Key/value labels the caller attached to the request.",
}


def filter_label(text: str, key: str) -> ui.Tag:
    """A filter's label with a small ? that explains it on hover or focus."""
    return ui.span(
        text,
        ui.tooltip(
            ui.tags.span("?", class_="help-ico", tabindex="0", role="img"),
            FILTER_HELP[key],
            placement="right",
        ),
        class_="filter-label",
    )


def table_header(title: str | ui.Tag, dl_id: str, help: ui.Tag | None = None) -> ui.CardItem:
    """Card header with a right-aligned CSV export button, and an optional ?."""
    return ui.card_header(
        ui.div(
            ui.div(ui.span(title), help, class_="hdr-flex-left") if help else ui.span(title),
            ui.download_button(dl_id, "Export CSV", class_="btn-sm export-btn"),
            class_="hdr-flex",
        )
    )


def chart_row(question, bars_id: str, share_id: str, share_title: str = "Share") -> ui.Tag:
    """One question, answered over the whole range: totals as bars, and the split
    of those totals as a donut."""
    return ui.div(
        ui.h6(question, class_="row-question"),
        ui.layout_columns(
            ui.card(ui.card_header("Total"), output_widget(bars_id)),
            ui.card(ui.card_header(share_title), output_widget(share_id)),
            col_widths=[7, 5],
        ),
        class_="question-block",
    )


# The nav rail: section label, then the pages under it. A page's value is its
# title (what `ui.nav_panel` uses when it is given no explicit value).
NAV_SECTIONS = [
    ("Gateway", ["Overview", "Adoption", "Models", "Attribution", "Conversations", "Principals", "Raw Data"]),
]
NAV_FIRST = NAV_SECTIONS[0][1][0]


def nav_item(name: str, current: str) -> ui.Tag:
    # A plain button that sets an input; the server turns it into a tab switch.
    # The rail is rendered from the server so that a restored bookmark lights
    # the right item.
    return ui.tags.button(
        name,
        type="button",
        class_="nav-item" + (" active" if name == current else ""),
        onclick=f"Shiny.setInputValue('nav_go', {name!r}, {{priority: 'event'}})",
    )


def nav_rail_ui(current: str) -> ui.Tag:
    out = []
    for label, names in NAV_SECTIONS:
        out.append(ui.div(label, class_="nav-section"))
        out.extend(nav_item(n, current) for n in names)
    return ui.tags.nav(*out, class_="nav-rail")


def plural(n, word: str) -> str:
    return f"{word}s" if n != 1 else word


def page_header(title: str, subtitle: str = "", help: ui.Tag | None = None) -> ui.Tag:
    return ui.div(
        ui.div(ui.h2(title, class_="page-title"), help, class_="hdr-flex-left")
        if help
        else ui.h2(title, class_="page-title"),
        ui.div(subtitle, class_="page-sub") if subtitle else None,
        class_="page-header",
    )


def _drawer(*children, **_) -> ui.Tag:
    """The filters that do not earn a place in the strip.

    Always in the DOM and merely hidden, never a popover: a popover only binds
    its inputs once it has been opened, and these feed every query.
    """
    return ui.div(
        ui.div(
            ui.span("Filters", class_="drawer-title"),
            ui.tags.button(
                "✕",
                type="button",
                class_="drawer-close",
                aria_label="Close filters",
                onclick="document.body.classList.remove('drawer-open')",
            ),
            class_="drawer-head",
        ),
        ui.div(*children, class_="drawer-body sidebar"),
        class_="filter-drawer",
    )


def _shell(
    *panels, sidebar, id, header, title, catalog, presets, default_preset, preset_range, **_
) -> ui.Tag:
    """Page chrome: top bar, nav rail, global filter strip, then the page.

    The pages are a hidden navset so that the existing `main_tabs` id (and the
    bookmark that stores it) keeps working; the rail is what a reader clicks.
    """
    c = catalog
    topbar = ui.div(
        ui.div(title, class_="topbar-brand"),
        ui.output_ui("topbar_status"),
        ui.div(
            ui.output_ui("topbar_meta", inline=True),
            ui.input_action_button(
                "refresh", "Refresh data", class_="btn-sm refresh-btn"
            ),
            class_="topbar-load",
        ),
        ui.input_dark_mode(id="dark_mode", mode="dark"),
        class_="topbar",
    )
    strip = ui.div(
        ui.input_radio_buttons("preset", None, presets, selected=default_preset, inline=True),
        ui.input_date_range(
            "date_range",
            None,
            start=preset_range(default_preset, c["start"], c["end"])[0],
            end=c["end"],
            min=c["start"],
            max=c["end"],
            separator="to",
        ),
        ui.tags.button(
            "Filters",
            ui.span(class_="filter-dot"),
            type="button",
            class_="strip-btn ms-auto",
            onclick="document.body.classList.toggle('drawer-open')",
        ),
        class_="filterstrip",
    )
    return ui.page_fluid(
        header,
        ui.div(
            topbar,
            ui.div(
                ui.output_ui("nav_rail"),
                ui.div(strip, ui.navset_hidden(*panels, id=id), class_="content"),
                class_="shell-body",
            ),
            sidebar,
            class_="app-shell",
        ),
        class_="p-0",
        title="Posit Team Gateway AI Usage",
    )

