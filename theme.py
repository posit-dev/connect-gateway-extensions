"""Chart chrome, validated palette, and small UI components (stat tiles, meters).

Palette is the dataviz reference instance; the 5-slot categorical order was
validated with scripts/validate_palette.js in both light and dark modes.
Light-mode aqua/yellow/magenta sit below 3:1 contrast -> relief rule: every
chart is paired with direct labels and/or a table view.
"""

from __future__ import annotations

import plotly.graph_objects as go
from shiny import ui

TOKENS = {
    "light": {
        "surface": "#ffffff",
        "page": "#f5f6f7",
        "ink": "#1f2d3d",
        "ink2": "#5a6673",
        "muted": "#8a949f",
        "grid": "#e4e7ea",
        "axis": "#d3d8dd",
        "series": [
            "#3d85c6",
            "#6a9e4f",
            "#e0a72c",
            "#e07b39",
            "#c0413b",
            "#2f7f7a",
            "#6b5ca5",
            "#a8577f",
        ],
        "seq": [
            "#e3eef8",
            "#c2dbf0",
            "#9cc3e4",
            "#74a9d6",
            "#4d8ac2",
            "#2f6ba3",
            "#1c4d78",
        ],
    },
    "dark": {
        "surface": "#1c2128",
        "page": "#14181d",
        "ink": "#eef1f4",
        "ink2": "#b3bcc6",
        "muted": "#8a949f",
        "grid": "#2e353d",
        "axis": "#3d454e",
        "series": [
            "#6aa8dd",
            "#8bbf6c",
            "#e0b552",
            "#e8945c",
            "#d9615b",
            "#4fa39c",
            "#9083c9",
            "#c47ba0",
        ],
        "seq": [
            "#1c4d78",
            "#2f6ba3",
            "#4d8ac2",
            "#74a9d6",
            "#9cc3e4",
            "#c2dbf0",
            "#e3eef8",
        ],
    },
}

STATUS = {
    "good": "#4a8b32",
    "warning": "#e0a72c",
    "serious": "#e07b39",
    "critical": "#c0413b",
}

FONT = (
    '"Open Sans", "Source Sans Pro", system-ui, -apple-system, '
    '"Segoe UI", Roboto, sans-serif'
)

# fixed slot assignments -- color follows the entity, never its rank
MODEL_ORDER = [
    "claude-sonnet-5",
    "gpt-5",
    "claude-opus-5",
    "gpt-5-mini",
    "claude-haiku-4-5",
]
PROVIDER_ORDER = ["anthropic", "openai", "aws.bedrock", "azure.openai", "google.vertex"]

_SLOTS = {"model": MODEL_ORDER, "provider": PROVIDER_ORDER}


def color(kind: str, name: str, mode: str) -> str:
    order = _SLOTS[kind]
    series = TOKENS[mode]["series"]
    return (
        series[order.index(name) % len(series)]
        if name in order
        else TOKENS[mode]["muted"]
    )


def seq_colorscale(mode: str):
    ramp = TOKENS[mode]["seq"]
    n = len(ramp) - 1
    return [[i / n, c] for i, c in enumerate(ramp)]


def apply(
    fig: go.Figure,
    mode: str,
    *,
    height: int = 320,
    time_axis: bool = False,
    showlegend: bool | None = None,
    y_title: str | None = None,
) -> go.Figure:
    t = TOKENS[mode]
    fig.update_layout(
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font={"family": FONT, "size": 12, "color": t["ink2"]},
        height=height,
        margin={"l": 8, "r": 8, "t": 34 if showlegend else 12, "b": 8},
        barcornerradius=3,
        colorway=t["series"],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": t["surface"],
            "bordercolor": t["grid"],
            "font": {"family": FONT, "size": 12, "color": t["ink"]},
        },
        hovermode="x unified" if time_axis else "closest",
    )
    grid = {
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "linecolor": t["axis"],
        "linewidth": 1,
        "zeroline": False,
        "tickfont": {"color": t["muted"], "size": 11},
        "title_font": {"color": t["muted"], "size": 11},
    }
    fig.update_xaxes(**grid, showgrid=False)
    fig.update_yaxes(**grid, title_text=y_title)
    if time_axis:
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor=t["axis"],
            spikethickness=1,
            spikedash="solid",
        )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def empty_fig(
    mode: str, msg: str = "No data in the selected range", height: int = 170
) -> go.Figure:
    """A placeholder that says why a chart is blank.

    Shorter than a real chart on purpose: a sentence does not need the height
    an axis does, and a full-size empty panel pushes everything below it off
    the screen.
    """
    t = TOKENS[mode]
    fig = go.Figure()
    fig.add_annotation(
        text=msg, showarrow=False, font={"color": t["muted"], "size": 13}
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply(fig, mode, height=height)


def fmt_usd(v: float) -> str:
    if v is None:
        return "–"
    if abs(v) >= 10_000:
        return f"${v / 1000:,.1f}k"
    if abs(v) >= 100:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def fmt_num(v: float) -> str:
    if v is None:
        return "–"
    if abs(v) >= 1_000_000:
        return f"{v / 1e6:,.1f}M"
    if abs(v) >= 10_000:
        return f"{v / 1e3:,.1f}K"
    return f"{v:,.0f}"


# The KPI tiles are wide enough for the whole number, so they spell it out where
# fmt_usd and fmt_num abbreviate: those two are still what chart labels, axis
# ticks and inline footnotes want, where space is tight.
def card_usd(v: float) -> str:
    """Whole dollars — cents are noise against a range total."""
    return "–" if v is None else f"${v:,.0f}"


def card_num(v: float) -> str:
    """A full count with thousands separators, unabbreviated."""
    return "–" if v is None else f"{v:,.0f}"


def stat_tile(
    label: str,
    value: str,
    note: str | None = None,
    status: str | None = None,
    help: ui.Tag | None = None,
) -> ui.Tag:
    """Stat tile: label / value / optional secondary note. Optional status dot.

    `help` takes a built help icon (see `app.help_icon`) and sits beside the
    label, where a reader looks when the number itself needs explaining. It is
    passed in rather than built here so the wording stays with the tile.
    """
    dot = (
        ui.span(class_="tile-dot", style=f"background:{STATUS[status]}")
        if status
        else None
    )
    return ui.div(
        ui.div(dot, label, help, class_="tile-label"),
        ui.div(value, class_="tile-value"),
        ui.div(note or "", class_="tile-note"),
        class_="stat-tile",
    )


def verdict(answer: str, ok: bool | None) -> ui.Tag:
    """Yes/no verdict chip for the spend-threshold cards. ok=None renders neutral."""
    cls = "verdict-neutral" if ok is None else ("verdict-ok" if ok else "verdict-flag")
    return ui.span(answer, class_=f"verdict {cls}")
