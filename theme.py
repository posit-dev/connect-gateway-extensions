"""Chart chrome, validated palette, and small UI components (stat tiles, meters).

Palette is the dataviz reference instance; the 5-slot categorical order was
validated with scripts/validate_palette.js in both light and dark modes.
Light-mode aqua/yellow/magenta sit below 3:1 contrast -> relief rule: every
chart is paired with direct labels and/or a table view.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Callable, cast

import plotly.graph_objects as go
import shinywidgets
from shiny import ui

# shinywidgets.render_widget, typed for what this app hands it. It renders plotly
# figures at runtime, but its annotations only admit ipywidgets.
render_widget = cast(Callable[[Callable[[], Any]], Any], shinywidgets.render_widget)

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
        "surface": "#131a2b",
        "page": "#0a0e1a",
        "ink": "#e8ecf4",
        "ink2": "#9aa6bd",
        "muted": "#6b7896",
        "grid": "#1f2a44",
        "axis": "#2a3656",
        # Saturated on purpose: thin bars on a near-black card need the punch.
        "series": [
            "#3b82f6",
            "#a855f7",
            "#10b981",
            "#f97316",
            "#ec4899",
            "#22c1dc",
            "#eab308",
            "#ef4444",
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

FONT = '"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# fixed slot assignments -- color follows the entity, never its rank
MODEL_ORDER = [
    "claude-sonnet-5",
    "gpt-5",
    "claude-opus-5",
    "gpt-5-mini",
    "claude-haiku-4-5",
]
PROVIDER_ORDER = [
    "anthropic",
    "openai",
    "bedrock",  # what the export writes for AWS Bedrock
    "aws.bedrock",
    "azure.openai",
    "google.vertex",
]

# Unlisted names take the next free slot the first time they are seen, so two of
# them never share a colour (a hash would collide) and each keeps its colour for
# the life of the process.
_UNLISTED: dict[tuple[str, str], int] = {}

ORIGIN_ORDER = ["connect", "workbench", "local"]
TOOL_ORDER = ["Posit Assistant", "Claude Code", "GitHub Copilot", "Codex CLI", "Pi",
              "R (ellmer)", "API SDKs", "Other"]

_SLOTS = {"model": MODEL_ORDER, "provider": PROVIDER_ORDER, "origin": ORIGIN_ORDER, "tool": TOOL_ORDER}


# Buckets that stand for "not known" rather than a real value: gray, like
# "Other", so they never pass for a model, origin or harness.
UNKNOWN_NAMES = {"(unidentified)", "(no user agent)", "(no model)", "other"}


def color(kind: str, name: str, mode: str) -> str:
    order = _SLOTS[kind]
    series = TOKENS[mode]["series"]
    if name in UNKNOWN_NAMES:
        return TOKENS[mode]["muted"]
    if name in order:
        return series[order.index(name) % len(series)]
    # A name with no reserved slot still gets a colour of its own, kept
    # for the life of the process, so it is the same on every render and chart. Gray made
    # every unlisted model look like "other".
    # The count is per kind: one shared counter let a model's slot push a provider
    # round onto a colour another provider already held.
    if (kind, name) not in _UNLISTED:
        _UNLISTED[(kind, name)] = sum(1 for k, _ in _UNLISTED if k == kind)
    slot = _UNLISTED[(kind, name)]
    return series[(len(order) + slot) % len(series)]


def seq_colorscale(mode: str):
    """Low to high, for a heatmap whose minimum is zero.

    Zero is the grid colour, so an empty cell reads as "nothing" rather than as
    the first step of the scale. The ramp then runs faint to strong: dark blue
    to near-white on the dark card, near-white to dark blue on the light one.
    """
    ramp = TOKENS[mode]["seq"]
    # The light ramp's first step is as pale as the light grid; start one step
    # in, so the smallest real value still reads as "something".
    if mode == "light":
        ramp = ramp[1:]
    n = len(ramp) - 1
    # Any value above zero starts at the first step, a hair above the empty cell.
    eps = 1e-6
    return [[0.0, TOKENS[mode]["grid"]], [eps, ramp[0]]] + [
        [eps + (1 - eps) * i / n, c] for i, c in enumerate(ramp) if i
    ]


def usd_axis(fig: go.Figure, axis: str = "x", **kw) -> go.Figure:
    """A dollar axis with plain numbers on it.

    Plotly's default SI suffixes write $0.00002 as "20\u00b5"; a micro-dollar
    reads as nonsense on a spend chart.
    """
    opts = {"tickprefix": "$", "exponentformat": "none", "separatethousands": True, **kw}
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(**opts)
    return fig


def apply(
    fig: go.Figure,
    mode: str,
    *,
    height: int = 320,
    time_axis: bool = False,
    showlegend: bool | None = None,
    y_title: str | None = None,
    hover: bool = False,
) -> go.Figure:
    """Chart chrome. `hover` turns tooltips on.

    Off by default: bars and donuts carry their values as labels, so a tooltip
    only repeats them. Turn it on where it is the only way to read a value (a
    dense line, a stack, a heatmap, an unlabelled scatter).
    """
    t = TOKENS[mode]
    fig.update_layout(
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font={"family": FONT, "size": 14, "color": t["ink2"]},
        height=height,
        margin={"l": 16, "r": 20, "t": 34 if showlegend else 12, "b": 8},
        barcornerradius=3,
        colorway=t["series"],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 15, "color": t["ink"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            # Black with white text, matching the name label on bars and donuts.
            "bgcolor": "#000000",
            "bordercolor": "rgba(255,255,255,0.35)",
            "font": {"family": FONT, "size": 14, "color": "#ffffff"},
        },
        hovermode=("x unified" if time_axis else "closest") if hover else False,
    )
    grid = {
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "linecolor": t["axis"],
        "linewidth": 1,
        "zeroline": False,
        "tickfont": {"color": t["ink"], "size": 15},
        "title_font": {"color": t["ink2"], "size": 14},
    }
    fig.update_xaxes(**grid, showgrid=False)
    fig.update_yaxes(**grid, title_text=y_title)
    # Air between the labels and the axis line: invisible outward ticks push the
    # tick labels away from the plot.
    fig.update_yaxes(ticks="outside", ticklen=10, tickcolor="rgba(0,0,0,0)")
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
    # Annotations do not wrap, and a long reason ran off the card.
    text = "<br>".join(textwrap.wrap(msg, 64)) if "<br>" not in msg else msg
    fig.add_annotation(
        text=text, showarrow=False, font={"color": t["muted"], "size": 13}
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply(fig, mode, height=height)


def _finite(v) -> float | None:
    """v as a float, or None when there is no figure: None, NaN (an empty SUM or
    mean), or an infinite ratio."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def missing(v) -> bool:
    """No figure to show: None, NaN, or infinity."""
    return _finite(v) is None


def fmt_usd(v: float | None) -> str:
    v = _finite(v)
    if v is None:
        return "–"
    # Spend too small to show in cents is still spend: "$0.00" read as free.
    if 0 < abs(v) < 0.005:
        return "<$0.01"
    if abs(v) >= 10_000:
        return f"${v / 1000:,.1f}k"
    if abs(v) >= 100:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def fmt_num(v: float | None) -> str:
    v = _finite(v)
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
def card_usd(v: float | None) -> str:
    """Whole dollars — cents are noise against a range total.

    Under a dollar the cents are the whole figure, so they are kept: "$0" would
    say a principal that spent 40 cents spent nothing.
    """
    v = _finite(v)
    if v is None:
        return "–"
    if 0 < abs(v) < 1:
        return fmt_usd(v)
    return f"${v:,.0f}"


def card_num(v: float | None) -> str:
    """A full count with thousands separators, unabbreviated."""
    v = _finite(v)
    return "–" if v is None else f"{v:,.0f}"


def fmt_pct(v: float | None, digits: int = 0) -> str:
    """A percentage already scaled to 0-100, or a dash when there is none."""
    v = _finite(v)
    return "–" if v is None else f"{v:,.{digits}f}%"


# Longer than this, a tile's value is a name rather than a figure.
TILE_TEXT_MAX = 10


def stat_tile(
    label: str,
    value: str,
    note: str | ui.Tag | None = None,
    status: str | None = None,
    help: ui.Tag | None = None,
) -> ui.Tag:
    """Stat tile: label / value / optional secondary note. Optional status dot.

    `help` takes a built help icon (see `ui_shell.help_icon`) and sits beside the
    label, where a reader looks when the number itself needs explaining. It is
    passed in rather than built here so the wording stays with the tile.
    """
    dot = (
        ui.span(class_="tile-dot", style=f"background:{STATUS[status]}")
        if status
        else None
    )
    # A name (a model, a content title) is longer than any figure and would wrap
    # at its hyphens into a tall, ragged tile. It gets a smaller size on one
    # line, cut with an ellipsis if it still does not fit, and the full name on
    # hover. Figures keep the large size.
    long_text = isinstance(value, str) and len(value) > TILE_TEXT_MAX
    return ui.div(
        ui.div(dot, label, help, class_="tile-label"),
        ui.div(
            value,
            class_="tile-value tile-value-text" if long_text else "tile-value",
            title=value if long_text else None,
        ),
        ui.div(note or "", class_="tile-note"),
        class_="stat-tile",
    )


