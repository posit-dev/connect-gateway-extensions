"""Conversation grouping, cost-driver definitions, and spend visualizations.

These functions consume dataframes and never read Shiny inputs or query DuckDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import theme
from theme import apply, empty_fig

# ---- higher-cost flag ---------------------------------------------------------
# A conversation is higher-cost when it costs more than HIGH_PCT percent of the
# same principal's conversations. Below MIN_BASELINE conversations a percentile
# is just the most expensive one, so nothing is flagged. Principals uses these
# too; they live here so that module can import this one without a cycle.
HIGH_PCT = 75
MIN_BASELINE = 20


def higher_cost(
    est: pd.Series, by: pd.Series | None = None, pct: float = HIGH_PCT
) -> pd.DataFrame:
    """Per row: `high` (flagged) and `line` (the principal's `pct`-th
    percentile, HIGH_PCT unless the caller sets its own).

    `by` names each row's principal; without it every row shares one line.
    A fixed dollar line flags whatever is expensive on a pricey model and misses a
    loop on a cheap one; a principal's own percentile moves with them. Costs
    are long-tailed, so a multiple of the median flagged a quarter of them.
    """
    keys = by.fillna("") if by is not None else pd.Series(0, index=est.index)
    g = est.groupby(keys)
    line = g.transform("quantile", pct / 100)
    high = (g.transform("size") >= MIN_BASELINE) & (est > line)
    return pd.DataFrame({"high": high, "line": line})


# ---- cost drivers -----------------------------------------------------------
# The Conversations tab breaks each conversation's cost into things someone can
# change. All of them are read from request metadata alone: token counts, the
# order and timing of a conversation's requests, model, status, finish reason.
#
# Anthropic's default prompt cache lives 5 minutes from its last use. A request
# that arrives later than that finds it gone and writes the whole history again.
CACHE_TTL_SECONDS = 300
# A conversation whose typical pause is this short, over at least
# AGENT_MIN_CALLS requests, is being driven by a program rather than a person.
AGENT_GAP_SECONDS = 20
AGENT_MIN_CALLS = 20
# The turn limit the agent-loop lever proposes.
AGENT_TURN_CAP = 100
# "Quick" for the model lever: the conversation's whole length in requests.
QUICK_MAX_CALLS = 5
# A conversation is "higher cost" above this percentile of all conversations in
# view. The same cut as the Principals tab's per-user HIGH_PCT; this one is
# taken over the whole population, that one over a principal's own.
CONV_HIGH_PCT = HIGH_PCT
# The context-length lever's compaction point, in input tokens per request.
COMPACT_AT = 100_000
# What the conversation spend chart can group by.
CONV_BY = {
    "context": "Peak context", "model": "Model", "duration": "Duration", "turns": "Turns",
    "pace": "Pace", "cache": "Cache hit", "tool": "Harness",
}
# The model view names this many models and folds the rest into "other".
CONV_BY_MODELS = 7
# The numeric views cut conversations at these percentiles, taken from the data
# in view, so the bands fit whatever traffic a server has: three bands, the
# bottom half, the next 40%, and the top 10%. Uneven on purpose: spend sits in
# the tail, so the tail gets a band of its own rather than half the chart.
CONV_BY_QUANTILES = (0.5, 0.9)
# A percentile cut is kept only if the band above it would hold at least this
# many conversations: on a tiny server the top-10% band would be one or two.
CONV_BY_MIN_BAND = 5


def _time_unit(s: float) -> tuple[float, str]:
    """The unit a length of time reads best in: seconds, minutes or hours."""
    return (1, "s") if s < 60 else (60, "min") if s < 3600 else (3600, "h")


def fmt_secs(s: float) -> str:
    """A length of time, short: 40 s, 12 min, 2.5 h."""
    size, unit = _time_unit(s)
    return f"{s / size:.2g} {unit}" if unit != "s" else f"{s:.0f} s"


def fmt_tokens(v: float) -> str:
    """A token count, short: 800, 45k, 1.2M."""
    if v >= 1e6:
        return f"{v / 1e6:.2g}M"
    if v >= 1e3:
        return f"{v / 1e3:.2g}k"
    return f"{v:.0f}"


def _up2(v: float) -> float:
    """v rounded up to two significant figures."""
    step = 10 ** np.floor(np.log10(v)) / 10
    return float(np.ceil(round(v / step, 9)) * step)


def _nice(v: float, kind: str) -> float:
    """A cut point rounded up to something a person would say.

    Times round in the unit they are shown in, so a cut reads 2.4 min rather
    than the 2.33 min that rounding 140 seconds would give.
    """
    if kind == "int":
        return float(np.ceil(v))
    if kind == "share":
        return float(np.ceil(round(v * 100, 9)) / 100)
    if kind == "tokens":
        return _up2(v) if v > 0 else 0.0
    if v <= 0:
        return 0.0
    size, _unit = _time_unit(v)
    return _up2(v / size) * size


def quantile_bands(
    vals: pd.Series, kind: str, tail: str = "high"
) -> tuple[pd.Series, list[str]]:
    """Band each value at CONV_BY_QUANTILES of the values themselves.

    `kind` says how to round and name the cuts: "int" (counts), "share"
    (0-1), "secs" (seconds). `tail` is the end that gets the narrow bands:
    "high" cuts at p50/90, "low" mirrors that to p10/50, for a
    measure whose costly cases are the small values (a fast pace). Returns
    each value's label and the labels in order. Cuts that round to the same
    number merge, so a mostly-constant column (every conversation one request
    long, say) gives fewer bands.
    """
    n = len(vals)
    if tail == "high":
        qs = [q for q in CONV_BY_QUANTILES if (1 - q) * n >= CONV_BY_MIN_BAND]
    else:
        qs = [1 - q for q in CONV_BY_QUANTILES if (1 - q) * n >= CONV_BY_MIN_BAND]
    cuts = sorted({_nice(float(vals.quantile(q)), kind) for q in qs})
    cuts = [c for c in cuts if c < vals.max()]  # nothing would sit above it
    say = {
        "int": lambda v: f"{v:,.0f}",
        "share": lambda v: f"{100 * v:.0f}%",
        "secs": fmt_secs,
        "tokens": fmt_tokens,
    }[kind]
    while True:
        edges = [-np.inf, *cuts, np.inf]
        labels = _band_labels(vals, edges, kind, say)
        if len(set(labels)) == len(labels):
            return pd.cut(vals, edges, right=True, labels=labels), labels
        # Two cuts that read the same: keep one.
        dup = next(i for i in range(1, len(labels)) if labels[i] in labels[:i])
        cuts.pop(min(dup, len(cuts)) - 1)


SINGLE = "single request"


def conv_keys(df: pd.DataFrame, by: str) -> tuple[pd.Series, list[str]]:
    """Each conversation's group for the chosen kind, and the groups in order.

    `df` has one row per conversation: est, main_model, duration_min, calls,
    cache_hit, peak_input, med_idle. Every grouping comes from the data in view: models are
    ranked by spend with the tail folded into "other", and the numeric views
    are cut at percentiles of the conversations themselves (see quantile_bands).
    """
    if df.empty:
        return pd.Series(dtype=object), []
    if by == "model":
        key = df.main_model.fillna("(no model)").replace("", "(no model)")
        ranked = df.groupby(key).est.sum().sort_values(ascending=False).index
        keep = set(ranked[:CONV_BY_MODELS])
        key = key.where(key.isin(keep), "other")
        order = [m for m in ranked if m in keep] + (["other"] if len(ranked) > len(keep) else [])
    elif by == "tool":
        # The harness most of the conversation's spend came through, read from the
        # User-Agent. No header is its own group, not a guess.
        key = df.main_tool.fillna("(no user agent)").replace("", "(no user agent)")
        ranked = df.groupby(key).est.sum().sort_values(ascending=False).index
        keep = set(ranked[:CONV_BY_MODELS])
        key = key.where(key.isin(keep), "other")
        order = [m for m in ranked if m in keep] + (["other"] if len(ranked) > len(keep) else [])
    elif by == "pace":
        # A one-request conversation has no pause to measure; it gets its
        # own group rather than a made-up pace, and stays out of the cuts.
        idle = df.med_idle.clip(lower=0)
        paced = idle.notna()
        key = pd.Series(SINGLE, index=df.index, dtype=object)
        order = [SINGLE] if (~paced).any() else []
        if paced.any():
            # Fast is the costly end here (agent loops), so the narrow
            # bands go to the low tail.
            bands, labels = quantile_bands(idle[paced], "secs", tail="low")
            key[paced] = bands.astype(str)
            order += labels
    else:
        vals, kind = {
            "duration": (df.duration_min * 60, "secs"),
            "turns": (df.calls, "int"),
            "cache": (df.cache_hit.fillna(0), "share"),
            "context": (df.peak_input.fillna(0), "tokens"),
        }[by]
        key, order = quantile_bands(vals, kind)
        key = key.astype(str)
    return key, order


def conv_groups(df: pd.DataFrame, key: pd.Series, order: list[str]) -> pd.DataFrame:
    """Totals per group, then the same split by the higher-cost flag (spend_hi,
    n_hi for flagged conversations; spend_lo, n_lo for the rest)."""
    if df.empty:
        return pd.DataFrame(columns=["label", "spend", "n"])
    hi = df.high.astype(bool)
    g = df.groupby(key).agg(spend=("est", "sum"), n=("est", "size"))
    for suffix, part in (("hi", df[hi]), ("lo", df[~hi])):
        s = part.groupby(key[part.index]).agg(spend=("est", "sum"), n=("est", "size"))
        g[f"spend_{suffix}"] = s.spend.reindex(g.index, fill_value=0)
        g[f"n_{suffix}"] = s.n.reindex(g.index, fill_value=0)
    return g.reindex(order, fill_value=0).rename_axis("label").reset_index()


CONV_BY_AXIS = {
    "model": "model the conversation spent most on",
    "tool": "harness the conversation spent most through (from User-Agent)",
    "duration": "first to last request",
    "turns": "requests in the conversation",
    "pace": "typical pause between requests (median)",
    "cache": "input read from the prompt cache",
    "context": "peak context: largest input in the conversation (tokens, cached included)",
}


def conv_spend_fig(df: pd.DataFrame, by: str, mode: str, pct: float) -> go.Figure:
    """Two bars per group: higher-cost conversations and the rest. Each bar is
    plotted and labelled as its share of its own series' spend (so each series
    sums to 100%, and bar heights match the labels); share of conversations sits under the group's name."""
    total = float(df.spend.sum()) if len(df) else 0.0
    if not total:
        return empty_fig(mode, "No conversation spend in this view")
    t = theme.TOKENS[mode]
    count = max(int(df.n.sum()), 1)
    ticks = [f"{lab}<br><sub>{100 * n / count:.0f}% of convs</sub>"
             for lab, n in zip(df.label.astype(str), df.n)]
    fig = go.Figure()
    # Normal first, so it sits on the left of each pair.
    for suffix, name, colour in (
        ("lo", "normal", t["series"][0]),
        ("hi", f"higher cost (above p{pct:g})", theme.STATUS["warning"]),
    ):
        spend, n = df[f"spend_{suffix}"], df[f"n_{suffix}"]
        # Each series is a share of its own total, so normal sums to 100% and
        # higher cost sums to 100%.
        series_total = float(spend.sum())
        share = spend / series_total * 100 if series_total else spend * 0.0
        fig.add_bar(
            x=ticks, y=share, name=name, marker={"color": colour},
            # a sliver would only read "0%"; hover still has its figure
            text=[f"{p:.0f}%" if p >= 0.5 else "" for p in share],
            textposition="outside", textfont={"color": t["ink"], "size": 13},
            customdata=[(k, s / k if k else 0, s) for s, k in zip(spend, n)],
            hovertemplate=(
                f"{name}: <b>%{{y:.1f}}%</b> of its spend"
                "<br>%{customdata[2]:$,.2f} \u00b7 %{customdata[0]:,} conversations"
                "<br>avg %{customdata[1]:$,.2f} each<extra></extra>"
            ),
        )
    top = max(
        (float(df[f"spend_{x}"].max()) / float(df[f"spend_{x}"].sum()) * 100
         for x in ("lo", "hi") if df[f"spend_{x}"].sum()),
        default=100.0,
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="share of the series' spend", ticksuffix="%",
                     tickformat=".0f", range=[0, top * 1.18])
    fig.update_xaxes(title_text=CONV_BY_AXIS[by])
    return apply(fig, mode, height=340, showlegend=True, hover=True)


def conv_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group, each share taken of this table's own totals."""
    total, count = float(df.spend.sum()), max(int(df.n.sum()), 1)
    return pd.DataFrame({
        "group": df.label.astype(str),
        "conversations": df.n,
        "pct_of_conversations": (100 * df.n / count).round(1),
        "est_spend_usd": df.spend.round(2),
        "pct_of_spend": (100 * df.spend / total).round(1) if total else 0.0,
        "avg_usd": (df.spend / df.n.replace(0, np.nan)).round(2),
        "higher_cost_conversations": df.n_hi,
        "higher_cost_spend_usd": df.spend_hi.round(2),
    })


def _band_labels(vals: pd.Series, edges: list, kind: str, say) -> list[str]:
    """Names for the bands between consecutive edges (right-inclusive)."""
    def span(a: str, b: str) -> str:
        """a\u2013b, saying a shared unit once: 2.4\u20136.7 min, 40\u201379%."""
        if a.endswith("%") and b.endswith("%"):
            return f"{a[:-1]}\u2013{b}"
        ua, ub = a.rpartition(" ")[2], b.rpartition(" ")[2]
        if " " in a and ua == ub:
            return f"{a.rpartition(' ')[0]}\u2013{b}"
        return f"{a}\u2013{b}"

    labels = []
    top = vals.max()
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if kind == "int":
            first = int(vals.min()) if i == 0 else int(lo) + 1
            last = int(top) if hi == np.inf else int(hi)
            if first == last:
                labels.append(say(first))
            elif hi == np.inf and len(edges) > 2:
                labels.append(f"{say(first)}+")
            else:
                labels.append(span(say(first), say(last)))
        elif i == 0:
            if hi == np.inf:
                labels.append(say(top) if vals.min() == top else "all")
            else:
                labels.append(say(hi) if hi <= 0 else f"\u2264 {say(hi)}")
        elif hi == np.inf:
            labels.append(f"> {say(lo)}")
        elif lo <= 0:
            # The band above an exact-zero band: everything above zero up to hi.
            labels.append(f"\u2264 {say(hi)}")
        else:
            labels.append(span(say(lo), say(hi)))
    return labels

# Each lever: key (also the savings column, sav_<key>), name, how the saving is
# estimated, and what to do about it. Levers can apply to the same request, so
# their savings overlap and are never summed.
LEVERS = [
    (
        "rewarm", "Cache expired mid-conversation",
        f"A request that came more than {CACHE_TTL_SECONDS // 60} minutes after the "
        "previous one ended found the prompt cache gone, and paid to write the "
        "history again. Saving: those re-written tokens priced as cache reads.",
        "Use the 1-hour cache for long interactive sessions, or start a fresh "
        "conversation after a break instead of resuming a long one.",
    ),
    (
        "context", "Context past the compaction point",
        "Input above the compaction threshold, priced at what that request paid "
        "per input token. An upper bound: it ignores the cost of the summary "
        "a compaction makes.",
        "Compact or clear history sooner (/compact in Claude Code, or a history "
        "window in the app).",
    ),
    (
        "uncached", "No prompt caching",
        "Requests that carried the previous turn's history with no cache read "
        "or write, on a model that offers caching. Saving: that history priced "
        "as a cache read, less the premium for writing the new part.",
        "Turn on prompt caching in the client or app (cache_control on the "
        "system prompt and history).",
    ),
    (
        "model", "Quick chats on a pricier model",
        f"Conversations of {QUICK_MAX_CALLS} requests or fewer on a model listed "
        "above the comparison model, repriced at the comparison model's rates.",
        "Make the comparison model the default for short chats; keep the "
        "bigger model for long work.",
    ),
    (
        "waste", "Failed or cut-off requests",
        "The full estimated cost of requests that failed (HTTP 4xx/5xx) or "
        "stopped at the output limit, which callers usually re-run.",
        "Fix the failing callers; raise max_tokens where answers are cut off.",
    ),
    (
        "loop", "Agent loops past the turn cap",
        f"Conversations paced like a program (median pause \u2264 "
        f"{AGENT_GAP_SECONDS}s over {AGENT_MIN_CALLS}+ requests). Saving: every "
        f"request after turn {AGENT_TURN_CAP}.",
        "Give autonomous agents a turn or budget limit.",
    ),
]
LEVER_NAME = {k: name for k, name, *_ in LEVERS}

