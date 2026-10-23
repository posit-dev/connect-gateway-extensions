"""Conversation grouping and spend visualizations.

These functions consume dataframes and never read Shiny inputs or query DuckDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import theme
from theme import apply, empty_fig

# ---- higher-cost flag ---------------------------------------------------------
# A conversation is higher-cost when it costs more than HIGH_PCT percent of the
# same principal's conversations. Below MIN_BASELINE conversations a percentile
# is just the most expensive one, so nothing is flagged. Profiles uses these
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


# ---- grouping -----------------------------------------------------------------
# What the conversation spend chart can group by.
CONV_BY = {
    "context": "Peak context", "model": "Model", "duration": "Duration", "turns": "Turns",
    "pace": "Pace", "cache": "Cache misses", "tool": "Harness",
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
    cache_miss, peak_input, med_idle. Every grouping comes from the data in view: models are
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
            "cache": (df.cache_miss.fillna(0), "tokens"),
            "context": (df.peak_input.fillna(0), "tokens"),
        }[by]
        key, order = quantile_bands(vals, kind)
        key = key.astype(str)
    return key, order


def conv_groups(df: pd.DataFrame, key: pd.Series, order: list[str]) -> pd.DataFrame:
    """Totals per group, then the same split by the higher-cost flag (spend_hi,
    n_hi for flagged conversations; spend_lo, n_lo for the rest)."""
    if df.empty:
        # Every column the chart and table read, so a principal with no
        # conversations in view gets an empty table rather than an error.
        return pd.DataFrame(
            columns=["label", "spend", "n", "spend_hi", "n_hi", "spend_lo", "n_lo"]
        )
    hi = df.high.astype(bool)
    g = df.groupby(key).agg(spend=("est", "sum"), n=("est", "size"))
    for suffix, part in (("hi", df[hi]), ("lo", df[~hi])):
        s = part.groupby(key[part.index]).agg(spend=("est", "sum"), n=("est", "size"))
        g[f"spend_{suffix}"] = s.spend.reindex(g.index, fill_value=0)
        g[f"n_{suffix}"] = s.n.reindex(g.index, fill_value=0)
    return g.reindex(order, fill_value=0).rename_axis("label").reset_index()


# A group's donut number turns amber when its higher-cost share is at least this
# far above the overall share: a few points either way is noise.
DONUT_ABOVE = 0.05


def conv_donuts_fig(df: pd.DataFrame, mode: str, pct: float) -> go.Figure:
    """A row of small donuts for the categorical views (model, harness): one for
    all conversations, then one per group, each split into normal and
    higher-cost spend. The number inside is the higher-cost share, amber when it
    is well above the overall one; under it, the group's share of all spend.
    `df` is conv_groups' table. With nothing flagged (too few conversations)
    every ring is one colour and the number is the group's share of spend."""
    total = float(df.spend.sum()) if len(df) else 0.0
    if not total:
        return empty_fig(mode, "No conversation spend in this view")
    t = theme.TOKENS[mode]
    flagged = bool(df.spend_hi.sum())
    overall = float(df.spend_hi.sum()) / total
    rows = [("all conversations", total, float(df.spend_lo.sum()), float(df.spend_hi.sum()),
             int(df.n.sum()))]
    rows += [(str(lab), float(s), float(lo), float(hi), int(k)) for lab, s, lo, hi, k in zip(
        df.label, df.spend.tolist(), df.spend_lo.tolist(), df.spend_hi.tolist(), df.n.tolist())]
    n = len(rows)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "domain"}] * n],
                        subplot_titles=[lab if len(lab) <= 18 else lab[:17] + "\u2026"
                                        for lab, *_ in rows])
    names = ("normal", f"higher cost (above p{pct:g})")
    for i, (lab, spend, lo, hi, convs) in enumerate(rows):
        if not spend:
            # Conversations, but none on a priced model: a grey ring, said so.
            labels, values, colours = ["unpriced"], [1.0], [t["grid"]]
            text = f"<b>unpriced</b><br><span style='font-size:12px'>{convs:,} convs</span>"
        else:
            labels, values = list(names), [lo, hi]
            colours = [t["series"][0], theme.STATUS["warning"]]
            share = hi / spend if flagged else spend / total
            amber = flagged and i > 0 and share - overall >= DONUT_ABOVE
            colour = theme.STATUS["warning"] if amber else t["ink"]
            sub = "of all spend" if i == 0 else f"{spend / total:.0%} of spend"
            if not flagged:
                sub = f"{convs:,} convs"
            text = (f"<b style='color:{colour}'>{share:.0%}</b><br>"
                    f"<span style='font-size:12px'>{sub}</span>")
        fig.add_pie(
            labels=labels, values=values,
            marker={"colors": colours, "line": {"color": t["surface"], "width": 2}},
            hole=0.62, sort=False, direction="clockwise", textinfo="none",
            showlegend=i == 0 and flagged, row=1, col=i + 1,
            title={"text": text, "font": {"size": 17, "color": t["ink"]}},
            customdata=[lab] * len(values),
            hovertemplate=("%{customdata}<br>%{label}: <b>%{value:$,.2f}</b>"
                           " \u00b7 %{percent}<extra></extra>") if spend else "%{customdata}: "
                          "no estimated spend<extra></extra>",
        )
    fig.update_annotations(font={"size": 13, "color": t["ink2"]})
    apply(fig, mode, height=250, showlegend=flagged, hover=True)
    # The legend goes under the row: above it, it ran into the donut titles.
    fig.update_layout(legend={"yanchor": "top", "y": -0.04, "x": 0},
                      margin={"t": 30, "b": 40 if flagged else 8})
    return fig


# The numeric views the cumulative chart draws: how to read the measure from a
# conversation, the axis type, the axis title and the hover format. Model and
# harness have no order to draw a line along, so they keep the bars.
CURVE_X = {
    "context": (lambda d: d.peak_input, "log",
                "peak context: largest input in the conversation (tokens, cached included)",
                ",.0f"),
    "duration": (lambda d: d.duration_min, "log", "first to last request (minutes)", ",.1f"),
    "turns": (lambda d: d.calls, "log", "requests in the conversation", ",.0f"),
    # A pause of 0 s has no place on a log axis; a tenth of a second reads the same.
    "pace": (lambda d: d.med_idle.clip(lower=0.1), "log",
             "typical pause between requests (seconds, median)", ",.1f"),
    "cache": (lambda d: d.cache_miss, "log",
              "input not read from the prompt cache (tokens, summed over the conversation)",
              ",.0f"),
}


def conv_cumulative_fig(df: pd.DataFrame, by: str, mode: str, pct: float) -> go.Figure:
    """Cumulative spend along one measure, one line for the higher-cost
    conversations and one for the rest. Each line is the share of its own
    spend that sits in conversations at or below each value, so both run from
    0 to 100%. Where a line climbs is where that spend sits; the further apart
    the two lines, the more the measure separates the costly conversations
    from the rest. `df` has one row per conversation: est, high and the
    columns CURVE_X reads."""
    get, axis_type, title, fmt = CURVE_X[by]
    x = get(df) if len(df) else pd.Series(dtype=float)
    est = df.est if len(df) else pd.Series(dtype=float)
    ok = x.notna() & (est > 0)
    if axis_type == "log":
        ok &= x > 0
    d = pd.DataFrame({"x": x[ok], "est": est[ok],
                      "high": df.high[ok].astype(bool) if len(df) else []})
    if d.empty:
        return empty_fig(mode, "No conversation spend in this view")
    t = theme.TOKENS[mode]
    # With nothing flagged (too few conversations) there is one line, not two.
    series = (
        [(False, "normal", t["series"][0]),
         (True, f"higher cost (above p{pct:g})", theme.STATUS["warning"])]
        if d.high.any() else [(False, "all conversations", t["series"][0])]
    )
    fig = go.Figure()
    for flag, name, colour in series:
        part = d[d.high == flag]
        if part.empty:
            continue
        g = part.groupby("x", sort=True).est.agg(["sum", "size"])
        cum = (g["sum"].cumsum() / g["sum"].sum() * 100).to_numpy()
        n = g["size"].cumsum().to_numpy()
        # Start at 0% on the smallest value, so the first step shows as a rise.
        fig.add_scatter(
            x=np.r_[g.index[0], g.index], y=np.r_[0, cum], customdata=np.r_[0, n],
            mode="lines", line_shape="hv", name=name,
            line={"color": colour, "width": 2.5},
            hovertemplate=(
                f"{name}<br>at or below %{{x:{fmt}}}: <b>%{{y:.0f}}%</b> of its spend"
                "<br>%{customdata:,.0f} conversations<extra></extra>"
            ),
        )
    # Half of each line's spend sits left of where it crosses this.
    fig.add_hline(y=50, line={"color": t["axis"], "width": 1, "dash": "dot"})
    fig.update_xaxes(type=axis_type, title_text=title)
    # Ticks at powers of ten (1k, 10k, 100k), not the 2s and 5s between them,
    # unless the data sits inside one decade, where that would leave no ticks.
    if axis_type == "log" and d.x.max() / d.x.min() >= 10:
        fig.update_xaxes(dtick=1, tickformat="~s")
    fig.update_yaxes(title_text="share of spend at or below", ticksuffix="%",
                     range=[0, 104])
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

