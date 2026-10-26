"""Conversation grouping and spend visualizations.

These functions consume dataframes and never read Shiny inputs or query DuckDB.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import theme
from theme import apply, empty_fig, fmt_usd

# ---- higher-cost flag ---------------------------------------------------------
# A conversation is higher-cost when it costs more than HIGH_PCT percent of the
# same principal's conversations. Below MIN_BASELINE conversations a percentile
# is just the most expensive one, so nothing is flagged. Attribution uses these
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
# What the conversation spend chart can show. Compare is the two groups side by
# side on several measures; the others group conversations by a category.
CONV_BY = {"compare": "Compare", "model": "Model", "tool": "Harness"}
# The model view names this many models and folds the rest into "other".
CONV_BY_MODELS = 7


def conv_keys(df: pd.DataFrame, by: str) -> tuple[pd.Series, list[str]]:
    """Each conversation's group for the chosen kind (model or tool), and the
    groups in order: ranked by spend, the tail folded into "other".

    `df` has one row per conversation: est, and main_model or main_tool."""
    if df.empty:
        return pd.Series(dtype=object), []
    if by == "model":
        key = df.main_model.fillna("(no model)").replace("", "(no model)")
    else:
        # The harness most of the conversation's spend came through, read from the
        # User-Agent. No header is its own group, not a guess.
        key = df.main_tool.fillna("(no user agent)").replace("", "(no user agent)")
    ranked = df.groupby(key).est.sum().sort_values(ascending=False).index
    keep = set(ranked[:CONV_BY_MODELS])
    key = key.where(key.isin(keep), "other")
    order = [m for m in ranked if m in keep] + (["other"] if len(ranked) > len(keep) else [])
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


def conv_bars_fig(df: pd.DataFrame, mode: str, pct: float) -> go.Figure:
    """Stacked bars for the categorical views (model, harness): one row for all
    conversations, then one per group. Every bar is the same width and split
    into higher-cost and normal spend, so the amber part of one row compares
    straight against the next. A group's share of all spend is in its label.
    `df` is conv_groups' table. With nothing flagged (too few conversations)
    every bar is one colour and the text is the conversation count."""
    total = float(df.spend.sum()) if len(df) else 0.0
    if not total:
        return empty_fig(mode, "No conversation spend in this view")
    t = theme.TOKENS[mode]
    flagged = bool(df.spend_hi.sum())
    rows = [("all conversations", total, float(df.spend_lo.sum()), float(df.spend_hi.sum()),
             int(df.n.sum()))]
    rows += [(str(lab), float(s), float(lo), float(hi), int(k)) for lab, s, lo, hi, k in zip(
        df.label, df.spend.tolist(), df.spend_lo.tolist(), df.spend_hi.tolist(), df.n.tolist())]
    labels = [(lab if len(lab) <= 28 else lab[:27] + "\u2026")
              + ("" if i == 0 or not spend else f"  \u00b7  {spend / total:.0%} of spend")
              for i, (lab, spend, *_) in enumerate(rows)]
    names = (f"higher cost (above p{pct:g})", "normal")
    colours = (theme.STATUS["warning"], t["series"][0])
    fig = go.Figure()
    for j, (name, colour) in enumerate(zip(names, colours)):
        if j == 0 and not flagged:
            continue
        shares, text, hover = [], [], []
        for lab, spend, lo, hi, convs in rows:
            part = (hi, lo)[j]
            share = part / spend if spend else 0.0
            shares.append(share)
            text.append(f"{share:.0%}" if spend and share >= 0.08 else "")
            hover.append(f"{lab}<br>{name}: <b>{fmt_usd(part)}</b> \u00b7 {share:.0%}"
                         if spend else f"{lab}: no estimated spend")
        fig.add_bar(
            y=labels, x=shares, orientation="h", name=name, marker_color=colour,
            text=text, textposition="inside", insidetextanchor="middle",
            textfont={"color": "#ffffff" if j == 0 else t["ink"], "size": 13},
            hovertext=hover, hoverinfo="text", showlegend=flagged,
        )
    # A group with conversations but no priced model: a grey bar, said so.
    unpriced = [i for i, r in enumerate(rows) if not r[1]]
    if unpriced:
        fig.add_bar(
            y=[labels[i] for i in unpriced], x=[1.0] * len(unpriced), orientation="h",
            marker_color=t["grid"], text=[f"unpriced \u00b7 {rows[i][4]:,} convs" for i in unpriced],
            textposition="inside", insidetextanchor="middle", textfont={"color": t["ink"], "size": 13},
            hovertext=[f"{rows[i][0]}: no estimated spend" for i in unpriced], hoverinfo="text",
            showlegend=False,
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="share of spend")
    apply(fig, mode, height=max(200, 38 * len(rows) + 90), showlegend=flagged, hover=True)
    fig.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0})
    return fig


# The measures the compare table sets side by side: label, how to read each from a
# conversation, the unit after a value, what a big gap suggests, and whether it
# is a yes/no per conversation (shown as the share of conversations where it is
# yes) rather than a number (shown as a median).
COMPARE = [
    ("Peak context", lambda d: d.peak_input, "", "Compact or reset the conversation sooner.", False),
    ("Duration", lambda d: d.duration_min, " min", "Start a new conversation sooner.", False),
    ("Using the prompt cache", lambda d: d.uses_cache, "%",
     "A conversation that never reads or writes the cache pays full price for every token "
     "it resends. Check whether the client or integration turns prompt caching on.",
     True),
    ("With a cache miss", lambda d: d.miss_tokens > 0, "%",
     "A cache miss is a request that had to read most of the prompt again at full price. "
     "Breaks over about 5 min cause most of them; so does changing the start of the prompt.",
     True),
]
# A size measure counts as separating the two groups when their medians differ by
# this much. For a share the same bar is ten points.
COMPARE_RATIO = 1.5


def _short_num(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e4:
        return f"{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def conv_compare_rows(df: pd.DataFrame) -> list[dict]:
    """What higher-cost conversations look like next to normal ones, one dict per
    row of the compare table: section, label, normal and high (text), gap (text,
    "" when there is none) and hint (what to do, only for a big gap). Medians,
    not means: a few huge conversations would drag a mean up. `df` has one row
    per conversation: est, high and the columns COMPARE reads. Empty when nothing
    is flagged (too few conversations to split)."""
    if df.empty or not df.high.any():
        return []
    hi = df.high.astype(bool)
    n_hi, n_lo = int(hi.sum()), int((~hi).sum())
    n = n_hi + n_lo
    rows = [dict(section="Split of the total", label="Conversations", gap="", hint="",
                 normal=f"{n_lo:,} \u00b7 {n_lo / n:.0%}", high=f"{n_hi:,} \u00b7 {n_hi / n:.0%}")]
    s_hi, s_lo = float(df.est[hi].sum()), float(df.est[~hi].sum())
    if s_hi + s_lo > 0:
        tot = s_hi + s_lo
        rows.append(dict(section="Split of the total", label="Spend", gap="", hint="",
                         normal=f"{fmt_usd(s_lo)} \u00b7 {s_lo / tot:.0%}",
                         high=f"{fmt_usd(s_hi)} \u00b7 {s_hi / tot:.0%}"))
    for label, get, unit, hint, share in COMPARE:
        x = get(df).where(df.est > 0)
        if not share:
            x = x.where(x > 0)
        lo_v, hi_v = x[~hi].dropna(), x[hi].dropna()
        if lo_v.empty or hi_v.empty:
            continue
        if share:
            lo_n, hi_n = 100 * float(lo_v.astype(bool).mean()), 100 * float(hi_v.astype(bool).mean())
            diff = hi_n - lo_n
            far = 1 + abs(diff) / 20  # ten points apart counts as far as 1.5x does
            gap = "about the same" if far < 1.1 else f"{diff:+.0f} points"
            normal, high = f"{lo_n:.0f}%", f"{hi_n:.0f}%"
        else:
            lo_n, hi_n = float(lo_v.median()), float(hi_v.median())
            ratio = hi_n / lo_n
            far = max(ratio, 1 / ratio)
            gap = "about the same" if far < 1.1 else f"{far:.1f}\u00d7 " + ("higher" if ratio >= 1 else "lower")
            normal, high = f"{_short_num(lo_n)}{unit}", f"{_short_num(hi_n)}{unit}"
        rows.append(dict(section="How often" if share else "Typical conversation (median)",
                         label=label, normal=normal, high=high, gap=gap,
                         hint=hint if far >= COMPARE_RATIO else ""))
    return rows
