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

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from starlette.requests import Request

import principals
import theme
import usage
from theme import apply, card_num, card_usd, color, empty_fig, fmt_num, fmt_usd

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


def _build_est_cost(prices: dict, uncached: bool = False) -> str:
    """The est_cost SQL for one price catalog.

    `uncached` prices the same traffic as if no prompt cache existed: cache
    reads and writes are charged at the fresh-input rate. The difference from
    the real figure is what caching saved (net of any write premium).
    """

    def term(i, o, cw, cr):
        return _rate_term(i, o, i, i) if uncached else _rate_term(i, o, cw, cr)

    return _build_case(prices, term)


def _build_case(prices: dict, term) -> str:
    """A CASE over the catalog that yields `term(i, o, cw, cr)` for each row, /1e6.

    The CASE tries the most specific branch first: the request's own region,
    then the provider's base rate for the model, then -- for a provider the
    catalog does not price -- the model's Anthropic rate, then $0. est_cost and
    the per-part rates (rate_sql) share this ladder, so the parts always add up
    to the estimate.
    """
    region_whens, base_whens, model_default = [], [], {}

    for (provider, region, model), (i, o, cw, cr) in prices.items():
        names = ", ".join(
            _sql_str(a) for a in _PROVIDER_ALIASES.get(provider, (provider,))
        )
        if region:
            region_whens.append(
                f"WHEN provider IN ({names}) AND region = {_sql_str(region)}"
                f" AND model = {_sql_str(model)} THEN {term(i, o, cw, cr)}"
            )
        else:
            base_whens.append(
                f"WHEN provider IN ({names}) AND model = {_sql_str(model)}"
                f" THEN {term(i, o, cw, cr)}"
            )
            if provider == "anthropic":
                model_default[model] = (i, o, cw, cr)
    # Model-only fallback, tried after every provider-keyed branch.
    fallback = [
        f"WHEN model = {_sql_str(m)} THEN {term(*r)}"
        for m, r in model_default.items()
    ]
    whens = region_whens + base_whens + fallback
    return f"CASE {' '.join(whens)} ELSE 0 END / 1e6"


# (prices object the SQL was built from, the SQL). The cache holds the object
# itself, not id(prices): a freed catalog can pass its id() to a new object and
# cause a wrong hit, but an `is` test on the object cannot.
_EST_COST_CACHE: tuple = (object(), "")
_NOCACHE_COST_CACHE: tuple = (object(), "")


_RATES_CACHE: tuple = (object(), {})

# The four prices a request pays, in the order the catalog stores them.
RATE_PARTS = ("i", "o", "cw", "cr")


def rate_sql() -> dict[str, str]:
    """USD per token for each part (fresh input, output, cache write, cache read).

    Each is a SQL expression over provider, region and model only, so it can be
    evaluated once per distinct model rather than once per row.
    """
    global _RATES_CACHE
    prices = usage.PRICES
    if _RATES_CACHE[0] is not prices:
        _RATES_CACHE = (
            prices,
            {
                p: _build_case(prices, lambda *r, k=k: f"{float(r[k])}")
                for k, p in enumerate(RATE_PARTS)
            },
        )
    return _RATES_CACHE[1]


def model_rates(model: str) -> tuple | None:
    """(input, output, cache write, cache read) in USD per token for a model's
    base list rate, Anthropic's first, or None when the catalog has no price."""
    hits = {p: r for (p, region, m), r in usage.PRICES.items() if m == model and not region}
    rates = hits.get("anthropic") or next(iter(hits.values()), None)
    return tuple(v / 1e6 for v in rates) if rates else None


def est_cost_uncached() -> str:
    """est_cost with no prompt-cache discount (see _build_est_cost)."""
    global _NOCACHE_COST_CACHE
    prices = usage.PRICES
    if _NOCACHE_COST_CACHE[0] is not prices:
        _NOCACHE_COST_CACHE = (prices, _build_est_cost(prices, uncached=True))
    return _NOCACHE_COST_CACHE[1]


def est_cost() -> str:
    """Estimated spend per row, as a SQL expression over gateway_events.

    The SQL is rebuilt only when the catalog changes. A refresh replaces
    usage.PRICES with a new object, which misses the cache and forces one
    rebuild; every call in between reuses the string. With a catalog, the price
    comes from (provider, region, model): the request's own region if it has a
    listed rate, else the provider's base rate, else the model's Anthropic rate,
    so a provider-name mismatch does not read $0. usage.load() refuses to start
    without a catalog, so there is always one here.
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
# view. The same cut as the Principals tab's per-user principals.HIGH_PCT; this
# one is taken over the whole population, that one over a principal's own.
CONV_HIGH_PCT = principals.HIGH_PCT
# The context-length lever's compaction point, in input tokens per request.
COMPACT_AT = 100_000
# What the conversation spend chart can group by.
CONV_BY = {
    "model": "Model", "duration": "Duration", "turns": "Turns",
    "pace": "Pace", "cache": "Cache hit", "tool": "Tool",
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
    cuts = sorted({_nice(vals.quantile(q), kind) for q in qs})
    cuts = [c for c in cuts if c < vals.max()]  # nothing would sit above it
    say = {
        "int": lambda v: f"{v:,.0f}",
        "share": lambda v: f"{100 * v:.0f}%",
        "secs": fmt_secs,
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
    cache_hit, med_idle. Every grouping comes from the data in view: models are
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
        # The tool most of the conversation's spend came through, read from the
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
    "tool": "tool the conversation spent most through (from User-Agent)",
    "duration": "first to last request",
    "turns": "requests in the conversation",
    "pace": "typical pause between requests (median)",
    "cache": "input read from the prompt cache",
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

MEASURES = {"spend": "Dollars", "tokens": "Tokens"}

# Where the call ran. Connect content always stamps a content guid; a Workbench
# session and a direct API caller never do. Both is the union, and it groups by
# user so one person's Connect and Workbench usage add up.
SOURCES = {"both": "Both", "connect": "Connect", "workbench": "Workbench"}

# Only Connect can group by content: a Workbench call has no content item, and a
# Both view has to use the axis the two sources share.
GROUP_BY = {"username": "User", "content_guid": "Content"}

GROUP_NOUN = {"username": "user", "content_guid": "content"}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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
            for r in usage.q("""SELECT username, MAX(user_email) AS email
                                FROM gateway_events WHERE username IS NOT NULL
                                GROUP BY 1 ORDER BY 1""").itertuples()
        },
        # Workbench sessions: the id is the value, the name (when there is one)
        # the label, so typing either finds the session.
        "session_labels": {
            r.sid: (f"{r.name} \u00b7 {r.sid}" if r.name and r.name != r.sid else r.sid)
            for r in usage.q("""SELECT workbench_session_id AS sid,
                                      MAX(workbench_session_name) AS name
                               FROM gateway_events
                               WHERE workbench_session_id IS NOT NULL
                               GROUP BY 1 ORDER BY 2, 1""").itertuples()
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
            for r in usage.q("""SELECT content_guid AS guid,
                                      MAX(content_owner_username) AS owner
                               FROM gateway_events WHERE content_guid IS NOT NULL
                               GROUP BY 1 ORDER BY 1""").itertuples()
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


def sql_str(x: str) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def meta_expr(key: str) -> str:
    """SQL for one metadata key's value, as text."""
    path = '$."' + key.replace('"', '\\"') + '"'
    return f"json_extract_string(request_metadata, {sql_str(path)})"


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
            out[meta_id(key, v)] = f"{key}: {v}"
    return out


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
    """How many models the Connect price catalog prices, for the sidebar."""
    n = info.get("priced_models") or 0
    return f"prices: {n} {plural(n, 'model')} from Connect catalog"


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


def name_content(df: pd.DataFrame, dim: str, col: str = "principal") -> pd.DataFrame:
    """Show a content item by title, keeping its guid in a column of its own.

    Only the content axis changes. A title is not unique and a deleted item has
    none, so the guid stays beside it (and in any CSV export) as the identifier.
    """
    if dim != "content_guid" or df.empty:
        return df
    df = df.copy()
    guids = df[col].astype(str)
    df.insert(df.columns.get_loc(col) + 1, "guid", guids)
    df[col] = [usage.content_name(g) or g for g in guids]
    return df


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
    "tokens": lambda v: f"{v:,.0f}",
    "input_tokens": lambda v: f"{v:,.0f}",
    "output_tokens": lambda v: f"{v:,.0f}",
    "cache_read_tokens": lambda v: f"{v:,.0f}",
    "cache_pct": lambda v: f"{v:,.1f}%",
    "cache_savings_usd": lambda v: f"${v:,.0f}",
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
}


def formatted(df: pd.DataFrame) -> pd.DataFrame:
    """A display copy of the frame, with numbers rendered as strings.

    Only what is drawn is touched: the CSV exports read the unformatted frames.
    """
    out = df.copy()
    for col, fmt in DISPLAY_FMT.items():
        if col in out.columns:
            out[col] = [None if pd.isna(v) else fmt(v) for v in out[col]]
    return out


def table_header(title: str, dl_id: str, help: ui.Tag | None = None) -> ui.Tag:
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


def _shell(*panels, sidebar, id, header, title, **_) -> ui.Tag:
    """Page chrome: top bar, nav rail, global filter strip, then the page.

    The pages are a hidden navset so that the existing `main_tabs` id (and the
    bookmark that stores it) keeps working; the rail is what a reader clicks.
    """
    c = CATALOG
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
        ui.input_radio_buttons("preset", None, PRESETS, selected=DEFAULT_PRESET, inline=True),
        ui.input_date_range(
            "date_range",
            None,
            start=preset_range(DEFAULT_PRESET, c["start"], c["end"])[0],
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


def app_ui(request: Request):
    return _shell(
        ui.nav_panel(
            "Overview",
            page_header("Consumption Overview"),
            ui.output_ui("kpis"),
            ui.div(
                ui.div(
                    ui.h6("Estimated spend by", class_="row-question"),
                    ui.div(
                        ui.input_radio_buttons(
                            "origin_tool", None, {"origin": "Origin", "tool": "Tool"},
                            selected="origin", inline=True,
                        ),
                        class_="seg",
                    ),
                    help_icon(
                        ui.p(
                            ui.strong("Origin"), " is where the request came from: "
                            " Connect, a Workbench, or "
                            "client running locally."
                        ),
                        ui.p(
                            ui.strong("Tool"), " is the software that made the call. This is derived "
                            "from the User-Agent header the caller sent.",
                            class_="mb-0",
                        ),
                    ),
                    class_="hdr-flex-left origin-tool-head",
                ),
                ui.layout_columns(
                    ui.card(ui.card_header("Total"), output_widget("origin_total")),
                    ui.card(ui.card_header("Share"), output_widget("origin_share")),
                    col_widths=[7, 5],
                ),
                class_="question-block",
            ),
            chart_row("Estimated spend by provider", "provider_total", "provider_share"),
            ui.div(
                ui.h6("Estimated spend by model", class_="row-question"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.span("Over time"),
                                ui.div(
                                    ui.input_radio_buttons(
                                        "grain",
                                        None,
                                        {"day": "Daily", "week": "Weekly", "month": "Monthly"},
                                        selected="day",
                                        inline=True,
                                    ),
                                    class_="seg",
                                ),
                                class_="hdr-flex",
                            )
                        ),
                        output_widget("model_time"),
                    ),
                    ui.card(ui.card_header("Share"), output_widget("model_share")),
                    col_widths=[7, 5],
                ),
                class_="question-block",
            ),
        ),
        ui.nav_panel(
            "Attribution",
            page_header("Attribution"),
            ui.div(
                ui.span("Show", class_="strip-label"),
                ui.div(
                    ui.input_radio_buttons(
                        "attr_kind",
                        None,
                        {"all": "All", "content": "Content", "user": "Users", "session": "Sessions"},
                        selected="all",
                        inline=True,
                    ),
                    class_="seg",
                ),
                ui.span("Order by", class_="strip-label"),
                ui.div(
                    ui.input_radio_buttons(
                        "attr_order",
                        None,
                        {"spend": "Spend", "type": "Type"},
                        selected="spend",
                        inline=True,
                    ),
                    class_="seg",
                ),
                class_="page-switch",
            ),
            ui.output_ui("attr_kpis"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Top 20 by estimated spend"),
                    output_widget("attr_bars"),
                ),
                ui.card(
                    ui.card_header("Top 20 by estimated cost per request"),
                    output_widget("attr_rate_bars"),
                ),
                col_widths=[6, 6],
            ),
            ui.card(
                ui.card_header("Outliers: requests against estimated spend"),
                output_widget("attr_scatter"),
            ),
            ui.card(
                table_header("Outliers", "dl_attr"),
                ui.card_body(
                    ui.output_data_frame("attr_table"),
                ),
            ),
        ),
        ui.nav_panel(
            "Conversations",
            page_header(
                "Conversations",
                help=help_icon(
                    "What is a conversation?",
                    ui.p(
                        "The gateway groups requests by conversation id. A client "
                        "can send its own in a Session-Id header; when it does not, "
                        "Connect derives one from the user, the content and the "
                        "first message of the request."
                    ),
                    ui.p(
                        "A derived id stays the same only while the client resends "
                        "the same history. If a client compacts or rewrites its "
                        "history, one working session shows up as several "
                        "conversations, so long agent sessions can be "
                        "undercounted. A request with neither id nor first message "
                        "belongs to no conversation and is left out here.",
                        class_="mb-0",
                    ),
                ),
            ),
            ui.output_ui("conv_kpis"),
            ui.h6("Where does the money go?", class_="row-question"),
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.div(
                            ui.span("Conversation spend by"),
                            class_="hdr-flex-left",
                        ),
                        ui.div(
                            ui.input_radio_buttons(
                                "conv_by", None, CONV_BY, selected="model", inline=True
                            ),
                            class_="seg",
                        ),
                        class_="hdr-flex",
                    )
                ),
                output_widget("conv_spend_by"),
                ui.output_ui("conv_spend_note"),
            ),
            ui.card(
                table_header(
                    "All conversations",
                    "dl_conversations",
                ),
                ui.card_body(
                    ui.output_ui("conv_detail"),
                    ui.output_data_frame("top_sessions"),
                ),
            ),
        ),
        ui.nav_panel(
            "Principals",
            # Prototype drill-down: see principals.py.
            *principals.panel(page_header, help_icon, table_header, CONV_BY),
        ),
        ui.nav_panel(
            "Adoption",
            page_header(
                "Adoption",
                help=help_icon(
                    "What counts as active?",
                    ui.p(
                        "A user is active on a day if the gateway recorded at "
                        "least one request charged to them. Each line is a "
                        "rolling window ending on that day:"
                    ),
                    ui.tags.ul(
                        ui.tags.li("DAU: that day only."),
                        ui.tags.li("WAU: that day and the 6 before it."),
                        ui.tags.li("MAU: that day and the 29 before it."),
                    ),
                    ui.p(
                        "Windows can reach back before the first day on the "
                        "chart, but never before the loaded data, so the first "
                        "weeks of WAU and MAU are understated.",
                        class_="mb-0",
                    ),
                ),
            ),
            ui.output_ui("adoption_kpis"),
            ui.card(
                ui.card_header("Active users over time"),
                output_widget("adoption_lines"),
            ),
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("Tools over time"),
                        ui.div(
                            ui.input_radio_buttons(
                                "tool_grain",
                                None,
                                {"day": "Daily", "week": "Weekly", "month": "Monthly"},
                                selected="week",
                                inline=True,
                            ),
                            class_="seg",
                        ),
                        class_="hdr-flex",
                    )
                ),
                output_widget("tool_time"),
            ),
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("When is it used? (UTC)"),
                        ui.div(
                            ui.input_radio_buttons(
                                "heat_measure",
                                None,
                                {"requests": "Requests", "spend": "Spend"},
                                selected="requests",
                                inline=True,
                            ),
                            class_="seg",
                        ),
                        class_="hdr-flex",
                    )
                ),
                output_widget("usage_heatmap"),
            ),
            ui.card(
                table_header("Active users by day", "dl_adoption"),
                ui.card_body(ui.output_data_frame("adoption_table")),
            ),
        ),
        ui.nav_panel(
            "Models",
            page_header("Models"),
            ui.output_ui("model_kpis"),
            ui.card(ui.card_header("Estimated spend by model"), output_widget("models_bars")),
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("Model adoption over time"),
                        ui.div(
                            ui.input_radio_buttons(
                                "adopt_grain",
                                None,
                                {"day": "Daily", "week": "Weekly", "month": "Monthly"},
                                selected="day",
                                inline=True,
                            ),
                            class_="seg",
                        ),
                        class_="hdr-flex",
                    )
                ),
                output_widget("models_adoption"),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span("Estimated cache savings over time by tool"),
                            help_icon(
                                "How are savings estimated?",
                                ui.p(
                                    "The same tokens priced twice: once as billed, once "
                                    "with every cached token charged at the fresh-input "
                                    "rate. The difference is what caching saved. Cache "
                                    "writes cost more than fresh input, so that premium "
                                    "is already netted out.",
                                    class_="mb-0",
                                ),
                            ),
                            class_="hdr-flex",
                        )
                    ),
                    output_widget("cache_savings_time"),
                ),
                ui.card(
                    ui.card_header("Cache read share over time"),
                    output_widget("cache_rate_time"),
                ),
                col_widths=[6, 6],
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span("Latency over time by model"),
                            help_icon(
                                "How is latency measured?",
                                ui.p(
                                    "The gateway's recorded duration of each request, "
                                    "failed ones included, p95 is the slow tail. ",
                                    class_="mb-0",
                                ),
                            ),
                            class_="hdr-flex",
                        )
                    ),
                    output_widget("latency_time"),
                ),
                ui.card(
                    ui.card_header("Error rate over time by model"),
                    output_widget("errors_time"),
                ),
                col_widths=[6, 6],
            ),
            ui.card(
                table_header("Model summary", "dl_models"),
                ui.card_body(
                    ui.output_data_frame("models_table"),
                ),
            ),
        ),
        ui.nav_panel(
            "Raw Data",
            page_header("Raw Data"),
            ui.div(
                ui.card(
                    table_header(ui.output_text("records_title", inline=True), "dl_records"),
                    ui.card_body(
                        ui.output_data_frame("records_table"),
                        ui.output_ui("records_capped"),
                    ),
                ),
                class_="question-block",
            ),
        ),
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
        "attr_order",
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
            ui.span(f"{int(r.users):,} {plural(r.users, 'user')}", class_="pill pill-blue"),
            ui.span(
                f"{int(r.contents):,} content {plural(r.contents, 'item')}",
                class_="pill pill-purple",
            ),
            ui.span(
                f"{int(r.sessions):,} {plural(r.sessions, 'session')}",
                class_="pill pill-amber",
            ),
            ui.span(f"{card_usd(r.est)} est. spend", class_="pill pill-green"),
            unpriced_pill(),
            class_="topbar-right",
        )

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
        # Applies to every origin. A Workbench call has a null content_guid, so
        # picking content naturally leaves Workbench out of the view.
        if f["contents"] and set(f["contents"]) != set(f["all_contents"]):
            conds.append(sql_in("content_guid", f["contents"]))
        if f["sessions"]:
            conds.append(sql_in("workbench_session_id", f["sessions"]))
        if f["providers"] and set(f["providers"]) != set(f["all_providers"]):
            conds.append(sql_in("provider", f["providers"]))
        if f["models"] and set(f["models"]) != set(f["all_models"]):
            conds.append(sql_in("model", f["models"]))
        # Values of one key are alternatives; separate keys all have to match.
        by_key: dict[str, list[str]] = {}
        for pick in f["meta"]:
            try:
                k, v = json.loads(pick)
            except (ValueError, TypeError):
                continue
            by_key.setdefault(k, []).append(v)
        for k, vals in by_key.items():
            conds.append(f"{meta_expr(k)} IN ({','.join(sql_str(v) for v in vals)})")
        return " AND ".join(conds) + extra

    # ---- shared chart shapes ------------------------------------------------
    def daily_lines(f, m, dim: str, order, color_kind: str | None):
        """One line per category, measured per day."""
        # A null key means the record has no value on this axis (no user, or no
        # content). It is not a series — plotting it yields a "nan" line.
        df = usage.q(f"""SELECT day, {dim} AS k, SUM({m["expr"]}) v FROM gateway_events
                         WHERE {where(f)} AND {dim} IS NOT NULL
                         GROUP BY 1,2 ORDER BY 1""")
        if df.empty or not df.v.sum():
            return empty_fig(f["mode"], f"No {m['noun']} recorded in this view")
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
                mode="lines+markers",
                line={"width": 2, "color": c},
                marker={"size": 5, "color": c},
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
        df = df[df.v > 0]
        if df.empty:
            return empty_fig(f["mode"], f"No {m['noun']} recorded in this view")
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
                text=["\u00a0\u00a0" + m["fmt"](v) for v in df.v],
                textposition="outside",
                textfont={"color": theme.TOKENS[f["mode"]]["ink"], "size": 15},
                customdata=[
                    f"{lab}\n{m['fmt'](v)}"
                    for lab, v in zip(chart_labels(dim, df.k), df.v)
                ],
                hoverinfo="none",  # the name follows the mouse; see nametip below
            )
        )
        fig.update_layout(meta="nametip")
        fig.update_xaxes(range=[0, df.v.max() * 1.25], title_text=m["total_axis"])
        return apply(fig, f["mode"], height=height, showlegend=False, hover=True)

    # ---- overview -----------------------------------------------------------
    @render.ui
    def kpis():
        f = flt()
        r = usage.q(f"""SELECT COALESCE(SUM({est_cost()}),0) est,
                            COUNT(*) requests,
                            COUNT(DISTINCT username) users,
                            COUNT(DISTINCT content_guid) contents,
                            COUNT(DISTINCT workbench_session_id) sessions
                     FROM gateway_events WHERE {where(f)}""").iloc[0]
        # Org-wide: the busiest model across everyone in the date range and
        # origin, not narrowed by the user, content, provider or model filters,
        # so it stays the answer to "what does this organisation run on".
        org = {
            **f,
            "users": [],
            "contents": [],
            "sessions": [],
            "providers": [],
            "models": [],
        }
        top = usage.q(f"""SELECT model, SUM({est_cost()}) est FROM gateway_events
                          WHERE {where(org)} AND model IS NOT NULL AND model <> ''
                          GROUP BY 1 ORDER BY 2 DESC LIMIT 1""")
        users, contents, sessions = int(r.users), int(r.contents), int(r.sessions)

        def avg(n: int, noun: str) -> str:
            if not n:
                return f"no {noun}s in range"
            return f"{card_usd(r.est / n)} avg / {noun}"

        return ui.layout_columns(
            theme.stat_tile(
                "Estimated spend", card_usd(r.est), f"{card_num(int(r.requests))} requests"
            ),
            theme.stat_tile(
                "Users",
                card_num(users),
                avg(users, "user"),
            ),
            theme.stat_tile(
                "Content",
                card_num(contents),
                avg(contents, "content"),
            ),
            theme.stat_tile(
                "Sessions",
                card_num(sessions),
                avg(sessions, "session"),
            ),
            theme.stat_tile(
                "Top model",
                top.model.iloc[0] if not top.empty else "\u2013",
                f"{card_usd(top.est.iloc[0])} in range" if not top.empty else "",
            ),
            col_widths=[2, 2, 2, 2, 4],
            class_="kpi-row",
        )

    def unpriced_pill():
        """Topbar warning for models with no price, or None. One place, every page."""
        df = unpriced_df()
        if df.empty:
            return None
        n, reqs = len(df), int(df.requests.sum())
        shown = df.head(10)
        rows = [
            ui.tags.li(ui.tags.code(r.model), f" \u00b7 {int(r.requests):,} {plural(r.requests, 'request')}")
            for r in shown.itertuples()
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

    @render_widget
    def provider_total():
        f = flt()
        return total_bars(f, measure_of(f), "provider", "provider", height=300)

    @render_widget
    def model_time():
        """Spend per day, week or month, stacked by model."""
        f, m, grain = flt(), measure_of(flt()), input.grain() or "day"
        df = usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b, model AS k,
                                SUM({m["expr"]}) v
                         FROM gateway_events
                         WHERE {where(f)} AND model IS NOT NULL AND model <> ''
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty or not df.v.sum():
            return empty_fig(f["mode"], f"No {m['noun']} recorded in this view")
        t = theme.TOKENS[f["mode"]]
        # The six biggest models get a stack of their own; the tail is "Other".
        order = df.groupby("k").v.sum().sort_values(ascending=False).index.tolist()
        shown = order[:6]
        df["k"] = df.k.where(df.k.isin(shown), "Other")
        df = df.groupby(["b", "k"], as_index=False).v.sum()
        fig = go.Figure()
        for k in shown + (["Other"] if "Other" in set(df.k) else []):
            d = df[df.k == k]
            fig.add_bar(
                x=d.b,
                y=d.v,
                name=k,
                marker={"color": t["muted"] if k == "Other" else color("model", k, f["mode"])},
                hovertemplate=f"<b>{m['prefix']}%{{y:{m['tick']}}}</b> {k}<extra></extra>",
            )
        fig.update_layout(barmode="stack", bargap=0.25)
        return apply(
            fig, f["mode"], height=300, time_axis=True, showlegend=True,
            y_title=m["axis"].replace("/ day", f"/ {grain}"), hover=True,
        )

    def share_donut(f, m, dim: str, color_kind: str | None, top: int = 8, empty=None):
        """The split of one axis, largest slices first and the tail as "Other"."""
        df = usage.q(f"""SELECT {dim} AS k, SUM({m["expr"]}) v FROM gateway_events
                         WHERE {where(f)} AND {dim} IS NOT NULL AND {dim} <> ''
                         GROUP BY 1 ORDER BY 2 DESC""")
        df = df[df.v > 0]
        if df.empty:
            return empty_fig(f["mode"], empty or f"No {m['noun']} recorded in this view")
        t = theme.TOKENS[f["mode"]]
        head, rest = df.head(top), df.v.iloc[top:].sum()
        keys = head.k.tolist()
        labels = chart_labels(dim, keys)
        colors = [
            color(color_kind, k, f["mode"]) if color_kind else t["series"][i % len(t["series"])]
            for i, k in enumerate(keys)
        ]
        values = head.v.tolist()
        if rest > 0:
            labels, colors, values = labels + ["Other"], colors + [t["muted"]], values + [rest]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                marker={"colors": colors, "line": {"color": t["surface"], "width": 2}},
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                textfont={"color": "#ffffff", "size": 15},
                customdata=[
                    f"{lab}\n{m['prefix']}{v:{m['tick']}}\n{100 * v / sum(values):.1f}%"
                    for lab, v in zip(labels, values)
                ],
                hoverinfo="none",  # the name follows the mouse; see nametip below
            )
        )
        apply(fig, f["mode"], height=300, showlegend=True, hover=True)
        fig.update_layout(
            meta="nametip",
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "x": 1.0},
        )
        return fig

    def _origin_tool() -> tuple[str, str]:
        """The column and colour kind the Origin / Tool toggle is on."""
        return ("tool", "tool") if input.origin_tool() == "tool" else ("source", "origin")

    @render_widget
    def origin_total():
        f = flt()
        dim, kind = _origin_tool()
        return total_bars(f, measure_of(f), dim, kind, height=260)

    @render_widget
    def origin_share():
        f = flt()
        dim, kind = _origin_tool()
        return share_donut(f, measure_of(f), dim, kind)

    @render_widget
    def provider_share():
        f = flt()
        return share_donut(f, measure_of(f), "provider", "provider")

    @render_widget
    def model_share():
        f = flt()
        return share_donut(f, measure_of(f), "model", "model")

    # ---- attribution --------------------------------------------------------
    # Users and content share one ranking. An outlier is judged against the rest of
    # its own kind, on a log scale (spend is long-tailed, so a straight-line fence
    # would flag half the top). A row is flagged if its total spend, or its spend
    # per request, is above the third quartile by more than 1.5 times the
    # interquartile range, and at least twice the median. Too few rows of a kind
    # make a quartile meaningless, so under this many nothing of that kind is
    # flagged.
    OUTLIER_MIN_N = 8
    KIND_LABEL = {"user": "user", "content": "content", "session": "session"}
    # What can be ranked: the column that identifies it, and what to call it. A
    # Workbench session is identified by the session id its request metadata
    # carries (caller-supplied, unverified), so it exists only for Workbench calls.
    PRINCIPAL_KINDS = [
        ("username", "user"),
        ("content_guid", "content"),
        ("workbench_session_id", "session"),
    ]
    # Sessions also say who ran them and for which team, which is what turns an
    # opaque id into something a reader can act on.
    SESSION_EXTRA = """, MAX(workbench_session_name) AS session_name,
                        MAX(username) AS owner, MAX(team) AS team"""

    def principal_parts(want: str, build) -> list[pd.DataFrame]:
        return [build(dim, kind) for dim, kind in PRINCIPAL_KINDS if want in ("all", kind)]

    def principal_label(row, want: str) -> str:
        """What an axis shows for one row."""
        k = str(row.k)
        if row.kind == "user":
            return k
        if row.kind == "session":
            name = row.session_name if isinstance(row.session_name, str) else k[:GUID_LABEL_CHARS]
            return f"{name} \u00b7 {row.owner}" if isinstance(row.owner, str) else name
        # A title says what the content is; without one a guid fragment stands in.
        # The kind is shown by bar colour/legend and the table's kind column, not
        # by a prefix on every label.
        return usage.content_name(k) or k[:GUID_LABEL_CHARS]

    def uniquify(df: pd.DataFrame) -> pd.DataFrame:
        """Make axis labels unique.

        Plotly draws two bars with the same category label as one row, stacked on
        top of each other. Sessions are the usual offender: "RStudio Pro Session 7"
        for one user on two different days is two sessions with one name.
        """
        dup = df.label.duplicated(keep=False)
        df.loc[dup, "label"] = [
            f"{lab} ({str(k)[:6]})" for lab, k in zip(df.label[dup], df.k[dup])
        ]
        return df

    def principal_detail(row) -> str:
        if row.kind == "user":
            return ""
        if row.kind == "content":
            return str(row.k)  # the full guid, which the axis label replaces
        team = row.team if isinstance(row.team, str) else ""
        return f"{row.k} \u00b7 {team}" if team else str(row.k)

    def attribution_part(dim: str, kind: str) -> pd.DataFrame:
        f = flt()
        extra = SESSION_EXTRA if kind == "session" else ""
        df = usage.q(f"""SELECT {dim} AS k, SUM({est_cost()}) AS spend,
                                COUNT(*) AS requests, SUM({TOKEN_COUNT}) AS tokens{extra}
                         FROM gateway_events
                         WHERE {where(f)} AND {dim} IS NOT NULL AND {dim} <> ''
                         GROUP BY 1 HAVING SUM({est_cost()}) > 0""")
        df.insert(0, "kind", kind)
        return df

    @reactive.calc
    def attr_df() -> pd.DataFrame:
        """The selected list: users, content, or both together, flagged and ranked."""
        want = input.attr_kind() or "all"
        parts = [d for d in principal_parts(want, attribution_part) if not d.empty]
        if not parts:
            return pd.DataFrame(
                columns=["kind", "k", "spend", "requests", "tokens", "label", "detail",
                         "per_request", "x_median", "why"]
            )
        df = pd.concat(parts, ignore_index=True)
        for col in ("session_name", "owner", "team"):
            if col not in df:
                df[col] = None
        df["label"] = [principal_label(r, want) for r in df.itertuples()]
        df = uniquify(df)
        df["detail"] = [principal_detail(r) for r in df.itertuples()]
        df = df.sort_values("spend", ascending=False, ignore_index=True)
        if input.attr_order() == "type":
            order = {k: i for i, (_, k) in enumerate(PRINCIPAL_KINDS)}
            df = df.sort_values(
                ["kind", "spend"], key=lambda s: s.map(order) if s.name == "kind" else -s,
                ignore_index=True,
            )
        df["per_request"] = df.spend / df.requests
        df["x_median"] = 0.0
        df["why"] = ""
        # Judged within each kind, even in the mixed view: a content item and a
        # person are different sizes of thing, and set against each other every
        # content item would look like a heavy user.
        for kind in df.kind.unique():
            rows = df.index[df.kind == kind]
            part = df.loc[rows]
            df.loc[rows, "x_median"] = part.spend / part.spend.median()
            if len(part) < OUTLIER_MIN_N:
                continue
            why = {i: [] for i in rows}
            for col, label in [
                ("spend", "high estimated spend"),
                ("per_request", "high estimated cost per request"),
            ]:
                lg = np.log10(part[col])
                q1, q3 = lg.quantile(0.25), lg.quantile(0.75)
                material = part[col] >= 2 * part[col].median()
                for i in rows[((lg > q3 + 1.5 * (q3 - q1)) & material).to_numpy()]:
                    why[i].append(label)
            df.loc[rows, "why"] = [" + ".join(why[i]) for i in rows]
        return df

    @render.ui
    def attr_kpis():
        df = attr_df()
        if df.empty:
            return empty_note("No users or content with estimated spend in this view.")
        n = df.kind.value_counts()
        out = df[df.why != ""]
        flagged = out.kind.value_counts()

        def part(kind: str, label: str, noun: str) -> ui.Tag:
            have, bad = int(n.get(kind, 0)), int(flagged.get(kind, 0))
            return theme.stat_tile(
                label,
                card_num(bad),
                f"of {card_num(have)} {noun} with estimated spend",
                status="critical" if bad else "good",
            )

        return ui.layout_columns(
            theme.stat_tile(
                "Outliers",
                card_num(len(out)),
                f"of {card_num(len(df))} with estimated spend",
                status="critical" if len(out) else "good",
                help=help_icon(
                    "How outliers are flagged",
                    ui.p(
                        "Flagged when total estimated spend, or spend per request, is "
                        "unusually high compared with the others in the list.",
                        class_="mb-0",
                    ),
                ),
            ),
            part("user", "Outlier users", "users"),
            part("content", "Outlier content", "content items"),
            part("session", "Outlier sessions", "sessions"),
            col_widths=[3, 3, 3, 3],
            class_="kpi-row",
        )

    def _kind_color(t: dict, kind: str) -> str:
        return t["series"][{"user": 0, "content": 1, "session": 2}[kind]]

    def rank_bars(f, df, col: str, flag: str, fmt, axis: str):
        """Top 20 by one measure. Red marks the rows flagged on that measure only."""
        t = theme.TOKENS[f["mode"]]
        ranked = df.sort_values(col, ascending=False)
        if input.attr_order() == "type":
            # Grouped: each kind gets its own block, best first within it.
            kinds = [k for _, k in PRINCIPAL_KINDS if (ranked.kind == k).any()]
            per = max(5, 20 // max(len(kinds), 1))
            top = pd.concat([ranked[ranked.kind == k].head(per) for k in kinds])
        else:
            top = ranked.head(20)
        top = top.iloc[::-1]  # plotly draws the first row at the bottom
        fig = go.Figure()
        for kind in [k for _, k in PRINCIPAL_KINDS]:
            d = top[top.kind == kind]
            if d.empty:
                continue
            fig.add_bar(
                x=d[col],
                y=d.label,
                name=KIND_LABEL[kind],
                orientation="h",
                width=0.6,
                marker={
                    "color": [
                        t["series"][7] if flag in w else _kind_color(t, kind) for w in d.why
                    ]
                },
                text=["\u00a0\u00a0" + fmt(v) for v in d[col]],
                textposition="outside",
                textfont={"color": t["ink"], "size": 13},
                customdata=[f"{lab}\n{fmt(v)}" for lab, v in zip(d.label, d[col])],
                hoverinfo="none",  # the name follows the mouse; see nametip
            )
        fig.update_layout(meta="nametip", barmode="overlay")
        fig.update_yaxes(categoryorder="array", categoryarray=list(top.label))
        fig.update_xaxes(range=[0, top[col].max() * 1.25], title_text=axis)
        return apply(
            fig, f["mode"], height=max(260, 24 * len(top) + 70),
            showlegend=top.kind.nunique() > 1, hover=True,
        )

    @render_widget
    def attr_bars():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        return rank_bars(f, df, "spend", "high estimated spend", fmt_usd, "est. USD")

    @render_widget
    def attr_rate_bars():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        return rank_bars(
            f, df, "per_request", "cost per request", lambda v: f"${v:,.4f}",
            "est. USD per request",
        )

    @render_widget
    def attr_scatter():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        t = theme.TOKENS[f["mode"]]
        fig = go.Figure()
        for kind, symbol in [("user", "circle"), ("content", "diamond"), ("session", "square")]:
            for flag in (False, True):
                d = df[(df.kind == kind) & ((df.why != "") == flag)]
                if d.empty:
                    continue
                crowded = (df.kind == kind).sum() > 150
                # Name only the biggest five outliers: twenty overlapping labels read
                # as noise. The table below lists every one.
                named = set(d.sort_values("spend", ascending=False).label.head(5)) if flag else set()
                fig.add_scatter(
                    x=d.requests,
                    y=d.spend,
                    name=f"{KIND_LABEL[kind]} outlier" if flag else KIND_LABEL[kind],
                    mode="markers+text" if flag else "markers",
                    text=[lab if lab in named else "" for lab in d.label] if flag else None,
                    textposition="top center",
                    textfont={"size": 12, "color": t["ink2"]},
                    marker={
                        "symbol": symbol,
                        "size": (9 if crowded else 11) if flag else (5 if crowded else 8),
                        "color": t["series"][7] if flag else _kind_color(t, kind),
                        "opacity": 0.9 if flag else 0.55,
                        "line": {"width": 1, "color": t["surface"]},
                    },
                    customdata=[
                        f"{lab}\n{fmt_usd(s)}\n{int(r):,} requests"
                        for lab, s, r in zip(d.label, d.spend, d.requests)
                    ],
                    hoverinfo="none",  # the name follows the mouse; see nametip
                )
        fig.update_layout(meta="nametip")
        fig.update_xaxes(type="log", title_text="requests", tickformat=",.0f")
        fig.update_yaxes(type="log", tickprefix="$", title_text="est. USD", tickformat=",.2~f")
        # Ticks at powers of ten, unless the data sits inside one decade, where
        # that would leave the axis with no ticks at all.
        if df.requests.max() / max(df.requests.min(), 1) >= 10:
            fig.update_xaxes(dtick=1)
        if df.spend.max() / df.spend.min() >= 10:
            fig.update_yaxes(dtick=1)
        return apply(
            fig, f["mode"], height=max(260, 24 * min(len(df), 20) + 70), showlegend=True,
            hover=True,
        )

    def attr_out_df() -> pd.DataFrame:
        df = attr_df()
        out = df[df.why != ""][
            ["kind", "label", "detail", "why", "spend", "requests", "per_request", "x_median", "tokens"]
        ]
        # The detail column only earns its place when something fills it (sessions).
        if not (out.detail != "").any():
            out = out.drop(columns="detail")
        return out.rename(columns={"label": "name", "spend": "est_spend_usd"})

    @render.data_frame
    def attr_table():
        out = attr_out_df()
        return render.DataGrid(
            _pretty(formatted(out)), height=None if len(out) <= 10 else "420px",
            selection_mode="row",
        )

    # Picking an outlier opens it on the Principals tab.
    @reactive.effect
    @reactive.event(attr_table.cell_selection)
    def _attr_open():
        rows = (attr_table.cell_selection() or {}).get("rows") or ()
        if rows:
            r = attr_df()[lambda d: d.why != ""].iloc[rows[0]]
            open_profile(r.kind, r.k)


    # ---- adoption -----------------------------------------------------------
    @reactive.calc
    def adoption_df() -> pd.DataFrame:
        """Rolling daily, weekly and monthly active users, one row per day in range.

        Windows look back past the start of the range (so a 30D chart does not
        start with an empty MAU), but not past the loaded data. The date range
        is the only filter lifted; everything else still applies.
        """
        f = flt()
        wide = {**f, "start": cat()["start"]}
        return usage.q(f"""
            WITH days AS (
                SELECT CAST(d AS DATE) AS d
                FROM generate_series(DATE '{f["start"]}', DATE '{f["end"]}',
                                     INTERVAL 1 DAY) t(d)),
            ev AS (SELECT DISTINCT day, username FROM gateway_events
                   WHERE {where(wide)} AND username IS NOT NULL)
            SELECT days.d AS day,
                   COUNT(DISTINCT CASE WHEN ev.day = days.d THEN ev.username END) AS dau,
                   COUNT(DISTINCT CASE WHEN ev.day > days.d - INTERVAL 7 DAY
                                       THEN ev.username END) AS wau,
                   COUNT(DISTINCT ev.username) AS mau
            FROM days LEFT JOIN ev
              ON ev.day <= days.d AND ev.day > days.d - INTERVAL 30 DAY
            GROUP BY 1 ORDER BY 1""")

    @render.ui
    def adoption_kpis():
        df = adoption_df()
        if df.empty:
            return empty_note("No users in this range.")
        last = df.iloc[-1]
        avg_dau = df.dau.mean()
        return ui.layout_columns(
            theme.stat_tile(
                "Avg DAU",
                f"{avg_dau:,.1f}",
                f"daily average over {len(df)} {plural(len(df), 'day')}",
                help=help_icon(
                    "Average daily users",
                    ui.p(
                        "On a typical day, this many people used the gateway. ",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "WAU",
                card_num(last.wau),
                f"7 days to {last.day:%b %d}",
                help=help_icon(
                    "Weekly users",
                    ui.p(
                        "How many different people used the gateway in the last 7 days.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "MAU",
                card_num(last.mau),
                f"30 days to {last.day:%b %d}",
                help=help_icon(
                    "Monthly users",
                    ui.p(
                        "How many different people used the gateway in the last "
                        "30 days.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "Stickiness",
                f"{100 * avg_dau / last.mau:,.0f}%" if last.mau else "\u2013",
                "avg DAU / MAU",
                help=help_icon(
                    "Stickiness",
                    ui.p(
                        "Of the people who used the gateway this month, the share "
                        "who show up on a typical day.",
                        class_="mb-0",
                    ),
                ),
            ),
            col_widths=[3, 3, 3, 3],
            class_="kpi-row",
        )

    @render.data_frame
    def adoption_table():
        df = adoption_df().copy()
        df["day"] = df.day.astype(str).str[:10]
        # Newest first: the day a reader is most likely looking for.
        df = df.iloc[::-1]
        return render.DataGrid(df, height="420px" if len(df) > 12 else None)

    @render_widget
    def adoption_lines():
        f, df = flt(), adoption_df()
        if df.empty or not df.mau.max():
            return empty_fig(f["mode"], "No active users in this view")
        series = theme.TOKENS[f["mode"]]["series"]
        fig = go.Figure()
        for col, name, c in [("dau", "DAU", series[0]), ("wau", "WAU", series[2]), ("mau", "MAU", series[1])]:
            fig.add_scatter(
                x=df.day, y=df[col], name=name, mode="lines",
                line={"width": 2.5, "color": c},
                hovertemplate=f"<b>%{{y:,}}</b> {name}<extra></extra>",
            )
        fig.update_yaxes(rangemode="tozero")
        return apply(fig, f["mode"], height=380, time_axis=True, showlegend=True, y_title="active users", hover=True)

    @render_widget
    def tool_time():
        """Distinct active users per tool, per day, week or month."""
        f, grain = flt(), input.tool_grain() or "week"
        df = usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b, tool AS k,
                                COUNT(DISTINCT username) AS users, COUNT(*) AS requests
                         FROM gateway_events
                         WHERE {where(f)} AND tool IS NOT NULL AND username IS NOT NULL
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty:
            return empty_fig(f["mode"], "No tool activity in this view")
        order = df.groupby("k").users.sum().sort_values(ascending=False).index.tolist()
        fig = go.Figure()
        for k in order:
            d = df[df.k == k]
            fig.add_scatter(
                x=d.b, y=d.users, name=k, mode="lines+markers",
                line={"width": 2.5, "color": color("tool", k, f["mode"])},
                marker={"size": 6},
                customdata=d.requests,
                hovertemplate=f"<b>%{{y:,}}</b> users \u00b7 %{{customdata:,}} requests \u00b7 {k}<extra></extra>",
            )
        fig.update_yaxes(rangemode="tozero")
        return apply(fig, f["mode"], height=340, time_axis=True, showlegend=True,
                     y_title=f"active users / {grain}", hover=True)

    # ---- models -------------------------------------------------------------
    # The model's own spend, whatever the Count toggle says: the table always
    # shows dollars and tokens side by side, so it does not need the toggle.
    def models_df() -> pd.DataFrame:
        f = flt()
        df = usage.q(f"""
            SELECT model,
                   string_agg(DISTINCT provider, ', ' ORDER BY provider) AS providers,
                   SUM({est_cost()}) AS est_spend_usd,
                   COUNT(*) AS requests,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(cache_read_tokens) AS cache_read_tokens,
                   -- input_tokens already counts cache writes and reads (see
                   -- usage._SHAPE), so it is the whole of the input.
                   100.0 * SUM(COALESCE(cache_read_tokens, 0))
                       / NULLIF(SUM(COALESCE(input_tokens, 0)), 0) AS cache_pct,
                   SUM({est_cost_uncached()} - {est_cost()}) AS cache_savings_usd,
                   100.0 * SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END)
                       / COUNT(*) AS error_pct,
                   quantile_cont(duration_ms, .95) AS p95_ms
            FROM gateway_events
            WHERE {where(f)} AND model IS NOT NULL AND model <> ''
            GROUP BY 1 ORDER BY est_spend_usd DESC, requests DESC""")
        total = df.est_spend_usd.sum()
        df.insert(
            3, "pct_of_total", 100 * df.est_spend_usd / total if total else 0.0
        )
        return df

    def _pretty(df: pd.DataFrame) -> pd.DataFrame:
        nice = {"est_spend_usd": "estimated spend (usd)", "cache_savings_usd": "cache savings (usd)", "cache_pct": "cache %", "error_pct": "error %", "pct_of_total": "% of total",
                "p95_ms": "p95 latency (ms)"}
        return df.rename(
            lambda x: nice.get(x, x.replace("_", " ")), axis="columns"
        )

    @render.ui
    def model_kpis():
        df = models_df()
        if df.empty:
            return empty_note("No model traffic in this view.")
        top = df.iloc[0]
        # input_tokens is all input, cached included (see usage._SHAPE).
        cached = df.cache_read_tokens.fillna(0)
        cache_in = cached.sum() / max(1, df.input_tokens.fillna(0).sum()) * 100
        errs = (df.error_pct * df.requests).sum() / df.requests.sum()
        return ui.layout_columns(
            theme.stat_tile("Models in use", card_num(len(df))),
            theme.stat_tile(
                "Top model",
                top.model,
                f"{card_usd(top.est_spend_usd)} \u00b7 {top.pct_of_total:.0f}% of spend",
            ),
            theme.stat_tile(
                "Cache read share",
                f"{cache_in:.1f}%",
                help=help_icon(
                    "Cache read share",
                    ui.p(
                        "The share of input the models reused from memory instead "
                        "of reading it again. Reused input costs far less, so a "
                        "higher share means lower spend.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "Cache savings",
                card_usd(df.cache_savings_usd.sum()),
                f"{100 * df.cache_savings_usd.sum() / max(1e-9, df.cache_savings_usd.sum() + df.est_spend_usd.sum()):.0f}% off the uncached price",
            ),
            theme.stat_tile(
                "Error rate",
                f"{errs:.2f}%",
                "across all models",
                status="critical" if errs > 2 else "good",
                help=help_icon(
                    "Error rate",
                    ui.p(
                        "The share of requests that failed instead of getting an "
                        "answer. It turns red when more than 2% fail.",
                        class_="mb-0",
                    ),
                ),
            ),
            class_="kpi-row",
        )

    @render_widget
    def models_bars():
        f = flt()
        return total_bars(f, measure_of(f), "model", "model", limit=12, height=340)

    @render_widget
    def models_adoption():
        """Distinct users per model in each day, week or month."""
        f, grain = flt(), input.adopt_grain() or "day"
        base = f"""FROM gateway_events WHERE {where(f)} AND username IS NOT NULL
                   AND model IS NOT NULL AND model <> ''"""
        order = usage.q(f"""SELECT model, COUNT(DISTINCT username) n {base}
                            GROUP BY 1 ORDER BY 2 DESC""").model.tolist()
        if not order:
            return empty_fig(f["mode"], "No model usage in this view")
        shown = order[:6]
        keep = ",".join(sql_str(m) for m in shown)
        df = usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b,
                                CASE WHEN model IN ({keep}) THEN model ELSE 'Other' END AS k,
                                COUNT(DISTINCT username) v
                         {base} GROUP BY 1, 2 ORDER BY 1""")
        t = theme.TOKENS[f["mode"]]
        fig = go.Figure()
        for k in shown + (["Other"] if "Other" in set(df.k) else []):
            d = df[df.k == k]
            c = t["muted"] if k == "Other" else color("model", k, f["mode"])
            fig.add_bar(
                x=d.b, y=d.v, name=k, marker={"color": c},
                hovertemplate=f"<b>%{{y:,}}</b> {k}<extra></extra>",
            )
        # A user on two models counts once in each, so a stack's height is
        # model-users, not people.
        fig.update_layout(barmode="stack", barnorm="percent", bargap=0.25)
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        return apply(
            fig, f["mode"], height=320, time_axis=True, showlegend=True,
            y_title=f"share of model-users / {grain}", hover=True,
        )

    def _model_series(f, extra_where=""):
        """Top 6 models by requests plus an SQL expression bucketing the rest."""
        order = usage.q(f"""SELECT model, COUNT(*) n FROM gateway_events
                            WHERE {where(f)} AND model IS NOT NULL AND model <> ''
                            {extra_where} GROUP BY 1 ORDER BY 2 DESC""").model.tolist()
        shown = order[:6]
        keep = ",".join(sql_str(m) for m in shown) or "''"
        expr = f"CASE WHEN model IN ({keep}) THEN model ELSE 'Other' END"
        return shown, expr

    def _model_color(k, mode):
        return theme.TOKENS[mode]["muted"] if k == "Other" else color("model", k, mode)

    def _model_lines(f, df, col, fmt, y_title, yaxis=None, dash_col=None):
        shown = [k for k in df.k.unique()]
        fig = go.Figure()
        for k in shown:
            d = df[df.k == k]
            c = _model_color(k, f["mode"])
            fig.add_scatter(
                x=d.day, y=d[col], name=k, mode="lines+markers",
                line={"width": 2.5, "color": c}, marker={"size": 5, "color": c},
                hovertemplate=f"<b>{fmt}</b> {k}<extra></extra>",
            )
        if yaxis:
            fig.update_yaxes(**yaxis)
        return apply(fig, f["mode"], height=300, time_axis=True, showlegend=True,
                     y_title=y_title, hover=True)

    # ---- caching ------------------------------------------------------------
    @render_widget
    def cache_savings_time():
        f = flt()
        df = usage.q(f"""SELECT day, tool AS k, SUM({est_cost_uncached()} - {est_cost()}) v
                         FROM gateway_events
                         WHERE {where(f)} AND tool IS NOT NULL
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty or not df.v.abs().sum():
            return empty_fig(f["mode"], "No cache savings recorded in this view")
        df["v"] = df.v.clip(lower=0)
        fig = go.Figure()
        # largest saver first, so the stack and legend read in the same order
        for m in df.groupby("k").v.sum().sort_values(ascending=False).index:
            d = df[df.k == m]
            fig.add_bar(
                x=d.day, y=d.v, name=m, marker={"color": color("tool", m, f["mode"])},
                hovertemplate=f"<b>$%{{y:,.2f}}</b> saved \u00b7 {m}<extra></extra>",
            )
        fig.update_layout(barmode="stack", barnorm="percent", bargap=0.25)
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        return apply(fig, f["mode"], height=300, time_axis=True, showlegend=True,
                     y_title="share of est. savings / day", hover=True)

    @render_widget
    def cache_rate_time():
        f = flt()
        df = usage.q(f"""SELECT day, 100.0 * SUM(COALESCE(cache_read_tokens, 0))
                                / NULLIF(SUM(COALESCE(input_tokens, 0)), 0) AS v
                         FROM gateway_events WHERE {where(f)} GROUP BY 1 ORDER BY 1""").dropna()
        if df.empty:
            return empty_fig(f["mode"], "No input tokens recorded in this view")
        c = theme.TOKENS[f["mode"]]["series"][0]
        fig = go.Figure(
            go.Scatter(
                x=df.day, y=df.v, mode="lines+markers", line={"width": 2.5, "color": c},
                marker={"size": 5, "color": c},
                hovertemplate="<b>%{y:.1f}%</b> of input read from cache<extra></extra>",
            )
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        return apply(fig, f["mode"], height=300, time_axis=True, showlegend=False,
                     y_title="cache read share", hover=True)

    # ---- reliability --------------------------------------------------------
    @render_widget
    def latency_time():
        f = flt()
        _, k = _model_series(f)
        df = usage.q(f"""SELECT day, {k} AS k, quantile_cont(duration_ms, .95) p95
                         FROM gateway_events
                         WHERE {where(f)} AND duration_ms IS NOT NULL
                           AND model IS NOT NULL AND model <> ''
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty:
            return empty_fig(f["mode"], "No latency recorded in this view")
        df["p95"] = df.p95 / 1000
        return _model_lines(f, df, "p95", "%{y:,.2f} s", "seconds (p95)",
                            {"rangemode": "tozero"})

    @render_widget
    def errors_time():
        f = flt()
        _, k = _model_series(f)
        df = usage.q(f"""SELECT day, {k} AS k,
                                100.0 * SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END)
                                    / COUNT(*) AS err
                         FROM gateway_events
                         WHERE {where(f)} AND status_code IS NOT NULL
                           AND model IS NOT NULL AND model <> ''
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty:
            return empty_fig(f["mode"], "No requests in this view")
        return _model_lines(f, df, "err", "%{y:.2f}%", "requests failed (status 400+)",
                            {"rangemode": "tozero", "ticksuffix": "%"})

    # ---- usage patterns -----------------------------------------------------
    @render_widget
    def usage_heatmap():
        """Hour of day against weekday. The session runs in UTC (see usage.py)."""
        f = flt()
        spend = input.heat_measure() == "spend"
        expr = est_cost() if spend else "1"
        df = usage.q(f"""SELECT isodow(ts) - 1 AS d, hour(ts) AS h, SUM({expr}) AS v
                         FROM gateway_events WHERE {where(f)} GROUP BY 1, 2""")
        if df.empty or not df.v.sum():
            return empty_fig(f["mode"], "No activity in this view")
        grid = np.zeros((7, 24))
        for r in df.itertuples():
            grid[int(r.d), int(r.h)] = r.v
        fmt = "$%{z:,.2f}" if spend else "%{z:,.0f} requests"
        fig = go.Figure(
            go.Heatmap(
                z=grid, x=[f"{h:02d}" for h in range(24)], y=WEEKDAYS,
                colorscale=theme.seq_colorscale(f["mode"]), xgap=2, ygap=2,
                colorbar={"thickness": 10, "outlinewidth": 0},
                hovertemplate="%{y} %{x}:00 \u00b7 <b>" + fmt + "</b><extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", title_text=None)
        fig.update_xaxes(title_text="hour of day (UTC)", type="category")
        return apply(fig, f["mode"], height=320, showlegend=False, hover=True)

    @render.data_frame
    def models_table():
        df = models_df()
        return render.DataGrid(
            _pretty(formatted(df)),
            height=None if len(df) <= 12 else "420px",
        )


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
                m["noun"].capitalize(), m["card_fmt"](r.v), "all origins"
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

    # ---- cost drivers -------------------------------------------------------
    def compact_at() -> int:
        return COMPACT_AT

    @reactive.calc
    def cmp_model() -> str | None:
        """The model the quick-chat lever re-prices at: of the priced models in the
        data, the newest Sonnet by name, else the one in the middle by price."""
        priced = []
        for m in cat()["models"]:
            r = model_rates(m) if m else None
            if r and r[0] > 0:
                priced.append((r[0], m))
        names = [m for _r, m in sorted(priced)]
        sonnets = sorted(m for m in names if "sonnet" in m)
        if sonnets:
            return sonnets[-1]
        return names[len(names) // 2] if names else None

    def conv_sql(f, extra: str = "") -> str:
        """A WITH clause ending in `req`: one row per request in a conversation.

        Each row carries its turn number, how long the conversation sat idle
        before it, its cost split into parts (c_fresh + c_cw + c_cr + c_out =
        cost, the same figure est_cost gives), and what each lever would have
        saved on it (sav_<lever>). Rates are looked up once per distinct model,
        not once per row.
        """
        r = rate_sql()
        k_ctx, ttl = compact_at(), CACHE_TTL_SECONDS
        tgt = model_rates(cmp_model() or "")
        if tgt:
            ti, to, tw, tr = tgt
            sav_model = (
                f"CASE WHEN conv_calls <= {QUICK_MAX_CALLS} AND ri > {ti} THEN GREATEST(cost - "
                f"((tin - tcw - tcr) * {ti} + tcw * {tw} + tcr * {tr} + tout * {to}), 0) ELSE 0 END"
            )
        else:
            sav_model = "0.0"
        return f"""
            WITH ev AS (
                SELECT conversation_id, ts, response_id, {f["dim"]} AS principal,
                       provider, region, model, tool, status_code, finish_reason,
                       COALESCE(duration_ms, 0) AS duration_ms,
                       COALESCE(input_tokens, 0) AS tin, COALESCE(output_tokens, 0) AS tout,
                       COALESCE(cache_creation_tokens, 0) AS tcw,
                       COALESCE(cache_read_tokens, 0) AS tcr
                FROM gateway_events
                WHERE {where(f)} AND conversation_id IS NOT NULL{priced_only()}{extra}),
            rates AS (
                SELECT provider, region, model,
                       {r["i"]} AS ri, {r["o"]} AS ro, {r["cw"]} AS rcw, {r["cr"]} AS rcr
                FROM (SELECT DISTINCT provider, region, model FROM ev)),
            seq AS (
                SELECT ev.*, ri, ro, rcw, rcr,
                       ROW_NUMBER() OVER w AS turn,
                       COUNT(*) OVER (PARTITION BY conversation_id) AS conv_calls,
                       -- from the end of the previous request, which is when the
                       -- prompt cache was last touched
                       (epoch_ms(ts) - LAG(epoch_ms(ts) + duration_ms) OVER w) / 1000.0 AS idle_s,
                       LAG(tin) OVER w AS prev_in,
                       LAG(tin + tout) OVER w AS prev_ctx
                FROM ev JOIN rates
                  ON ev.provider IS NOT DISTINCT FROM rates.provider
                 AND ev.region IS NOT DISTINCT FROM rates.region
                 AND ev.model IS NOT DISTINCT FROM rates.model
                WINDOW w AS (PARTITION BY conversation_id ORDER BY ts, response_id)),
            pace AS (
                SELECT conversation_id, median(idle_s) AS med_idle
                FROM seq GROUP BY 1),
            costed AS (
                SELECT seq.*, med_idle,
                       (tin - tcw - tcr) * ri AS c_fresh, tcw * rcw AS c_cw,
                       tcr * rcr AS c_cr, tout * ro AS c_out,
                       (tin - tcw - tcr) * ri + tcw * rcw + tcr * rcr + tout * ro AS cost,
                       COALESCE(idle_s > {ttl} AND tcw > 0 AND prev_in > 0, FALSE) AS rewarm,
                       COALESCE(status_code >= 400 OR finish_reason IN ('max_tokens', 'length'), FALSE)
                           AS wasted,
                       COALESCE(med_idle <= {AGENT_GAP_SECONDS} AND conv_calls >= {AGENT_MIN_CALLS}, FALSE)
                           AS agent
                FROM seq JOIN pace USING (conversation_id)),
            req AS (
                SELECT *,
                       CASE WHEN rewarm THEN LEAST(tcw, prev_in) * GREATEST(rcw - rcr, 0)
                            ELSE 0 END AS sav_rewarm,
                       CASE WHEN tin > {k_ctx}
                            THEN (c_fresh + c_cw + c_cr) * (tin - {k_ctx}) / tin
                            ELSE 0 END AS sav_context,
                       CASE WHEN prev_ctx > 0 AND tcw = 0 AND tcr = 0 AND rcr < ri
                                 AND idle_s <= {ttl}
                            THEN GREATEST(LEAST(prev_ctx, tin) * (ri - rcr)
                                 - GREATEST(tin - prev_ctx, 0) * GREATEST(rcw - ri, 0), 0)
                            ELSE 0 END AS sav_uncached,
                       {sav_model} AS sav_model,
                       CASE WHEN wasted THEN cost ELSE 0 END AS sav_waste,
                       CASE WHEN agent AND turn > {AGENT_TURN_CAP} THEN cost
                            ELSE 0 END AS sav_loop
                FROM costed)
        """

    def lever_label(k: str) -> str:
        """A lever's name, with the threshold or model it is set to."""
        if k == "context":
            return f"Context past {compact_at() // 1000}k tokens"
        if k == "model":
            return f"Quick chats priced above {cmp_model() or 'the comparison model'}"
        return LEVER_NAME[k]

    @reactive.calc
    def sessions_df() -> pd.DataFrame:
        """Every conversation in range, cheapest included.

        (The function keeps its old name; the data is one row per conversation id.)

        There is no cost floor: gating at $0.25 emptied the panel on a quiet
        range, and the cloud of ordinary sessions is what gives a higher-cost
        one something to stand out against.
        """
        f = flt()
        savings = ",\n".join(f"SUM(sav_{k}) AS sav_{k}" for k, *_ in LEVERS)
        df = usage.q(conv_sql(f) + f"""
            SELECT conversation_id,
                   -- a label, not a key: a null here would reach plotly as NaN
                   COALESCE(ANY_VALUE(principal), 'unnamed') AS principal,
                   MIN(ts) AS started,
                   GREATEST(date_diff('second', MIN(ts), MAX(ts)) / 60.0, 0.5) AS duration_min,
                   COUNT(*) AS calls, SUM(cost) AS est,
                   string_agg(DISTINCT model, ', ' ORDER BY model)
                       FILTER (WHERE model <> '') AS models,
                   -- the model the conversation spent most on, for grouping
                   arg_max(model, cost) FILTER (WHERE model <> '') AS main_model,
                   arg_max(tool, cost) FILTER (WHERE tool IS NOT NULL) AS main_tool,
                   SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                   MAX(tin) AS peak_input,
                   SUM(tcr) / NULLIF(SUM(tin), 0) AS cache_hit,
                   BOOL_OR(agent) AS agent,
                   ANY_VALUE(med_idle) AS med_idle,
                   {savings}
            FROM req
            GROUP BY 1
            ORDER BY est DESC""")
        # Judged against every conversation in view, not the principal's own: the
        # question here is which conversations are expensive, whoever ran them. (The
        # Principals tab keeps the per-principal line, where that is the point.)
        df[["high", "line"]] = principals.higher_cost(df.est, pct=CONV_HIGH_PCT)
        if f["dim"] == "content_guid":
            df["principal"] = [usage.content_name(g) or g for g in df.principal]
        return df

    @render.ui
    def conv_kpis():
        f, df = flt(), sessions_df()
        r = usage.q(f"""SELECT COUNT(*) n, COUNT(conversation_id) c,
                            COUNT(*) FILTER (
                                WHERE regexp_full_match(conversation_id, '[0-9a-f]{{32}}')
                            ) derived
                     FROM gateway_events WHERE {where(f)}{priced_only()}""").iloc[0]
        if not r.n or not r.c:
            return empty_note("No request in this view carries a conversation id.")
        flagged = int(df.high.sum()) if not df.empty else 0
        line = float(df.line.iloc[0]) if not df.empty else None
        return ui.layout_columns(
            theme.stat_tile(
                "Conversations",
                card_num(len(df)),
                f"{df.calls.mean():,.1f} requests avg" if len(df) else "",
            ),
            theme.stat_tile(
                "Median cost",
                fmt_usd(df.est.median()) if len(df) else "\u2013",
                "per conversation",
            ),
            theme.stat_tile(
                "Higher cost",
                card_num(flagged),
                (
                    f"above {fmt_usd(line)}, p{CONV_HIGH_PCT:g} of all conversations"
                    if line is not None and len(df) >= principals.MIN_BASELINE
                    else f"needs {principals.MIN_BASELINE}+ conversations"
                ),
                status="warning" if flagged else "good",
                help=help_icon(
                    "Higher-cost conversations",
                    ui.p(
                        "A conversation is flagged when it costs more than "
                        f"{CONV_HIGH_PCT:g}% of all conversations in view, "
                        "whoever ran them. The line moves with the date range "
                        f"and filters. Below {principals.MIN_BASELINE} "
                        "conversations nothing is flagged.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "Coverage",
                f"{100 * r.c / r.n:.1f}%",
                f"of requests carry an id \u00b7 {100 * r.derived / r.c:.0f}% derived",
                help=help_icon(
                    "Coverage",
                    ui.p(
                        "The share of requests that are linked to a conversation. ",
                        class_="mb-0",
                    ),
                ),
            ),
            class_="kpi-row",
        )

    # ---- where the money goes -----------------------------------------------
    @reactive.calc
    def conv_spend_df() -> pd.DataFrame:
        """Conversation spend per group of the chosen kind, in display order."""
        df, by = sessions_df(), input.conv_by() or "model"
        key, order = conv_keys(df, by)
        return conv_groups(df, key, order)

    COST_PARTS = [
        ("fresh", "new input"), ("cw", "cache write"), ("cr", "cache read"), ("out", "output"),
    ]

    def cost_part_bars(fig: go.Figure, x, df: pd.DataFrame, mode: str) -> None:
        """The four parts of a cost as one stack, in the same colours everywhere."""
        t = theme.TOKENS[mode]
        for i, (col, name) in enumerate(COST_PARTS):
            fig.add_bar(
                x=x, y=df[col], name=name, marker={"color": t["series"][i]},
                hovertemplate=f"{name}: <b>%{{y:$,.2f}}</b><extra></extra>",
            )
        fig.update_layout(barmode="stack")

    @render_widget
    def conv_spend_by():
        f = flt()
        return conv_spend_fig(
            conv_spend_df(), input.conv_by() or "model", f["mode"], CONV_HIGH_PCT
        )

    @render.ui
    def conv_spend_note():
        """The group whose share of spend most outruns its share of conversations."""
        df = conv_spend_df()
        total, count = (float(df.spend.sum()), int(df.n.sum())) if len(df) else (0.0, 0)
        if not total or not count:
            return None
        df = df.assign(s=df.spend / total, c=df.n / count)
        hot = df[(df.s >= 0.1) & (df.c > 0)]
        if hot.empty:
            return None
        r = hot.loc[(hot.s / hot.c).idxmax()]
        if r.s / r.c < 1.5:
            return ui.div("Spend is spread roughly in line with conversation counts.",
                          class_="card-note")
        lab = str(r.label)
        name = {
            "model": f"Conversations mostly on {lab}",
            "duration": f"Conversations lasting {lab}",
            "turns": "Single-request conversations" if lab == "1"
            else f"Conversations of {lab} requests",
            "pace": "Single-request conversations" if lab == SINGLE
            else f"Conversations pausing {lab} between requests",
            "cache": f"Conversations with {lab} of input cached",
            "tool": f"Conversations mostly through {lab}",
        }[input.conv_by() or "model"]
        return ui.div(
            f"{name} are {100 * r.c:.0f}% of conversations but {100 * r.s:.0f}% of "
            f"spend, at {fmt_usd(r.spend / r.n)} each against "
            f"{fmt_usd(total / count)} overall.",
            class_="card-note",
        )

    # The conversation picked in the table, shown turn by turn above it.
    picked_conv = reactive.value(None)

    @reactive.calc
    def conv_matches() -> pd.DataFrame:
        """Every conversation in view, largest first."""
        return sessions_df()

    @reactive.calc
    def one_conv() -> str | None:
        """The conversation id picked in the table, if it is still in view."""
        cid = picked_conv()
        hit = conv_matches()
        return cid if cid and (hit.conversation_id == cid).any() else None

    @render_widget
    def conv_turns():
        """One conversation, turn by turn: what each request cost, and why."""
        f, cid = flt(), one_conv()
        if not cid:
            return empty_fig(f["mode"], "Pick one conversation")
        d = usage.q(conv_sql(f, f" AND conversation_id = {sql_str(cid)}") + """
            SELECT turn, ts, idle_s, tin, c_fresh AS fresh, c_cw AS cw, c_cr AS cr,
                   c_out AS out, cost, rewarm, wasted, status_code, finish_reason
            FROM req ORDER BY turn""")
        t = theme.TOKENS[f["mode"]]
        fig = go.Figure()
        cost_part_bars(fig, d.turn, d, f["mode"])
        fig.add_scatter(
            x=d.turn, y=d.tin, name="input tokens", yaxis="y2", mode="lines",
            line={"color": t["ink"], "width": 2},
            hovertemplate="input <b>%{y:,.0f}</b> tokens<extra></extra>",
        )
        top = float(d.cost.max() or 0)
        for mask, name, colour, why in [
            (d.rewarm, "cache expired", theme.STATUS["warning"],
             [f"idle {v / 60:,.0f} min before this turn" for v in d.idle_s]),
            (d.wasted, "failed / cut off", theme.STATUS["critical"],
             [f"HTTP {s}, {r}" for s, r in zip(d.status_code, d.finish_reason)]),
        ]:
            m = mask.fillna(False).astype(bool)
            if m.any():
                fig.add_scatter(
                    x=d.turn[m], y=d.cost[m] + 0.06 * top, name=name, mode="markers",
                    marker={"symbol": "triangle-down", "size": 11, "color": colour},
                    customdata=[w for w, k in zip(why, m) if k],
                    hovertemplate=f"{name}: %{{customdata}}<extra></extra>",
                )
        apply(fig, f["mode"], height=300, showlegend=True, hover=True)
        fig.update_layout(
            hovermode="x unified", bargap=0.15,
            yaxis={"title_text": "est. USD per request", "tickformat": "$.2~f"},
            yaxis2={
                "overlaying": "y", "side": "right", "showgrid": False, "rangemode": "tozero",
                "title_text": "input tokens", "tickformat": "~s",
                "tickfont": {"color": t["ink"]}, "title_font": {"color": t["ink2"]},
            },
            xaxis={"title_text": "turn"},
        )
        return fig

    @render.ui
    def conv_detail():
        """What the picked conversation cost."""
        cid = one_conv()
        if not cid:
            return None
        hit = conv_matches()
        c = hit[hit.conversation_id == cid].iloc[0]
        t = usage.q(f"""SELECT SUM(input_tokens) i, SUM(output_tokens) o,
                               SUM(cache_read_tokens) cr, MAX(input_tokens) peak
                        FROM gateway_events
                        WHERE conversation_id = {sql_str(c.conversation_id)}""").iloc[0]
        tiles = ui.layout_columns(
            theme.stat_tile("Estimated cost", fmt_usd(c.est), c.principal),
            theme.stat_tile(
                "Requests", card_num(c.calls),
                f"{card_num(c.errors)} failed" if c.errors else "none failed",
            ),
            theme.stat_tile(
                "Duration", f"{c.duration_min:,.0f} min",
                f"started {c.started:%Y-%m-%d %H:%M} UTC",
            ),
            theme.stat_tile(
                "Tokens", f"{fmt_num(t.i)} in",
                f"{fmt_num(t.o)} out \u00b7 {fmt_num(t.cr)} cached \u00b7 "
                f"peak {fmt_num(t.peak)} per request",
            ),
            theme.stat_tile("Models", (c.models or "\u2013").split(", ")[0],
                            c.models if c.models and "," in c.models else ""),
            class_="kpi-row",
        )
        found = sorted(
            ((c[f"sav_{k}"], k) for k, *_ in LEVERS if c[f"sav_{k}"] > 0.005), reverse=True
        )
        levers = (
            ui.div(
                ui.strong("What would have cut this one: "),
                "; ".join(
                    f"{lever_label(k).lower()} {fmt_usd(v)} ({100 * v / c.est:.0f}%)"
                    for v, k in found
                ) + ".",
                class_="card-note",
            )
            if found and c.est
            else ui.div("No lever applies to this conversation.", class_="card-note")
        )
        return ui.div(tiles, levers, output_widget("conv_turns"), class_="conv-one")

    @render.data_frame
    def top_sessions():
        """Every conversation, largest first — not just the top five.

        The five-row cap hid how many sessions there were and gave no way to
        reach the rest; the export was the only path to them.
        """
        cols = [
            "conversation",
            "principal",
            "started",
            "duration_min",
            "calls",
            "models",
            "est_spend_usd",
            "peak_input",
            "cache_hit_pct",
            "pace",
            "status",
        ]
        df = conv_matches().copy()
        if df.empty:
            # An empty grid still draws its header, so give it the header the
            # filled table has rather than the raw query's column names.
            return fitted_grid(pd.DataFrame(columns=cols))
        df["conversation"] = df.conversation_id
        df["started"] = df.started.dt.strftime("%Y-%m-%d %H:%M")
        df["usd_per_min"] = (df.est / df.duration_min).round(2)
        df["duration_min"] = df.duration_min.round(0)
        df["est_spend_usd"] = df.est.round(2)
        df["status"] = [
            f"▲ above p{CONV_HIGH_PCT:g}" if h else "" for h in df.high
        ]
        df["cache_hit_pct"] = 100 * df.cache_hit.fillna(0)
        df["pace"] = ["agent" if a else "interactive" for a in df.agent]
        return fitted_grid(formatted(df[cols]), selection_mode="row")

    # Picking a conversation opens it: the detail above the table shows it turn
    # by turn.
    @reactive.effect
    @reactive.event(top_sessions.cell_selection)
    def _conv_pick():
        rows = (top_sessions.cell_selection() or {}).get("rows") or ()
        df = conv_matches()
        picked_conv.set(df.iloc[rows[0]].conversation_id if rows and rows[0] < len(df) else None)


    # ---- audit log ----------------------------------------------------------
    # The derived principal answers "by whom", and the three raw identities
    # behind it are kept beside it: audit evidence must show what the gateway
    # recorded, not only what this dashboard concluded from it.
    # `status` is a string, not a number: an HTTP code is a label rather than a
    # quantity, so sorting or ranging over it ("status > 300") would say nothing.
    RECORD_COLS = """ts, source AS origin, tool, user_agent, username, user_kind AS via,
                     user_email AS email,
                     authenticated_username, viewer_username,
                     content_owner_username, content_guid,
                     (SELECT n.content_name FROM content_names n
                      WHERE n.content_guid = gateway_events.content_guid) AS content_name,
                     job_key,
                     model, provider, integration_auth_type AS auth_type,
                     CAST(status_code AS VARCHAR) AS status,
                     ROUND(duration_ms) AS ms,
                     input_tokens AS tok_in, output_tokens AS tok_out,
                     cache_read_tokens AS tok_cache_read,
                     ROUND(EST_COST_EXPR,4) AS est_cost_usd,
                     finish_reason, response_id, conversation_id,
                     workbench_session_id, workbench_session_name,
                     workbench_project_name, workbench_project_path,
                     team, cost_center, tier, enforced,
                     CAST(request_metadata AS VARCHAR) AS request_metadata"""

    @reactive.calc
    def audit_where() -> str:
        """The sidebar's WHERE."""
        return where(flt())

    def records_sql(where_sql: str, limit: str = "") -> str:
        cols = RECORD_COLS.replace("EST_COST_EXPR", est_cost())
        return f"SELECT {cols} FROM gateway_events WHERE {where_sql} ORDER BY ts DESC{limit}"

    # The grid is sent to the browser whole, so it lists only the most recent
    # rows. The CSV is never capped: it is streamed from DuckDB with every record.
    RECORDS_GRID_MAX = 2000

    @reactive.calc
    def records_shown() -> pd.DataFrame:
        return usage.q(records_sql(audit_where(), f" LIMIT {RECORDS_GRID_MAX}"))

    def records_sql_all() -> str:
        """The export is every filtered row — the table only shows the head."""
        return records_sql(audit_where())

    @render.data_frame
    def records_table():
        return render.DataGrid(records_shown(), height="500px")

    @render.text
    def records_title():
        total = int(
            usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {audit_where()}")
            or 0
        )
        if total > RECORDS_GRID_MAX:
            return f"Gateway records: latest {RECORDS_GRID_MAX:,} of {total:,} shown"
        return f"Gateway records ({total:,})"

    @render.ui
    def records_capped():
        total = (
            usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {audit_where()}")
            or 0
        )
        # An invisible marker for the script in the page head.
        if total > RECORDS_GRID_MAX:
            return ui.span(id="records-capped", style="display:none")
        return None

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

    register_sql_export("records", records_sql_all, "gateway_records.csv")

    open_profile = principals.server(
        input, flt, where, est_cost, records_sql, formatted, fitted_grid,
        register_sql_export, register_export,
        conv_tools=SimpleNamespace(
            by=CONV_BY, keys=conv_keys, groups=conv_groups, fig=conv_spend_fig,
            table=conv_group_table, pct=CONV_HIGH_PCT,
        ),
    )
    register_export("adoption", adoption_df, "adoption_active_users.csv")

    for _name, _fn, _file in [
        ("conversations", sessions_df, "conversations.csv"),
        (
            "attr",
            lambda: attr_df()[
                ["kind", "k", "label", "detail", "why", "spend", "requests", "per_request", "x_median", "tokens"]
            ].rename(columns={"k": "id", "label": "name", "spend": "est_spend_usd"}),
            "attribution.csv",
        ),
        ("models", models_df, "models.csv"),
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
