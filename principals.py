"""Profiles: pick one user, content item, Workbench session or Connect group,
and drill into it.

A prototype, built to find out which of these views are worth keeping. Every
figure respects the sidebar's filters and date range, and is set against the
other principals of the same kind ("peers"), because a number on its own says
little about whether it is unusual.

The SQL and chart builders are plain functions of a WHERE clause, so they can be
exercised without a browser; `server()` is only the glue to Shiny.

Each principal is measured on spend, conversation cost and peak context. The
ones in the top 5% of their kind on any of the three are outliers: listed first
in the picker and tagged on the tiles.

Groups come from the Connect groups API (see usage._fetch_groups). A user can be
in several groups, so group totals overlap: the same request counts towards every
group its user is in, and group spend does not add up to the whole.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, req, ui
from shinywidgets import output_widget

import conversations as conv_tools
import theme
import usage
from conversations import HIGH_PCT, MIN_BASELINE, higher_cost
from filters import TOKEN_COUNT, dim_sql
from pages.context import PageContext
from theme import apply, card_num, card_usd, empty_fig, fmt_usd, render_widget
from ui_shell import empty_note, export_button, help_icon, page_header, table_header

KINDS = {"user": "User", "content": "Content", "session": "Session", "group": "Group"}
NOUN = {"user": "user", "content": "content item", "session": "session", "group": "group"}
# The column that identifies a principal of each kind, on gateway_events. A group
# is not a column: it is reached through user_groups (see kind_source).
KEY_COL = {"user": "username", "content": "content_guid", "session": "workbench_session_id"}
DASH, UP, DOWN = "\u2014", "\u25b2", "\u25bc"


def sq(x) -> str:
    """A SQL string literal."""
    return "'" + str(x).replace("'", "''") + "'"


def kind_source(kind: str) -> tuple[str, str]:
    """(FROM clause, key expression) giving one group of rows per principal."""
    if kind == "group":
        return "gateway_events JOIN user_groups USING (user_guid)", "group_guid"
    return "gateway_events", KEY_COL[kind]


def pred(kind: str, key: str) -> str:
    """The rows of one principal, as a condition on plain gateway_events."""
    if kind == "group":
        return f"user_guid IN (SELECT user_guid FROM user_groups WHERE group_guid = {sq(key)})"
    return f"{KEY_COL[kind]} = {sq(key)}"


def short(s, n: int = 8) -> str:
    return str(s)[:n]


# ---- data -----------------------------------------------------------------------
# What a principal can stand out on, and how the header and charts say so.
OUTLIER_MEASURES = [
    ("spend", "high estimated spend"),
    ("conv_p75", "high conversation cost"),
    ("peak_p75", "high peak context"),
]
# A principal at or above this percentile of its kind, on a measure, is an outlier.
OUTLIER_PCT = 95
# With fewer peers than this a percentile means little, so nobody is flagged.
OUTLIER_MIN_N = 8


def peers_df(kind: str, w: str, cost: str) -> pd.DataFrame:
    """Every principal of a kind in view, ranked by spend, with the measures
    the KPI tiles and outlier flags judge."""
    src, key = kind_source(kind)
    has_key = f"{key} IS NOT NULL AND CAST({key} AS VARCHAR) <> ''"
    extra = ", MAX(group_name) AS gname" if kind == "group" else ""
    df = usage.q(f"""
        SELECT {key} AS k, COALESCE(SUM({cost}), 0) AS spend, COUNT(*) AS requests,
               COALESCE(SUM({TOKEN_COUNT}), 0) AS tokens,
               MAX(username) AS owner, MAX(workbench_session_name) AS sname{extra}
        FROM {src}
        WHERE {w} AND {has_key}
        GROUP BY 1""")
    # Conversation cost and peak context (the largest input of a conversation):
    # a percentile of each across the principal's conversations.
    peaks = usage.q(f"""
        SELECT k, quantile_cont(est, {HIGH_PCT / 100}) AS conv_p75,
               quantile_cont(peak, {HIGH_PCT / 100}) AS peak_p75
        FROM (SELECT {key} AS k, SUM({cost}) AS est, MAX(input_tokens) AS peak FROM {src}
              WHERE {w} AND {has_key} AND conversation_id IS NOT NULL
              GROUP BY {key}, conversation_id)
        GROUP BY k""")
    df = df.merge(peaks, on="k", how="left")
    df = df.sort_values("spend", ascending=False, ignore_index=True)
    df["label"] = [peer_label(kind, r) for r in df.itertuples()]
    # Two sessions can share a name, and a chart draws two bars with one label
    # as one.
    dup = df.label.duplicated(keep=False)
    df.loc[dup, "label"] = [f"{lab} ({short(k, 6)})" for lab, k in zip(df.label[dup], df.k[dup])]
    df["why"] = outliers(df)
    return df


def outliers(df: pd.DataFrame) -> pd.Series:
    """Why each principal stands out from its peers, or "" when it does not.

    A principal is flagged on a measure when it is at or above the OUTLIER_PCT
    percentile of its kind, and above the median (so a tie at the top of a flat
    field is not an outlier). Each measure compares only the principals with a
    value above zero for it.
    """
    why: dict = {i: [] for i in df.index}
    for col, label in OUTLIER_MEASURES:
        vals = peer_values(df, col)
        if len(vals) < OUTLIER_MIN_N:
            continue
        high = (vals >= vals.quantile(OUTLIER_PCT / 100)) & (vals > vals.median())
        for i in vals.index[high.to_numpy()]:
            why[i].append(label)
    return pd.Series([" + ".join(why[i]) for i in df.index], index=df.index, dtype=object)


def peer_values(peers: pd.DataFrame, col: str) -> pd.Series:
    """The values of one measure that peers are compared on: those above zero."""
    vals = peers[col].dropna()
    return vals[vals > 0]


def principal_spend(w: str, cost: str) -> pd.DataFrame:
    """One row per user and per content item in view, with its spend.

    Unpriced traffic is kept (spend $0, requests counted): the point is a
    complete list, not only the ones with a price.
    """
    parts = []
    for dim, kind, extra in [
        ("username", "user", "MAX(user_email)"),
        ("content_guid", "content", "MAX(content_owner_username)"),
    ]:
        df = usage.q(f"""SELECT {dim} AS id, {extra} AS detail, COUNT(*) AS requests,
                                COALESCE(SUM({TOKEN_COUNT}), 0) AS tokens,
                                COALESCE(SUM({cost}), 0) AS est_spend_usd
                         FROM gateway_events
                         WHERE {w} AND {dim} IS NOT NULL AND {dim} <> ''
                         GROUP BY 1""")
        df.insert(0, "kind", kind)
        df.insert(2, "name", [
            str(i) if kind == "user" else (usage.content_name(i) or "")
            for i in df.id
        ])
        # What the second column means depends on the kind.
        df = df.rename(columns={"detail": "email" if kind == "user" else "owner"})
        parts.append(df)
    cols = ["kind", "id", "name", "email", "owner", "requests", "tokens", "est_spend_usd"]
    out = pd.concat(parts, ignore_index=True).reindex(columns=cols)
    out["est_spend_usd"] = out.est_spend_usd.round(4)
    return out.sort_values(["est_spend_usd", "requests"], ascending=False, ignore_index=True)


def peer_label(kind: str, r) -> str:
    if kind == "user":
        return str(r.k)
    if kind == "content":
        return usage.content_name(r.k) or short(r.k)
    if kind == "session":
        name = r.sname if isinstance(r.sname, str) else short(r.k)
        return f"{name} \u00b7 {r.owner}" if isinstance(r.owner, str) else name
    return str(r.gname)


def identity(kind: str, key: str, w: str, cost: str) -> dict:
    """What the header says about one principal."""
    p = pred(kind, key)
    out: dict = {}
    if kind == "user":
        r = usage.q(f"""SELECT MAX(user_email) email, string_agg(DISTINCT team, ', ') teams
                        FROM gateway_events WHERE {p}""").iloc[0]
        out["email"] = r.email
        out["teams"] = r.teams
        out["groups"] = usage.q(f"""
            SELECT DISTINCT group_name FROM user_groups WHERE user_guid IN
              (SELECT DISTINCT user_guid FROM gateway_events WHERE {p})
            ORDER BY 1""").group_name.tolist()
    elif kind == "content":
        r = usage.q(f"""
            SELECT MAX(content_owner_username) AS owner,
                   string_agg(DISTINCT integration_auth_type, ', ') AS auth,
                   COUNT(DISTINCT viewer_username) AS viewers,
                   COUNT(DISTINCT job_key) AS jobs,
                   COUNT(DISTINCT username) AS charged
            FROM gateway_events WHERE {w} AND {p}""").iloc[0]
        out.update(r.to_dict())
        out["title"] = usage.content_name(key)
    elif kind == "session":
        r = usage.q(f"""
            SELECT MAX(workbench_session_name) AS name, MAX(username) AS owner,
                   MAX(team) AS team, MAX(workbench_project_name) AS project,
                   MAX(workbench_project_path) AS path, MIN(ts) AS first, MAX(ts) AS last
            FROM gateway_events WHERE {p}""").iloc[0]
        out.update(r.to_dict())
    else:
        r = usage.q(f"""
            SELECT MAX(group_name) AS name, COUNT(DISTINCT user_guid) AS members
            FROM user_groups WHERE group_guid = {sq(key)}""").iloc[0]
        out.update(r.to_dict())
        out["active"] = int(usage.scalar(
            f"SELECT COUNT(DISTINCT user_guid) FROM gateway_events WHERE {w} AND {p}") or 0)
        out["multi"] = int(usage.scalar(f"""
            SELECT COUNT(*) FROM (SELECT user_guid FROM user_groups
              WHERE user_guid IN (SELECT user_guid FROM user_groups WHERE group_guid = {sq(key)})
              GROUP BY 1 HAVING COUNT(DISTINCT group_guid) > 1)""") or 0)
    return out


def daily(w: str, p: str, cost: str, grain: str = "day") -> pd.DataFrame:
    """This principal's spend per day, week or month."""
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, SUM({cost}) v
                       FROM gateway_events WHERE {w} AND {p} GROUP BY 1 ORDER BY 1""")


def relations(kind: str, key: str, w: str, cost: str) -> list[dict]:
    """Two tables of what this principal connects to. Each: title, note, df, label."""
    p = pred(kind, key)
    if kind == "user":
        content = usage.q(f"""
            SELECT content_guid, string_agg(DISTINCT user_kind, ', ') AS via,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {p} AND content_guid IS NOT NULL
            GROUP BY 1 ORDER BY est_spend_usd DESC""")
        content.insert(0, "content", [usage.content_name(g) or short(g) for g in content.content_guid])
        sessions = usage.q(f"""
            SELECT MAX(workbench_session_name) AS session, workbench_session_id AS id,
                   MAX(workbench_project_name) AS project,
                   strftime(MIN(ts), '%Y-%m-%d') AS first, strftime(MAX(ts), '%Y-%m-%d') AS last,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {p} AND workbench_session_id IS NOT NULL
            GROUP BY 2 ORDER BY est_spend_usd DESC""")
        return [
            dict(title="Content", df=content, label="content",
                 note="How they reached it: as a viewer, or as the owner of content "
                      "running on its own credentials.",
                 empty="No Connect content used in this view: all of this user's "
                       "requests came from outside content."),
            dict(title="Sessions", df=sessions, label="session", note="",
                 empty="No Workbench sessions in this view. A session is known only "
                       "when Workbench names it in the request's Gateway-Metadata "
                       "header (workbench-session-id)."),
        ]
    if kind == "content":
        users = usage.q(f"""
            SELECT username AS user, string_agg(DISTINCT user_kind, ', ') AS via,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {p} GROUP BY 1 ORDER BY est_spend_usd DESC""")
        top = users.est_spend_usd.iloc[0] / users.est_spend_usd.sum() if users.est_spend_usd.sum() else 0
        groups = usage.q(f"""
            SELECT group_name AS "group", COUNT(DISTINCT user_guid) users,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events JOIN user_groups USING (user_guid)
            WHERE {w} AND {p} GROUP BY 1 ORDER BY est_spend_usd DESC""")
        return [
            dict(title="Who drives it", df=users, label="user",
                 note=f"The top user accounts for {top:.0%} of its spend." if len(users) else "",
                 empty="No user was charged for this content in this view."),
            dict(title="By the charged users' Connect groups", df=groups, label="group",
                 note="A user in several groups counts in each, so these overlap.",
                 empty="None of the charged users is in a Connect group."),
        ]
    if kind == "session":
        owner = usage.scalar(f"SELECT MAX(username) FROM gateway_events WHERE {p}")
        mine = f"username = {sq(owner)}"
        others = usage.q(f"""
            SELECT MAX(workbench_session_name) AS session, workbench_session_id AS id,
                   strftime(MIN(ts), '%Y-%m-%d') AS first, strftime(MAX(ts), '%Y-%m-%d') AS last,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {mine} AND workbench_session_id IS NOT NULL
            GROUP BY 2 ORDER BY est_spend_usd DESC""")
        projects = usage.q(f"""
            SELECT COALESCE(workbench_project_name, '(no project)') AS project,
                   COUNT(DISTINCT workbench_session_id) sessions,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {mine} AND source = 'workbench'
            GROUP BY 1 ORDER BY est_spend_usd DESC""")
        return [
            dict(title=f"All sessions by {owner}", df=others, label="session",
                 note=""),
            dict(title=f"{owner}'s Workbench projects", df=projects, label="project", note="",
                 empty="No Workbench project named in this view."),
        ]
    members = usage.q(f"""
        SELECT a.username AS user, o.other_groups, a.requests, a.est_spend_usd
        FROM (SELECT user_guid, MAX(username) username, COUNT(*) requests,
                     SUM({cost}) est_spend_usd
              FROM gateway_events WHERE {w} AND {p} GROUP BY 1) a
        LEFT JOIN (SELECT user_guid, string_agg(group_name, ', ' ORDER BY group_name) other_groups
                   FROM user_groups WHERE group_guid <> {sq(key)} GROUP BY 1) o
        USING (user_guid) ORDER BY a.est_spend_usd DESC""")
    overlap = usage.q(f"""
        SELECT group_name AS "group", COUNT(DISTINCT user_guid) shared_members,
               COUNT(*) requests, SUM({cost}) est_spend_usd
        FROM gateway_events JOIN user_groups USING (user_guid)
        WHERE {w} AND {p} AND group_guid <> {sq(key)}
        GROUP BY 1 ORDER BY est_spend_usd DESC""")
    total = members.est_spend_usd.sum()
    shared = members[members.other_groups.notna()].est_spend_usd.sum()
    return [
        dict(title="Members by spend", df=members, label="user", note="",
             empty="No member of this group made a request in this view."),
        dict(title="Overlap with other groups", df=overlap, label="group",
             note=(f"{fmt_usd(shared)} of this group's {fmt_usd(total)} ({shared / total:.0%}) "
                   "is also counted under another group.") if total else "",
             empty="No member of this group is in another group."),
    ]


def conversations(w: str, p: str, cost: str) -> pd.DataFrame:
    df = usage.q(f"""
        WITH ev AS (
            SELECT *, {cost} AS cost,
                   -- from the end of the previous request: when the cache was last touched
                   (epoch_ms(ts) - LAG(epoch_ms(ts) + COALESCE(duration_ms, 0))
                        OVER (PARTITION BY conversation_id ORDER BY ts, response_id)) / 1000.0
                       AS idle_s
            FROM gateway_events
            WHERE {w} AND {p} AND conversation_id IS NOT NULL)
        SELECT conversation_id AS conversation, MIN(ts) AS started,
               date_diff('second', MIN(ts), MAX(ts)) / 60.0 AS minutes,
               COUNT(*) AS turns, SUM(cost) AS est,
               MAX(input_tokens) AS peak_context,
               100.0 * SUM(cache_read_tokens) / NULLIF(SUM(input_tokens), 0) AS cache_pct,
               SUM(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS errors,
               SUM(CASE WHEN finish_reason = 'tool_use' THEN 1 ELSE 0 END) AS tool_calls,
               mode(model) AS model,
               arg_max(model, cost) FILTER (WHERE model <> '') AS main_model,
               arg_max(tool, cost) FILTER (WHERE tool IS NOT NULL) AS main_tool,
               median(idle_s) AS med_idle,
               MAX(username) AS who, MAX(content_guid) AS content
        FROM ev
        GROUP BY 1 ORDER BY est DESC""")
    # Judged against this principal's own conversations, so every figure below
    # is relative to the principal itself, not to the whole server.
    df[["high", "line"]] = higher_cost(df.est)
    return df


# Within this fraction of the typical peer, a principal reads as "in line" rather
# than above or below: a few percent either way is noise, not a habit.
VERSUS_EVEN = 0.05


def versus_peers(e: SimpleNamespace, col: str, enough: bool = True) -> ui.Tag | None:
    """A badge saying where this principal's figure for one measure sits against
    the typical (median) peer, and an outlier tag when the principal is in the
    top 5% of its kind on it. Lower is better for every measure (spend, cost,
    context), so below is green and above is amber. With `enough` false the
    principal's own figure is too thin to compare, so only the tag is shown.
    The figures behind them are in the tile's help (`typical_help`)."""
    vals = peer_values(e.peers, col)
    if vals.empty:
        return None
    org, r = float(vals.median()), e.peers.iloc[e.rank - 1]
    mine = r[col]
    flag = dict(OUTLIER_MEASURES)[col] in r.why.split(" + ")
    tag = ui.span("outlier", class_="versus versus-outlier") if flag else None
    if not enough or pd.isna(mine):
        return ui.div(tag, class_="versus-row") if tag else None
    d = mine / org - 1 if org else 0.0
    if abs(d) < VERSUS_EVEN:
        cls, text = "even", "\u2248 typical"
    elif d < 0:
        cls, text = "below", f"\u25bc {-d:.0%} below typical"
    else:
        cls, text = "above", f"\u25b2 {d:.0%} above typical"
    return ui.div(ui.span(text, class_=f"versus versus-{cls}"), tag, class_="versus-row")


def typical_help(e: SimpleNamespace, col: str, say, about: str, *extra) -> ui.Tag:
    """The help beside a KPI tile: what the measure is, the typical peer it is
    compared with, and where the outlier line sits."""
    vals = peer_values(e.peers, col)
    body: list = [ui.p(about)]
    if not vals.empty:
        body.append(ui.p(
            ui.strong(f"Typical {e.noun}: {say(float(vals.median()))}"),
            f", the median of the {len(vals):,} {e.noun}s in view. The badge says how "
            f"far this {e.noun} is above or below it."))
    if len(vals) >= OUTLIER_MIN_N:
        line = say(float(vals.quantile(OUTLIER_PCT / 100)))
        body.append(ui.p(
            ui.strong("Outlier"),
            f" marks the top {100 - OUTLIER_PCT}% of {e.noun}s: {line} or more."))
    else:
        body.append(ui.p(
            f"With fewer than {OUTLIER_MIN_N} {e.noun}s to compare, nobody is "
            "marked an outlier."))
    body.extend(ui.p(x) for x in extra if x)
    body[-1] = ui.p(*body[-1].children, class_="mb-0")
    return help_icon(*body)


# ---- figures ------------------------------------------------------------------------
def fig_trend(me, f, grain: str = "day") -> go.Figure:
    mode = f["mode"]
    if me.empty:
        return empty_fig(mode, "No estimated spend in this view")
    t = theme.TOKENS[mode]
    me = me.set_index(pd.to_datetime(me.day)).v
    if grain == "day":
        # Quiet days are zero-height bars, not gaps.
        days = pd.date_range(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), freq="D")
        me = me.reindex(days, fill_value=0)
    days = me.index
    fig = go.Figure()
    fig.add_bar(x=days, y=me.values, marker_color=t["series"][0],
                hovertemplate="$%{y:,.2f}<extra></extra>")
    theme.usd_axis(fig, "y", title_text=f"est. USD / {grain}")
    fig.update_layout(bargap=0.25)
    return apply(fig, mode, height=300, time_axis=True, showlegend=False, hover=True)


def fig_split(w, p, cost, mode, col: str, kind: str, top: int = 8) -> go.Figure:
    """Where this principal's estimated spend went, split by one column.

    Bars, not a donut: with several models the slices were too thin to read or
    tell apart, and a bar carries its name and figure on the axis.
    """
    k = dim_sql(col)
    df = usage.q(f"""SELECT {k} AS k, SUM({cost}) v, COUNT(*) n FROM gateway_events
                     WHERE {w} AND {p} AND {k} IS NOT NULL AND {k} <> ''
                     GROUP BY 1 ORDER BY v DESC""")
    df = df[df.v > 0]
    if df.empty:
        return empty_fig(mode, "No estimated spend in this view")
    t = theme.TOKENS[mode]
    head, rest = df.head(top), df.v.iloc[top:].sum()
    labels = [str(x) for x in head.k]
    colors = [theme.color(kind, lab, mode) for lab in labels]
    values = head.v.tolist()
    if rest > 0:
        labels, colors, values = labels + ["Other"], colors + [t["muted"]], values + [rest]
    total = sum(values)
    # plotly draws the first bar at the bottom; reverse so the biggest is on top
    labels, colors, values = labels[::-1], colors[::-1], values[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=[lab[:28] for lab in labels], orientation="h", width=0.6,
        marker={"color": colors},
        # The share alone: the card is a third of the page, and a model name
        # plus "$59.21 \u00b7 50%" ran off its edge. Hover has the dollars.
        text=[f"\u00a0{v / total:.0%}" for v in values],
        textposition="outside", textfont={"color": t["ink"], "size": 13},
        customdata=[f"{lab}\n{fmt_usd(v)} \u00b7 {v / total:.1%}" for lab, v in zip(labels, values)],
        hoverinfo="none",  # the name follows the mouse; see nametip
    ))
    fig.update_layout(meta="nametip")
    fig.update_yaxes(type="category")
    fig.update_xaxes(range=[0, max(values) * 1.3], showticklabels=False)
    return apply(fig, mode, height=max(200, 34 * len(labels) + 60), showlegend=False,
                 hover=True)


VIA_SLOT = {"viewer": 0, "owner": 3, "authenticated": 2}


def fig_relation(rel: dict, mode: str) -> go.Figure:
    df = rel["df"]
    if df.empty or not df.est_spend_usd.sum():
        msg = rel.get("empty", "Nothing here in this view")
        if not df.empty:
            msg = f"No estimated spend: {len(df):,} {rel['label']}s, all on unpriced models"
        return empty_fig(mode, msg)
    t = theme.TOKENS[mode]
    top = df.head(12).iloc[::-1]
    labels = [str(x)[:40] for x in top[rel["label"]]]
    top = top.assign(_y=labels)
    # Content is coloured by how the user reached it (viewer / owner / ...).
    groups = [(v, top[top.via == v]) for v in top.via.dropna().unique()] \
        if "via" in top else [(None, top)]
    fig = go.Figure()
    for via, d in groups:
        # "owner, viewer": reached both ways, so it gets a colour of its own.
        slot = VIA_SLOT.get(via, 5) if via else 1
        fig.add_bar(
            x=d.est_spend_usd, y=d._y, orientation="h", width=0.6,
            name=via or "", showlegend=bool(via), marker_color=t["series"][slot],
            text=["\u00a0" + fmt_usd(v) for v in d.est_spend_usd],
            textposition="outside", textfont={"color": t["ink"], "size": 13},
            hovertemplate="%{y}<br>$%{x:,.2f}" + (f" \u00b7 {via}" if via else "")
                          + "<extra></extra>",
        )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels)
    theme.usd_axis(fig, range=[0, top.est_spend_usd.max() * 1.3], title_text="est. USD")
    return apply(fig, mode, height=max(200, 26 * len(top) + 70),
                 showlegend=any(v for v, _ in groups), hover=True)


# ---- UI ---------------------------------------------------------------------------
def panel():
    """The Profiles tab, a prototype drill-down."""
    return ui.nav_panel(
        "Profiles",
        page_header("Profiles", "Drill down into details for users, groups, content and sessions."),
        ui.div(
            ui.div(
                ui.input_radio_buttons("pf_kind", None, KINDS, selected="user", inline=True),
                class_="seg",
            ),
            ui.div(
                # input_select, not input_selectize: the latter restores a bookmarked
                # value as the *choices* and crashes the page on reload. The choices
                # are filled in by the server (`_choices`).
                ui.input_select(
                    "pf_pick", None, {}, selectize=True,
                    options={"placeholder": "pick one\u2026"},
                ),
                class_="pf-pick",
            ),
            # Every user and every content item with its spend, for an admin to
            # take into another tool. Not tied to the kind or the pick.
            ui.div(
                export_button("dl_principal_spend", "Export spend by principal"),
                help_icon(
                    "Spend by principal",
                    ui.p(
                        "A CSV with one row per user and one per content item in "
                        "view: its kind (user or content), id, name, requests, "
                        "tokens and estimated spend, largest first. It follows the "
                        "date range and filters."
                    ),
                    ui.p(
                        "A request from content is charged to both the content item "
                        "and a user, so the user rows and the content rows each add "
                        "up to their own total; do not add the two together.",
                        class_="mb-0",
                    ),
                ),
                class_="hdr-flex-left ms-auto",
            ),
            class_="page-switch",
        ),
        ui.output_ui("pf_header"),
        ui.output_ui("pf_kpis"),
        # Hidden until something is picked: with no principal of the kind in view
        # (no sessions, say) the tabs were rows of empty cards under the note.
        ui.panel_conditional("input.pf_pick", ui.div(ui.navset_underline(
            ui.nav_panel(
                "Conversations",
                ui.output_ui("pf_conv_summary"),
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.div(
                                ui.span(ui.output_text("pf_conv_chart_title", inline=True)),
                                ui.output_ui("pf_conv_help", inline=True),
                                class_="hdr-flex-left",
                            ),
                            ui.div(
                                ui.div(
                                    ui.input_radio_buttons(
                                        "pf_conv_by", None, conv_tools.CONV_BY,
                                        selected="context", inline=True,
                                    ),
                                    class_="seg",
                                ),
                                export_button("dl_pf_conv_groups"),
                                class_="hdr-flex-left",
                            ),
                            class_="hdr-flex",
                        )
                    ),
                    output_widget("pf_conv_spend"),
                ),
            ),
            ui.nav_panel(
                "Spending",
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span("Over time"),
                            ui.div(
                                ui.div(
                                    ui.input_radio_buttons(
                                        "pf_grain", None,
                                        {"day": "Daily", "week": "Weekly", "month": "Monthly"},
                                        selected="day", inline=True,
                                    ),
                                    class_="seg",
                                ),
                                # Every gateway record of the pick, uncapped.
                                export_button("dl_pf_records", "Export records"),
                                class_="hdr-flex-left",
                            ),
                            class_="hdr-flex",
                        )
                    ),
                    output_widget("pf_trend"),
                ),
                ui.layout_columns(
                    ui.card(ui.card_header("By model"), output_widget("pf_models")),
                    ui.card(ui.card_header("By provider"), output_widget("pf_providers")),
                    ui.card(ui.card_header("By origin"), output_widget("pf_origins")),
                    col_widths=[4, 4, 4],
                ),
            ),
            ui.nav_panel(
                "Connections",
                # The chart shows the top rows; the CSV holds all of them.
                ui.layout_columns(
                    *[
                        ui.card(
                            table_header(ui.output_text(f"pf_rel_{i}_title", inline=True),
                                         f"dl_pf_rel_{i}"),
                            ui.card_body(
                                ui.output_ui(f"pf_rel_{i}_note"),
                                output_widget(f"pf_rel_{i}_chart"),
                            ),
                        )
                        for i in (0, 1)
                    ],
                    col_widths=[6, 6],
                ),
            ),
        ), class_="pf-tabs")),
    )


# ---- server ------------------------------------------------------------------------
def server(input, ctx: PageContext, records_sql):
    """Wire the tab. `records_sql` is the Raw Data page's SQL builder."""
    flt, where, est_cost = ctx.flt, ctx.where, ctx.est_cost
    register_export, register_sql_export = ctx.register_export, ctx.register_sql_export

    @reactive.calc
    def peers() -> pd.DataFrame:
        return peers_df(input.pf_kind(), where(flt()), est_cost())

    @reactive.effect
    def _choices():
        df = peers()
        with reactive.isolate():
            cur = input.pf_pick()
        # Outliers first, in a group of their own, so the page opens on the
        # biggest one.
        def options(d: pd.DataFrame) -> dict[str, str]:
            return {str(k): f"{lab} \u2014 {card_usd(s)}"
                    for k, lab, s in zip(d.k, d.label, d.spend)}

        flagged = df.why != ""
        choices: dict[str, Any] = options(df)
        if flagged.any():
            choices = {f"Outliers ({flagged.sum():,} of {len(df):,})": options(df[flagged])}
            if not flagged.all():
                choices["Others"] = options(df[~flagged])
        keys = [str(k) for k in pd.concat([df.k[flagged], df.k[~flagged]])]
        sel = cur if cur in keys else next(iter(keys), None)
        ui.update_selectize("pf_pick", choices=choices, selected=sel)

    @reactive.calc
    def ent() -> SimpleNamespace:
        kind, key, df = input.pf_kind(), input.pf_pick(), peers()
        req(key and key in set(df.k))
        f = flt()
        rank = int(df.index[df.k == key][0]) + 1
        return SimpleNamespace(
            kind=kind, key=key, f=f, w=where(f), p=pred(kind, key), cost=est_cost(),
            label=df.label[rank - 1], rank=rank, peers=df, noun=NOUN[kind],
        )

    @render.ui
    def pf_header():
        kind = input.pf_kind()
        if peers().empty:
            why = {
                "session": " A session is known only when Workbench names it in the "
                           "Gateway-Metadata header (workbench-session-id), and no "
                           "request in view carries one.",
                "group": " Groups come from Connect; none of the users in view is in one.",
            }.get(kind, "")
            return empty_note(f"No {NOUN[kind]}s in this view.{why}")
        e = ent()
        info = identity(e.kind, e.key, e.w, e.cost)
        facts: list[tuple[str, ui.TagChild]] = []
        sub = KINDS[e.kind]
        if e.kind == "user":
            sub = info["email"] or "user"
            if info["groups"]:
                facts.append(("Groups", chips(info["groups"])))
        elif e.kind == "content":
            sub = e.key
            link = f"{usage.connect_server()}/connect/#/apps/{e.key}"
            facts += [
                ("owner", info["owner"] or "unknown"),
                ("auth", info["auth"] or "unknown"),
                ("viewers", f"{int(info['viewers']):,} named \u00b7 "
                            f"{int(info['charged']):,} charged"),
                ("job keys", f"{int(info['jobs']):,} processes"),
                ("open", ui.a("in Connect \u2197", href=link, target="_blank")),
            ]
        elif e.kind == "session":
            sub = e.key
            dur = (info["last"] - info["first"]) if pd.notna(info["first"]) else None
            facts += [
                ("owner", info["owner"]),
                ("team", info["team"] or DASH),
                ("project", " ".join(x for x in (info["project"], info["path"]) if x) or DASH),
                ("active", f"{info['first']:%Y-%m-%d %H:%M} \u2192 {info['last']:%Y-%m-%d %H:%M}"
                           + (f" ({dur.days}d {dur.seconds // 3600}h)" if dur is not None else "")),
            ]
        else:
            sub = "Connect group"
            facts += [
                ("members", f"{int(info['members']):,} \u00b7 {info['active']:,} active in view"),
                ("overlap", f"{info['multi']:,} of its members are also in other groups, "
                            "so their usage counts there too"),
            ]
        return ui.div(
            ui.div(e.label, class_="pf-title"),
            ui.div(sub, class_="pf-sub"),
            ui.tags.dl(*[x for k, v in facts for x in (ui.tags.dt(k), ui.tags.dd(v))],
                       class_="pf-facts"),
            class_="pf-head",
        )

    @render.ui
    def pf_kpis():
        # Three tiles, each this principal's figure set against the typical
        # (median) peer of its kind, with an outlier tag in the top 5%. The
        # figures are the same ones peers_df ranks and flags; the typical
        # figure and the outlier line are in each tile's help.
        e = ent()
        r = e.peers.iloc[e.rank - 1]
        thin = (f"This {e.noun} has fewer than {MIN_BASELINE} conversations, too few "
                "for its own percentile to compare.")
        spend = theme.stat_tile(
            "Estimated spend", card_usd(r.spend), versus_peers(e, "spend"),
            help=typical_help(e, "spend", fmt_usd,
                              f"This {e.noun}'s estimated spend in view."))
        cv = convs()
        if cv.empty:
            conv = theme.stat_tile("Conversation cost", DASH,
                                   "no conversation ids in view")
        else:
            # How much of the spend sits in the costliest conversations, those
            # above this principal's HIGH_PCT percentile.
            run, total = cv[cv.high], cv.est.sum()
            enough = len(cv) >= MIN_BASELINE
            share = (
                f"{run.est.sum() / total:.0%} of this {e.noun}'s spend "
                f"({fmt_usd(run.est.sum())}) is in its {len(run):,} conversations "
                f"above p{HIGH_PCT:g}." if enough and total and len(run) else None)
            conv = theme.stat_tile(
                "Conversation cost",
                DASH if pd.isna(r.conv_p75) else fmt_usd(r.conv_p75),
                versus_peers(e, "conv_p75", enough=enough),
                help=typical_help(
                    e, "conv_p75", fmt_usd,
                    f"The p{HIGH_PCT:g} of the estimated cost of this {e.noun}'s "
                    "conversations; the typical figure is the median of each "
                    f"{e.noun}'s p{HIGH_PCT:g}.",
                    share, None if enough else thin))
        peaks = cv.peak_context.dropna() if not cv.empty else pd.Series(dtype=float)
        if len(peaks) == 0:
            ctx = theme.stat_tile("Peak context", DASH,
                                  "no conversation ids in view")
        else:
            enough = len(peaks) >= MIN_BASELINE
            ctx = theme.stat_tile(
                "Peak context", card_num(r.peak_p75),
                versus_peers(e, "peak_p75", enough=enough),
                help=typical_help(
                    e, "peak_p75", lambda v: f"{card_num(v)} tokens",
                    "Peak context is the largest input any request in a conversation "
                    f"sent. This is the p{HIGH_PCT:g} of it across this {e.noun}'s "
                    f"conversations; the typical figure is the median of each "
                    f"{e.noun}'s p{HIGH_PCT:g}.",
                    None if enough else thin))
        return ui.layout_columns(spend, conv, ctx, col_widths=[4, 4, 4], class_="kpi-row")

    @render_widget
    def pf_trend():
        e = ent()
        return fig_trend(daily(e.w, e.p, e.cost, input.pf_grain() or "day"), e.f,
                         input.pf_grain() or "day")

    @render_widget
    def pf_models():
        e = ent()
        return fig_split(e.w, e.p, e.cost, e.f["mode"], "model", "model")

    @render_widget
    def pf_providers():
        e = ent()
        return fig_split(e.w, e.p, e.cost, e.f["mode"], "provider", "provider")

    @render_widget
    def pf_origins():
        e = ent()
        return fig_split(e.w, e.p, e.cost, e.f["mode"], "source", "origin")

    @reactive.calc
    def rels() -> list[dict]:
        e = ent()
        return relations(e.kind, e.key, e.w, e.cost)

    def rel_outputs(i: int):
        # A renderer registers under its function's name, so name each one first.
        def named(fn, name):
            fn.__name__ = name
            return fn

        def _title():
            return rels()[i]["title"]

        def _note():
            note = rels()[i]["note"]
            return ui.p(note, class_="pf-note") if note else None

        def _chart():
            return fig_relation(rels()[i], flt()["mode"])

        render.text(named(_title, f"pf_rel_{i}_title"))
        render.ui(named(_note, f"pf_rel_{i}_note"))
        render_widget(named(_chart, f"pf_rel_{i}_chart"))
        register_export(f"pf_rel_{i}", lambda: rels()[i]["df"],
                        ("pf_connections_%d.csv" % (i + 1)))

    for _i in (0, 1):
        rel_outputs(_i)

    @reactive.calc
    def convs() -> pd.DataFrame:
        e = ent()
        return conversations(e.w, e.p, e.cost)

    @reactive.calc
    def conv_by() -> str:
        return input.pf_conv_by() or "context"

    @reactive.calc
    def conv_grouped() -> SimpleNamespace:
        """This principal's conversations, each placed in a group of the chosen
        kind, and the totals per group. Cut at percentiles of its own
        conversations, so the bands fit this principal and not the server."""
        df = convs().copy()
        # the frame conv_keys groups
        df = df.assign(main_model=df.main_model.fillna(df.model), calls=df.turns,
                       duration_min=df.minutes.clip(lower=0.5),
                       cache_hit=df.cache_pct / 100, peak_input=df.peak_context)
        key, order = conv_tools.conv_keys(df, conv_by())
        groups = conv_tools.conv_groups(df, key, order)
        return SimpleNamespace(df=df, key=key, order=order, groups=groups)

    @render.text
    def pf_conv_chart_title():
        return f"Conversation spend for {ent().label} by"

    @render_widget
    def pf_conv_spend():
        g, by, mode = conv_grouped(), conv_by(), flt()["mode"]
        if by in conv_tools.CURVE_X:
            return conv_tools.conv_cumulative_fig(g.df, by, mode, HIGH_PCT)
        return conv_tools.conv_donuts_fig(g.groups, mode, HIGH_PCT)

    @render.ui
    def pf_conv_summary():
        return conv_summary(convs(), ent().noun, flt()["mode"])

    @render.ui
    def pf_conv_help():
        by = conv_by()
        what = CONV_VIEW_HELP[by]
        if by in conv_tools.CURVE_X:
            how = ("Each line adds up its own spend from left to right, so a line that "
                   "climbs late means that spend sits at the high end. The further apart "
                   "the two lines, the more this measure sets the costly conversations "
                   "apart from the rest.")
        else:
            how = ("Each donut splits spend into higher-cost and normal conversations. The "
                   "number inside is the higher-cost share, amber when it is "
                   f"{conv_tools.DONUT_ABOVE:.0%} or more above the share for all "
                   "conversations.")
        return help_icon(conv_tools.CONV_BY[by], ui.p(what), ui.p(how, class_="mb-0"))

    register_export("pf_conv_groups",
                    lambda: conv_tools.conv_group_table(conv_grouped().groups),
                    "profile_conversation_groups.csv")

    # Every record of the picked principal, uncapped: DuckDB writes the file.
    register_sql_export(
        "pf_records",
        lambda: records_sql(f"{ent().w} AND {ent().p}"),
        "profile_records.csv",
    )

    register_export("principal_spend", lambda: principal_spend(where(flt()), est_cost()),
                    "spend_by_principal.csv")


# What each view of the conversation chart measures, for its help.
CONV_VIEW_HELP = {
    "context": "Peak context: the largest input any request in the conversation sent, "
               "cached and uncached tokens alike.",
    "model": "Each conversation counts under the model it spent most on.",
    "duration": "Duration: the time from a conversation's first request to its last.",
    "turns": "Turns: the number of requests in the conversation.",
    "pace": "Pace: the typical (median) pause between a conversation's requests. "
            "Single-request conversations have no pause and are left out.",
    "cache": "Caching: the share of the conversation's input read from the prompt cache.",
    "tool": "Each conversation counts under the harness it spent most through, read "
            "from the User-Agent header.",
}


def conv_summary(cv: pd.DataFrame, noun: str, mode: str) -> ui.Tag:
    """The box at the top of the Conversations tab: how many conversations
    there are, where the higher-cost line sits, and how the two camps split
    the count and the spend."""
    n = len(cv)
    if not n:
        return ui.div(f"No conversations for this {noun} in view.", class_="conv-summary")
    priced = cv[cv.est > 0]
    unpriced = n - len(priced)
    count = ui.div(
        ui.div(f"{n:,}", class_="conv-summary-num"),
        ui.div("conversations", class_="conv-summary-label"),
        ui.div(f"{unpriced:,} with no estimated spend", class_="conv-summary-sub")
        if unpriced else None,
        class_="conv-summary-count",
    )
    if n < MIN_BASELINE:
        return ui.div(count, ui.div(
            f"Too few to split by cost: a conversation is marked higher cost against "
            f"this {noun}'s p{HIGH_PCT:g}, which needs {MIN_BASELINE}+ conversations.",
            class_="conv-summary-how"), class_="conv-summary")
    total, line = float(priced.est.sum()), float(cv.line.iloc[0])
    hi, lo = cv[cv.high], cv[~cv.high]
    colours = (theme.STATUS["warning"], theme.TOKENS[mode]["series"][0])

    def split(label: str, parts: list[tuple[float, str]]) -> ui.Tag:
        """One labelled bar, higher cost then normal, each part as wide as its share."""
        whole = sum(v for v, _ in parts) or 1
        segs = []
        for (v, text), colour, cls in zip(parts, colours, ("hi", "lo")):
            share = v / whole
            segs.append(ui.div(
                # A thin part keeps its title; its text would not fit.
                f"{text} \u00b7 {share:.0%}" if share >= 0.12 else "",
                class_=f"conv-split-seg conv-split-{cls}",
                style=f"width:{share:.2%};background:{colour}",
                title=f"{text} \u00b7 {share:.0%}",
            ))
        return ui.div(ui.div(label, class_="conv-split-label"),
                      ui.div(*segs, class_="conv-split-bar"), class_="conv-split-row")

    def key(colour: str, name: str) -> ui.Tag:
        return ui.span(ui.span(class_="tile-dot", style=f"background:{colour}"),
                       f" {name}", class_="conv-summary-key")

    return ui.div(
        count,
        ui.div(
            split("conversations", [(len(hi), f"{len(hi):,}"), (len(lo), f"{len(lo):,}")]),
            split("spend", [(float(hi.est.sum()), fmt_usd(float(hi.est.sum()))),
                            (float(lo.est.sum()), fmt_usd(float(lo.est.sum())))]),
            ui.div(
                key(colours[0], "higher cost"),
                key(colours[1], "normal"),
                help_icon(
                    "Higher cost and normal",
                    ui.p(
                        ui.strong("Higher cost"),
                        f" conversations cost more than {HIGH_PCT:g}% of this {noun}'s "
                        f"own conversations: over {fmt_usd(line)}, its p{HIGH_PCT:g}. ",
                        ui.strong("Normal"), " is the rest.",
                    ),
                    ui.p(
                        f"The line is drawn from this {noun}'s own conversations, "
                        "so it fits their usage rather than a fixed dollar amount. A "
                        "conversation's cost is the estimated cost of all its requests.",
                        class_="mb-0",
                    ),
                ),
                class_="conv-summary-keys",
            ),
            class_="conv-summary-splits",
        ),
        class_="conv-summary",
    ) if total else ui.div(count, class_="conv-summary")


def chips(names: list[str]):
    if not names:
        return None
    return ui.span(*[ui.span(n, class_="pf-chip") for n in names])
