"""Principals: pick one user, content item, Workbench session or Connect group,
and drill into it.

A prototype, built to find out which of these views are worth keeping. Every
figure respects the sidebar's filters and date range, and is set against the
other principals of the same kind ("peers"), because a number on its own says
little about whether it is unusual.

The SQL and chart builders are plain functions of a WHERE clause, so they can be
exercised without a browser; `server()` is only the glue to Shiny.

Groups come from the Connect groups API (see usage._fetch_groups). A user can be
in several groups, so group totals overlap: the same request counts towards every
group its user is in, and group spend does not add up to the whole.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, req, ui
from shinywidgets import output_widget, render_widget

import conversations as conv_tools
import theme
import usage
from conversations import HIGH_PCT, MIN_BASELINE, higher_cost
from pages.context import PageContext
from theme import apply, card_num, card_usd, empty_fig, fmt_usd
from ui_shell import page_header, table_header

KINDS = {"user": "User", "content": "Content", "session": "Session", "group": "Group"}
NOUN = {"user": "user", "content": "content item", "session": "session", "group": "group"}
# The column that identifies a principal of each kind, on gateway_events. A group
# is not a column: it is reached through user_groups (see kind_source).
KEY_COL = {"user": "username", "content": "content_guid", "session": "workbench_session_id"}
RECORDS_MAX = 2000  # rows in the grid; the CSV export is never capped
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
def peers_df(kind: str, w: str, cost: str) -> pd.DataFrame:
    """Every principal of a kind in view, with what the KPIs compare against."""
    src, key = kind_source(kind)
    extra = ", MAX(group_name) AS gname" if kind == "group" else ""
    df = usage.q(f"""
        SELECT {key} AS k, COALESCE(SUM({cost}), 0) AS spend, COUNT(*) AS requests,
               SUM(cache_read_tokens) / NULLIF(SUM(input_tokens), 0) AS cache_share,
               AVG(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS err,
               MAX(username) AS owner, MAX(workbench_session_name) AS sname{extra}
        FROM {src}
        WHERE {w} AND {key} IS NOT NULL AND CAST({key} AS VARCHAR) <> ''
        GROUP BY 1""")
    df = df.sort_values("spend", ascending=False, ignore_index=True)
    df["label"] = [peer_label(kind, r) for r in df.itertuples()]
    return df


def peer_label(kind: str, r) -> str:
    if kind == "user":
        return str(r.k)
    if kind == "content":
        return usage.content_name(r.k) or short(r.k)
    if kind == "session":
        name = r.sname if isinstance(r.sname, str) else short(r.k)
        return f"{name} \u00b7 {r.owner}" if isinstance(r.owner, str) else name
    return str(r.gname)


def stats(w: str, p: str, cost: str) -> pd.Series:
    return usage.q(f"""
        SELECT COALESCE(SUM({cost}), 0) AS spend, COUNT(*) AS requests,
               SUM(input_tokens) AS tin, SUM(output_tokens) AS tout,
               SUM(cache_read_tokens) AS cr,
               SUM(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS errors,
               SUM(CASE WHEN finish_reason = 'max_tokens' THEN 1 ELSE 0 END) AS truncated,
               quantile_cont(duration_ms, 0.5) AS p50,
               quantile_cont(duration_ms, 0.95) AS p95,
               COUNT(DISTINCT day) AS days, MIN(ts) AS first, MAX(ts) AS last,
               COUNT(DISTINCT conversation_id) AS convs
        FROM gateway_events WHERE {w} AND {p}""").iloc[0]


def identity(kind: str, key: str, w: str, cost: str) -> dict:
    """What the header says about one principal."""
    p = pred(kind, key)
    out: dict = {}
    if kind == "user":
        r = usage.q(f"""SELECT MAX(user_email) email, string_agg(DISTINCT team, ', ') teams
                        FROM gateway_events WHERE {p}""").iloc[0]
        out["email"] = r.email
        out["teams"] = r.teams
        out["via"] = usage.q(f"""SELECT user_kind AS via, SUM({cost}) s FROM gateway_events
                                 WHERE {w} AND {p} GROUP BY 1 ORDER BY s DESC""")
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
                      "running on its own credentials."),
            dict(title="Sessions", df=sessions, label="session", note=""),
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
                 note=f"The top user accounts for {top:.0%} of its spend." if len(users) else ""),
            dict(title="By the charged users' Connect groups", df=groups, label="group",
                 note="A user in several groups counts in each, so these overlap."),
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
        others["session"] = [
            f"\u25b6 {s}" if i == key else s for s, i in zip(others.session, others.id)
        ]
        projects = usage.q(f"""
            SELECT COALESCE(workbench_project_name, '(no project)') AS project,
                   COUNT(DISTINCT workbench_session_id) sessions,
                   COUNT(*) requests, SUM({cost}) est_spend_usd
            FROM gateway_events WHERE {w} AND {mine} AND source = 'workbench'
            GROUP BY 1 ORDER BY est_spend_usd DESC""")
        return [
            dict(title=f"All sessions by {owner}", df=others, label="session",
                 note="\u25b6 marks this one."),
            dict(title=f"{owner}'s Workbench projects", df=projects, label="project", note=""),
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
        dict(title="Members by spend", df=members, label="user", note=""),
        dict(title="Overlap with other groups", df=overlap, label="group",
             note=(f"{fmt_usd(shared)} of this group's {fmt_usd(total)} ({shared / total:.0%}) "
                   "is also counted under another group.") if total else ""),
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
    fig.update_yaxes(tickprefix="$", title_text=f"est. USD / {grain}")
    fig.update_layout(bargap=0.25)
    return apply(fig, mode, height=300, time_axis=True, showlegend=False, hover=True)


def fig_donut(w, p, cost, mode, col: str, kind: str, top: int = 8) -> go.Figure:
    """Where this principal's estimated spend went, split by one column."""
    df = usage.q(f"""SELECT {col} AS k, SUM({cost}) v, COUNT(*) n FROM gateway_events
                     WHERE {w} AND {p} AND {col} IS NOT NULL AND {col} <> ''
                     GROUP BY 1 ORDER BY v DESC""")
    df = df[df.v > 0]
    if df.empty:
        return empty_fig(mode, "No estimated spend in this view")
    t = theme.TOKENS[mode]
    head, rest = df.head(top), df.v.iloc[top:].sum()
    labels = head.k.tolist()
    colors = [theme.color(kind, k, mode) for k in labels]
    values = head.v.tolist()
    if rest > 0:
        labels, colors, values = labels + ["Other"], colors + [t["muted"]], values + [rest]
    total = sum(values)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58, sort=False,
        marker={"colors": colors, "line": {"color": t["surface"], "width": 2}},
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        textfont={"color": "#ffffff", "size": 14},
        hovertemplate="%{label}<br>$%{value:,.2f} \u00b7 %{percent}<extra></extra>",
    ))
    apply(fig, mode, height=300, showlegend=True, hover=True)
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        legend={"orientation": "h", "yanchor": "top", "y": -0.02, "x": 0.5, "xanchor": "center"},
    )
    return fig


VIA_SLOT = {"viewer": 0, "owner": 3, "authenticated": 2}


def fig_relation(rel: dict, mode: str) -> go.Figure:
    df = rel["df"]
    if df.empty or not df.est_spend_usd.sum():
        return empty_fig(mode, "Nothing here in this view")
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
    fig.update_xaxes(range=[0, top.est_spend_usd.max() * 1.3], title_text="est. USD")
    return apply(fig, mode, height=max(200, 26 * len(top) + 70),
                 showlegend=any(v for v, _ in groups), hover=True)


# ---- UI ---------------------------------------------------------------------------
def panel():
    """The Principals tab, a prototype drill-down."""
    return ui.nav_panel(
        "Principals",
        page_header("Principals", "One user, content item, session or group, drilled into."),
        ui.div(
            ui.span("Kind", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons("pf_kind", None, KINDS, selected="user", inline=True),
                class_="seg",
            ),
            ui.div(
                ui.input_selectize("pf_pick", None, {}, options={"placeholder": "pick one\u2026"}),
                class_="pf-pick",
            ),
            ui.span("ranked by estimated spend in view", class_="page-switch-note"),
            class_="page-switch",
        ),
        ui.output_ui("pf_header"),
        ui.output_ui("pf_kpis"),
        ui.div(ui.navset_underline(
            ui.nav_panel(
                "Spending",
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span("Over time"),
                            ui.div(
                                ui.input_radio_buttons(
                                    "pf_grain", None,
                                    {"day": "Daily", "week": "Weekly", "month": "Monthly"},
                                    selected="day", inline=True,
                                ),
                                class_="seg",
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
                ui.card(
                    table_header(ui.output_text("pf_records_title", inline=True),
                                 "dl_pf_records"),
                    ui.card_body(ui.output_data_frame("pf_records")),
                ),
            ),
            ui.nav_panel(
                "Connections",
                # Charts in one row and tables in the next, so a short chart never
                # leaves its neighbour's table floating at the bottom of a tall card.
                ui.layout_columns(
                    *[
                        ui.card(
                            ui.card_header(ui.output_text(f"pf_rel_{i}_title", inline=True)),
                            ui.card_body(
                                ui.output_ui(f"pf_rel_{i}_note"),
                                output_widget(f"pf_rel_{i}_chart"),
                            ),
                        )
                        for i in (0, 1)
                    ],
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    *[
                        ui.card(
                            table_header(ui.output_text(f"pf_rel_{i}_ttitle", inline=True),
                                         f"dl_pf_rel_{i}"),
                            ui.card_body(ui.output_data_frame(f"pf_rel_{i}")),
                        )
                        for i in (0, 1)
                    ],
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Conversations",
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span(ui.output_text("pf_conv_chart_title", inline=True)),
                            ui.div(
                                ui.input_radio_buttons(
                                    "pf_conv_by", None, conv_tools.CONV_BY, selected="model",
                                    inline=True,
                                ),
                                class_="seg",
                            ),
                            class_="hdr-flex",
                        )
                    ),
                    output_widget("pf_conv_spend"),
                ),
                ui.card(
                    table_header(ui.output_text("pf_conv_groups_title", inline=True),
                                 "dl_pf_conv_groups"),
                    ui.card_body(
                        ui.output_ui("pf_conv_groups_note"),
                        ui.output_data_frame("pf_conv_groups"),
                    ),
                ),
            ),
        ), class_="pf-tabs"),
    )


# ---- server ------------------------------------------------------------------------
def server(input, ctx: PageContext, records_sql):
    """Wire the tab. `records_sql` is the Raw Data page's SQL builder.

    Returns open_profile(kind, key), to jump here from elsewhere.
    """
    flt, where, est_cost = ctx.flt, ctx.where, ctx.est_cost
    formatted, fitted_grid = ctx.formatted, ctx.fitted_grid
    register_export, register_sql_export = ctx.register_export, ctx.register_sql_export
    pending: reactive.Value = reactive.value(None)

    @reactive.calc
    def peers() -> pd.DataFrame:
        return peers_df(input.pf_kind(), where(flt()), est_cost())

    @reactive.effect
    def _choices():
        df = peers()
        with reactive.isolate():
            cur, want = input.pf_pick(), pending.get()
        choices = {k: f"{lab} \u2014 {card_usd(s)}" for k, lab, s in zip(df.k, df.label, df.spend)}
        sel = want if want in choices else cur if cur in choices else next(iter(choices), None)
        pending.set(None)
        ui.update_selectize("pf_pick", choices=choices, selected=sel)

    def open_profile(kind: str, key: str):
        pending.set(key)
        ui.update_radio_buttons("pf_kind", selected=kind)
        if input.pf_kind() == kind:
            ui.update_selectize("pf_pick", selected=key)
        ui.update_navset("main_tabs", selected="Principals")

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
        e = ent()
        info = identity(e.kind, e.key, e.w, e.cost)
        n = len(e.peers)
        top_pct = math.ceil(100 * e.rank / n) / 100 if n else 1
        facts: list[tuple[str, object]] = [
            ("rank", f"#{e.rank} of {n:,} {e.noun}s by estimated spend \u00b7 top {top_pct:.0%}")
        ]
        sub = KINDS[e.kind]
        if e.kind == "user":
            sub = info["email"] or "user"
            via = info["via"]
            tot = via.s.sum() or 1
            facts.append(("charged as", " \u00b7 ".join(
                f"{v} {s / tot:.0%}" for v, s in zip(via.via, via.s))))
            facts.append(("Connect groups", chips(info["groups"]) or "none found"))
        elif e.kind == "content":
            sub = e.key
            link = f"{usage.CONNECT_SERVER.rstrip('/')}/connect/#/apps/{e.key}"
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
        e = ent()
        s = stats(e.w, e.p, e.cost)
        cache = (s.cr / s.tin) if s.tin else 0
        cv = convs()
        if cv.empty:
            conv = theme.stat_tile("Conversation cost", DASH, "no conversation ids in view")
        else:
            # One tile: the median as the figure (a few expensive conversations would
            # drag a mean far above what one usually costs), and under it how
            # much of the spend sits in the costliest conversations, those above
            # the HIGH_PCT percentile of this principal's own.
            run, total = cv[cv.high], cv.est.sum()
            if len(cv) < MIN_BASELINE:
                flagged = f"needs {MIN_BASELINE}+ conversations to flag the most expensive"
            elif total and len(run):
                flagged = ui.span(
                    f"{run.est.sum() / total:.0%} of spend ({fmt_usd(run.est.sum())}) is in "
                    f"the {len(run):,} conversations above p{HIGH_PCT:g}",
                    title=f"Conversations costing more than {HIGH_PCT:g}% of this "
                          f"{NOUN[e.kind]}'s conversations.")
            else:
                flagged = f"none above p{HIGH_PCT:g}"
            conv = theme.stat_tile(
                "Conversation cost (median)", fmt_usd(cv.est.median()),
                ui.div(flagged))
        return ui.layout_columns(
            theme.stat_tile("Estimated spend", card_usd(s.spend),
                            f"{card_num(s.requests)} requests"),
            theme.stat_tile("Cache hit", f"{cache:.0%}", "of input read from the prompt cache"),
            conv,
            col_widths=[3, 3, 6], class_="kpi-row",
        )

    @render_widget
    def pf_trend():
        e = ent()
        return fig_trend(daily(e.w, e.p, e.cost, input.pf_grain() or "day"), e.f,
                         input.pf_grain() or "day")

    @render_widget
    def pf_models():
        e = ent()
        return fig_donut(e.w, e.p, e.cost, e.f["mode"], "model", "model")

    @render_widget
    def pf_providers():
        e = ent()
        return fig_donut(e.w, e.p, e.cost, e.f["mode"], "provider", "provider")

    @render_widget
    def pf_origins():
        e = ent()
        return fig_donut(e.w, e.p, e.cost, e.f["mode"], "source", "origin")

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

        def _table():
            df = rels()[i]["df"].copy()
            df["est_spend_usd"] = df.est_spend_usd.round(2)
            return fitted_grid(formatted(df), scroll_after=8)

        def _ttitle():
            return f"{rels()[i]['title']}: all rows"

        render.text(named(_title, f"pf_rel_{i}_title"))
        render.text(named(_ttitle, f"pf_rel_{i}_ttitle"))
        render.ui(named(_note, f"pf_rel_{i}_note"))
        render_widget(named(_chart, f"pf_rel_{i}_chart"))
        render.data_frame(named(_table, f"pf_rel_{i}"))
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
        return input.pf_conv_by() or "model"

    @reactive.calc
    def conv_grouped() -> SimpleNamespace:
        """This principal's conversations, each placed in a group of the chosen
        kind, and the totals per group. Cut at percentiles of its own
        conversations, so the bands fit this principal and not the server."""
        df = convs().copy()
        # the frame the Conversations page's grouping works on
        df = df.assign(main_model=df.main_model.fillna(df.model), calls=df.turns,
                       duration_min=df.minutes.clip(lower=0.5),
                       cache_hit=df.cache_pct / 100)
        key, order = conv_tools.conv_keys(df, conv_by())
        groups = conv_tools.conv_groups(df, key, order)
        return SimpleNamespace(df=df, key=key, order=order, groups=groups)

    @render.text
    def pf_conv_chart_title():
        return f"Conversation spend for {ent().label} by"

    @render_widget
    def pf_conv_spend():
        g = conv_grouped()
        return conv_tools.conv_spend_fig(g.groups, conv_by(), flt()["mode"], HIGH_PCT)

    @render.text
    def pf_conv_groups_title():
        return f"Conversation spend by {conv_tools.CONV_BY[conv_by()].lower()}"

    @render.ui
    def pf_conv_groups_note():
        n = len(conv_grouped().df)
        if n < MIN_BASELINE:
            return ui.div(
                f"Only {n} conversations: too few to flag the most expensive "
                f"(needs {MIN_BASELINE}+).", class_="card-note")
        return ui.div(
            f"Shares are of this {NOUN[input.pf_kind()]}'s own conversations. "
            f"Higher cost means above p{HIGH_PCT} of them.", class_="card-note")

    @render.data_frame
    def pf_conv_groups():
        t = conv_tools.conv_group_table(conv_grouped().groups)
        return fitted_grid(formatted(t))

    register_export("pf_conv_groups",
                    lambda: conv_tools.conv_group_table(conv_grouped().groups),
                    "principal_conversation_groups.csv")

    @render.text
    def pf_records_title():
        e = ent()
        n = int(usage.scalar(
            f"SELECT COUNT(*) FROM gateway_events WHERE {e.w} AND {e.p}") or 0)
        if n > RECORDS_MAX:
            return (f"Gateway records for {e.label}: latest {RECORDS_MAX:,} of {n:,} shown")
        return f"Gateway records for {e.label} ({n:,})"

    @render.data_frame
    def pf_records():
        e = ent()
        return render.DataGrid(
            usage.q(records_sql(f"{e.w} AND {e.p}", f" LIMIT {RECORDS_MAX}")), height="420px"
        )

    # Every record of the picked principal, uncapped: DuckDB writes the file.
    register_sql_export(
        "pf_records",
        lambda: records_sql(f"{ent().w} AND {ent().p}"),
        "principal_records.csv",
    )

    return open_profile


def chips(names: list[str]):
    if not names:
        return None
    return ui.span(*[ui.span(n, class_="pf-chip") for n in names])
