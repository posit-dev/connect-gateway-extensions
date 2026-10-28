"""Attribution: pick one user, content item, Workbench session or Connect group,
and drill into it.

A prototype, built to find out which of these views are worth keeping. Every
figure respects the sidebar's filters and date range, and is set against the
other principals of the same kind ("peers"), because a number on its own says
little about whether it is unusual.

The SQL and chart builders are plain functions of a WHERE clause, so they can be
exercised without a browser; `server()` is only the glue to Shiny.

The ones in the top 5% of their kind by spend are outliers: listed first in the
table and tagged.

Groups come from the Connect groups API (see usage._fetch_groups). A user can be
in several groups, so group totals overlap: the same request counts towards every
group its user is in, and group spend does not add up to the whole.
"""

from __future__ import annotations

from types import SimpleNamespace

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
from theme import apply, empty_fig, fmt_usd, render_widget
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
# A principal at or above this percentile of its kind, by spend, is an outlier.
OUTLIER_PCT = 95
# With fewer peers than this a percentile means little, so nobody is flagged.
OUTLIER_MIN_N = 8


def peers_df(kind: str, w: str, cost: str) -> pd.DataFrame:
    """Every principal of a kind in view, ranked by spend, with the model each
    spent most on and whether it is an outlier."""
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
    models = usage.q(f"""
        SELECT k, arg_max(model, v) AS top_model
        FROM (SELECT {key} AS k, model, COALESCE(SUM({cost}), 0) AS v
              FROM {src} WHERE {w} AND {has_key} GROUP BY 1, 2)
        GROUP BY k""")
    df = df.merge(models, on="k", how="left")
    df = df.sort_values("spend", ascending=False, ignore_index=True)
    df["label"] = [peer_label(kind, r) for r in df.itertuples()]
    # Two sessions can share a name, and a chart draws two bars with one label
    # as one.
    dup = df.label.duplicated(keep=False)
    df.loc[dup, "label"] = [f"{lab} ({short(k, 6)})" for lab, k in zip(df.label[dup], df.k[dup])]
    df["outlier"] = outliers(df)
    return df


def outliers(df: pd.DataFrame) -> pd.Series:
    """Whether each principal's spend stands out from its peers.

    A principal is an outlier when its spend is at or above the OUTLIER_PCT
    percentile of its kind, and above the median (so a tie at the top of a flat
    field is not an outlier). Only principals with some spend are compared.
    """
    vals = df.spend[df.spend > 0]
    if len(vals) < OUTLIER_MIN_N:
        return pd.Series(False, index=df.index)
    return (df.spend >= vals.quantile(OUTLIER_PCT / 100)) & (df.spend > vals.median())


def peer_label(kind: str, r) -> str:
    if kind == "user":
        return str(r.k)
    if kind == "content":
        return usage.content_name(r.k) or short(r.k)
    if kind == "session":
        name = r.sname if isinstance(r.sname, str) else short(r.k)
        return f"{name} \u00b7 {r.owner}" if isinstance(r.owner, str) else name
    return str(r.gname)


def daily(w: str, p: str, cost: str, grain: str = "day") -> pd.DataFrame:
    """This principal's spend per day, week or month."""
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, SUM({cost}) v
                       FROM gateway_events WHERE {w} AND {p} GROUP BY 1 ORDER BY 1""")


def relations(kind: str, key: str, w: str, cost: str) -> list[dict]:
    """Two tables of what this principal connects to. Each: title, note, df, label."""
    p = pred(kind, key)
    if kind == "user":
        content = usage.q(f"""
            SELECT content_guid, string_agg(DISTINCT user_kind, ', ' ORDER BY user_kind) AS via,
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
                 note="How they reached it: as a viewer, as the owner of content running "
                      "on its own credentials, or direct, with their own credentials.",
                 empty="No Connect content used in this view: all of this user's "
                       "requests came from outside content."),
            dict(title="Sessions", df=sessions, label="session", note="",
                 empty="No Workbench sessions in this view. A session is known only "
                       "when Workbench names it in the request's Gateway-Metadata "
                       "header (workbench-session-id)."),
        ]
    if kind == "content":
        users = usage.q(f"""
            SELECT username AS user, string_agg(DISTINCT user_kind, ', ' ORDER BY user_kind) AS via,
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
                   LAG(input_tokens) OVER (PARTITION BY conversation_id
                                           ORDER BY ts, response_id) AS prev_input,
                   MAX(cache_read_tokens) OVER (PARTITION BY conversation_id) AS cache_used
            FROM gateway_events
            WHERE {w} AND {p} AND conversation_id IS NOT NULL)
        SELECT conversation_id AS conversation, MIN(ts) AS started,
               date_diff('second', MIN(ts), MAX(ts)) / 60.0 AS minutes,
               COUNT(*) AS turns, SUM(cost) AS est,
               MAX(input_tokens) AS peak_context,
               SUM(cache_read_tokens + cache_creation_tokens) > 0 AS uses_cache,
               -- A cache miss: a request that read back under half of what the
               -- one before it sent, in a conversation that uses the cache at
               -- all. The tokens are what had to be read again. Below 1024
               -- tokens nothing is cached, so a short prompt is no miss.
               SUM(CASE WHEN prev_input >= 1024 AND cache_used > 0
                             AND cache_read_tokens < 0.5 * prev_input
                        THEN prev_input - cache_read_tokens ELSE 0 END) AS miss_tokens,
               SUM(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS errors,
               SUM(CASE WHEN finish_reason = 'tool_use' THEN 1 ELSE 0 END) AS tool_calls,
               mode(model) AS model,
               arg_max(model, cost) FILTER (WHERE model <> '') AS main_model,
               arg_max(tool, cost) FILTER (WHERE tool IS NOT NULL) AS main_tool,
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
# How the chart names each kind. "authenticated" says how the credential was
# checked, not who called; it means the person used their own credentials.
VIA_LABEL = {"authenticated": "direct"}


def via_label(via: str) -> str:
    return ", ".join(VIA_LABEL.get(k, k) for k in via.split(", "))


def fig_relation(rel: dict, mode: str) -> go.Figure:
    df = rel["df"]
    if df.empty or not df.est_spend_usd.sum():
        msg = rel.get("empty", "Nothing here in this view")
        if not df.empty:
            msg = f"No estimated spend: {len(df):,} {rel['label']}s, all on unpriced models"
        return empty_fig(mode, msg)
    t = theme.TOKENS[mode]
    top = df.head(12).iloc[::-1].reset_index(drop=True)
    labels = [str(x)[:40] for x in top[rel["label"]]]
    # Bars sit at row positions, not at their label: two items can share a title
    # (content with the same name) and each still gets its own bar. The id of the
    # row, when it has one, shows on hover.
    id_col = next((c for c in ("content_guid", "id") if c in top), None)
    top = top.assign(_y=range(len(top)),
                     _id=top[id_col].astype(str) if id_col else "")
    # Content is coloured by how the user reached it (viewer / owner / ...).
    groups = [(v, top[top.via == v]) for v in top.via.dropna().unique()] \
        if "via" in top else [(None, top)]
    fig = go.Figure()
    for via, d in groups:
        # "owner, viewer": reached both ways, so it gets a colour of its own.
        slot = VIA_SLOT.get(via, 5) if via else 1
        fig.add_bar(
            x=d.est_spend_usd, y=d._y, orientation="h", width=0.6,
            name=via_label(via) if via else "", showlegend=bool(via), marker_color=t["series"][slot],
            text=["\u00a0" + fmt_usd(v) for v in d.est_spend_usd],
            textposition="outside", textfont={"color": t["ink"], "size": 13},
            customdata=d[["_id"]].assign(label=[labels[i] for i in d._y]),
            hovertemplate="%{customdata[1]}<br>$%{x:,.2f}" + (f" \u00b7 {via_label(via)}" if via else "")
                          + ("<br>%{customdata[0]}" if id_col else "")
                          + "<extra></extra>",
        )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(top))), ticktext=labels,
                     range=[-0.5, len(top) - 0.5])
    theme.usd_axis(fig, range=[0, top.est_spend_usd.max() * 1.3], title_text="est. USD")
    return apply(fig, mode, height=max(200, 26 * len(top) + 70),
                 showlegend=any(v for v, _ in groups), hover=True)


# ---- UI ---------------------------------------------------------------------------
def panel():
    """The Attribution tab, a prototype drill-down."""
    return ui.nav_panel(
        "Attribution",
        page_header("Attribution", "Compare users, groups, content and sessions, "
                                   "then pick one to see its details."),
        ui.div(
            ui.div(
                ui.input_radio_buttons("pf_kind", None, KINDS, selected="user", inline=True),
                class_="seg",
            ),
            # The pick is set by clicking a row of the table (`pf_row`); the
            # select only holds it, so bookmarks restore it.
            # input_select, not input_selectize: the latter restores a bookmarked
            # value as the *choices* and crashes the page on reload. The choices
            # are filled in by the server (`_choices`).
            ui.div(ui.input_select("pf_pick", None, {}, selectize=True), class_="d-none"),
            class_="page-switch",
        ),
        ui.card(
            ui.card_header(ui.output_text("pf_table_title", inline=True)),
            ui.card_body(ui.output_ui("pf_table")),
            class_="peer-card",
        ),
        # One box for the picked principal: its name, then its sub-tabs. Hidden
        # until something is picked: with no principal of the kind in view (no
        # sessions, say) it was rows of empty cards under the table's note.
        ui.panel_conditional("input.pf_pick", ui.div(ui.output_ui("pf_header"), ui.navset_underline(
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
                "Conversations",
                ui.card(
                    ui.card_header(ui.div(
                        ui.span("Higher cost and normal conversations"),
                        CONV_HELP["compare"], class_="hdr-flex-left")),
                    ui.card_body(ui.output_ui("pf_conv_compare")),
                ),
                ui.card(
                    ui.card_header(ui.div(ui.span("Spend by model"), CONV_HELP["model"],
                                          class_="hdr-flex-left")),
                    output_widget("pf_conv_models"),
                ),
                ui.card(
                    ui.card_header(ui.div(ui.span("Spend by harness"), CONV_HELP["tool"],
                                          class_="hdr-flex-left")),
                    output_widget("pf_conv_tools"),
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
        # peers_df ranks by spend, so the page opens on the biggest spender.
        choices = {str(k): str(lab) for k, lab in zip(df.k, df.label)}
        sel = cur if cur in choices else next(iter(choices), None)
        ui.update_selectize("pf_pick", choices=choices, selected=sel)

    # The peer table's sort: a column, and whether it runs high to low.
    sort: reactive.Value[tuple[str, bool]] = reactive.value(("", True))

    @reactive.effect
    @reactive.event(input.pf_sort)
    def _sort():
        col = input.pf_sort()
        cur, desc = sort.get()
        # Names read best A to Z; figures high to low. A second click flips it.
        sort.set((col, not desc if col == cur else col != "label"))

    @reactive.effect
    @reactive.event(input.pf_row)
    def _row():
        ui.update_selectize("pf_pick", selected=input.pf_row())

    @render.text
    def pf_table_title():
        return f"Compare {NOUN[input.pf_kind()]}s"

    @render.ui
    def pf_table():
        df, kind = peers(), input.pf_kind()
        if df.empty:
            why = {
                "session": " A session is known only when Workbench names it in the "
                           "Gateway-Metadata header (workbench-session-id), and no "
                           "request in view carries one.",
                "group": " Groups come from Connect; none of the users in view is in one.",
            }.get(kind, "")
            return empty_note(f"No {NOUN[kind]}s in this view.{why}")
        col, desc = sort.get()
        return peer_table(df, kind, input.pf_pick(), col, desc)

    @reactive.calc
    def ent() -> SimpleNamespace:
        kind, key, df = input.pf_kind(), input.pf_pick(), peers()
        req(key and key in set(df.k))
        f = flt()
        return SimpleNamespace(
            kind=kind, key=key, f=f, w=where(f), p=pred(kind, key), cost=est_cost(),
            label=df.label[df.k == key].iloc[0], noun=NOUN[kind],
        )

    @render.ui
    def pf_header():
        # The title of the details box, so the sub-tabs read as one principal's
        # details rather than as more of the table.
        e = ent()
        return ui.div(
            ui.span(f"{KINDS[e.kind]} details", class_="pf-eyebrow"),
            ui.span(e.label, class_="pf-title"),
            class_="pf-head",
        )

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
    def conv_frame() -> pd.DataFrame:
        """This principal's conversations, with the columns the views read."""
        df = convs().copy()
        return df.assign(main_model=df.main_model.fillna(df.model),
                         duration_min=df.minutes.clip(lower=0.5), peak_input=df.peak_context)

    def conv_bars(by: str) -> go.Figure:
        df = conv_frame()
        key, order = conv_tools.conv_keys(df, by)
        groups = conv_tools.conv_groups(df, key, order)
        return conv_tools.conv_bars_fig(groups, flt()["mode"], HIGH_PCT)

    @render_widget
    def pf_conv_models():
        return conv_bars("model")

    @render_widget
    def pf_conv_tools():
        return conv_bars("tool")

    @render.ui
    def pf_conv_compare():
        return conv_compare(conv_frame(), ent().noun, flt()["mode"])

    # Every record of the picked principal, uncapped: DuckDB writes the file.
    register_sql_export(
        "pf_records",
        lambda: records_sql(f"{ent().w} AND {ent().p}"),
        "profile_records.csv",
    )


# What each part of the Conversations tab shows, for its help.
CONV_HELP = {
    "compare": help_icon(
        "Higher cost and normal",
        ui.p("A conversation is higher cost when it costs more than the stated share of "
             "this principal's own conversations, so the line fits their usage and not a "
             "fixed dollar amount. Normal is the rest."),
        ui.p("Typical conversation: the median of each group. Peak context: the largest "
             "input any request sent, cached tokens included. Duration: first request to "
             "last. How often: the share of each group's conversations with the trait. "
             "The gap is higher cost against normal; the last column says what to do "
             "about a big one.", class_="mb-0"),
    ),
    "model": help_icon(
        "Spend by model",
        ui.p("Each conversation counts under the model it spent most on."),
        ui.p("Each bar splits spend into higher-cost and normal conversations, so the "
             "amber part compares across rows. The top bar is all conversations; each "
             "label gives that group's share of all spend.", class_="mb-0"),
    ),
    "tool": help_icon(
        "Spend by harness",
        ui.p("Each conversation counts under the harness it spent most through, read "
             "from the User-Agent header."),
        ui.p("Each bar splits spend into higher-cost and normal conversations, so the "
             "amber part compares across rows.", class_="mb-0"),
    ),
}


def conv_compare(cv: pd.DataFrame, noun: str, mode: str) -> ui.Tag:
    """The Compare view: higher-cost and normal conversations side by side, as
    a table of how many there are, what they cost and how they differ."""
    n = len(cv)
    if not n:
        return ui.div(f"No conversations for this {noun} in view.", class_="empty-note")
    rows = conv_tools.conv_compare_rows(cv)
    if not rows:
        return ui.div(
            f"{n:,} conversations: too few to split by cost. A conversation is marked "
            f"higher cost against this {noun}'s p{HIGH_PCT:g}, which needs "
            f"{MIN_BASELINE}+ conversations.", class_="empty-note")
    unpriced = n - int((cv.est > 0).sum())
    colours = (theme.TOKENS[mode]["series"][0], theme.STATUS["warning"])

    def head(text: str, colour: str | None = None) -> ui.Tag:
        dot = ui.span(class_="tile-dot", style=f"background:{colour}") if colour else None
        return ui.tags.th(dot, f" {text}" if dot else text)

    body: list = []
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            body.append(ui.tags.tr(ui.tags.th(section, colspan="5"), class_="cmp-section"))
        body.append(ui.tags.tr(
            ui.tags.td(r["label"], class_="cmp-label"),
            ui.tags.td(r["normal"]), ui.tags.td(r["high"]),
            ui.tags.td(r["gap"], class_="cmp-gap"),
            ui.tags.td(r["hint"], class_="cmp-hint"),
        ))
    return ui.div(
        ui.div(
            f"{n:,} conversations" + (f", {unpriced:,} with no estimated spend" if unpriced else "")
            + f". Higher cost is over {fmt_usd(float(cv.line.iloc[0]))}, this {noun}'s "
            f"p{HIGH_PCT:g}; normal is the rest.",
            class_="cmp-note"),
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                head(""), head("Normal", colours[0]), head("Higher cost", colours[1]),
                head("Gap"), head("What to do"))),
            ui.tags.tbody(*body), class_="cmp-table"),
        class_="cmp")


def peer_table(df: pd.DataFrame, kind: str, sel: str | None, sort: str, desc: bool) -> ui.Tag:
    """Every principal of a kind side by side; clicking a row picks it.

    `sort` is "label", "spend" or "top_model", or "" for outliers first then spend.
    """
    if sort:
        df = df.sort_values(sort, ascending=not desc, na_position="last")
    else:
        df = df.sort_values(["outlier", "spend"], ascending=False)
    top = float(df.spend.max() or 1)

    def head(col: str, text: str) -> ui.Tag:
        arrow = (" \u25bc" if desc else " \u25b2") if col == sort else ""
        return ui.tags.th(
            text + arrow, class_="sortable" + (" sorted" if col == sort else ""),
            onclick=f"Shiny.setInputValue('pf_sort', {col!r}, {{priority: 'event'}})")

    rows = [
        ui.tags.tr(
            ui.tags.td(
                ui.span(str(label), class_="pt-name"),
                ui.span("outlier", class_="versus versus-outlier") if out else None),
            ui.tags.td(
                ui.span(ui.span(class_="pt-bar", style=f"width:{100 * spend / top:.1f}%"),
                        class_="pt-track"),
                ui.span(fmt_usd(spend)), class_="pt-spend"),
            ui.tags.td(str(model) if pd.notna(model) else DASH),
            class_="pt-row" + (" pt-sel" if str(k) == sel else ""),
            onclick=f"Shiny.setInputValue('pf_row', {str(k)!r}, {{priority: 'event'}})",
        )
        for k, label, spend, model, out in zip(
            df.k, df.label, df.spend.astype(float), df.top_model, df.outlier)
    ]
    n = int(df.outlier.sum())
    return ui.div(
        ui.div(
            f"{len(df):,} {NOUN[kind]}s in view"
            + (f", {n:,} outliers (top {100 - OUTLIER_PCT}% of {NOUN[kind]}s by spend)"
               if n else "")
            + ". Click a column to sort, a row to see its details below.",
            class_="cmp-note"),
        ui.div(ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                head("label", KINDS[kind]), head("spend", "Est. spend"),
                head("top_model", "Top model"))),
            ui.tags.tbody(*rows), class_="cmp-table peer-table"), class_="pt-scroll"),
    )

