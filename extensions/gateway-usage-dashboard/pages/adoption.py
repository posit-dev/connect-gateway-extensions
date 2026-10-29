"""Adoption page: active users or content, and harness and model activity.

Preserves the public input/output IDs and registers its CSV export. A Count
toggle switches every chart between people (`username`) and Connect content
items (`content_guid`), the two kinds of principal the gateway charges.
"""

import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget

import theme
import usage
from filters import dim_sql, sql_str
from pages.context import PageContext
from theme import apply, card_num, color, empty_fig, render_widget
from ui_shell import empty_note, help_icon, page_header, plural, table_header

# What the Count toggle counts: the column a principal of that kind is keyed
# by, the plural noun, and the line under the toggle saying what is counted.
KINDS = {"user": "Users", "content": "Content"}
KIND_COL = {"user": "username", "content": "content_guid"}
KIND_NOUN = {"user": "users", "content": "content items"}
KIND_NOTE = {
    "user": "People who made requests in Workbench or Connect. A request from "
            "Connect content counts for the user it was charged to, the viewer "
            "or the content's owner.",
    "content": "Connect content items that made requests. Workbench traffic has "
               "no content item and is left out.",
}


def active_names(kind: str) -> dict[str, str]:
    """The rolling-window columns, named in full: readers did not know the acronyms."""
    noun = KIND_NOUN[kind]
    return {"dau": f"daily active {noun}", "wau": f"weekly active {noun}",
            "mau": f"monthly active {noun}"}


# The user names, kept for callers that do not pass a kind.
ACTIVE_NAMES = active_names("user")


def over_time_card(title: str, grain_id: str, output_id: str) -> ui.Tag:
    return ui.card(
        ui.card_header(
            ui.div(
                ui.span(title),
                ui.div(
                    ui.input_radio_buttons(
                        grain_id,
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
        output_widget(output_id),
    )


def panel():
    return ui.nav_panel(
        "Adoption",
        page_header(
            "Adoption",
            help=help_icon(
                "What counts as active?",
                ui.p(
                    "A user or content item is active if it made at least one "
                    "request through the gateway. Each point on a line counts "
                    "who was active in the days leading up to it:"
                ),
                ui.tags.ul(
                    ui.tags.li("Daily: active on that day."),
                    ui.tags.li("Weekly: active at least once in the past 7 days."),
                    ui.tags.li("Monthly: active at least once in the past 30 days."),
                ),
                ui.p(
                    "Near the start of the data, the weekly and monthly counts "
                    "read low, because there are fewer earlier days to look back on.",
                    class_="mb-0",
                ),
            ),
        ),
        ui.div(
            ui.span("Count", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons("adoption_kind", None, KINDS, selected="user",
                                       inline=True),
                class_="seg",
            ),
            ui.output_text("adoption_kind_note", inline=True).add_class("page-switch-note"),
            class_="page-switch",
        ),
        ui.output_ui("adoption_kpis"),
        ui.card(
            table_header(ui.output_text("adoption_lines_title", inline=True), "dl_adoption"),
            output_widget("adoption_lines"),
        ),
        ui.layout_columns(
            over_time_card("Harnesses over time", "tool_grain", "tool_time"),
            over_time_card("Models over time", "model_grain", "model_time"),
            col_widths=[6, 6],
        ),
    )


def server(input, ctx: PageContext):
    flt, where, cat = ctx.flt, ctx.where, ctx.cat

    def kind() -> str:
        k = input.adoption_kind()
        return k if k in KINDS else "user"

    @render.text
    def adoption_kind_note():
        return KIND_NOTE[kind()]

    @render.text
    def adoption_lines_title():
        return f"Active {KIND_NOUN[kind()]} over time"

    @reactive.calc
    def adoption_df() -> pd.DataFrame:
        """Rolling daily, weekly and monthly active principals, one row per day in range.

        Windows look back past the start of the range (so a 30D chart does not
        start with an empty MAU), but not past the loaded data. The date range
        is the only filter lifted; everything else still applies.
        """
        f, col = flt(), KIND_COL[kind()]
        wide = {**f, "start": cat()["start"]}
        return usage.q(f"""
            WITH days AS (
                SELECT CAST(d AS DATE) AS d
                FROM generate_series(DATE '{f["start"]}', DATE '{f["end"]}',
                                     INTERVAL 1 DAY) t(d)),
            ev AS (SELECT DISTINCT day, {col} AS who FROM gateway_events
                   WHERE {where(wide)} AND {col} IS NOT NULL)
            SELECT days.d AS day,
                   COUNT(DISTINCT CASE WHEN ev.day = days.d THEN ev.who END) AS dau,
                   COUNT(DISTINCT CASE WHEN ev.day > days.d - INTERVAL 7 DAY
                                       THEN ev.who END) AS wau,
                   COUNT(DISTINCT ev.who) AS mau
            FROM days LEFT JOIN ev
              ON ev.day <= days.d AND ev.day > days.d - INTERVAL 30 DAY
            GROUP BY 1 ORDER BY 1""")

    @render.ui
    def adoption_kpis():
        df, noun = adoption_df(), KIND_NOUN[kind()]
        who = "people" if kind() == "user" else "content items"
        if df.empty:
            return empty_note(f"No {noun} in this range.")
        last = df.iloc[-1]
        avg_dau = df.dau.mean()
        return ui.layout_columns(
            theme.stat_tile(
                f"Average daily active {noun}",
                f"{avg_dau:,.1f}",
                f"daily average over {len(df)} {plural(len(df), 'day')}",
                help=help_icon(
                    f"Average daily active {noun}",
                    ui.p(
                        f"The average number of active {noun} per day. On a "
                        f"typical day, this many {who} used the gateway.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                f"Weekly active {noun}",
                card_num(last.wau),
                f"7 days to {last.day:%b %d}",
                help=help_icon(
                    f"Weekly active {noun}",
                    ui.p(
                        f"How many different {who} used the gateway in the last 7 days.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                f"Monthly active {noun}",
                card_num(last.mau),
                f"30 days to {last.day:%b %d}",
                help=help_icon(
                    f"Monthly active {noun}",
                    ui.p(
                        f"How many different {who} used the gateway in the last "
                        "30 days.",
                        class_="mb-0",
                    ),
                ),
            ),
            theme.stat_tile(
                "Stickiness",
                f"{100 * avg_dau / last.mau:,.0f}%" if last.mau else "\u2013",
                f"daily \u00f7 monthly active {noun}",
                help=help_icon(
                    "Stickiness",
                    ui.p(
                        f"Average daily active {noun} divided by monthly active "
                        f"{noun}: of the {who} that used the gateway this month, "
                        "the share that show up on a typical day.",
                        class_="mb-0",
                    ),
                ),
            ),
            col_widths=[3, 3, 3, 3],
            class_="kpi-row",
        )

    @render_widget
    def adoption_lines():
        f, df, noun = flt(), adoption_df(), KIND_NOUN[kind()]
        if df.empty or not df.mau.max():
            return empty_fig(f["mode"], f"No active {noun} in this view")
        series = theme.TOKENS[f["mode"]]["series"]
        names = active_names(kind())
        fig = go.Figure()
        for col, c in [("dau", series[0]), ("wau", series[2]), ("mau", series[1])]:
            name = names[col].capitalize()
            fig.add_scatter(
                x=df.day, y=df[col], name=name, mode="lines",
                line={"width": 2.5, "color": c},
                hovertemplate=f"<b>%{{y:,}}</b> {name}<extra></extra>",
            )
        fig.update_yaxes(rangemode="tozero")
        return apply(fig, f["mode"], height=380, time_axis=True, showlegend=True,
                     y_title=f"active {noun}", hover=True)

    def active_by(dim: str, grain: str, nothing: str):
        """Distinct active users (or content items) per harness or model, per
        day, week or month. The six with the most get a line; the rest are "Other"."""
        f = flt()
        col, noun = KIND_COL[kind()], KIND_NOUN[kind()]
        # Requests with no User-Agent are a line of their own: on a server whose
        # callers send none, leaving them out made the chart look empty.
        k = dim_sql(dim)
        base = f"FROM gateway_events WHERE {where(f)} AND {col} IS NOT NULL AND {k} <> ''"
        top = usage.q(f"""SELECT {k} AS k {base} GROUP BY 1
                          ORDER BY COUNT(DISTINCT {col}) DESC LIMIT 6""").k.tolist()
        if not top:
            return empty_fig(f["mode"], nothing)
        keep = ",".join(sql_str(x) for x in top)
        df = usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b,
                                CASE WHEN {k} IN ({keep}) THEN {k} ELSE 'Other' END AS k,
                                COUNT(DISTINCT {col}) AS users, COUNT(*) AS requests
                         {base} GROUP BY 1, 2 ORDER BY 1""")
        fig = go.Figure()
        for k in top + (["Other"] if "Other" in set(df.k) else []):
            d = df[df.k == k]
            c = theme.TOKENS[f["mode"]]["muted"] if k == "Other" else color(dim, k, f["mode"])
            fig.add_scatter(
                x=d.b, y=d.users, name=k, mode="lines+markers",
                line={"width": 2.5, "color": c},
                marker={"size": 6},
                customdata=d.requests,
                hovertemplate=f"<b>%{{y:,}}</b> {noun} \u00b7 %{{customdata:,}} requests \u00b7 {k}<extra></extra>",
            )
        fig.update_yaxes(rangemode="tozero")
        return apply(fig, f["mode"], height=340, time_axis=True, showlegend=True,
                     y_title=f"active {noun} / {grain}", hover=True)

    @render_widget
    def tool_time():
        return active_by("tool", input.tool_grain() or "week", "No harness activity in this view")

    @render_widget
    def model_time():
        return active_by("model", input.model_grain() or "week", "No model usage in this view")

    ctx.register_export(
        "adoption",
        lambda: adoption_df().rename(
            columns={k: v.replace(" ", "_") for k, v in active_names(kind()).items()}
        ),
        lambda: f"adoption_active_{KIND_NOUN[kind()].replace(' ', '_')}.csv",
    )
