"""Adoption page: active users, harness activity, and usage patterns.

Preserves the public input/output IDs and registers its CSV export.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget

import theme
import usage
from pages.context import PageContext
from theme import apply, card_num, color, empty_fig, render_widget
from ui_shell import empty_note, help_icon, page_header, plural, table_header

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def panel():
    return ui.nav_panel(
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
                    ui.span("Harnesses over time"),
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
    )


def server(input, ctx: PageContext):
    flt, where, cat = ctx.flt, ctx.where, ctx.cat
    est_cost = ctx.est_cost

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
        """Distinct active users per harness, per day, week or month."""
        f, grain = flt(), input.tool_grain() or "week"
        df = usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b, tool AS k,
                                COUNT(DISTINCT username) AS users, COUNT(*) AS requests
                         FROM gateway_events
                         WHERE {where(f)} AND tool IS NOT NULL AND username IS NOT NULL
                         GROUP BY 1, 2 ORDER BY 1""")
        if df.empty:
            return empty_fig(f["mode"], "No harness activity in this view")
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
        for r in usage.rows(df):
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

    ctx.register_export("adoption", adoption_df, "adoption_active_users.csv")
