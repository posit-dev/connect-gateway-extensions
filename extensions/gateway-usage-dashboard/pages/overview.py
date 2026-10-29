"""Overview page: spend, caching and errors, grouped by model, provider, origin or harness."""

import pandas as pd
import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget

import queries
import theme
from filters import NO_USER_AGENT, UNIDENTIFIED
from pages.context import PageContext
from theme import apply, card_num, card_usd, color, empty_fig, render_widget
from ui_shell import help_icon, page_header

# What the page can be grouped by: the column, and its colour kind. "none" is
# no grouping at all, one series of the whole spend.
GROUP_BY = {"none": "None", "model": "Model", "provider": "Provider", "source": "Origin",
            "tool": "Harness"}
COLOR_KIND = {"model": "model", "provider": "provider", "source": "origin", "tool": "tool"}
# The time buckets every chart on the page is drawn in.
GRAINS = {"day": "Daily", "week": "Weekly", "month": "Monthly"}


def panel():
    return ui.nav_panel(
        "Overview",
        page_header("Consumption Overview"),
        ui.div(
            ui.span("Group by", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons(
                    "group_by", None, GROUP_BY, selected="none", inline=True,
                ),
                class_="seg",
            ),
            help_icon(
                ui.p(
                    ui.strong("Origin"), " is where the request came from: "
                    "Connect content, a Workbench session, or a client running "
                    "locally. ", ui.strong(UNIDENTIFIED), " is a request made "
                    "outside content by a client that did not name its host "
                    "(Claude Code, Copilot, a script, or no User-Agent at all), so "
                    "Connect cannot tell whether it ran in Workbench or on a laptop."
                ),
                ui.p(
                    ui.strong("Harness"), " is the client software that made the "
                    "call, read from the User-Agent header the caller sent. ",
                    ui.strong(NO_USER_AGENT), " is a request that sent none.",
                    class_="mb-0",
                ),
            ),
            class_="page-switch",
        ),
        ui.div(
            ui.span("Interval", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons("grain", None, GRAINS, selected="day", inline=True),
                class_="seg",
            ),
            class_="page-switch",
        ),
        ui.output_ui("kpis"),
        ui.card(
            ui.card_header("Spend"),
            output_widget("spend_time"),
            class_="question-block",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("Estimated cache savings"),
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
                ui.card_header("Cache read share"),
                output_widget("cache_rate_time"),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Error rate"),
            output_widget("errors_time"),
        ),
    )


def server(input, ctx: PageContext):
    flt = ctx.flt

    def group_by() -> str:
        by = input.group_by()
        return by if by in GROUP_BY else "none"

    def grain() -> str:
        g = input.grain()
        return g if g in GRAINS else "day"

    def group_color(by: str, key: str, mode: str) -> str:
        if by == "none":
            return theme.TOKENS[mode]["series"][0]
        if key == "Other":
            return theme.TOKENS[mode]["muted"]
        return color(COLOR_KIND[by], key, mode)

    @render.ui
    def kpis():
        f = flt()
        r = queries.overview_totals(f)
        err = 0.0 if pd.isna(r.err) else float(r.err)
        users, contents, sessions = int(r.users), int(r.contents), int(r.sessions)

        def avg(n: int, spend: float, noun: str) -> str:
            if not n:
                return f"no {noun}s in range"
            return f"{card_usd(spend / n)} avg / {noun}"

        # No session ids anywhere in the loaded data means Workbench is not
        # attaching them (or none of the traffic is from Workbench), not that
        # nobody used a session. Say that rather than show a bare 0.
        reported = bool(ctx.cat()["session_labels"])
        sessions_help = help_icon(
            "What is a session?",
            ui.p(
                "A Workbench session: one running RStudio Pro, Positron or VS Code "
                "session. Workbench names it in the Gateway-Metadata header "
                "(workbench-session-id) of each request it sends."
            ),
            ui.p(
                "Requests from Connect content, from a laptop, or from a Workbench "
                "that does not send the header carry no session, so they do not "
                "count here.",
                class_="mb-0",
            ),
        )
        if reported:
            sessions_tile = theme.stat_tile(
                "Sessions", card_num(sessions), avg(sessions, r.session_est, "session"),
                help=sessions_help,
            )
        else:
            sessions_tile = theme.stat_tile(
                "Sessions", "\u2013", "not reported by Workbench", help=sessions_help,
            )

        return ui.layout_columns(
            theme.stat_tile(
                "Estimated spend", card_usd(r.est), f"{card_num(int(r.requests))} requests"
            ),
            theme.stat_tile(
                "Users",
                card_num(users),
                avg(users, r.user_est, "user"),
            ),
            theme.stat_tile(
                "Content",
                card_num(contents),
                avg(contents, r.content_est, "content item"),
            ),
            sessions_tile,
            theme.stat_tile(
                "Cache savings",
                card_usd(r.saved),
                f"{100 * r.saved / max(1e-9, r.saved + r.est):.0f}% off the uncached price",
            ),
            theme.stat_tile(
                "Error rate",
                f"{err:.2f}%",
                "of requests failed",
                status="critical" if err > 2 else "good",
                help=help_icon(
                    "Error rate",
                    ui.p(
                        "The share of requests that failed instead of getting an "
                        "answer. It turns red when more than 2% fail.",
                        class_="mb-0",
                    ),
                ),
            ),
            col_widths=[2, 2, 2, 2, 2, 2],
            class_="kpi-row",
        )

    @render_widget
    def spend_time():
        """Spend per day, week or month, stacked by the chosen group."""
        f, g, by = flt(), grain(), group_by()
        df = queries.spend_by_period(f, g, by, "Estimated spend")
        if df.empty or not df.v.sum():
            return empty_fig(f.mode, "No estimated spend recorded in this view")
        t = theme.TOKENS[f.mode]
        # The six biggest get a stack of their own; the tail is "Other".
        order = df.groupby("k").v.sum().sort_values(ascending=False).index.tolist()
        shown = order[:6]
        df["k"] = df.k.where(df.k.isin(shown), "Other")
        df = df.groupby(["b", "k"], as_index=False).v.sum()
        fig = go.Figure()
        for key in shown + (["Other"] if "Other" in set(df.k) - set(shown) else []):
            d = df[df.k == key]
            fig.add_bar(
                x=d.b,
                y=d.v,
                name=key,
                marker={"color": group_color(by, key, f.mode) if key in shown
                        else t["muted"]},
                hovertemplate=f"<b>$%{{y:,.2f}}</b> {key}<extra></extra>",
            )
        fig.update_layout(barmode="stack", bargap=0.25)
        apply(
            fig, f.mode, height=300, time_axis=True, showlegend=by != "none",
            y_title=f"est. USD / {g}", hover=True,
        )
        return theme.usd_axis(fig, "y")

    def group_lines(f, by, df, fmt, y_title, yaxis):
        """One line per group, `df` holding day, k and v."""
        fig = go.Figure()
        keys = df.groupby("k").v.size().sort_values(ascending=False).index
        for key in sorted(keys, key=lambda x: x == "Other"):
            d = df[df.k == key]
            c = group_color(by, key, f.mode)
            fig.add_scatter(
                x=d.day, y=d.v, name=key, mode="lines+markers",
                line={"width": 2.5, "color": c}, marker={"size": 5, "color": c},
                hovertemplate=f"<b>{fmt}</b> {key}<extra></extra>",
            )
        fig.update_yaxes(**yaxis)
        return apply(fig, f.mode, height=300, time_axis=True, showlegend=by != "none",
                     y_title=y_title, hover=True)

    # ---- caching --------------------------------------------------------------
    @render_widget
    def cache_savings_time():
        f, g, by = flt(), grain(), group_by()
        df = queries.cache_savings_by_period(f, g, by, "Saved")
        df["v"] = df.v.fillna(0).clip(lower=0)
        if df.empty or not df.v.sum():
            return empty_fig(f.mode, "No cache savings recorded in this view")
        fig = go.Figure()
        # largest saver first, so the stack and legend read in the same order
        for key in df.groupby("k").v.sum().sort_values(ascending=False).index:
            d = df[df.k == key]
            fig.add_bar(
                x=d.day, y=d.v, name=key,
                marker={"color": group_color(by, key, f.mode)},
                hovertemplate=f"<b>$%{{y:,.2f}}</b> saved \u00b7 {key}<extra></extra>",
            )
        fig.update_layout(barmode="stack", bargap=0.25)
        apply(fig, f.mode, height=300, time_axis=True, showlegend=by != "none",
              y_title=f"est. USD saved / {g}", hover=True)
        return theme.usd_axis(fig, "y")

    @render_widget
    def cache_rate_time():
        f, g, by = flt(), grain(), group_by()
        df = queries.cache_read_share_by_period(f, g, by, "Cache read share").dropna()
        if df.empty:
            return empty_fig(f.mode, "No input tokens recorded in this view")
        return group_lines(f, by, df, "%{y:.1f}%", "input read from cache",
                           {"range": [0, 100], "ticksuffix": "%"})

    # ---- errors ---------------------------------------------------------------
    @render_widget
    def errors_time():
        f, g, by = flt(), grain(), group_by()
        df = queries.error_rate_by_period(f, g, by, "Error rate")
        if df.empty:
            return empty_fig(f.mode, "No requests in this view")
        return group_lines(f, by, df, "%{y:.2f}%", "requests failed (status 400+)",
                           {"rangemode": "tozero", "ticksuffix": "%"})
