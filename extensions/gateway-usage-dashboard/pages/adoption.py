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
import queries
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
    flt, cat = ctx.flt, ctx.cat

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
        return queries.active_principals(flt(), KIND_COL[kind()], cat()["start"])

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
            return empty_fig(f.mode, f"No active {noun} in this view")
        series = theme.TOKENS[f.mode]["series"]
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
        return apply(fig, f.mode, height=380, time_axis=True, showlegend=True,
                     y_title=f"active {noun}", hover=True)

    def active_by(dim: str, grain: str, nothing: str):
        """Distinct active users (or content items) per harness or model, per
        day, week or month. The six with the most get a line; the rest are "Other"."""
        f = flt()
        col, noun = KIND_COL[kind()], KIND_NOUN[kind()]
        top = queries.top_by_active(f, col, dim)
        if not top:
            return empty_fig(f.mode, nothing)
        df = queries.active_by_bucket(f, col, dim, grain, top)
        fig = go.Figure()
        for k in top + (["Other"] if "Other" in set(df.k) else []):
            d = df[df.k == k]
            c = theme.TOKENS[f.mode]["muted"] if k == "Other" else color(dim, k, f.mode)
            fig.add_scatter(
                x=d.b, y=d.users, name=k, mode="lines+markers",
                line={"width": 2.5, "color": c},
                marker={"size": 6},
                customdata=d.requests,
                hovertemplate=f"<b>%{{y:,}}</b> {noun} \u00b7 %{{customdata:,}} requests \u00b7 {k}<extra></extra>",
            )
        fig.update_yaxes(rangemode="tozero")
        return apply(fig, f.mode, height=340, time_axis=True, showlegend=True,
                     y_title=f"active {noun} / {grain}", hover=True)

    @render_widget
    def tool_time():
        grain = input.tool_grain()
        return active_by("tool", grain if grain in ("day", "week", "month") else "week",
                         "No harness activity in this view")

    @render_widget
    def model_time():
        grain = input.model_grain()
        return active_by("model", grain if grain in ("day", "week", "month") else "week",
                         "No model usage in this view")

    ctx.register_export(
        "adoption",
        lambda: adoption_df().rename(
            columns={k: v.replace(" ", "_") for k, v in active_names(kind()).items()}
        ),
        lambda: f"adoption_active_{KIND_NOUN[kind()].replace(' ', '_')}.csv",
    )
