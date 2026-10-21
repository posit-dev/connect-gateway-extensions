"""Overview page: spend KPIs and spend by origin/harness, provider, and model."""

import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget

import theme
import usage
from pages.context import PageContext
from theme import apply, card_num, card_usd, color, empty_fig, render_widget
from ui_shell import chart_row, help_icon, page_header


def panel():
    return ui.nav_panel(
        "Overview",
        page_header("Consumption Overview"),
        ui.output_ui("kpis"),
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
        ui.div(
            ui.div(
                ui.h6("Estimated spend by", class_="row-question"),
                ui.div(
                    ui.input_radio_buttons(
                        "origin_tool", None, {"origin": "Origin", "tool": "Harness"},
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
                        ui.strong("Harness"), " is the client software that made the call. This is derived "
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
    )


def server(input, ctx: PageContext):
    flt, where, est_cost = ctx.flt, ctx.where, ctx.est_cost
    total_bars, measure_of, chart_labels = ctx.total_bars, ctx.measure_of, ctx.chart_labels

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
