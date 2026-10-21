"""Models page: model adoption, spend, caching, latency, and reliability."""

import pandas as pd
import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget

import theme
import usage
from filters import sql_str
from pages.context import PageContext
from table_format import pretty_columns
from theme import apply, card_num, card_usd, color, empty_fig, render_widget
from ui_shell import empty_note, help_icon, page_header, table_header


def panel():
    return ui.nav_panel(
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
                        ui.span("Estimated cache savings over time by harness"),
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
    )


def server(input, ctx: PageContext):
    flt, where = ctx.flt, ctx.where
    est_cost, est_cost_uncached = ctx.est_cost, ctx.est_cost_uncached
    total_bars, measure_of, formatted = ctx.total_bars, ctx.measure_of, ctx.formatted

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

    @render.data_frame
    def models_table():
        df = models_df()
        return render.DataGrid(
            pretty_columns(formatted(df)),
            height=None if len(df) <= 12 else "420px",
        )

    ctx.register_export("models", models_df, "models.csv")
