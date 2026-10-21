"""Conversations page: conversation spend, cost drivers, and one conversation turn by turn.

The grouping and cost-driver definitions live in the top-level `conversations`
module; this module owns the page's UI, queries and outputs. Requests to
unpriced models are left out through `ctx.priced_only`, which app.py shares with
the topbar's unpriced-model warning.
"""

import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget

import theme
import usage
from conversations import (
    AGENT_GAP_SECONDS, AGENT_MIN_CALLS, AGENT_TURN_CAP, CACHE_TTL_SECONDS,
    COMPACT_AT, CONV_BY, CONV_HIGH_PCT, LEVERS, LEVER_NAME, MIN_BASELINE,
    QUICK_MAX_CALLS, SINGLE, conv_groups, conv_keys, conv_spend_fig, higher_cost,
)
from filters import sql_str
from pages.context import PageContext
from pricing import model_rates, rate_sql
from theme import apply, card_num, empty_fig, fmt_num, fmt_usd, render_widget
from ui_shell import empty_note, help_icon, page_header, table_header


def panel():
    return ui.nav_panel(
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
    )


def server(input, ctx: PageContext):
    flt, where, cat, formatted = ctx.flt, ctx.where, ctx.cat, ctx.formatted
    fitted_grid, priced_only = ctx.fitted_grid, ctx.priced_only

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
        df[["high", "line"]] = higher_cost(df.est, pct=CONV_HIGH_PCT)
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
                    if line is not None and len(df) >= MIN_BASELINE
                    else f"needs {MIN_BASELINE}+ conversations"
                ),
                status="warning" if flagged else "good",
                help=help_icon(
                    "Higher-cost conversations",
                    ui.p(
                        "A conversation is flagged when it costs more than "
                        f"{CONV_HIGH_PCT:g}% of all conversations in view, "
                        "whoever ran them. The line moves with the date range "
                        f"and filters. Below {MIN_BASELINE} "
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
        r = hot.loc[(hot.s / hot.c).idxmax()]  # pyright: ignore[reportCallIssue, reportArgumentType]
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
            "context": f"Conversations peaking at {lab} tokens of context",
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

    ctx.register_export("conversations", sessions_df, "conversations.csv")
