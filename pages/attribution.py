"""Attribution page: users, content and sessions ranked by spend, with outliers.

Picking an outlier in the table opens it on the Principals tab, through the
`open_profile` callback the Principals server returns.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

import theme
import usage
from pages.context import PageContext
from filters import TOKEN_COUNT
from table_format import GUID_LABEL_CHARS, pretty_columns
from theme import apply, card_num, empty_fig, fmt_usd
from ui_shell import empty_note, help_icon, page_header, table_header


def panel():
    return ui.nav_panel(
        "Attribution",
        page_header("Attribution"),
        ui.div(
            ui.span("Show", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons(
                    "attr_kind",
                    None,
                    {"all": "All", "content": "Content", "user": "Users", "session": "Sessions"},
                    selected="all",
                    inline=True,
                ),
                class_="seg",
            ),
            ui.span("Order by", class_="strip-label"),
            ui.div(
                ui.input_radio_buttons(
                    "attr_order",
                    None,
                    {"spend": "Spend", "type": "Type"},
                    selected="spend",
                    inline=True,
                ),
                class_="seg",
            ),
            class_="page-switch",
        ),
        ui.output_ui("attr_kpis"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Top 20 by estimated spend"),
                output_widget("attr_bars"),
            ),
            ui.card(
                ui.card_header("Top 20 by estimated cost per request"),
                output_widget("attr_rate_bars"),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Outliers: requests against estimated spend"),
            output_widget("attr_scatter"),
        ),
        ui.card(
            table_header("Outliers", "dl_attr"),
            ui.card_body(
                ui.output_data_frame("attr_table"),
            ),
        ),
    )


def server(input, ctx: PageContext, open_profile):
    flt, where, est_cost, formatted = ctx.flt, ctx.where, ctx.est_cost, ctx.formatted

    # Users and content share one ranking. An outlier is judged against the rest of
    # its own kind, on a log scale (spend is long-tailed, so a straight-line fence
    # would flag half the top). A row is flagged if its total spend, or its spend
    # per request, is above the third quartile by more than 1.5 times the
    # interquartile range, and at least twice the median. Too few rows of a kind
    # make a quartile meaningless, so under this many nothing of that kind is
    # flagged.
    OUTLIER_MIN_N = 8
    KIND_LABEL = {"user": "user", "content": "content", "session": "session"}
    # What can be ranked: the column that identifies it, and what to call it. A
    # Workbench session is identified by the session id its request metadata
    # carries (caller-supplied, unverified), so it exists only for Workbench calls.
    PRINCIPAL_KINDS = [
        ("username", "user"),
        ("content_guid", "content"),
        ("workbench_session_id", "session"),
    ]
    # Sessions also say who ran them and for which team, which is what turns an
    # opaque id into something a reader can act on.
    SESSION_EXTRA = """, MAX(workbench_session_name) AS session_name,
                        MAX(username) AS owner, MAX(team) AS team"""

    def principal_parts(want: str, build) -> list[pd.DataFrame]:
        return [build(dim, kind) for dim, kind in PRINCIPAL_KINDS if want in ("all", kind)]

    def principal_label(row, want: str) -> str:
        """What an axis shows for one row."""
        k = str(row.k)
        if row.kind == "user":
            return k
        if row.kind == "session":
            name = row.session_name if isinstance(row.session_name, str) else k[:GUID_LABEL_CHARS]
            return f"{name} \u00b7 {row.owner}" if isinstance(row.owner, str) else name
        # A title says what the content is; without one a guid fragment stands in.
        # The kind is shown by bar colour/legend and the table's kind column, not
        # by a prefix on every label.
        return usage.content_name(k) or k[:GUID_LABEL_CHARS]

    def uniquify(df: pd.DataFrame) -> pd.DataFrame:
        """Make axis labels unique.

        Plotly draws two bars with the same category label as one row, stacked on
        top of each other. Sessions are the usual offender: "RStudio Pro Session 7"
        for one user on two different days is two sessions with one name.
        """
        dup = df.label.duplicated(keep=False)
        df.loc[dup, "label"] = [
            f"{lab} ({str(k)[:6]})" for lab, k in zip(df.label[dup], df.k[dup])
        ]
        return df

    def principal_detail(row) -> str:
        if row.kind == "user":
            return ""
        if row.kind == "content":
            return str(row.k)  # the full guid, which the axis label replaces
        team = row.team if isinstance(row.team, str) else ""
        return f"{row.k} \u00b7 {team}" if team else str(row.k)

    def attribution_part(dim: str, kind: str) -> pd.DataFrame:
        f = flt()
        extra = SESSION_EXTRA if kind == "session" else ""
        df = usage.q(f"""SELECT {dim} AS k, SUM({est_cost()}) AS spend,
                                COUNT(*) AS requests, SUM({TOKEN_COUNT}) AS tokens{extra}
                         FROM gateway_events
                         WHERE {where(f)} AND {dim} IS NOT NULL AND {dim} <> ''
                         GROUP BY 1 HAVING SUM({est_cost()}) > 0""")
        df.insert(0, "kind", kind)
        return df

    @reactive.calc
    def attr_df() -> pd.DataFrame:
        """The selected list: users, content, or both together, flagged and ranked."""
        want = input.attr_kind() or "all"
        parts = [d for d in principal_parts(want, attribution_part) if not d.empty]
        if not parts:
            return pd.DataFrame(
                columns=["kind", "k", "spend", "requests", "tokens", "label", "detail",
                         "per_request", "x_median", "why"]
            )
        df = pd.concat(parts, ignore_index=True)
        for col in ("session_name", "owner", "team"):
            if col not in df:
                df[col] = None
        df["label"] = [principal_label(r, want) for r in df.itertuples()]
        df = uniquify(df)
        df["detail"] = [principal_detail(r) for r in df.itertuples()]
        df = df.sort_values("spend", ascending=False, ignore_index=True)
        if input.attr_order() == "type":
            order = {k: i for i, (_, k) in enumerate(PRINCIPAL_KINDS)}
            df = df.sort_values(
                ["kind", "spend"], key=lambda s: s.map(order) if s.name == "kind" else -s,
                ignore_index=True,
            )
        df["per_request"] = df.spend / df.requests
        df["x_median"] = 0.0
        df["why"] = ""
        # Judged within each kind, even in the mixed view: a content item and a
        # person are different sizes of thing, and set against each other every
        # content item would look like a heavy user.
        for kind in df.kind.unique():
            rows = df.index[df.kind == kind]
            part = df.loc[rows]
            df.loc[rows, "x_median"] = part.spend / part.spend.median()
            if len(part) < OUTLIER_MIN_N:
                continue
            why = {i: [] for i in rows}
            for col, label in [
                ("spend", "high estimated spend"),
                ("per_request", "high estimated cost per request"),
            ]:
                lg = np.log10(part[col])
                q1, q3 = lg.quantile(0.25), lg.quantile(0.75)
                material = part[col] >= 2 * part[col].median()
                for i in rows[((lg > q3 + 1.5 * (q3 - q1)) & material).to_numpy()]:
                    why[i].append(label)
            df.loc[rows, "why"] = [" + ".join(why[i]) for i in rows]
        return df

    @render.ui
    def attr_kpis():
        df = attr_df()
        if df.empty:
            return empty_note("No users or content with estimated spend in this view.")
        n = df.kind.value_counts()
        out = df[df.why != ""]
        flagged = out.kind.value_counts()

        def part(kind: str, label: str, noun: str) -> ui.Tag:
            have, bad = int(n.get(kind, 0)), int(flagged.get(kind, 0))
            return theme.stat_tile(
                label,
                card_num(bad),
                f"of {card_num(have)} {noun} with estimated spend",
                status="critical" if bad else "good",
            )

        return ui.layout_columns(
            theme.stat_tile(
                "Outliers",
                card_num(len(out)),
                f"of {card_num(len(df))} with estimated spend",
                status="critical" if len(out) else "good",
                help=help_icon(
                    "How outliers are flagged",
                    ui.p(
                        "Flagged when total estimated spend, or spend per request, is "
                        "unusually high compared with the others in the list.",
                        class_="mb-0",
                    ),
                ),
            ),
            part("user", "Outlier users", "users"),
            part("content", "Outlier content", "content items"),
            part("session", "Outlier sessions", "sessions"),
            col_widths=[3, 3, 3, 3],
            class_="kpi-row",
        )

    def _kind_color(t: dict, kind: str) -> str:
        return t["series"][{"user": 0, "content": 1, "session": 2}[kind]]

    def rank_bars(f, df, col: str, flag: str, fmt, axis: str):
        """Top 20 by one measure. Red marks the rows flagged on that measure only."""
        t = theme.TOKENS[f["mode"]]
        ranked = df.sort_values(col, ascending=False)
        if input.attr_order() == "type":
            # Grouped: each kind gets its own block, best first within it.
            kinds = [k for _, k in PRINCIPAL_KINDS if (ranked.kind == k).any()]
            per = max(5, 20 // max(len(kinds), 1))
            top = pd.concat([ranked[ranked.kind == k].head(per) for k in kinds])
        else:
            top = ranked.head(20)
        top = top.iloc[::-1]  # plotly draws the first row at the bottom
        fig = go.Figure()
        for kind in [k for _, k in PRINCIPAL_KINDS]:
            d = top[top.kind == kind]
            if d.empty:
                continue
            fig.add_bar(
                x=d[col],
                y=d.label,
                name=KIND_LABEL[kind],
                orientation="h",
                width=0.6,
                marker={
                    "color": [
                        t["series"][7] if flag in w else _kind_color(t, kind) for w in d.why
                    ]
                },
                text=["\u00a0\u00a0" + fmt(v) for v in d[col]],
                textposition="outside",
                textfont={"color": t["ink"], "size": 13},
                customdata=[f"{lab}\n{fmt(v)}" for lab, v in zip(d.label, d[col])],
                hoverinfo="none",  # the name follows the mouse; see nametip
            )
        fig.update_layout(meta="nametip", barmode="overlay")
        fig.update_yaxes(categoryorder="array", categoryarray=list(top.label))
        fig.update_xaxes(range=[0, top[col].max() * 1.25], title_text=axis)
        return apply(
            fig, f["mode"], height=max(260, 24 * len(top) + 70),
            showlegend=top.kind.nunique() > 1, hover=True,
        )

    @render_widget
    def attr_bars():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        return rank_bars(f, df, "spend", "high estimated spend", fmt_usd, "est. USD")

    @render_widget
    def attr_rate_bars():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        return rank_bars(
            f, df, "per_request", "cost per request", lambda v: f"${v:,.4f}",
            "est. USD per request",
        )

    @render_widget
    def attr_scatter():
        f, df = flt(), attr_df()
        if df.empty:
            return empty_fig(f["mode"], "No estimated spend in this view")
        t = theme.TOKENS[f["mode"]]
        fig = go.Figure()
        for kind, symbol in [("user", "circle"), ("content", "diamond"), ("session", "square")]:
            for flag in (False, True):
                d = df[(df.kind == kind) & ((df.why != "") == flag)]
                if d.empty:
                    continue
                crowded = (df.kind == kind).sum() > 150
                # Name only the biggest five outliers: twenty overlapping labels read
                # as noise. The table below lists every one.
                named = set(d.sort_values("spend", ascending=False).label.head(5)) if flag else set()
                fig.add_scatter(
                    x=d.requests,
                    y=d.spend,
                    name=f"{KIND_LABEL[kind]} outlier" if flag else KIND_LABEL[kind],
                    mode="markers+text" if flag else "markers",
                    text=[lab if lab in named else "" for lab in d.label] if flag else None,
                    textposition="top center",
                    textfont={"size": 12, "color": t["ink2"]},
                    marker={
                        "symbol": symbol,
                        "size": (9 if crowded else 11) if flag else (5 if crowded else 8),
                        "color": t["series"][7] if flag else _kind_color(t, kind),
                        "opacity": 0.9 if flag else 0.55,
                        "line": {"width": 1, "color": t["surface"]},
                    },
                    customdata=[
                        f"{lab}\n{fmt_usd(s)}\n{int(r):,} requests"
                        for lab, s, r in zip(d.label, d.spend, d.requests)
                    ],
                    hoverinfo="none",  # the name follows the mouse; see nametip
                )
        fig.update_layout(meta="nametip")
        fig.update_xaxes(type="log", title_text="requests", tickformat=",.0f")
        fig.update_yaxes(type="log", tickprefix="$", title_text="est. USD", tickformat=",.2~f")
        # Ticks at powers of ten, unless the data sits inside one decade, where
        # that would leave the axis with no ticks at all.
        if df.requests.max() / max(df.requests.min(), 1) >= 10:
            fig.update_xaxes(dtick=1)
        if df.spend.max() / df.spend.min() >= 10:
            fig.update_yaxes(dtick=1)
        return apply(
            fig, f["mode"], height=max(260, 24 * min(len(df), 20) + 70), showlegend=True,
            hover=True,
        )

    def attr_out_df() -> pd.DataFrame:
        df = attr_df()
        out = df[df.why != ""][
            ["kind", "label", "detail", "why", "spend", "requests", "per_request", "x_median", "tokens"]
        ]
        # The detail column only earns its place when something fills it (sessions).
        if not (out.detail != "").any():
            out = out.drop(columns="detail")
        return out.rename(columns={"label": "name", "spend": "est_spend_usd"})

    @render.data_frame
    def attr_table():
        out = attr_out_df()
        return render.DataGrid(
            pretty_columns(formatted(out)), height=None if len(out) <= 10 else "420px",
            selection_mode="row",
        )

    # Picking an outlier opens it on the Principals tab.
    @reactive.effect
    @reactive.event(attr_table.cell_selection)
    def _attr_open():
        rows = (attr_table.cell_selection() or {}).get("rows") or ()
        if rows:
            r = attr_df()[lambda d: d.why != ""].iloc[rows[0]]
            open_profile(r.kind, r.k)

    ctx.register_export(
        "attr",
        lambda: attr_df()[
            ["kind", "k", "label", "detail", "why", "spend", "requests", "per_request", "x_median", "tokens"]
        ].rename(columns={"k": "id", "label": "name", "spend": "est_spend_usd"}),
        "attribution.csv",
    )
