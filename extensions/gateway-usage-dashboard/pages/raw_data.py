"""Raw gateway records, recent grid, and uncapped CSV download."""

import pandas as pd
from shiny import reactive, render, ui

import usage
from pages.context import PageContext
from ui_shell import page_header, table_header


def panel():
    return ui.nav_panel(
        "Raw Data",
        page_header("Raw Data"),
        ui.div(
            ui.card(
                table_header(ui.output_text("records_title", inline=True), "dl_records"),
                ui.card_body(
                    ui.output_data_frame("records_table"),
                    ui.output_ui("records_capped"),
                ),
            ),
            class_="question-block",
        ),
    )


def server(input, ctx: PageContext):
    flt, where, est_cost = ctx.flt, ctx.where, ctx.est_cost

    # The derived principal answers "by whom", and the three raw identities
    # behind it are kept beside it: audit evidence must show what the gateway
    # recorded, not only what this dashboard concluded from it.
    # `status` is a string, not a number: an HTTP code is a label rather than a
    # quantity, so sorting or ranging over it ("status > 300") would say nothing.
    RECORD_COLS = """ts, source AS origin, tool, user_agent, username, user_kind AS via,
                     user_email AS email,
                     authenticated_username, viewer_username,
                     content_owner_username, content_guid,
                     (SELECT n.content_name FROM content_names n
                      WHERE n.content_guid = gateway_events.content_guid) AS content_name,
                     job_key,
                     model, provider, integration_auth_type AS auth_type,
                     CAST(status_code AS VARCHAR) AS status,
                     ROUND(duration_ms) AS ms,
                     input_tokens AS tok_in, output_tokens AS tok_out,
                     cache_read_tokens AS tok_cache_read,
                     ROUND(EST_COST_EXPR,4) AS est_cost_usd,
                     finish_reason, response_id, conversation_id,
                     workbench_session_id, workbench_session_name,
                     workbench_project_name, workbench_project_path,
                     team, cost_center, tier, enforced,
                     CAST(request_metadata AS VARCHAR) AS request_metadata"""

    @reactive.calc
    def audit_where() -> str:
        """The sidebar's WHERE."""
        return where(flt())

    def records_sql(where_sql: str, limit: str = "") -> str:
        cols = RECORD_COLS.replace("EST_COST_EXPR", est_cost())
        return f"SELECT {cols} FROM gateway_events WHERE {where_sql} ORDER BY ts DESC{limit}"

    # The grid is sent to the browser whole, so it lists only the most recent
    # rows. The CSV is never capped: it is streamed from DuckDB with every record.
    RECORDS_GRID_MAX = 2000

    @reactive.calc
    def records_shown() -> pd.DataFrame:
        return usage.q(records_sql(audit_where(), f" LIMIT {RECORDS_GRID_MAX}"))

    def records_sql_all() -> str:
        """The export is every filtered row — the table only shows the head."""
        return records_sql(audit_where())

    @render.data_frame
    def records_table():
        return render.DataGrid(records_shown(), height="500px")

    @render.text
    def records_title():
        total = int(
            usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {audit_where()}")
            or 0
        )
        if total > RECORDS_GRID_MAX:
            return f"Gateway records: latest {RECORDS_GRID_MAX:,} of {total:,} shown"
        return f"Gateway records ({total:,})"

    @render.ui
    def records_capped():
        total = (
            usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {audit_where()}")
            or 0
        )
        # An invisible marker for the script in the page head.
        if total > RECORDS_GRID_MAX:
            return ui.span(id="records-capped", style="display:none")
        return None

    ctx.register_sql_export("records", records_sql_all, "gateway_records.csv")
    return records_sql
