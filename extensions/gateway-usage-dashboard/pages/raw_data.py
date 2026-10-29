"""Raw gateway records, recent grid, and uncapped CSV download."""

import pandas as pd
from shiny import reactive, render, ui

import queries
from filters import Query
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
    flt = ctx.flt

    # The grid is sent to the browser whole, so it lists only the most recent
    # rows. The CSV is never capped: it is streamed from DuckDB with every record.
    RECORDS_GRID_MAX = 2000

    @reactive.calc
    def records_shown() -> pd.DataFrame:
        return queries.records(flt(), RECORDS_GRID_MAX)

    def records_query_all() -> Query:
        """The export is every filtered row — the table only shows the head."""
        return queries.records_query(flt())

    @render.data_frame
    def records_table():
        return render.DataGrid(records_shown(), height="500px")

    @render.text
    def records_title():
        total = queries.record_count(flt())
        if total > RECORDS_GRID_MAX:
            return f"Gateway records: latest {RECORDS_GRID_MAX:,} of {total:,} shown"
        return f"Gateway records ({total:,})"

    @render.ui
    def records_capped():
        total = queries.record_count(flt())
        # An invisible marker for the script in the page head.
        if total > RECORDS_GRID_MAX:
            return ui.span(id="records-capped", style="display:none")
        return None

    ctx.register_sql_export("records", records_query_all, "gateway_records.csv")
