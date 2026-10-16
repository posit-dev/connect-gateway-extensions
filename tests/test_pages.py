"""Page servers run against the fake Connect's data, outside a browser session.

Each page server gets the real shared services from app.py through a
PageContext, and its inputs as plain values. The exports a page registers are
called directly, so these check what the CSVs (and the tables built from the
same frames) contain.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NoReturn

import pandas as pd
import pytest
from shiny import reactive, ui

import conversations
import filters
import pricing
import principals
from pages import adoption, attribution, models, raw_data
from pages import conversations as conversations_page
from pages.context import PageContext


def unused(*args, **kwargs) -> NoReturn:
    """A PageContext service the page under test must not call."""
    raise AssertionError("unexpected call to an unused PageContext service")


@pytest.fixture
def make_ctx(f):
    """A PageContext over `f`. Exports land in the returned dicts."""
    import app

    def make(priced_only: str = "", sql_exports: dict | None = None):
        exports: dict = {}
        ctx = PageContext(
            flt=lambda: f, where=filters.where, cat=app.catalog,
            est_cost=pricing.est_cost,
            register_export=lambda name, fn, filename: exports.update({name: (fn, filename)}),
            register_sql_export=(
                unused if sql_exports is None
                else lambda name, fn, filename: sql_exports.update({name: (fn, filename)})
            ),
            est_cost_uncached=pricing.est_cost_uncached,
            total_bars=unused, measure_of=app.measure_of, formatted=app.formatted,
            chart_labels=app.chart_labels, fitted_grid=app.fitted_grid,
            priced_only=lambda: priced_only,
        )
        return ctx, exports

    return make


def test_attribution_ranking_and_export(loaded, f, make_ctx):
    def attribution_export(attr_order: str):
        ctx, exports = make_ctx()
        inputs = SimpleNamespace(attr_kind=lambda: "all", attr_order=lambda: attr_order)
        attribution.server(inputs, ctx, lambda kind, key: None)
        with reactive.isolate():
            return exports, exports["attr"][0]()

    exports, df = attribution_export("spend")
    assert list(exports) == ["attr"]
    assert exports["attr"][1] == "attribution.csv"
    assert df.columns.tolist() == [
        "kind", "id", "name", "detail", "why", "est_spend_usd", "requests",
        "per_request", "x_median", "tokens",
    ]
    assert set(df.kind) == {"user", "content", "session"}
    assert (df.est_spend_usd > 0).all()
    assert df.est_spend_usd.tolist() == sorted(df.est_spend_usd, reverse=True)
    users = df[df.kind == "user"]
    user_spend = loaded.scalar(f"""SELECT SUM(s) FROM (
        SELECT SUM({pricing.est_cost()}) s FROM gateway_events
        WHERE {filters.where(f)} AND username IS NOT NULL AND username <> ''
        GROUP BY username HAVING SUM({pricing.est_cost()}) > 0)""")
    assert users.est_spend_usd.sum() == pytest.approx(user_spend)
    assert users.id.tolist() == users.name.tolist()

    _, grouped = attribution_export("type")
    assert grouped.kind.drop_duplicates().tolist() == ["user", "content", "session"]
    assert len(grouped.kind.ne(grouped.kind.shift()).cumsum().unique()) == 3


def test_conversations_export_and_unpriced_exclusion(loaded, f, make_ctx):
    def conversations_export(priced_only: str):
        ctx, exports = make_ctx(priced_only)
        conversations_page.server(SimpleNamespace(conv_by=lambda: "model"), ctx)
        with reactive.isolate():
            return exports, exports["conversations"][0]()

    exports, df = conversations_export("")
    assert list(exports) == ["conversations"]
    assert exports["conversations"][1] == "conversations.csv"
    assert not df.empty
    for col in ("conversation_id", "principal", "calls", "est", "main_model", "high", "line"):
        assert col in df.columns
    for k, *_ in conversations.LEVERS:
        assert f"sav_{k}" in df.columns
    assert df.est.tolist() == sorted(df.est, reverse=True)
    # The per-part cost the page sums equals the catalog's own estimate.
    total = loaded.scalar(f"""SELECT SUM({pricing.est_cost()}) FROM gateway_events
        WHERE {filters.where(f)} AND conversation_id IS NOT NULL""")
    assert df.est.sum() == pytest.approx(total, abs=1e-6)
    assert df.calls.sum() == loaded.scalar(
        f"""SELECT COUNT(*) FROM gateway_events
            WHERE {filters.where(f)} AND conversation_id IS NOT NULL""")

    # The shared unpriced-model fragment from app.py narrows the page's query.
    top = df.main_model.dropna().iloc[0]
    _, narrowed = conversations_export(
        f" AND (model IS NULL OR model NOT IN ({filters.sql_str(top)}))"
    )
    assert top not in set(narrowed.main_model.dropna())
    assert narrowed.calls.sum() < df.calls.sum()


def test_principals_panel_outputs():
    html = str(ui.TagList(ui.navset_tab(principals.panel(), id="main_tabs")))
    assert 'data-value="Principals"' in html
    for output_id in (
        "pf_kind", "pf_pick", "pf_header", "pf_kpis", "pf_grain", "pf_trend",
        "pf_models", "pf_providers", "pf_origins", "pf_records_title", "pf_records",
        "dl_pf_records", "pf_rel_0_chart", "pf_rel_1", "dl_pf_rel_1",
        "pf_conv_by", "pf_conv_spend", "pf_conv_groups", "dl_pf_conv_groups",
    ):
        assert f'id="{output_id}"' in html


def test_principals_exports_use_raw_data_sql(loaded, f, make_ctx):
    sql_exports: dict = {}
    ctx, exports = make_ctx(sql_exports=sql_exports)
    peers = principals.peers_df("user", filters.where(f), pricing.est_cost())
    user = peers.k.iloc[0]
    inputs = SimpleNamespace(
        pf_kind=lambda: "user", pf_pick=lambda: user,
        pf_conv_by=lambda: "model", pf_grain=lambda: "day",
    )
    records_sql = raw_data.server(inputs, ctx)
    open_profile = principals.server(inputs, ctx, records_sql)
    assert callable(open_profile)
    assert {name: filename for name, (_, filename) in exports.items()} == {
        "pf_rel_0": "pf_connections_1.csv", "pf_rel_1": "pf_connections_2.csv",
        "pf_conv_groups": "principal_conversation_groups.csv",
    }
    assert sql_exports["pf_records"][1] == "principal_records.csv"

    with reactive.isolate():
        # The uncapped export is the Raw Data builder over this principal's rows.
        sql = sql_exports["pf_records"][0]()
        assert sql == records_sql(f"{filters.where(f)} AND {principals.pred('user', user)}")
        records = loaded.q(sql)
        groups = exports["pf_conv_groups"][0]()
        content = exports["pf_rel_0"][0]()
    assert len(records) == peers.requests.iloc[0]
    assert set(records.username) == {user}
    assert "LIMIT" not in sql
    assert groups.conversations.sum() == records.conversation_id.nunique()
    assert "content" in content.columns


def run_export(make_ctx, page, name: str, **inputs):
    """Start a page's server and call the export it registered."""
    ctx, exports = make_ctx()
    page.server(SimpleNamespace(**{k: (lambda v=v: v) for k, v in inputs.items()}), ctx)
    with reactive.isolate():
        return exports[name][0](), exports[name][1]


def test_adoption_export_counts_active_users(loaded, f, make_ctx):
    df, filename = run_export(make_ctx, adoption, "adoption", adopt_grain="day",
                              tool_grain="week", heat_measure="requests")
    assert filename == "adoption_active_users.csv"
    assert df.columns.tolist() == ["day", "dau", "wau", "mau"]
    days = (f["end"] - f["start"]).days + 1
    assert len(df) == days
    assert df.day.astype(str).str[:10].tolist() == sorted(df.day.astype(str).str[:10])
    # A wider window can only hold more of the same people.
    assert ((df.dau <= df.wau) & (df.wau <= df.mau)).all()

    # On the last day, MAU is the distinct users of the 30 days ending then.
    last = df.iloc[-1]
    day = f["end"]
    distinct = loaded.scalar(f"""SELECT COUNT(DISTINCT username) FROM gateway_events
        WHERE username IS NOT NULL AND ts >= TIMESTAMP '{day} 23:59:59.999' - INTERVAL 30 DAY
          AND ts <= TIMESTAMP '{day} 23:59:59.999'""")
    assert last.mau == pytest.approx(distinct, abs=1)  # the window edge is the day, not the second
    on_day = loaded.scalar(f"""SELECT COUNT(DISTINCT username) FROM gateway_events
        WHERE username IS NOT NULL AND CAST(ts AS DATE) = DATE '{day}'""")
    assert last.dau == on_day


def test_adoption_narrowed_to_one_user(loaded, f, make_ctx):
    user = loaded.q(f"""SELECT username FROM gateway_events WHERE {filters.where(f)}
                        AND username IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1""")
    f["users"] = [user.username[0]]
    df, _ = run_export(make_ctx, adoption, "adoption", adopt_grain="day",
                       tool_grain="week", heat_measure="requests")
    assert df.mau.max() == 1
    assert df.dau.max() == 1


def test_models_export_totals_and_unpriced_model(loaded, f, make_ctx):
    df, filename = run_export(make_ctx, models, "models")
    assert filename == "models.csv"
    assert df.columns.tolist() == [
        "model", "providers", "est_spend_usd", "pct_of_total", "requests", "input_tokens",
        "output_tokens", "cache_read_tokens", "cache_pct", "cache_savings_usd", "error_pct",
        "p95_ms",
    ]
    assert df.model.is_unique
    assert df.est_spend_usd.tolist() == sorted(df.est_spend_usd, reverse=True)
    assert df.pct_of_total.sum() == pytest.approx(100)
    assert df.cache_pct.dropna().between(0, 100).all()
    assert df.error_pct.between(0, 100).all()
    assert (df.cache_savings_usd >= -1e-9).all()  # caching never costs more than not caching

    base = f"{filters.where(f)} AND model IS NOT NULL AND model <> ''"
    assert df.requests.sum() == loaded.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {base}")
    assert df.input_tokens.sum() == loaded.scalar(
        f"SELECT SUM(input_tokens) FROM gateway_events WHERE {base}")
    assert df.est_spend_usd.sum() == pytest.approx(loaded.scalar(
        f"SELECT SUM({pricing.est_cost()}) FROM gateway_events WHERE {base}"))
    # A model with no catalog price still counts its requests, at $0.
    free = df[df.est_spend_usd == 0]
    assert not free.empty and (free.requests > 0).all()


def test_models_narrowed_to_one_model(loaded, f, make_ctx):
    full, _ = run_export(make_ctx, models, "models")
    f["models"] = [full.model.iloc[0]]
    narrowed, _ = run_export(make_ctx, models, "models")
    assert narrowed.model.tolist() == [full.model.iloc[0]]
    assert narrowed.requests.iloc[0] == full.requests.iloc[0]
    assert narrowed.pct_of_total.iloc[0] == pytest.approx(100)


def test_raw_data_records_follow_the_filter(loaded, f, make_ctx):
    def start():
        # A new server each time: the page caches its WHERE for the filter it saw.
        sql_exports: dict = {}
        ctx, _ = make_ctx(sql_exports=sql_exports)
        return raw_data.server(SimpleNamespace(), ctx), sql_exports["records"]

    def export() -> pd.DataFrame:
        _, (sql_fn, _) = start()
        with reactive.isolate():
            return loaded.q(sql_fn())

    records_sql, (_, filename) = start()
    assert filename == "gateway_records.csv"
    everything = export()
    assert len(everything) == loaded.scalar(
        f"SELECT COUNT(*) FROM gateway_events WHERE {filters.where(f)}")
    assert everything.ts.is_monotonic_decreasing
    assert {"origin", "tool", "username", "authenticated_username", "viewer_username",
            "content_owner_username", "via", "est_cost_usd", "request_metadata"} <= set(
        everything.columns)
    assert everything.est_cost_usd.sum() == pytest.approx(
        loaded.scalar(f"SELECT SUM({pricing.est_cost()}) FROM gateway_events "
                      f"WHERE {filters.where(f)}"), abs=0.1)  # each row is rounded to 4 places
    assert everything.response_id.is_unique  # no row is repeated by the grid's joins

    f["models"] = [everything.model.dropna().iloc[0]]
    narrowed = export()
    assert set(narrowed.model) == set(f["models"])
    assert 0 < len(narrowed) < len(everything)
    # The builder principals reuse takes any predicate, and caps nothing itself.
    assert "LIMIT" not in records_sql("TRUE")
