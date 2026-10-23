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

import filters
import pricing
import principals
from pages import adoption, attribution, overview, raw_data
from pages.context import PageContext


def unused(*args, **kwargs) -> NoReturn:
    """A PageContext service the page under test must not call."""
    raise AssertionError("unexpected call to an unused PageContext service")


@pytest.fixture
def make_ctx(f):
    """A PageContext over `f`. Exports land in the returned dicts."""
    import app

    def make(sql_exports: dict | None = None):
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
            formatted=app.formatted, fitted_grid=app.fitted_grid,
        )
        return ctx, exports

    return make


def test_attribution_ranking_and_export(loaded, f, make_ctx):
    def attribution_export():
        ctx, exports = make_ctx()
        inputs = SimpleNamespace(attr_kind=lambda: "all")
        attribution.server(inputs, ctx, lambda kind, key: None)
        with reactive.isolate():
            return exports, exports["attr"][0]()

    exports, df = attribution_export()
    assert list(exports) == ["principal_spend", "attr"]
    assert exports["attr"][1] == "attribution.csv"
    assert df.columns.tolist() == [
        "kind", "id", "name", "detail", "why", "est_spend_usd", "requests",
        "per_request", "x_median", "tokens", "peak_p75",
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


def test_spend_by_principal_export_lists_every_user_and_content_item(loaded, f, make_ctx):
    ctx, exports = make_ctx()
    attribution.server(SimpleNamespace(attr_kind=lambda: "session"), ctx, lambda k, v: None)
    with reactive.isolate():
        df = exports["principal_spend"][0]()
    assert exports["principal_spend"][1] == "spend_by_principal.csv"
    assert df.columns.tolist() == [
        "kind", "id", "name", "email", "owner", "requests", "tokens", "est_spend_usd"]
    # Not tied to the Show toggle: users and content, whatever it says.
    assert set(df.kind) == {"user", "content"}
    where = filters.where(f)
    for kind, col in (("user", "username"), ("content", "content_guid")):
        part = df[df.kind == kind]
        assert set(part.id) == set(loaded.q(
            f"SELECT DISTINCT {col} AS id FROM gateway_events "
            f"WHERE {where} AND {col} IS NOT NULL AND {col} <> ''").id)
        assert part.requests.sum() == loaded.scalar(
            f"SELECT COUNT(*) FROM gateway_events WHERE {where} AND {col} IS NOT NULL")
        assert part.est_spend_usd.sum() == pytest.approx(loaded.scalar(
            f"SELECT SUM({pricing.est_cost()}) FROM gateway_events "
            f"WHERE {where} AND {col} IS NOT NULL"), abs=0.01)
    # Every user's email is filled in; content gets an owner instead.
    assert df[df.kind == "user"].email.notna().all()
    assert df[df.kind == "content"].email.isna().all()
    assert df.est_spend_usd.tolist() == sorted(df.est_spend_usd, reverse=True)


def test_principals_panel_outputs():
    html = str(ui.TagList(ui.navset_tab(principals.panel(), id="main_tabs")))
    assert 'data-value="Profiles"' in html
    for output_id in (
        "pf_kind", "pf_pick", "pf_header", "pf_kpis", "pf_grain", "pf_trend",
        "pf_models", "pf_providers", "pf_origins",
        "dl_pf_records", "pf_rel_0_chart", "dl_pf_rel_1",
        "pf_conv_by", "pf_conv_spend", "pf_conv_groups_note", "dl_pf_conv_groups",
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
        "pf_conv_groups": "profile_conversation_groups.csv",
    }
    assert sql_exports["pf_records"][1] == "profile_records.csv"

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
        filename = exports[name][1]
        return exports[name][0](), filename() if callable(filename) else filename


def test_adoption_export_counts_active_users(loaded, f, make_ctx):
    df, filename = run_export(make_ctx, adoption, "adoption", adoption_kind="user",
                              tool_grain="week")
    assert filename == "adoption_active_users.csv"
    # Spelled out: the export is read by people who do not know the acronyms.
    assert df.columns.tolist() == [
        "day", "daily_active_users", "weekly_active_users", "monthly_active_users"]
    df = df.rename(columns={"daily_active_users": "dau", "weekly_active_users": "wau",
                            "monthly_active_users": "mau"})
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
    df, _ = run_export(make_ctx, adoption, "adoption", adoption_kind="user",
                       tool_grain="week")
    assert df.monthly_active_users.max() == 1
    assert df.daily_active_users.max() == 1


def test_adoption_counts_content_items_when_toggled(loaded, f, make_ctx):
    """The Content toggle counts distinct content items, not the people behind them."""
    df, filename = run_export(make_ctx, adoption, "adoption", adoption_kind="content",
                              tool_grain="week")
    assert filename == "adoption_active_content_items.csv"
    assert df.columns.tolist() == [
        "day", "daily_active_content_items", "weekly_active_content_items",
        "monthly_active_content_items"]
    day = f["end"]
    on_day = loaded.scalar(f"""SELECT COUNT(DISTINCT content_guid) FROM gateway_events
        WHERE content_guid IS NOT NULL AND CAST(ts AS DATE) = DATE '{day}'""")
    assert df.iloc[-1].daily_active_content_items == on_day
    assert (df.daily_active_content_items <= df.monthly_active_content_items).all()


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
