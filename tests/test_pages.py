"""Page servers run against the fake Connect's data, outside a browser session.

Each page server gets the real shared services from app.py through a
PageContext, and its inputs as plain values. The exports a page registers are
called directly, so these check what the CSVs (and the tables built from the
same frames) contain.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import NoReturn

import pandas as pd
import pytest
from shiny import reactive, ui

import pricing
import principals
import queries
from filters import Filters
from pages import adoption, overview, raw_data
from pages.context import PageContext


def unused(*args, **kwargs) -> NoReturn:
    """A PageContext service the page under test must not call."""
    raise AssertionError("unexpected call to an unused PageContext service")


@pytest.fixture
def make_ctx(f):
    """A PageContext over `f`. Exports land in the returned dicts."""
    import app

    def make(sql_exports: dict | None = None, narrowed: Filters | None = None):
        exports: dict = {}
        ctx = PageContext(
            flt=lambda: narrowed or f, cat=app.catalog,
            register_export=lambda name, fn, filename: exports.update({name: (fn, filename)}),
            register_sql_export=(
                unused if sql_exports is None
                else lambda name, fn, filename: sql_exports.update({name: (fn, filename)})
            ),
            fitted_grid=app.fitted_grid,
        )
        return ctx, exports

    return make


def test_peers_rank_and_flag_outliers(loaded, f):
    w, cost = f.where(), pricing.est_cost()
    users = principals.peers_df(f, "user")
    assert users.spend.tolist() == sorted(users.spend, reverse=True)
    assert users.spend.sum() == pytest.approx(loaded.scalar(
        f"SELECT SUM({cost}) FROM gateway_events WHERE {w} AND username IS NOT NULL"))
    assert users.label.is_unique
    # The fake Connect has a runaway content item: the top spender, and flagged.
    content = principals.peers_df(f, "content")
    assert content.outlier.iloc[0]
    assert content.outlier.sum() < len(content) / 2


def test_peer_table_lists_outliers_first_and_marks_the_pick(loaded, f):
    peers = principals.peers_df(f, "content")
    top = peers.k.iloc[0]
    html = str(principals.peer_table(peers, "content", str(top), "", True))
    # One row per content item; the runaway one leads, picked and flagged.
    assert html.count('class="pt-row') == len(peers)
    first = html[html.index('class="pt-row'):]
    first = first[:first.index("</tr>")]
    assert "pt-sel" in first and "versus-outlier" in first
    assert f"&apos;pf_row&apos;, &apos;{top}&apos;" in first
    # Sorted by a column, low to high.
    asc = str(principals.peer_table(peers, "content", None, "spend", False))
    assert "\u25b2" in asc and "pt-sel" not in asc


def test_conversation_compare_splits_the_two_camps():
    est = pd.Series([1.0] * 30 + [10.0] * 10)
    cv = pd.DataFrame({"est": est, "duration_min": 1.0, "peak_input": 1.0,
                       "miss_tokens": 0.0, "uses_cache": True})
    cv[["high", "line"]] = principals.higher_cost(cv.est)
    table = str(principals.conv_compare(cv, "user", "dark"))
    # 10 of 40 cost $10, the rest $1: $100 of $130 is in the higher-cost camp
    assert "40 conversations" in table and "Higher cost" in table and "Normal" in table
    assert "10 \u00b7 25%" in table and "\u00b7 77%" in table
    few = str(principals.conv_compare(cv.head(3), "user", "dark"))
    assert "too few to split" in few and "Normal" not in few


def test_principals_panel_outputs():
    html = str(ui.TagList(ui.navset_tab(principals.panel(), id="main_tabs")))
    assert 'data-value="Attribution"' in html
    for output_id in (
        "pf_kind", "pf_pick", "pf_table", "pf_header", "pf_grain", "pf_trend",
        "pf_models", "pf_providers", "pf_origins",
        "dl_pf_records", "pf_rel_0_chart", "dl_pf_rel_1",
        "pf_conv_compare", "pf_conv_models", "pf_conv_tools",
    ):
        assert f'id="{output_id}"' in html


def test_principals_exports_use_raw_data_sql(loaded, f, make_ctx):
    sql_exports: dict = {}
    ctx, exports = make_ctx(sql_exports=sql_exports)
    peers = principals.peers_df(f, "user")
    user = peers.k.iloc[0]
    inputs = SimpleNamespace(
        pf_kind=lambda: "user", pf_pick=lambda: user,
        pf_grain=lambda: "day", pf_sort=lambda: "", pf_row=lambda: "",
    )
    raw_data.server(inputs, ctx)
    principals.server(inputs, ctx)
    assert {name: filename for name, (_, filename) in exports.items()} == {
        "pf_rel_0": "pf_connections_1.csv", "pf_rel_1": "pf_connections_2.csv",
    }
    assert sql_exports["pf_records"][1] == "profile_records.csv"

    with reactive.isolate():
        # The uncapped export is the Raw Data builder over this principal's rows.
        sql = sql_exports["pf_records"][0]()
        assert sql == queries.records_sql(f"{f.where()} AND {queries.pred('user', user)}")
        records = loaded.q(sql)
        content = exports["pf_rel_0"][0]()
    assert len(records) == peers.requests.iloc[0]
    assert set(records.username) == {user}
    assert "LIMIT" not in sql
    assert "content" in content.columns


def run_export(make_ctx, page, name: str, narrowed: Filters | None = None, **inputs):
    """Start a page's server and call the export it registered."""
    ctx, exports = make_ctx(narrowed=narrowed)
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
    days = (f.end - f.start).days + 1
    assert len(df) == days
    assert df.day.astype(str).str[:10].tolist() == sorted(df.day.astype(str).str[:10])
    # A wider window can only hold more of the same people.
    assert ((df.dau <= df.wau) & (df.wau <= df.mau)).all()

    # On the last day, MAU is the distinct users of the 30 days ending then.
    last = df.iloc[-1]
    day = f.end
    distinct = loaded.scalar(f"""SELECT COUNT(DISTINCT username) FROM gateway_events
        WHERE username IS NOT NULL AND ts >= TIMESTAMP '{day} 23:59:59.999' - INTERVAL 30 DAY
          AND ts <= TIMESTAMP '{day} 23:59:59.999'""")
    assert last.mau == pytest.approx(distinct, abs=1)  # the window edge is the day, not the second
    on_day = loaded.scalar(f"""SELECT COUNT(DISTINCT username) FROM gateway_events
        WHERE username IS NOT NULL AND CAST(ts AS DATE) = DATE '{day}'""")
    assert last.dau == on_day


def test_adoption_narrowed_to_one_user(loaded, f, make_ctx):
    user = loaded.q(f"""SELECT username FROM gateway_events WHERE {f.where()}
                        AND username IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1""")
    df, _ = run_export(make_ctx, adoption, "adoption", adoption_kind="user",
                       tool_grain="week", narrowed=replace(f, users=(user.username[0],)))
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
    day = f.end
    on_day = loaded.scalar(f"""SELECT COUNT(DISTINCT content_guid) FROM gateway_events
        WHERE content_guid IS NOT NULL AND CAST(ts AS DATE) = DATE '{day}'""")
    assert df.iloc[-1].daily_active_content_items == on_day
    assert (df.daily_active_content_items <= df.monthly_active_content_items).all()


def test_raw_data_records_follow_the_filter(loaded, f, make_ctx):
    def start(narrowed: Filters | None = None):
        # A new server each time: the page caches its WHERE for the filter it saw.
        sql_exports: dict = {}
        ctx, _ = make_ctx(sql_exports=sql_exports, narrowed=narrowed)
        raw_data.server(SimpleNamespace(), ctx)
        return sql_exports["records"]

    def export(narrowed: Filters | None = None) -> pd.DataFrame:
        sql_fn, _ = start(narrowed)
        with reactive.isolate():
            return loaded.q(sql_fn())

    _, filename = start()
    assert filename == "gateway_records.csv"
    everything = export()
    assert len(everything) == loaded.scalar(
        f"SELECT COUNT(*) FROM gateway_events WHERE {f.where()}")
    assert everything.ts.is_monotonic_decreasing
    assert {"origin", "tool", "username", "authenticated_username", "viewer_username",
            "content_owner_username", "via", "est_cost_usd", "request_metadata"} <= set(
        everything.columns)
    assert everything.est_cost_usd.sum() == pytest.approx(
        loaded.scalar(f"SELECT SUM({pricing.est_cost()}) FROM gateway_events "
                      f"WHERE {f.where()}"), abs=0.1)  # each row is rounded to 4 places
    assert everything.response_id.is_unique  # no row is repeated by the grid's joins

    group = loaded.q("SELECT group_guid FROM user_groups ORDER BY 1 LIMIT 1").group_guid[0]
    members = set(loaded.q(f"""SELECT DISTINCT username FROM gateway_events WHERE user_guid IN
        (SELECT user_guid FROM user_groups WHERE group_guid = '{group}')""").username)
    narrowed = export(replace(f, groups=(group,)))
    assert set(narrowed.username) <= members
    assert 0 < len(narrowed) < len(everything)
    # The builder principals reuse takes any predicate, and caps nothing itself.
    assert "LIMIT" not in queries.records_sql("TRUE")
