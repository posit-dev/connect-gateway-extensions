"""Pure helpers: no Connect, no DuckDB session, no Shiny."""

from datetime import date

import duckdb
import pandas as pd
import pytest

import conversations
import principals
import theme
import usage
from filters import where


# ---- usage._catalog_prices ---------------------------------------------------
def _row(model: str, n: float) -> dict:
    return {
        "model": model,
        "input_usd_per_million_tokens": n,
        "output_usd_per_million_tokens": n * 2,
        "cache_creation_usd_per_million_tokens": n * 3,
        "cache_read_usd_per_million_tokens": n * 4,
    }


def _rates(n: float) -> tuple:
    return (n, n * 2, n * 3, n * 4)


def _catalog(*models: dict, provider: str = "aws.bedrock") -> dict:
    return {"providers": [{"provider": provider, "models": list(models)}]}


@pytest.mark.parametrize("order", [
    ["claude-x-v1:0", "global.claude-x-v1:0", "us.claude-x-v1:0"],
    ["us.claude-x-v1:0", "global.claude-x-v1:0", "claude-x-v1:0"],
    ["global.claude-x-v1:0", "us.claude-x-v1:0", "claude-x-v1:0"],
])
def test_default_is_base_then_global_then_region_whatever_the_order(order):
    price = {"claude-x-v1:0": 1, "global.claude-x-v1:0": 2, "us.claude-x-v1:0": 3}
    prices = usage._catalog_prices(_catalog(*[_row(m, price[m]) for m in order]))
    assert prices[("aws.bedrock", "", "claude-x")] == _rates(1)
    assert prices[("aws.bedrock", "us", "claude-x")] == _rates(3)
    assert prices[("aws.bedrock", "global", "claude-x")] == _rates(2)


def test_without_a_base_id_global_beats_a_specific_region():
    for order in (["global.m", "us.m"], ["us.m", "global.m"]):
        price = {"global.m": 2, "us.m": 3}
        prices = usage._catalog_prices(_catalog(*[_row(m, price[m]) for m in order]))
        assert prices[("aws.bedrock", "", "m")] == _rates(2)


def test_only_a_specific_region_still_gives_a_default():
    prices = usage._catalog_prices(_catalog(_row("eu.m", 5)))
    assert prices[("aws.bedrock", "", "m")] == _rates(5)
    assert prices[("aws.bedrock", "eu", "m")] == _rates(5)


def test_dated_snapshots_collapse_to_one_family_first_seen_wins():
    prices = usage._catalog_prices(_catalog(_row("m-20250101", 1), _row("m-20250601", 9)))
    assert prices == {("aws.bedrock", "", "m"): _rates(1)}


def test_providers_do_not_share_prices():
    pricing = {"providers": [
        {"provider": "anthropic", "models": [_row("m", 1)]},
        {"provider": "aws.bedrock", "models": [_row("m", 2)]},
    ]}
    prices = usage._catalog_prices(pricing)
    assert prices[("anthropic", "", "m")] == _rates(1)
    assert prices[("aws.bedrock", "", "m")] == _rates(2)


def test_rows_with_missing_or_bad_fields_are_skipped():
    bad_rate = _row("bad", 1) | {"output_usd_per_million_tokens": "n/a"}
    missing = {k: v for k, v in _row("gone", 1).items() if k != "cache_read_usd_per_million_tokens"}
    no_model = {k: v for k, v in _row("x", 1).items() if k != "model"}
    pricing = _catalog(bad_rate, missing, no_model, _row("ok", 1))
    pricing["providers"].append({"models": [_row("anon", 1)]})  # no provider
    assert usage._catalog_prices(pricing) == {("aws.bedrock", "", "ok"): _rates(1)}


def test_numeric_strings_are_accepted_as_rates():
    prices = usage._catalog_prices(_catalog(_row("m", 1) | {"input_usd_per_million_tokens": "1"}))
    assert prices[("aws.bedrock", "", "m")] == _rates(1)


def test_empty_catalogs_give_no_prices():
    assert usage._catalog_prices({}) == {}
    assert usage._catalog_prices({"providers": []}) == {}


@pytest.mark.parametrize("bad", [
    [],                                  # a list where an object was expected
    {"providers": 3},
    {"providers": [None]},
    {"providers": [{"provider": "p", "models": [5]}]},
    {"providers": [{"provider": "p", "models": None}]},
])
def test_malformed_shapes_raise_so_the_caller_can_return_none(bad):
    with pytest.raises((AttributeError, TypeError)):
        usage._catalog_prices(bad)


# ---- usage._list_pages (fake fetcher, no network) ------------------------------
def _pager(pages: list, *, log: list | None = None):
    """A get() serving pages[page_number - 1]; past the end, an empty page."""
    def get(url: str):
        number = int(url.rsplit("page_number=", 1)[1])
        if log is not None:
            log.append(number)
        return pages[number - 1] if number <= len(pages) else []
    return get


def test_list_pages_follows_short_last_page():
    pages = [[1, 2], [3, 4], [5]]
    log: list = []
    assert usage._list_pages("u", 2, get=_pager(pages, log=log)) == [1, 2, 3, 4, 5]
    assert log == [1, 2, 3]


def test_list_pages_full_last_page_costs_one_extra_empty_request():
    log: list = []
    assert usage._list_pages("u", 2, get=_pager([[1, 2], [3, 4]], log=log)) == [1, 2, 3, 4]
    assert log == [1, 2, 3]


def test_list_pages_uses_total_when_the_server_caps_page_size():
    pages = [{"results": [1, 2], "total": 5}, {"results": [3, 4], "total": 5},
             {"results": [5], "total": 5}]
    assert usage._list_pages("u", 100, get=_pager(pages)) == [1, 2, 3, 4, 5]


def test_list_pages_total_reached_stops_without_another_request():
    log: list = []
    pages = [{"results": [1, 2], "total": 2}]
    assert usage._list_pages("u", 2, get=_pager(pages, log=log)) == [1, 2]
    assert log == [1]


def test_list_pages_total_not_reached_by_an_empty_page_fails():
    pages = [{"results": [1], "total": 5}, {"results": [], "total": 5}]
    assert usage._list_pages("u", 1, get=_pager(pages)) is None


def test_list_pages_failed_page_returns_none_not_a_partial_list():
    def get(url: str):
        return [1, 2] if url.endswith("page_number=1") else None
    assert usage._list_pages("u", 2, get=get) is None


def test_list_pages_unexpected_shape_returns_none():
    assert usage._list_pages("u", 2, get=lambda url: {"oops": 1}) is None
    assert usage._list_pages("u", 2, get=lambda url: "text") is None


def test_list_pages_server_ignoring_page_number_returns_none():
    assert usage._list_pages("u", 2, get=lambda url: [1, 2]) is None


def test_list_pages_unpaged_endpoint_returns_everything_at_once():
    assert usage._list_pages("u", 2, get=lambda url: [1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_list_pages_wrong_current_page_returns_none():
    body = {"results": [1], "current_page": 2, "total": 9}
    assert usage._list_pages("u", 1, get=lambda url: body) is None


def test_list_pages_empty_list():
    assert usage._list_pages("u", 2, get=lambda url: []) == []


def test_list_pages_boolean_total_is_not_a_count():
    pages = [{"results": [1], "total": True}, {"results": []}]
    assert usage._list_pages("u", 1, get=_pager(pages)) == [1]


# ---- conversations ------------------------------------------------------------
@pytest.mark.parametrize("s, text", [
    (0, "0 s"), (40, "40 s"), (59.4, "59 s"),
    (60, "1 min"), (720, "12 min"), (3599, "60 min"),
    (3600, "1 h"), (9000, "2.5 h"),
])
def test_fmt_secs(s, text):
    assert conversations.fmt_secs(s) == text


@pytest.mark.parametrize("v, up", [
    (2.33, 2.4), (2.4, 2.4), (140, 140), (141, 150), (0.0123, 0.013), (99, 99), (99.1, 100),
])
def test_up2_rounds_up_to_two_significant_figures(v, up):
    assert conversations._up2(v) == pytest.approx(up)


@pytest.mark.parametrize("v, kind, nice", [
    (2.1, "int", 3), (3.0, "int", 3),
    (0.401, "share", 0.41), (0.4, "share", 0.4),
    (140, "secs", 150),          # 2.33 min -> 2.4 min = 144 s
    (0, "secs", 0), (-5, "secs", 0),
    (45, "secs", 45), (4000, "secs", 4320),
])
def test_nice(v, kind, nice):
    expected = 144.0 if (v, kind) == (140, "secs") else nice
    assert conversations._nice(v, kind) == pytest.approx(expected)


def test_quantile_bands_cuts_at_p50_and_p90():
    vals = pd.Series(range(1, 101), dtype=float)
    bands, labels = conversations.quantile_bands(vals, "int")
    assert len(labels) == 3 and labels[-1].endswith("+")
    assert list(bands.cat.categories) == labels
    counts = bands.value_counts().reindex(labels).tolist()
    assert counts == [51, 40, 9]   # cuts round up: p50=50.5 -> 51, p90=90.1 -> 91
    assert bands.iloc[0] == labels[0] and bands.iloc[-1] == labels[-1]


def test_quantile_bands_low_tail_mirrors_the_cuts():
    vals = pd.Series(range(1, 101), dtype=float)
    low, _ = conversations.quantile_bands(vals, "int", "low")
    counts = low.value_counts().reindex(low.cat.categories).tolist()
    assert counts == [11, 40, 49]   # p10=10.9 -> 11; the narrow band is at the low end


def test_quantile_bands_tiny_population_gets_one_band():
    vals = pd.Series([1.0, 2.0, 3.0])
    bands, labels = conversations.quantile_bands(vals, "int")
    assert labels == ["1\u20133"]
    assert bands.notna().all()


def test_quantile_bands_constant_column_merges_to_one_band():
    vals = pd.Series([1.0] * 50)
    bands, labels = conversations.quantile_bands(vals, "int")
    assert len(labels) == 1 and bands.nunique() == 1


def test_quantile_bands_labels_are_unique_and_cover_every_value():
    vals = pd.Series([1.0] * 60 + [2.0] * 30 + [50.0] * 10)
    for kind in ("int", "share", "secs"):
        v = vals / 100 if kind == "share" else vals
        bands, labels = conversations.quantile_bands(v, kind)
        assert len(set(labels)) == len(labels)
        assert bands.notna().all()


def test_band_labels_int_secs_and_share():
    vals = pd.Series([1, 2, 3, 10, 20, 30])
    inf = float("inf")
    say = lambda v: f"{v:,.0f}"  # noqa: E731
    assert conversations._band_labels(vals, [-inf, 3, 10, inf], "int", say) == ["1\u20133", "4\u201310", "11+"]
    assert conversations._band_labels(vals, [-inf, 3, inf], "int", say) == ["1\u20133", "4+"]
    assert conversations._band_labels(pd.Series([2, 2]), [-inf, inf], "int", say) == ["2"]
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    share = pd.Series([0.1, 0.5, 0.9])
    assert conversations._band_labels(share, [-inf, 0.4, 0.8, inf], "share", pct) == [
        "\u2264 40%", "40\u201380%", "> 80%"]
    secs = pd.Series([10.0, 200.0, 5000.0])
    assert conversations._band_labels(secs, [-inf, 120, 600, inf], "secs", conversations.fmt_secs) == [
        "\u2264 2 min", "2\u201310 min", "> 10 min"]
    assert conversations._band_labels(share, [-inf, inf], "share", pct) == ["all"]
    assert conversations._band_labels(pd.Series([0.0, 0.0]), [-inf, 0, inf], "share", pct)[0] == "0%"


def test_higher_cost_needs_a_baseline_and_flags_above_the_percentile():
    est = pd.Series([1.0] * 99 + [100.0])
    out = conversations.higher_cost(est)
    assert out.high.tolist() == [False] * 99 + [True]
    assert (out.line == est.quantile(0.75)).all()


def test_higher_cost_below_min_baseline_flags_nothing():
    est = pd.Series([1.0] * 18 + [100.0])
    assert not conversations.higher_cost(est).high.any()


def test_higher_cost_is_per_principal_and_treats_missing_as_one_group():
    est = pd.Series([1.0] * 40 + [100.0, 100.0])
    by = pd.Series(["a"] * 20 + ["b"] * 20 + ["a", "b"])
    out = conversations.higher_cost(est, by)
    assert out.high.tolist() == [False] * 40 + [True, True]
    for who in "ab":
        assert (out.line[by == who] == est[by == who].quantile(0.75)).all()
    small = conversations.higher_cost(pd.Series([1.0, 9.0]), pd.Series([None, None]))
    assert not small.high.any()


def test_higher_cost_custom_percentile():
    est = pd.Series(range(1, 41), dtype=float)
    p50 = conversations.higher_cost(est, pct=50)
    assert p50.high.sum() == 20
    assert (p50.line == est.quantile(0.5)).all()


# ---- principals quoting --------------------------------------------------------
def test_sq_quotes_and_escapes():
    assert principals.sq("plain") == "'plain'"
    assert principals.sq("O'Brien") == "'O''Brien'"
    assert principals.sq(5) == "'5'"
    assert principals.sq("'; DROP TABLE x; --") == "'''; DROP TABLE x; --'"


@pytest.mark.parametrize("kind", ["user", "group", "content"])
def test_pred_is_evaluable_sql_that_cannot_be_escaped(kind):
    if kind not in principals.KEY_COL and kind != "group":
        pytest.skip("kind not a principal kind here")
    hostile = "x' OR '1'='1"
    con = duckdb.connect()
    con.execute("CREATE TABLE gateway_events(user_guid VARCHAR, username VARCHAR, content_guid VARCHAR)")
    con.execute("CREATE TABLE user_groups(user_guid VARCHAR, group_guid VARCHAR)")
    con.execute("INSERT INTO gateway_events VALUES ('u1','u1','c1'), ('u2','u2','c2')")
    con.execute("INSERT INTO user_groups VALUES ('u1', 'g1')")
    count = lambda key: con.execute(  # noqa: E731
        f"SELECT count(*) FROM gateway_events WHERE {principals.pred(kind, key)}").fetchall()[0][0]
    assert count(hostile) == 0
    expected = {"group": ("g1", 1)}.get(kind)
    if expected:
        assert count(expected[0]) == expected[1]


# ---- theme formatters -----------------------------------------------------------
@pytest.mark.parametrize("v, usd, num, card_u, card_n", [
    (None, "\u2013", "\u2013", "\u2013", "\u2013"),
    # An empty SUM or mean (NaN) and a ratio over zero (inf) are no figure, not "nan".
    (float("nan"), "\u2013", "\u2013", "\u2013", "\u2013"),
    (float("inf"), "\u2013", "\u2013", "\u2013", "\u2013"),
    (0, "$0.00", "0", "$0", "0"),
    # Spend under a cent is still spend; under a dollar the tile keeps the cents.
    (0.00004, "<$0.01", "0", "<$0.01", "0"),
    (0.42, "$0.42", "0", "$0.42", "0"),
    (12.345, "$12.35", "12", "$12", "12"),
    (99.994, "$99.99", "100", "$100", "100"),
    (100, "$100", "100", "$100", "100"),
    (9_999, "$9,999", "9,999", "$9,999", "9,999"),
    (10_000, "$10.0k", "10.0K", "$10,000", "10,000"),
    (1_234_567, "$1,234.6k", "1.2M", "$1,234,567", "1,234,567"),
])
def test_number_formatters(v, usd, num, card_u, card_n):
    assert theme.fmt_usd(v) == usd  # pyright: ignore[reportArgumentType]
    assert theme.fmt_num(v) == num  # pyright: ignore[reportArgumentType]
    assert theme.card_usd(v) == card_u  # pyright: ignore[reportArgumentType]
    assert theme.card_num(v) == card_n  # pyright: ignore[reportArgumentType]


def test_negative_numbers_use_magnitude_for_the_unit():
    assert theme.fmt_usd(-20_000) == "$-20.0k"
    assert theme.fmt_num(-2_000_000) == "-2.0M"


def test_fmt_pct_dashes_a_missing_share():
    assert theme.fmt_pct(12.345, 1) == "12.3%"
    assert theme.fmt_pct(float("nan")) == "\u2013"
    assert theme.fmt_pct(None) == "\u2013"


# ---- chart categories -------------------------------------------------------------
def test_null_origin_and_harness_are_named_buckets():
    import filters

    assert filters.dim_sql("source") == "COALESCE(source, '(unidentified)')"
    assert filters.dim_sql("tool") == "COALESCE(tool, '(no user agent)')"
    assert filters.dim_sql("model") == "model"  # other columns are left alone
    with duckdb.connect() as db:
        got = db.sql(f"SELECT {filters.dim_sql('source')} FROM "
                     "(VALUES ('connect'), (NULL)) t(source)").fetchall()
    assert [r[0] for r in got] == ["connect", "(unidentified)"]


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_unknown_buckets_are_gray(mode):
    muted = theme.TOKENS[mode]["muted"]
    for kind, name in [("origin", "(unidentified)"), ("tool", "(no user agent)")]:
        assert theme.color(kind, name, mode) == muted
    assert theme.color("origin", "connect", mode) != muted


def test_empty_fig_wraps_a_long_reason():
    fig = theme.empty_fig("dark", "word " * 40)
    assert "<br>" in fig.layout.annotations[0].text


# ---- conversations with nothing in view -------------------------------------------
def test_conv_groups_of_no_conversations_still_make_a_table():
    """A principal with no conversations in view (one day, say) crashed the table."""
    empty = pd.DataFrame(columns=["est", "high"])
    groups = conversations.conv_groups(empty, pd.Series(dtype=object), [])
    table = conversations.conv_group_table(groups)
    assert table.empty
    assert "higher_cost_conversations" in table.columns
    assert conversations.conv_donuts_fig(groups, "dark", 75).layout.annotations


# ---- principals.outliers ----------------------------------------------------------
def _peers(spend: list[float]) -> pd.DataFrame:
    n = len(spend)
    return pd.DataFrame({"spend": spend, "conv_p75": [1.0] * n, "peak_p75": [100.0] * n})


def test_outliers_flag_the_top_5_percent():
    why = principals.outliers(_peers(list(range(1, 41))))
    # p95 of 1..40 is 38.05: the top two are flagged.
    assert why.tolist() == [""] * 38 + ["high estimated spend"] * 2


def test_outliers_ignore_a_flat_field():
    why = principals.outliers(_peers([1.0] * 10))
    assert (why == "").all()


def test_outliers_need_enough_priced_peers():
    # Seven with spend, and zero-spend rows do not count towards the eight.
    why = principals.outliers(_peers([1, 1, 1, 1, 1, 1, 50, 0, 0]))
    assert (why == "").all()


# ---- filters.where: date edges ----------------------------------------------------
@pytest.fixture
def f() -> dict:
    return {
        "start": date(2026, 9, 1), "end": date(2026, 9, 30),
        "users": [], "all_users": [], "contents": [], "all_contents": [],
        "sessions": [], "groups": [], "meta": [],
    }


def _matching(predicate: str, stamps: list[str]) -> list[str]:
    con = duckdb.connect()
    con.execute("CREATE TABLE gateway_events(ts TIMESTAMP)")
    for s in stamps:
        con.execute("INSERT INTO gateway_events VALUES (?)", [s])
    rows = con.execute(f"SELECT strftime(ts, '%Y-%m-%d %H:%M:%S.%g') FROM gateway_events "
                       f"WHERE {predicate} ORDER BY ts").fetchall()
    return [r[0] for r in rows]


def test_where_includes_both_edge_days_entirely(f):
    stamps = ["2026-08-31 23:59:59.999", "2026-09-01 00:00:00", "2026-09-15 12:00:00",
              "2026-09-30 23:59:59.999", "2026-10-01 00:00:00"]
    got = _matching(where(f), stamps)
    assert len(got) == 3
    assert got[0].startswith("2026-09-01 00:00:00")
    assert got[-1].startswith("2026-09-30 23:59:59")


def test_where_single_day_window(f):
    f["start"] = f["end"] = date(2026, 9, 5)
    stamps = ["2026-09-04 23:59:59", "2026-09-05 00:00:00", "2026-09-05 23:59:59.5",
              "2026-09-06 00:00:00"]
    assert len(_matching(where(f), stamps)) == 2
