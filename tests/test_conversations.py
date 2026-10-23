"""Conversation grouping, computed from data frames with no Shiny session."""

import pandas as pd

import conversations


def test_grouping_by_model_and_pace():
    df = pd.DataFrame({
        "est": [1.0] * 19 + [10.0],
        "main_model": ["small"] * 19 + ["large"],
        "high": [False] * 19 + [True],
        "calls": list(range(1, 21)),
        "duration_min": [1.0] * 20,
        "cache_miss": [0.0] * 20,
        "peak_input": [50_000.0] * 20,
        "med_idle": [None] + [3.0] * 19,
    })
    key, order = conversations.conv_keys(df, "model")
    assert order == ["small", "large"]
    grouped = conversations.conv_groups(df, key, order)
    assert grouped.spend.tolist() == [19.0, 10.0]
    assert grouped.n_hi.tolist() == [0, 1]
    table = conversations.conv_group_table(grouped)
    assert table.pct_of_spend.sum() == 100.0
    # A donut for all conversations, then one per model.
    data = conversations.conv_donuts_fig(grouped, "dark", 90).to_dict()["data"]
    assert len(data) == 3
    # The large model is all higher cost: 10 of its 10 dollars.
    assert list(data[2]["values"]) == [0.0, 10.0]
    assert "100%" in data[2]["title"]["text"] and "#e0a72c" in data[2]["title"]["text"]
    pace, pace_order = conversations.conv_keys(df, "pace")
    assert pace.iloc[0] == conversations.SINGLE
    assert pace_order[0] == conversations.SINGLE


def test_grouping_by_peak_context():
    df = pd.DataFrame({
        "est": [1.0] * 20,
        "high": [False] * 20,
        "calls": list(range(1, 21)),
        "duration_min": [1.0] * 20,
        "cache_miss": [0.0] * 20,
        "peak_input": [float(x) for x in range(10_000, 210_000, 10_000)],
    })
    key, order = conversations.conv_keys(df, "context")
    assert len(order) >= 2 and set(key) == set(order)
    assert order[-1].startswith(">") and order[-1].endswith("k")
    grouped = conversations.conv_groups(df, key, order)
    assert grouped.n.sum() == 20
    assert conversations.fmt_tokens(1_250_000) == "1.2M"


def _curve_frame(high: list[bool]) -> pd.DataFrame:
    n = len(high)
    return pd.DataFrame({
        "est": [10.0 if h else 1.0 for h in high],
        "high": high,
        "calls": list(range(1, n + 1)),
        "duration_min": [1.0] * n,
        "cache_miss": [5_000.0] * n,
        # the costly conversations are the big ones
        "peak_input": [200_000.0 if h else 10_000.0 * (i + 1) for i, h in enumerate(high)],
        "med_idle": [0.0] + [3.0] * (n - 1),
    })


def test_cumulative_curves_split_higher_cost_from_the_rest():
    df = _curve_frame([False] * 15 + [True] * 5)
    fig = conversations.conv_cumulative_fig(df, "context", "dark", 75)
    d = fig.to_dict()
    lines = {tr["name"]: tr for tr in d["data"]}
    assert set(lines) == {"normal", "higher cost (above p75)"}
    for tr in lines.values():
        # each line runs from 0 to all of its own spend
        assert tr["y"][0] == 0 and tr["y"][-1] == 100
        assert list(tr["y"]) == sorted(tr["y"])
    # every higher-cost conversation sits at 200k, so its line is one step there
    assert set(lines["higher cost (above p75)"]["x"]) == {200_000.0}
    assert d["layout"]["xaxis"]["type"] == "log"


def test_cumulative_curve_is_one_line_with_nothing_flagged():
    fig = conversations.conv_cumulative_fig(_curve_frame([False] * 5), "pace", "light", 75)
    data = fig.to_dict()["data"]
    assert [tr["name"] for tr in data] == ["all conversations"]
    # a pause of 0 s is kept, at the floor, rather than dropped off the log axis
    assert data[0]["customdata"][-1] == 5


def test_donut_for_an_unpriced_model_says_so():
    groups = pd.DataFrame({
        "label": ["priced", "gpt-5"], "spend": [4.0, 0.0], "n": [4, 3],
        "spend_hi": [1.0, 0.0], "n_hi": [1, 0], "spend_lo": [3.0, 0.0], "n_lo": [3, 3],
    })
    data = conversations.conv_donuts_fig(groups, "light", 75).to_dict()["data"]
    assert "unpriced" in data[2]["title"]["text"]


def test_cumulative_curve_with_no_spend_is_empty():
    df = _curve_frame([False] * 5).assign(est=0.0)
    fig = conversations.conv_cumulative_fig(df, "cache", "dark", 75)
    assert not fig.data  # pyright: ignore[reportArgumentType]
