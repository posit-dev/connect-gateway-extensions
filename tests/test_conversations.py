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
        "cache_hit": [0.0] * 20,
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
    fig = conversations.conv_spend_fig(grouped, "model", "dark", 90)
    assert len(fig.data) == 2  # pyright: ignore[reportArgumentType]
    pace, pace_order = conversations.conv_keys(df, "pace")
    assert pace.iloc[0] == conversations.SINGLE
    assert pace_order[0] == conversations.SINGLE


def test_grouping_by_peak_context():
    df = pd.DataFrame({
        "est": [1.0] * 20,
        "high": [False] * 20,
        "calls": list(range(1, 21)),
        "duration_min": [1.0] * 20,
        "cache_hit": [0.0] * 20,
        "peak_input": [float(x) for x in range(10_000, 210_000, 10_000)],
    })
    key, order = conversations.conv_keys(df, "context")
    assert len(order) >= 2 and set(key) == set(order)
    assert order[-1].startswith(">") and order[-1].endswith("k")
    grouped = conversations.conv_groups(df, key, order)
    assert grouped.n.sum() == 20
    assert conversations.fmt_tokens(1_250_000) == "1.2M"
