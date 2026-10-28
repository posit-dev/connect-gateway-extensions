"""Conversation grouping, computed from data frames with no Shiny session."""

import pandas as pd

import conversations


def test_grouping_by_model():
    df = pd.DataFrame({
        "est": [1.0] * 19 + [10.0],
        "main_model": ["small"] * 19 + ["large"],
        "high": [False] * 19 + [True],
        "duration_min": [1.0] * 20,
        "peak_input": [50_000.0] * 20,
    })
    key, order = conversations.conv_keys(df, "model")
    assert order == ["small", "large"]
    grouped = conversations.conv_groups(df, key, order)
    assert grouped.spend.tolist() == [19.0, 10.0]
    assert grouped.n_hi.tolist() == [0, 1]
    # A bar for all conversations, then one per model.
    data = conversations.conv_bars_fig(grouped, "dark", 90).to_dict()["data"]
    assert len(data[0]["y"]) == 3
    # The large model is all higher cost: 10 of its 10 dollars.
    assert data[0]["x"][2] == 1.0 and data[1]["x"][2] == 0.0


def _curve_frame(high: list[bool]) -> pd.DataFrame:
    n = len(high)
    return pd.DataFrame({
        "est": [10.0 if h else 1.0 for h in high],
        "high": high,
        "duration_min": [1.0] * n,
        # the costly conversations are the big ones
        "peak_input": [200_000.0 if h else 10_000.0 * (i + 1) for i, h in enumerate(high)],
        "miss_tokens": [0.0] * n,
        "uses_cache": [True] * n,
    })


def test_compare_rows_lead_with_the_split_then_peak_context():
    df = _curve_frame([False] * 15 + [True] * 5)
    rows = conversations.conv_compare_rows(df)
    assert [r["label"] for r in rows] == [
        "Conversation Count", "Total Spend", "Peak context", "Duration",
        "Using the prompt cache", "With a cache miss"]
    assert rows[0]["high"] == "5 \u00b7 25%" and rows[0]["normal"] == "15 \u00b7 75%"
    # peak context is 10k-150k against 200k; the other measures are the same
    peak = rows[2]
    assert "higher" in peak["gap"] and "Compact" in peak["hint"]
    assert rows[3]["gap"] == "about the same" and rows[3]["hint"] == ""


def test_compare_with_nothing_flagged_has_no_rows():
    assert conversations.conv_compare_rows(_curve_frame([False] * 5)) == []


def test_donut_for_an_unpriced_model_says_so():
    groups = pd.DataFrame({
        "label": ["priced", "gpt-5"], "spend": [4.0, 0.0], "n": [4, 3],
        "spend_hi": [1.0, 0.0], "n_hi": [1, 0], "spend_lo": [3.0, 0.0], "n_lo": [3, 3],
    })
    data = conversations.conv_bars_fig(groups, "light", 75).to_dict()["data"]
    assert any("unpriced" in t for tr in data for t in (tr["text"] or []))
