"""The spend SQL built from a price catalog, evaluated by DuckDB."""

import duckdb
import pytest

import pricing

# A model name with a quote in it, priced three ways: Anthropic's base rate,
# Bedrock's base rate, and Bedrock's us. inference-profile rate.
PRICES = {
    ("anthropic", "", "o'hare"): (2, 10, 3, 0.5),
    ("aws.bedrock", "", "o'hare"): (4, 12, 5, 1),
    ("aws.bedrock", "us", "o'hare"): (6, 14, 7, 2),
}

# 1M input tokens of which 100k are cache writes and 200k cache reads (so 700k
# fresh), and 200k output tokens.
EVENTS = """(VALUES
    ('anthropic', '', 'o''hare', 1000000, 200000, 100000, 200000),
    ('bedrock', 'us', 'o''hare', 1000000, 200000, 100000, 200000),
    ('bedrock', 'eu', 'o''hare', 1000000, 200000, 100000, 200000),
    ('other', '', 'o''hare', 1000000, 200000, 100000, 200000),
    ('other', '', 'unknown', 1000000, 200000, 100000, 200000)
) AS events(provider, region, model, input_tokens, output_tokens,
            cache_creation_tokens, cache_read_tokens)"""


@pytest.fixture(scope="module")
def priced() -> list[tuple]:
    """(billed, uncached) for each EVENTS row."""
    with duckdb.connect() as con:
        return con.sql(f"""SELECT
            {pricing._build_est_cost(PRICES)} AS billed,
            {pricing._build_est_cost(PRICES, uncached=True)} AS uncached
            FROM {EVENTS}""").fetchall()


def test_price_ladder(priced):
    # anthropic base; bedrock us rate; bedrock eu falls back to bedrock's base;
    # an unpriced provider falls back to the model's Anthropic rate; an
    # unpriced model costs nothing.
    assert [round(r[0], 2) for r in priced] == [3.8, 8.1, 5.9, 3.8, 0]


def test_uncached_charges_cache_tokens_at_the_input_rate(priced):
    assert [round(r[1], 2) for r in priced] == [4.0, 8.8, 6.4, 4.0, 0]
