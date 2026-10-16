"""Price gateway request tokens with the current Connect price catalog.

The SQL expressions operate on gateway_events rows. A refresh replaces
usage.PRICES, which invalidates the expression caches by object identity.
"""

from __future__ import annotations

import usage

# The export writes the provider as "anthropic" or "bedrock"; the price catalog
# names Bedrock "aws.bedrock". Match either so a provider-keyed rate still fires.
_PROVIDER_ALIASES = {"anthropic": ("anthropic",), "aws.bedrock": ("bedrock", "aws.bedrock")}


def _prices() -> dict:
    """usage.PRICES, which usage.load() always sets before anything prices a row."""
    prices = usage.PRICES
    assert prices is not None, "usage.load() has not run"
    return prices


def _rate_term(i: float, o: float, cw: float, cr: float) -> str:
    """The spend for one row at these four $/Mtok rates, before the /1e6.

    float() coerces each rate, so a string or null price cannot enter the SQL as
    a bare token.
    """
    return (
        f"((input_tokens - cache_read_tokens - cache_creation_tokens) * {float(i)}"
        f" + cache_creation_tokens * {float(cw)} + cache_read_tokens * {float(cr)}"
        f" + output_tokens * {float(o)})"
    )


def _sql_str(s: str) -> str:
    """One value as a SQL string literal, with embedded quotes doubled.

    Provider and model names come from the pricing endpoint, so the CASE text
    must not assume they are quote-free.
    """
    return "'" + str(s).replace("'", "''") + "'"


def _build_est_cost(prices: dict, uncached: bool = False) -> str:
    """The est_cost SQL for one price catalog.

    `uncached` prices the same traffic as if no prompt cache existed: cache
    reads and writes are charged at the fresh-input rate. The difference from
    the real figure is what caching saved (net of any write premium).
    """

    def term(i, o, cw, cr):
        return _rate_term(i, o, i, i) if uncached else _rate_term(i, o, cw, cr)

    return _build_case(prices, term)


def _build_case(prices: dict, term) -> str:
    """A CASE over the catalog that yields `term(i, o, cw, cr)` for each row, /1e6.

    The CASE tries the most specific branch first: the request's own region,
    then the provider's base rate for the model, then -- for a provider the
    catalog does not price -- the model's Anthropic rate, then $0. est_cost and
    the per-part rates (rate_sql) share this ladder, so the parts always add up
    to the estimate.
    """
    region_whens, base_whens, model_default = [], [], {}

    for (provider, region, model), (i, o, cw, cr) in prices.items():
        names = ", ".join(
            _sql_str(a) for a in _PROVIDER_ALIASES.get(provider, (provider,))
        )
        if region:
            region_whens.append(
                f"WHEN provider IN ({names}) AND region = {_sql_str(region)}"
                f" AND model = {_sql_str(model)} THEN {term(i, o, cw, cr)}"
            )
        else:
            base_whens.append(
                f"WHEN provider IN ({names}) AND model = {_sql_str(model)}"
                f" THEN {term(i, o, cw, cr)}"
            )
            if provider == "anthropic":
                model_default[model] = (i, o, cw, cr)
    # Model-only fallback, tried after every provider-keyed branch.
    fallback = [
        f"WHEN model = {_sql_str(m)} THEN {term(*r)}"
        for m, r in model_default.items()
    ]
    whens = region_whens + base_whens + fallback
    return f"CASE {' '.join(whens)} ELSE 0 END / 1e6"


# (prices object the SQL was built from, the SQL). The cache holds the object
# itself, not id(prices): a freed catalog can pass its id() to a new object and
# cause a wrong hit, but an `is` test on the object cannot.
_EST_COST_CACHE: tuple = (object(), "")
_NOCACHE_COST_CACHE: tuple = (object(), "")
_RATES_CACHE: tuple = (object(), {})

# The four prices a request pays, in the order the catalog stores them.
RATE_PARTS = ("i", "o", "cw", "cr")


def rate_sql() -> dict[str, str]:
    """USD per token for each part (fresh input, output, cache write, cache read).

    Each is a SQL expression over provider, region and model only, so it can be
    evaluated once per distinct model rather than once per row.
    """
    global _RATES_CACHE
    prices = _prices()
    if _RATES_CACHE[0] is not prices:
        _RATES_CACHE = (
            prices,
            {
                p: _build_case(prices, lambda *r, k=k: f"{float(r[k])}")
                for k, p in enumerate(RATE_PARTS)
            },
        )
    return _RATES_CACHE[1]


def model_rates(model: str) -> tuple | None:
    """(input, output, cache write, cache read) in USD per token for a model's
    base list rate, Anthropic's first, or None when the catalog has no price."""
    hits = {p: r for (p, region, m), r in _prices().items() if m == model and not region}
    rates = hits.get("anthropic") or next(iter(hits.values()), None)
    return tuple(v / 1e6 for v in rates) if rates else None


def est_cost_uncached() -> str:
    """est_cost with no prompt-cache discount (see _build_est_cost)."""
    global _NOCACHE_COST_CACHE
    prices = _prices()
    if _NOCACHE_COST_CACHE[0] is not prices:
        _NOCACHE_COST_CACHE = (prices, _build_est_cost(prices, uncached=True))
    return _NOCACHE_COST_CACHE[1]


def est_cost() -> str:
    """Estimated spend per row, as a SQL expression over gateway_events.

    The SQL is rebuilt only when the catalog changes. A refresh replaces
    usage.PRICES with a new object, which misses the cache and forces one
    rebuild; every call in between reuses the string. With a catalog, the price
    comes from (provider, region, model): the request's own region if it has a
    listed rate, else the provider's base rate, else the model's Anthropic rate,
    so a provider-name mismatch does not read $0. usage.load() refuses to start
    without a catalog, so there is always one here.
    """
    global _EST_COST_CACHE
    prices = _prices()
    cached_prices, cached_sql = _EST_COST_CACHE
    if cached_prices is prices:
        return cached_sql
    sql = _build_est_cost(prices)
    _EST_COST_CACHE = (prices, sql)
    return sql
