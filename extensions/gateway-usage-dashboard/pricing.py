"""Price gateway request tokens with the current Connect price catalog.

The SQL expressions operate on gateway_events rows. Each session's dataset keeps
the ones built from its own catalog, and drops them with the session.
"""

from __future__ import annotations

import usage

# The export writes the provider as "anthropic" or "bedrock"; the price catalog
# names Bedrock "aws.bedrock". Match either so a provider-keyed rate still fires.
_PROVIDER_ALIASES = {"anthropic": ("anthropic",), "aws.bedrock": ("bedrock", "aws.bedrock")}


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
    catalog does not price -- the model's Anthropic rate, then $0.
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


def _session_cost_sql(uncached: bool) -> str:
    """The session's spend SQL, built once from its catalog and kept on its dataset."""
    dataset = usage.current()
    sql = dataset.cost_sql.get(uncached)
    if sql is None:
        # Dataset.load() always sets the catalog before anything prices a row.
        assert dataset.prices is not None, "the dataset has not been loaded"
        sql = dataset.cost_sql[uncached] = _build_est_cost(dataset.prices, uncached)
    return sql


def est_cost_uncached() -> str:
    """est_cost with no prompt-cache discount (see _build_est_cost)."""
    return _session_cost_sql(uncached=True)


def est_cost() -> str:
    """Estimated spend per row, as a SQL expression over gateway_events.

    The price comes from (provider, region, model): the request's own region if
    it has a listed rate, else the provider's base rate, else the model's
    Anthropic rate, so a provider-name mismatch does not read $0.
    Dataset.load() refuses to finish without a catalog, so there is always one
    here.
    """
    return _session_cost_sql(uncached=False)
