"""Persistent DuckDB connection to the Connect AI Gateway usage export.

The gateway serves a window of its usage log as a single Parquet file:

    GET <CONNECT_SERVER>/__api__/v1/system/gateway/usage/export
        ?from=<rfc3339>&to=<rfc3339>&as_of=<rfc3339>[&format=parquet|json]
    Authorization: Key <CONNECT_API_KEY>

The key must belong to an administrator. Connect sets `CONNECT_SERVER` and
`CONNECT_API_KEY` for the content it runs, so a deployed copy needs no
configuration. Set both yourself to run the app locally.

All three time parameters are required:

  - `from` and `to` select whole partitions, not individual records, and the
    server widens the range by one day at each end. This module therefore
    filters on `start_time` as well.
  - `as_of` is an exclusive upper bound on `ingest_time`. It pins the bytes
    behind the URL, which matters because DuckDB reads one Parquet file with
    several range requests: the footer first, then the column chunks. Without
    a watermark the newest partition can grow between those requests and the
    query returns a plausible wrong answer. The server rejects an `as_of`
    newer than one minute ago, because a record can reach disk after the
    ingest time it carries.

The server declares the column types, so the export is read with no schema
probe. Conversation content (`system_instructions`, `input_messages`,
`output_messages`) is never exported, and this dashboard never asks for it.

The export is read one time, at start, into the local `gateway_events` table.
Every dashboard query then runs against local data. Restart the app, or press
"Refresh data", to read a newer window.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

CONNECT_SERVER = os.environ.get("CONNECT_SERVER")
CONNECT_API_KEY = os.environ.get("CONNECT_API_KEY")
EXPORT_PATH = "__api__/v1/system/gateway/usage/export"
LOOKBACK_DAYS = 90

# How far behind the clock the watermark sits. The server demands at least one
# minute; two absorbs the skew between this process and Connect.
AS_OF_LAG = timedelta(minutes=2)

if not CONNECT_SERVER:
    raise RuntimeError(
        "CONNECT_SERVER is not set. Connect sets it for deployed content; "
        "set it yourself to run locally, e.g. http://localhost:3939"
    )
if not CONNECT_API_KEY:
    raise RuntimeError(
        "CONNECT_API_KEY is not set. Connect sets it for deployed content; "
        "the export needs an administrator key"
    )

_EXPORT_ENDPOINT = f"{CONNECT_SERVER.rstrip('/')}/{EXPORT_PATH}"
# The gateway pricing endpoints share the export's base. `pricing` is the
# vendored litellm default catalog; `models` is the admin override table.
_GATEWAY_BASE = f"{CONNECT_SERVER.rstrip('/')}/__api__/v1/system/gateway"
_RFC3339 = "%Y-%m-%dT%H:%M:%SZ"

_lock = threading.Lock()
_con = duckdb.connect()
# The export carries TIMESTAMPTZ. Pin the session to UTC, or every cast to
# TIMESTAMP below shifts by the server's local offset and the "UTC" the charts
# claim is not the UTC they show.
_con.execute("SET TimeZone='UTC'")
_con.execute("INSTALL httpfs; LOAD httpfs;")
_con.execute(
    """CREATE OR REPLACE SECRET gateway (
           TYPE http, SCOPE ?, EXTRA_HTTP_HEADERS MAP {'Authorization': ?})""",
    [CONNECT_SERVER, f"Key {CONNECT_API_KEY}"],
)

# Set by load(): when the table was read, the window it covers, and its size.
INFO: dict = {}

# Set by load(): token prices keyed by (provider, model), each value a tuple of
# US dollars per million tokens (input, output, cache write, cache read). The
# provider is the catalog name ("anthropic", "aws.bedrock"); the model is the
# family that gateway_events.model carries. None means the server does not serve
# the pricing endpoint, or the fetch failed; the caller then falls back to its
# own list-price table.
PRICES: dict | None = None

_MODEL_DATE_REV = re.compile(r"(-\d{8})?(-v\d+(:\d+)?)?$")


def _model_family(raw: str) -> str:
    """Reduce a raw model id to the family that gateway_events.model carries.

    This must produce the same value as the `model` column in _SHAPE, so a
    catalogue price keyed by a dated, prefixed id still matches a request: drop
    any prefix up to the last dot, then a trailing -YYYYMMDD date and a -vN[:N]
    revision.
    """
    after_dot = raw.rsplit(".", 1)[-1]
    return _MODEL_DATE_REV.sub("", after_dot)


def _get_json(url: str):
    """GET one gateway endpoint as JSON, or None if it is absent or fails.

    A server without the pricing endpoints answers 404, which arrives here as
    an HTTPError; every transport or decode failure returns None so that a
    missing price catalogue never blocks a data load.
    """
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Key {CONNECT_API_KEY}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _rows_to_prices(rows) -> dict:
    """Key override or catalog rows by (provider, model family)."""
    return {
        (row["provider"], _model_family(row["model"])): (
            row["input_usd_per_million_tokens"],
            row["output_usd_per_million_tokens"],
            row["cache_creation_usd_per_million_tokens"],
            row["cache_read_usd_per_million_tokens"],
        )
        for row in rows
    }


def _fetch_prices() -> dict | None:
    """Merge the default catalog with the admin overrides, or return None.

    The catalog gives every model a default; the override table replaces the
    price for whichever (provider, model) it names. The server does not merge
    them, so this does. A missing catalog returns None; a missing override
    table just leaves the defaults in place.
    """
    pricing = _get_json(f"{_GATEWAY_BASE}/pricing")
    if not pricing:
        return None
    prices = {}
    for prov in pricing.get("providers", []):
        prices.update(
            _rows_to_prices(
                dict(m, provider=prov["provider"]) for m in prov.get("models", [])
            )
        )
    overrides = _get_json(f"{_GATEWAY_BASE}/models")
    if overrides:
        prices.update(_rows_to_prices(overrides.get("overrides", [])))
    return prices or None


def _window() -> tuple[str, str, str]:
    """The export window and its watermark, as RFC 3339 strings.

    GATEWAY_FROM and GATEWAY_TO pin the window; without them it trails the
    clock. The watermark is always derived, and truncated to the minute so
    that repeated reads of one window hit the server's conversion cache.
    """
    now = datetime.now(timezone.utc)
    as_of = (now - AS_OF_LAG).replace(second=0, microsecond=0)
    return (
        os.environ.get("GATEWAY_FROM")
        or (now - timedelta(days=LOOKBACK_DAYS)).strftime(_RFC3339),
        os.environ.get("GATEWAY_TO") or now.strftime(_RFC3339),
        as_of.strftime(_RFC3339),
    )


# The statement that reads the export into the local table.
#
# CREATE OR REPLACE, so a failed reload rolls back and leaves the data that is
# already loaded in place.
#
# Three identities can be attached to one call, and they are independent of the
# content axis:
#
#   authenticated_*  the person whose credentials authenticated the request: a
#                    Workbench session's user, or the viewer a Viewer
#                    credential was issued to. Null when the principal is not a
#                    person, such as content's own ephemeral key.
#   viewer_*         the viewer a request from content was made for. It equals
#                    the authenticated user when a Viewer credential was used.
#   content_owner_*  the owner of the content that made the call. Null for a
#                    caller that is not content.
#
# The username beside each GUID comes from a lookup that Connect performs at
# write time, and that an administrator can disable. A GUID with no name is
# therefore normal, so `username` falls back to the GUID rather than dropping
# the request out of the rankings.
#
# Connect writes an email beside each name. On a server whose usernames are
# already email addresses the two columns hold the same value.
_SHAPE = """
CREATE OR REPLACE TABLE gateway_events AS
SELECT
    start_time::TIMESTAMP                              AS ts,
    start_time::TIMESTAMP::DATE                        AS day,
    strftime(start_time::TIMESTAMP, '%Y-%m')           AS month,
    hour(start_time::TIMESTAMP)                        AS hour,
    dayofweek(start_time::TIMESTAMP) IN (0, 6)         AS is_weekend,
    node_name,
    provider,
    operation,
    -- Billed model, normalized to the family that EST_PRICES is keyed by.
    --
    -- response_model is EMPTY -- not null -- on failed calls and on some
    -- successful ones, so NULLIF is what makes the fallback to request_model
    -- fire. Normalizing runs on whichever source wins, because the upstream
    -- dates BOTH: Anthropic answers claude-haiku-4-5-20251001, and a Bedrock
    -- profile asks for us.anthropic.claude-opus-4-1-20250805-v1:0. Both must
    -- reach claude-haiku-4-5 and claude-opus-4-1, or the price lookup misses
    -- and the call silently costs zero.
    --
    -- regexp_extract drops any inference-profile prefix (everything up to the
    -- last dot); regexp_replace drops a trailing -YYYYMMDD date and a -vN[:N]
    -- revision.
    regexp_replace(
        regexp_extract(
            COALESCE(NULLIF(response_model, ''), request_model),
            '([^.]+)$', 1),
        '(-\\d{8})?(-v\\d+(:\\d+)?)?$', ''
    )                                                  AS model,
    -- The exact strings the call carried, kept for audit evidence.
    request_model                                      AS request_model_raw,
    response_model                                     AS response_model_raw,
    integration_guid,
    -- How the call reached the provider. integration_auth_type is the flow the
    -- integration is configured for ("Viewer", "Service Account", "Environment
    -- Variables"), and it is what explains which identity columns are filled.
    integration_type,
    integration_auth_type,
    duration_ms,
    status_code,
    finish_reasons[1]                                  AS finish_reason,
    response_id,
    server_address,
    server_port,
    input_tokens,
    output_tokens,
    cache_creation_input_tokens                        AS cache_creation_tokens,
    cache_read_input_tokens                            AS cache_read_tokens,
    -- The three identities, kept apart for the audit record table.
    authenticated_username,
    viewer_username,
    content_owner_username,
    -- The principal this call is charged to. The ladder runs on the GUIDs,
    -- which Connect always writes, and the name follows the branch the GUID
    -- picked so that one identity's name is never read beside another's.
    CASE WHEN viewer_user_guid IS NOT NULL        THEN 'viewer'
         WHEN authenticated_user_guid IS NOT NULL THEN 'authenticated'
         WHEN content_owner_guid IS NOT NULL      THEN 'owner'
    END                                                AS user_kind,
    CASE user_kind WHEN 'viewer'        THEN viewer_user_guid
                   WHEN 'authenticated' THEN authenticated_user_guid
                   WHEN 'owner'         THEN content_owner_guid
    END                                                AS user_guid,
    CASE user_kind WHEN 'viewer'        THEN viewer_email
                   WHEN 'authenticated' THEN authenticated_email
                   WHEN 'owner'         THEN content_owner_email
    END                                                AS user_email,
    COALESCE(
        CASE user_kind WHEN 'viewer'        THEN viewer_username
                       WHEN 'authenticated' THEN authenticated_username
                       WHEN 'owner'         THEN content_owner_username
        END,
        user_guid
    )                                                  AS username,
    -- content_guid is the axis the dashboard groups Connect traffic by. The
    -- export carries no content name, and an earlier version labelled items
    -- 'content #<id>', which named nothing an operator could look up. The guid
    -- is the identifier Connect itself uses, so it pastes straight into
    -- /connect/#/apps/<guid>.
    content_guid,
    content_id,
    -- job_key names the process behind the call. It is set only for content
    -- that authenticated with an ephemeral key, so it is null for a request a
    -- person made and for self-authenticating callers such as Bedrock.
    job_key,
    session_id                                         AS conversation_id,
    -- Where the call came from. The content fields are null for a request that
    -- did not originate in deployed content, which is what marks a Workbench
    -- session or a direct API caller.
    CASE WHEN content_guid IS NOT NULL THEN 'connect' ELSE 'workbench' END
                                                       AS source
FROM read_parquet(?)
WHERE start_time >= ?::TIMESTAMPTZ
  AND start_time <  ?::TIMESTAMPTZ
"""


# What each refusal from the export route means for whoever deployed this app.
_ADVICE = {
    400: "the dashboard sent an invalid window or watermark",
    401: "CONNECT_API_KEY was refused",
    403: "CONNECT_API_KEY is not an administrator key",
    404: "this Connect server does not serve the gateway usage export",
    413: f"the window is too large to convert; lower LOOKBACK_DAYS "
    f"(now {LOOKBACK_DAYS}) or pin a shorter one with GATEWAY_FROM and "
    f"GATEWAY_TO",
    503: "Connect is already converting as many exports as it can; wait a "
    "moment and refresh",
    504: "the conversion timed out; ask for a shorter window",
}


def _probe(url: str) -> tuple[int | None, str]:
    """Ask the endpoint directly what it objects to.

    httpfs reports 400, 401, 403 and 413 as "HTTP 0" and never keeps the
    response body, so the three failures an operator is most likely to hit --
    a refused key, a key that is not an administrator's, and a window too
    large -- all arrive indistinguishable. The route answers every refusal
    with a plain-text reason, so one small request recovers both the status
    and the server's own words. It runs only on the error path.

    Range caps the reply at one byte in case the endpoint answers after all.
    """
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Key {CONNECT_API_KEY}", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, ""
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace").strip()
    except OSError:
        # The endpoint is unreachable, which the caller's error already says.
        return None, ""


def _explain(url: str, e: duckdb.Error) -> RuntimeError:
    """Turn a transport failure into the operator's next move."""
    status = getattr(e, "status_code", None) or None
    reason = ""
    if status is None or status not in _ADVICE:
        status, reason = _probe(url)
    advice = _ADVICE.get(
        status,
        "check CONNECT_SERVER, CONNECT_API_KEY, and "
        "that the endpoint serves range requests",
    )
    detail = f"{status}: {reason}" if reason else (str(e).splitlines() or [""])[0]
    return RuntimeError(f"cannot read the usage export at {url} -- {advice} ({detail})")


def load() -> dict:
    """Read the export into the local table, and report what was loaded.

    Called at import and again for every refresh. A failure leaves the data
    already loaded in place, so a refresh that cannot reach the endpoint is
    not destructive.
    """
    window_from, window_to, as_of = _window()
    url = (
        f"{_EXPORT_ENDPOINT}?from={window_from}&to={window_to}"
        f"&as_of={as_of}&format=parquet"
    )
    try:
        with _lock:
            _con.execute(_SHAPE, [url, window_from, window_to])
            rows = _con.sql("SELECT COUNT(*) FROM gateway_events").fetchone()[0]
    except (duckdb.HTTPException, duckdb.IOException) as e:
        # httpfs reports an unreachable host, a refused key and a server-side
        # refusal all as transport errors.
        raise _explain(url, e) from e

    # Fetched outside the lock: it is network I/O, and the assignment is atomic.
    # A failure leaves PRICES as it was, so the caller keeps the last good
    # catalog rather than dropping to its fallback on one bad refresh.
    global PRICES
    fetched = _fetch_prices()
    if fetched is not None:
        PRICES = fetched

    INFO.update(
        loaded_at=datetime.now(timezone.utc),
        window_from=window_from,
        window_to=window_to,
        as_of=as_of,
        rows=rows,
        priced_models=None if PRICES is None else len(PRICES),
    )
    return INFO


load()


def q(sql: str) -> pd.DataFrame:
    with _lock:
        return _con.sql(sql).df()


def scalar(sql: str):
    df = q(sql)
    return None if df.empty else df.iloc[0, 0]
