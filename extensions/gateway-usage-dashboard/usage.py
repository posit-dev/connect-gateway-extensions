"""One visitor's DuckDB copy of the Connect AI Gateway usage export.

The gateway serves a window of its usage log as a single Parquet file:

    GET <CONNECT_SERVER>/__api__/v1/system/gateway/usage/export
        ?from=<rfc3339>&to=<rfc3339>&as_of=<rfc3339>[&format=parquet|json]
    Authorization: Key <the visitor's API key>

The key must belong to an administrator. Every request is made with the key of
the person looking at the dashboard (see access.py), so Connect itself refuses
anyone else, and nothing is read before a visitor arrives. Each session builds
its own `Dataset`, which holds the data in its own in-memory DuckDB database
and drops it when the session ends. Nothing is shared between visitors or kept
on disk.

Connect sets `CONNECT_SERVER` for the content it runs. To run the app locally,
set it and `CONNECT_API_KEY` yourself; with no visitor, that key is used.

All three time parameters are required:

  - `from` and `to` select whole partitions, not individual records, and the
    server widens the range by one day at each end. This module therefore
    filters on `start_time` as well.
  - `as_of` is an exclusive upper bound on `ingest_time`. It pins the bytes
    behind the URL, so one window and watermark always name the same file.
    The server rejects an `as_of` newer than one minute ago, because a record
    can reach disk after the ingest time it carries.

The export is fetched with ONE plain GET into a local temporary file, and
DuckDB reads that file. It is not read over HTTP with httpfs: httpfs makes a
HEAD, a footer range request, then one range request per column chunk (about
30 for a 1.6M-row export), and retries any range answered with a 200 before
falling back to a full GET. On a server that converts the export on demand
(17-30 s for a large window) every one of those requests can pay the
conversion, so a single download is both faster and independent of how the
server handles Range and caching.

The server declares the column types, so the export is read with no schema
probe. Conversation content (`system_instructions`, `input_messages`,
`output_messages`) is never exported, and this dashboard never asks for it.

The export is read into the local `gateway_events` table, and every dashboard
query runs against local data.
"""

from __future__ import annotations

import csv
import http.client
import io
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
from shiny.session import Session, get_current_session

from filters import Query, sql_str

EXPORT_PATH = "__api__/v1/system/gateway/usage/export"
LOOKBACK_DAYS = 90

# How long the export GET may wait on the socket: for the server to finish
# converting (17-30 s for 1.6M rows) before the first byte, and between chunks.
EXPORT_TIMEOUT = 300

# How far behind the clock the watermark sits. The server demands at least one
# minute; two absorbs the skew between this process and Connect.
AS_OF_LAG = timedelta(minutes=2)


# Configuration is read from the environment when it is used, not at import,
# so importing this module has no side effects: no network, no env checks. The
# app calls load() itself (see app.py), and tests import the helpers without a
# Connect server.
def connect_server() -> str:
    """CONNECT_SERVER without a trailing slash, or "" when it is unset."""
    return (os.environ.get("CONNECT_SERVER") or "").rstrip("/")


def _check_config(key: str) -> None:
    """Refuse to load without the server and the key to read it with."""
    if not connect_server():
        raise RuntimeError(
            "CONNECT_SERVER is not set. Connect sets it for deployed content; "
            "set it yourself to run locally, e.g. http://localhost:3939"
        )
    if not key:
        raise RuntimeError(
            "CONNECT_API_KEY is not set. To run locally, set it to an "
            "administrator key; the export needs one"
        )


def _export_endpoint() -> str:
    return f"{connect_server()}/{EXPORT_PATH}"


# The gateway pricing endpoints share the export's base. `pricing` is the
# vendored litellm default catalog; `models` is the admin override table.
def _gateway_base() -> str:
    return f"{connect_server()}/__api__/v1/system/gateway"


def _api_base() -> str:
    return f"{connect_server()}/__api__/v1"

_RFC3339 = "%Y-%m-%dT%H:%M:%SZ"

def new_connection() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection set up the way the app's own is.

    The export carries TIMESTAMPTZ. Pin the session to UTC, or every cast to
    TIMESTAMP below shifts by the server's local offset and the "UTC" the
    charts claim is not the UTC they show.
    """
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


_MODEL_DATE_REV = re.compile(r"(-\d{8})?(-v\d+(:\d+)?)?$")

# The region (us., us-gov., global., ...) and vendor (anthropic., openai., ...)
# prefixes of a Bedrock model id: up to two leading segments of words that
# have only letters, joined by hyphens. A model name always has a digit, and it
# can have dots (openai.gpt-5.5), so the match stops before it.
_MODEL_PREFIX = re.compile(r"^(?:[a-z]+(?:-[a-z]+)*\.){0,2}")


def _model_family(raw: str) -> str:
    """Reduce a raw model id to the family that gateway_events.model carries.

    This must produce the same value as the `model` column in _SHAPE, so a
    catalogue price keyed by a dated, prefixed id still matches a request: drop
    an inference-region prefix and a vendor prefix (see _MODEL_PREFIX), then a
    trailing -YYYYMMDD date and a -vN[:N] revision.
    """
    return _MODEL_DATE_REV.sub("", _MODEL_PREFIX.sub("", raw))


def _get_json(url: str, key: str):
    """GET one gateway endpoint as JSON, or None if it is absent or fails.

    A server without the pricing endpoints answers 404, which arrives here as
    an HTTPError; every transport or decode failure returns None so that a
    missing price catalogue never blocks a data load.
    """
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Key {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _list_pages(url: str, page_size: int = 100, *, get) -> list | None:
    """Read a whole Connect list, or fail without returning a partial list.

    Some endpoints return bare arrays; others include results and total. Use
    total when supplied, since the server may cap the requested page size.
    `get` fetches one page URL as JSON.
    """
    items: list = []
    previous: list | None = None
    page = 1
    while True:
        body = get(f"{url}?page_size={page_size}&page_number={page}")
        rows = _results(body)
        if rows is None or (previous is not None and rows == previous and rows):
            return None  # failed page, or a server ignoring page_number
        if isinstance(body, dict) and body.get("current_page") not in (None, page):
            return None
        items.extend(rows)
        total = body.get("total") if isinstance(body, dict) else None
        if isinstance(total, int) and not isinstance(total, bool):
            if len(items) >= total:
                return items
            if not rows:
                return None
        elif len(rows) < page_size or (page == 1 and len(rows) > page_size):
            # More than requested means this endpoint returned an unpaged list.
            return items
        previous = rows
        page += 1


def _fetch_content_names(key: str) -> dict[str, str] | None:
    """guid -> title (else name) for every content item, or None on failure.

    An unexpected shape also returns None, and the load then shows guids rather
    than half-read titles.
    """
    items = _list_pages(f"{_api_base()}/content", get=lambda u: _get_json(u, key))
    if items is None:
        return None
    names: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("guid"):
            continue
        label = (item.get("title") or "").strip() or (item.get("name") or "").strip()
        if label:
            names[item["guid"]] = label
    return names


def _results(body) -> list | None:
    """The rows of a Connect list response: a bare list or {"results": [...]}."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        return body["results"]
    return None


def _fetch_groups(key: str) -> list[tuple[str, str, str]] | None:
    """Every (user_guid, group_guid, group_name) membership, or None on failure.

    The group list and each group's members are paged. A partial response is
    unusable for group totals, so return None rather than some of them.
    """
    def get(u: str):
        return _get_json(u, key)

    groups = _list_pages(f"{_api_base()}/groups", get=get)
    if groups is None:
        return None
    out = []
    for g in groups:
        if not isinstance(g, dict) or not g.get("guid"):
            continue
        members = _list_pages(f"{_api_base()}/groups/{g['guid']}/members", get=get)
        if members is None:
            return None
        for m in members:
            if isinstance(m, dict) and m.get("guid"):
                out.append((m["guid"], g["guid"], g.get("name") or g["guid"]))
    return out


def content_name(guid) -> str | None:
    """The content's title, or None when it is unknown (deleted, or no list)."""
    return current().content_names.get(str(guid))


def _rates(row) -> tuple | None:
    """The four $/Mtok rates from a catalog or override row, in cost order.

    Returns None when a rate is missing or not a number, so the caller can drop
    the row. The rates go into SQL text later, so a bad value must stop here.
    """
    try:
        return (
            float(row["input_usd_per_million_tokens"]),
            float(row["output_usd_per_million_tokens"]),
            float(row["cache_creation_usd_per_million_tokens"]),
            float(row["cache_read_usd_per_million_tokens"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


# Bedrock lists a separate price for each cross-region inference profile (us.,
# eu., apac., ...), a few percent above the base/global list rate. The catalog
# id and the raw request model both keep the region, so the app prices a request
# at its own region's rate. When a region has no listed price the app falls back
# to the base (un-prefixed) rate, else global -- see _catalog_prices.
_REGIONS = {"us", "eu", "apac", "au", "jp", "global", "us-gov", "ca", "sa", "me"}


def _region_rank(raw: str) -> int:
    """Preference for one catalog id when several collapse to one family.

    0 = no region prefix (the base list rate), 1 = global, 2 = a specific
    region. The lowest rank wins, so the "" default is the base list rate.
    """
    first = raw.split(".", 1)[0]
    if first not in _REGIONS:
        return 0
    return 1 if first == "global" else 2


def _model_region(raw: str) -> str:
    """The inference-region token of a catalog id, or "" when it has none.

    A Bedrock profile prefixes the id with a region (us., eu., ...); every other
    id has no prefix. Keep this in step with _region_rank and _REGIONS.
    """
    first = raw.split(".", 1)[0]
    return first if first in _REGIONS else ""


def _catalog_prices(pricing) -> dict:
    """Rate per (provider, region, family), from the catalog.

    `region` is the inference-region token (us, eu, ...) for a Bedrock profile,
    or "" for the base id. Each region keeps its own list rate. The "" entry is
    the default the app charges when a request's region has no price of its own:
    the base list rate, else global -- the lowest _region_rank, chosen so it
    never depends on catalog order. Dated snapshots of one region keep the first
    seen.
    """
    per_region: dict = {}  # (provider, region, family) -> rates, region != ""
    best: dict = {}        # (provider, family) -> (rank, rates), for the "" default
    for prov in pricing.get("providers", []):
        provider = prov.get("provider")
        for m in prov.get("models", []):
            model = m.get("model")
            rates = _rates(m)
            if provider is None or model is None or rates is None:
                continue
            family = _model_family(model)
            region = _model_region(model)
            if region:
                per_region.setdefault((provider, region, family), rates)
            rank = _region_rank(model)
            fam = (provider, family)
            if fam not in best or rank < best[fam][0]:
                best[fam] = (rank, rates)
    prices = dict(per_region)
    for (provider, family), (_rank, rates) in best.items():
        prices[(provider, "", family)] = rates
    return prices


def _fetch_prices(key: str) -> dict | None:
    """Merge the default catalog with the admin overrides, or return None.

    The catalog gives every model a default; the override table replaces the
    price for whichever (provider, model) it names. The server does not merge
    them, so this does. A missing catalog returns None; a missing override
    table just leaves the defaults in place. An unexpected shape also returns
    None: a non-numeric rate, a renamed or missing field, or a list where an
    object was expected. The caller decides what a missing catalog means.
    """
    pricing = _get_json(f"{_gateway_base()}/pricing", key)
    if not pricing:
        return None
    try:
        prices = _catalog_prices(pricing)
        overrides = _get_json(f"{_gateway_base()}/models", key)
        if overrides:
            for row in overrides.get("overrides", []):
                provider, model, rates = row.get("provider"), row.get("model"), _rates(row)
                if provider is None or model is None or rates is None:
                    continue
                prices[(provider, _model_region(model), _model_family(model))] = rates
    except (AttributeError, TypeError):
        return None
    return prices or None


def _window(now: datetime | None = None) -> tuple[str, str, str]:
    """The export window and its watermark, as RFC 3339 strings.

    GATEWAY_FROM and GATEWAY_TO pin the window; without them it trails the
    watermark. Everything is truncated to the minute: Connect caches a
    conversion for ten minutes under (from, to, as_of), so a second process,
    or a restart, inside the same minute reuses the build instead of paying
    for another one. A window that carried the clock's seconds never hit.

    The trailing window ends at the watermark. Nothing newer can be in the
    export anyway: as_of bounds ingest_time, and a call starts before it is
    ingested.
    """
    now = now or datetime.now(timezone.utc)
    as_of = (now - AS_OF_LAG).replace(second=0, microsecond=0)
    return (
        os.environ.get("GATEWAY_FROM")
        or (as_of - timedelta(days=LOOKBACK_DAYS)).strftime(_RFC3339),
        os.environ.get("GATEWAY_TO") or as_of.strftime(_RFC3339),
        as_of.strftime(_RFC3339),
    )


# The SELECT that shapes one downloaded export into gateway_events rows, wrapped
# in CREATE OR REPLACE TABLE gateway_events.
#
# Parameters: the file, and the window's start and end (on start_time).
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
SELECT
    start_time::TIMESTAMP                              AS ts,
    -- When Connect wrote the record, which is when the call finished.
    ingest_time,
    start_time::TIMESTAMP::DATE                        AS day,
    strftime(start_time::TIMESTAMP, '%Y-%m')           AS month,
    hour(start_time::TIMESTAMP)                        AS hour,
    dayofweek(start_time::TIMESTAMP) IN (0, 6)         AS is_weekend,
    node_name,
    provider,
    operation,
    -- Billed model, normalized to the family that PRICES is keyed by.
    --
    -- response_model is EMPTY -- not null -- on failed calls and on some
    -- successful ones, so NULLIF is what makes the fallback to request_model
    -- fire. Normalizing runs on whichever source wins, because the upstream
    -- dates BOTH: Anthropic answers claude-haiku-4-5-20251001, and a Bedrock
    -- profile asks for us.anthropic.claude-opus-4-1-20250805-v1:0. Both must
    -- reach claude-haiku-4-5 and claude-opus-4-1, or the price lookup misses
    -- and the call silently costs zero.
    --
    -- The inner regexp_replace drops the region and vendor prefixes (see
    -- _MODEL_PREFIX); the outer one drops a trailing -YYYYMMDD date and a
    -- -vN[:N] revision.
    regexp_replace(
        regexp_replace(
            COALESCE(NULLIF(response_model, ''), request_model),
            '^([a-z]+(-[a-z]+)*\\.){0,2}', ''),
        '(-\\d{8})?(-v\\d+(:\\d+)?)?$', ''
    )                                                  AS model,
    -- The exact strings the call carried, kept for audit evidence.
    request_model                                      AS request_model_raw,
    response_model                                     AS response_model_raw,
    -- The cross-region inference profile the caller invoked, when there is one.
    -- Bedrock prefixes the request model id with a region (us., eu., ...), which
    -- sets the list price, so est_cost prices by it. request_model names the
    -- profile; response_model can be empty on a failed call. Keep the prefix
    -- only when it is a known region. Keep this set in step with usage._REGIONS.
    CASE
        WHEN split_part(COALESCE(NULLIF(request_model, ''), response_model), '.', 1)
             IN ('us', 'eu', 'apac', 'au', 'jp', 'global', 'us-gov', 'ca', 'sa', 'me')
        THEN split_part(COALESCE(NULLIF(request_model, ''), response_model), '.', 1)
    END                                                AS region,
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
    -- ALL input, cached included: Connect records the provider's input plus
    -- cache_creation and cache_read for both Anthropic and Bedrock. Fresh
    -- (uncached) input is input_tokens - cache_creation - cache_read, which is
    -- how est_cost prices it.
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
    -- before we started storing conversation IDs this field was called
    -- session_id and was always empty. there should never be both a session_id
    -- and a conversation_id.
    COLUMNS('^(conversation_id|session_id)$')          AS conversation_id,
    -- Caller-supplied attribution (the Gateway-Metadata request header): a JSON
    -- object such as {"team":"data-science","workbench-session-id":"d1ac60de"}.
    -- Connect keeps at most 16 entries and drops any key sent twice. NULL when
    -- the request carried none, which is every Connect-content call and any row
    -- from a server that predates the field. The caller wrote the values and
    -- Connect does not verify them, so they label usage; they never authorise it.
    __REQUEST_METADATA__                               AS request_metadata,
    -- The keys worth reading at a glance, pulled out of the JSON. Keys are 1 to
    -- 32 lowercase letters, digits and hyphens, hence the quoted paths.
    json_extract_string(request_metadata, '$."team"')                   AS team,
    json_extract_string(request_metadata, '$."cost-center"')            AS cost_center,
    json_extract_string(request_metadata, '$."tier"')                   AS tier,
    json_extract_string(request_metadata, '$."enforced"')               AS enforced,
    json_extract_string(request_metadata, '$."workbench-session-id"')   AS workbench_session_id,
    json_extract_string(request_metadata, '$."workbench-session-name"') AS workbench_session_name,
    -- Workbench also names the project the session has open, when it has one.
    json_extract_string(request_metadata, '$."workbench-project-id"')   AS workbench_project_id,
    json_extract_string(request_metadata, '$."workbench-project-name"') AS workbench_project_name,
    json_extract_string(request_metadata, '$."workbench-project-path"') AS workbench_project_path,
    -- The caller's raw User-Agent (at most 512 bytes), kept as sent. The client
    -- writes it, so it labels usage and never authorises it. NULL on a server
    -- that predates the field.
    __USER_AGENT__                                     AS user_agent,
    -- Which tool made the call, read from the User-Agent; the first match wins.
    -- A blank header is NULL, so the tool charts leave it out rather than
    -- showing a bucket that says nothing. "API SDKs" is a provider SDK or HTTP
    -- library with no tool behind it, i.e. scripts and apps calling the API.
    CASE
        WHEN COALESCE(__USER_AGENT__, '') = ''         THEN NULL
        WHEN regexp_matches(__USER_AGENT__, '(?i)positassistant') THEN 'Posit Assistant'
        WHEN regexp_matches(__USER_AGENT__, '(?i)claude')  THEN 'Claude Code'
        WHEN regexp_matches(__USER_AGENT__, '(?i)copilot') THEN 'GitHub Copilot'
        WHEN regexp_matches(__USER_AGENT__, '(?i)codex')   THEN 'Codex CLI'
        WHEN regexp_matches(__USER_AGENT__, '(?i)(^|[\\s(;])pi([/-]|-coding-agent)') THEN 'Pi'
        WHEN regexp_matches(__USER_AGENT__, '(?i)r-ellmer') THEN 'R (ellmer)'
        WHEN regexp_matches(__USER_AGENT__,
             '(?i)^(openai|anthropic|azure-openai|google-genai|litellm)/|python-(requests|httpx)|aiohttp|^python/|^node|axios|undici|^curl/|^go-http-client|okhttp')
                                                       THEN 'API SDKs'
        ELSE 'Other'
    END                                                AS tool,
    -- Where the call came from: connect (deployed content), workbench or local
    -- (a Posit client naming its host IDE in the User-Agent), else NULL. Connect
    -- derives it at export time. An older server sends neither column, so
    -- there anything that is not content is "workbench", as it always was.
    __ORIGIN__                                         AS source
FROM read_parquet(?)
WHERE start_time >= ?::TIMESTAMPTZ
  AND start_time <  ?::TIMESTAMPTZ
"""


# What each refusal from the export route means for whoever opened the dashboard.
_ADVICE = {
    400: "the dashboard sent an invalid window or watermark",
    401: "the API key was refused",
    403: "the API key is not an administrator key",
    404: "this Connect server does not serve the gateway usage export",
    413: f"the window is too large to convert; lower LOOKBACK_DAYS "
    f"(now {LOOKBACK_DAYS}) or pin a shorter one with GATEWAY_FROM and "
    f"GATEWAY_TO",
    503: "Connect is already converting as many exports as it can; wait a "
    "moment and reload",
    504: "the conversion timed out; ask for a shorter window",
}


def _download(url: str, key: str) -> Path:
    """GET the export once and stream it into a temporary Parquet file.

    The caller deletes the file. The body is copied in 1 MB pieces, so the
    export never sits in Python memory whole. A refusal arrives as an ordinary
    HTTP status with the server's plain-text reason in the body, which is what
    the error message quotes.
    """
    req = urllib.request.Request(
        url, method="GET", headers={"Authorization": f"Key {key}"}
    )
    fd, name = tempfile.mkstemp(suffix=".parquet")
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            req, timeout=EXPORT_TIMEOUT
        ) as r:
            received = 0
            while chunk := r.read(1 << 20):
                out.write(chunk)
                received += len(chunk)
            # read(n) returns what arrived when the connection drops instead of
            # raising, so a cut-off body would pass for a whole file.
            expected = r.headers.get("Content-Length")
            if expected is not None and expected.isdigit() and received < int(expected):
                raise http.client.IncompleteRead(b"", int(expected) - received)
    except urllib.error.HTTPError as err:
        path.unlink(missing_ok=True)
        # Connect answers in plain text; a proxy in front of it may send an
        # HTML page instead, so flatten and cap whatever came back.
        reason = " ".join(err.read(2000).decode("utf-8", "replace").split())[:300]
        advice = _ADVICE.get(err.code, "check CONNECT_SERVER and CONNECT_API_KEY")
        detail = f"{err.code}: {reason}" if reason else str(err.code)
        raise RuntimeError(
            f"cannot read the usage export at {url} -- {advice} ({detail})"
        ) from err
    except (OSError, http.client.HTTPException) as err:
        # Unreachable host, timeout, or a connection cut mid-body
        # (IncompleteRead is an HTTPException, not an OSError).
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cannot read the usage export at {url} -- check CONNECT_SERVER, and "
            f"that the export finishes within {EXPORT_TIMEOUT}s ({err})"
        ) from err
    return path


def _shape_into(
    con: duckdb.DuckDBPyConnection,
    target: str, path: Path, window_from: str, window_to: str,
) -> None:
    """Run _SHAPE over one downloaded export into `target` (a CREATE ... AS)."""
    args = [str(path), window_from, window_to]
    # A Connect server that predates a column has none of it in its export.
    # Read what it has and fill the rest: first without origin and user agent,
    # then also without request metadata. Each step is tried only if the last
    # one failed on a column it lacks.
    steps = [
        {"__REQUEST_METADATA__": "request_metadata::JSON",
         "__USER_AGENT__": "user_agent::VARCHAR",
         "__ORIGIN__": "origin::VARCHAR"},
        {"__REQUEST_METADATA__": "request_metadata::JSON",
         "__USER_AGENT__": "NULL::VARCHAR",
         "__ORIGIN__": "CASE WHEN content_guid IS NOT NULL THEN 'connect' ELSE 'workbench' END"},
        {"__REQUEST_METADATA__": "NULL::JSON",
         "__USER_AGENT__": "NULL::VARCHAR",
         "__ORIGIN__": "CASE WHEN content_guid IS NOT NULL THEN 'connect' ELSE 'workbench' END"},
    ]
    for i, subs in enumerate(steps):
        sql = _SHAPE
        for key, val in subs.items():
            sql = sql.replace(key, val)
        try:
            con.execute(target + sql, args)
            return
        except duckdb.BinderException as e:
            # Move on when the missing column is one a later step fills in. The
            # binder names the first missing column in SELECT order, which is
            # request_metadata when an export lacks all three.
            cols = ("user_agent", "origin", "request_metadata") if i == 0 else ("request_metadata",)
            if i == len(steps) - 1 or not any(c in str(e) for c in cols):
                raise


def _count(table: str, con: duckdb.DuckDBPyConnection) -> int:
    """Rows in a table."""
    rows = con.sql(f"SELECT COUNT(*) FROM {table}").fetchall()
    return rows[0][0]  # COUNT(*) always returns one row


def _read_export(
    path: Path, window_from: str, window_to: str, con: duckdb.DuckDBPyConnection,
) -> int:
    """Replace gateway_events with one downloaded export; return its row count."""
    _shape_into(
        con, "CREATE OR REPLACE TABLE gateway_events AS ", path, window_from, window_to
    )
    return _count("gateway_events", con)


def _export_url(window_from: str, window_to: str, as_of: str) -> str:
    return (
        f"{_export_endpoint()}?from={window_from}&to={window_to}"
        f"&as_of={as_of}&format=parquet"
    )


def _ingest(url: str, key: str, read) -> int:
    """Download one export and hand the file to `read`; always delete it."""
    path = _download(url, key)
    try:
        return read(path)
    except (duckdb.IOException, duckdb.InvalidInputException) as e:
        # The body arrived but is not a Parquet file DuckDB can read.
        raise RuntimeError(
            f"the usage export at {url} is not a readable Parquet file ({e})"
        ) from e
    finally:
        path.unlink(missing_ok=True)


# The export, the price catalog, the content list and the groups are independent
# requests, so load() runs the small ones while the export is still converting.
# One pool serves every session; its jobs are short.
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="usage-load")

# How many sessions may load at once. Each load asks Connect to convert an
# export and holds a copy of it in memory, so a crowd of visitors waits its turn
# instead of all asking at the same moment. One that waits longer than LOAD_WAIT
# seconds gives up.
MAX_LOADS = 4
LOAD_WAIT = EXPORT_TIMEOUT
_loads = threading.BoundedSemaphore(MAX_LOADS)


class Dataset:
    """What one visitor's session reads from, fetched with that visitor's key."""

    def __init__(self, key: str):
        self._key = key
        self._lock = threading.Lock()
        # Set by close(), under the lock. A load still running then stops
        # touching the connection and keeps what it read to itself.
        self._closed = False
        self.con = new_connection()
        # Set by load(): when the table was read, the window it covers, and its size.
        self.info: dict = {}
        # Set by load(): token prices keyed by (provider, region, model), each
        # value a tuple of US dollars per million tokens (input, output, cache
        # write, cache read). The provider is the catalog name ("anthropic",
        # "aws.bedrock", "aws.bedrock_mantle"); the region is a Bedrock
        # inference-region token ("us", "eu", ...) or "" for the base rate; the
        # model is the family that gateway_events.model carries. There is no
        # built-in price list: load() fails if the catalog cannot be read.
        self.prices: dict | None = None
        # Set by load(): the human-readable name of each content item, keyed by
        # guid. The export carries only the guid. The name comes from the content
        # list: its title, else its name. Empty means the list could not be
        # read, and callers fall back to showing the guid.
        self.content_names: dict[str, str] = {}
        # Set by load(): Connect group memberships as (user_guid, group_guid,
        # group_name). A user can be in many groups, so anything summed by group
        # double-counts the people who are in more than one. Empty when the
        # groups API could not be read.
        self.group_members: list[tuple[str, str, str]] = []
        # The spend SQL pricing.py builds from `prices`, keyed by whether it is
        # the uncached figure. load() empties it whenever it sets new prices.
        self.cost_sql: dict[bool, str] = {}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.con.close()

    def load(self) -> dict:
        """Read the export, prices, content names and groups, and report what loaded.

        It raises before any request when CONNECT_SERVER or the key is unset,
        and when MAX_LOADS other loads have run for longer than LOAD_WAIT. A
        load that outlives close() returns without setting anything.
        """
        _check_config(self._key)
        if not _loads.acquire(timeout=LOAD_WAIT):
            raise RuntimeError(
                f"{MAX_LOADS} other sessions were still loading the usage data "
                f"after {LOAD_WAIT}s -- reload the page to try again"
            )
        try:
            return self._load()
        finally:
            _loads.release()

    def _load(self) -> dict:
        """load(), once it has its turn.

        The query lock is held only while DuckDB reads the downloaded file, not
        during the download.
        """
        window_from, window_to, as_of = _window()
        prices_f = _pool.submit(_fetch_prices, self._key)
        names_f = _pool.submit(_fetch_content_names, self._key)
        groups_f = _pool.submit(_fetch_groups, self._key)

        def read(path: Path) -> int:
            with self._lock:
                if self._closed:
                    return 0
                return _read_export(path, window_from, window_to, self.con)

        try:
            rows = _ingest(_export_url(window_from, window_to, as_of), self._key, read)
        except BaseException:
            for job in (prices_f, names_f, groups_f):
                job.cancel()  # those not started yet; the rest finish unread
            raise

        prices = prices_f.result()
        content_names = names_f.result() or {}
        group_members = groups_f.result() or []
        with self._lock:
            if self._closed:
                return self.info
            # With no catalog at all the load fails, as it does for the export:
            # spend has no other source.
            if prices is None:
                raise RuntimeError(
                    f"cannot read the gateway price catalog at {_gateway_base()}/pricing -- "
                    "check that this Connect server serves it and that your API key "
                    "may read it"
                )
            self.prices = prices
            self.cost_sql = {}
            self.content_names = content_names
            self.group_members = group_members
            # A table, so SQL (the Raw Data grid) can join the names in.
            self.con.execute(
                "CREATE OR REPLACE TABLE content_names "
                "(content_guid VARCHAR, content_name VARCHAR)"
            )
            if self.content_names:
                self.con.executemany(
                    "INSERT INTO content_names VALUES (?, ?)",
                    list(self.content_names.items()),
                )
            self.con.execute(
                "CREATE OR REPLACE TABLE user_groups "
                "(user_guid VARCHAR, group_guid VARCHAR, group_name VARCHAR)"
            )
            if self.group_members:
                self.con.executemany(
                    "INSERT INTO user_groups VALUES (?, ?, ?)", self.group_members
                )
        self.info = dict(
            loaded_at=datetime.now(timezone.utc),
            window_from=window_from,
            window_to=window_to,
            as_of=as_of,
            rows=rows,
            priced_models=len({(p, f) for (p, _region, f) in prices}),
        )
        return self.info

    def q(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a query and return its rows. `params` bind the `$name`s in the SQL."""
        with self._lock:
            return self.con.sql(sql, params=params).df()

    def scalar(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        df = self.q(sql, params)
        return None if df.empty else df.iloc[0, 0]

    def export_csv(self, query: Query) -> Path:
        """Write a query's full result to a temporary CSV and return its path.

        DuckDB writes the file itself, so a million-row export never has to sit
        in Python memory as a DataFrame. The caller streams the file and deletes it.
        """
        fd, name = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        path = Path(name)
        try:
            with self._lock:
                self.con.execute(
                    f"COPY ({query.sql}) TO {sql_str(name)} (FORMAT CSV, HEADER)",
                    query.params,
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path


# The session each page's queries belong to is Shiny's current session, which
# is set around every render and effect. Keyed by session id.
_datasets: dict[str, Dataset] = {}


def attach(session: Session, dataset: Dataset) -> None:
    """Make `dataset` the one this session's queries read, until the session ends."""
    _datasets[session.id] = dataset

    def release() -> None:
        _datasets.pop(session.id, None)
        dataset.close()

    session.on_ended(release)


def current() -> Dataset:
    """The current session's dataset. Raises outside a session with one attached."""
    session = get_current_session()
    dataset = _datasets.get(session.id) if session else None
    if dataset is None:
        raise RuntimeError("no dataset is attached to the current session")
    return dataset


def q(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    return current().q(sql, params)


def scalar(sql: str, params: dict[str, Any] | None = None) -> Any:
    return current().scalar(sql, params)


def rows(df: pd.DataFrame) -> Iterable[Any]:
    """df.itertuples(), typed so a row's columns read as attributes."""
    return df.itertuples()



def safe_csv(source: Iterable[str]) -> Iterable[str]:
    """Stream a CSV without letting text cells become spreadsheet formulas."""
    out = io.StringIO()
    writer = csv.writer(out)
    for row in csv.reader(source):
        writer.writerow([
            "'" + cell if cell.lstrip(" \t\r\n").startswith(("=", "+", "-", "@"))
            and not re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", cell)
            else cell
            for cell in row
        ])
        if out.tell() >= 1 << 16:
            yield out.getvalue()
            out.seek(0)
            out.truncate()
    if out.tell():
        yield out.getvalue()

