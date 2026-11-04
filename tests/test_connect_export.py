"""The app against exports Connect itself built (tests/fixtures, `just connect-fixtures`).

The records behind the fixtures are in tests/fixtures/connect_export_fixture_test.go;
each one's response_id names the case it covers.
"""

import json
import urllib.request

import pandas as pd
import pytest

import pricing
import usage
from conftest import (
    CONNECT_EXPORT, EXPORT_SCHEMA, ROOT, WINDOW_FROM, WINDOW_TO, parquet_schema,
)


@pytest.fixture
def events(con) -> pd.DataFrame:
    """gateway_events from the full Connect export, indexed by response_id."""
    usage._read_export(CONNECT_EXPORT, WINDOW_FROM, WINDOW_TO, con)
    return con.sql("SELECT * FROM gateway_events").df().set_index("response_id")


def value(v):
    return None if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else v


def test_every_record_in_the_window_loads(events):
    # before-window started on 08-31, so the start_time filter drops it even
    # though its partition is in the window.
    assert sorted(events.index) == [
        "after-early", "claude-code", "failed-call", "late-ingest", "local-assistant",
        "near-cutoff", "no-name", "old-record", "service-content", "viewer-content",
        "workbench-assistant",
    ]


# response_id -> the gateway_events columns that case is about.
EXPECTED = {
    "workbench-assistant": dict(
        model="claude-sonnet-4-5", region=None, user_kind="authenticated", username="alice",
        tool="Posit Assistant", source="workbench", team="data-science",
        workbench_session_id="d1ac60de", conversation_id="conv-alice-1"),
    "local-assistant": dict(tool="Posit Assistant", source="local", team=None),
    "claude-code": dict(tool="Claude Code", source=None, team="platform", cost_center="cc-42",
                        conversation_id="conv-alice-2"),
    "viewer-content": dict(
        provider="aws.bedrock", model="claude-opus-4-1", region="us", user_kind="viewer",
        username="vic", user_email="vic@example.com", content_owner_username="olga",
        content_guid="content-churn", job_key="job-1", tool="R (ellmer)", source="connect",
        conversation_id=None),
    "service-content": dict(
        model="claude-haiku-4-5", user_kind="owner", username="olga", tool="API SDKs",
        source="connect", integration_auth_type="Service Account"),
    "failed-call": dict(model="claude-sonnet-4-5", status_code=429, finish_reason=None,
                        response_model_raw=""),
    "no-name": dict(user_kind="authenticated", username="guid-alice", user_email=None,
                    tool=None, source=None),
    # Written before user_agent, request_metadata and conversation_id existed.
    "old-record": dict(tool=None, source=None, request_metadata=None, team=None,
                       conversation_id=None, username="alice"),
}


@pytest.mark.parametrize("response_id", EXPECTED)
def test_record_shaping(events, response_id):
    row = events.loc[response_id]
    got = {col: value(row[col]) for col in EXPECTED[response_id]}
    assert got == EXPECTED[response_id]


def test_every_call_with_tokens_is_priced(events, con):
    # Against a real catalog snapshot, no model in the export may cost $0.
    prices = usage._catalog_prices(json.loads((ROOT / "demo" / "pricing.json").read_text()))
    cost = con.sql(f"""SELECT response_id, input_tokens, {pricing._build_est_cost(prices)} AS est
                       FROM gateway_events""").df().set_index("response_id")
    assert (cost[cost.input_tokens > 0].est > 0).all()
    assert cost.est["failed-call"] == 0
    # The Bedrock profile pays the us. rate for opus 4.1: 1000 fresh input,
    # 2000 cache writes, 9000 cache reads and 800 output tokens.
    i, o, cw, cr = prices[("aws.bedrock", "us", "claude-opus-4-1")]
    assert cost.est["viewer-content"] == pytest.approx(
        (1000 * i + 2000 * cw + 9000 * cr + 800 * o) / 1e6)


def test_fake_connect_serves_connect_export_schema(fake_connect, tmp_path):
    """The fake Connect's export has exactly the columns and types Connect's has."""
    url = usage._export_url(WINDOW_FROM, WINDOW_TO, WINDOW_TO)
    req = urllib.request.Request(url, headers={"Authorization": "Key test"})
    path = tmp_path / "fake.parquet"
    with urllib.request.urlopen(req, timeout=30) as r:
        path.write_bytes(r.read())
    assert parquet_schema(path) == EXPORT_SCHEMA
