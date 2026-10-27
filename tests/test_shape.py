"""How usage._SHAPE turns rows of the Connect usage export into gateway_events.

Each test writes a few export rows to a real Parquet file and reads it the way
a load does (see the `shaped` fixture). A mistake here does not crash; it
shows wrong numbers, such as a call priced at $0 or charged to the wrong user.
"""

import re

import pandas as pd
import pytest

import usage
from conftest import WINDOW_FROM, WINDOW_TO

# (request_model, response_model) -> the model family and inference region the
# row should carry. The family is what the price catalog is keyed by.
MODELS = [
    # Anthropic answers with a dated id.
    ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-haiku-4-5", None),
    # A failed call has an empty response_model; the request model is used.
    ("claude-sonnet-4-5", "", "claude-sonnet-4-5", None),
    # Bedrock inference profiles: region prefix, provider, date and revision.
    ("us.anthropic.claude-opus-4-1-20250805-v1:0", "", "claude-opus-4-1", "us"),
    ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
     "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5", "eu"),
    ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", "", "claude-sonnet-4-5", "global"),
    ("us-gov.anthropic.claude-3-haiku-20240307-v1:0", "", "claude-3-haiku", "us-gov"),
    # A Bedrock base id has a provider prefix but no region.
    ("anthropic.claude-3-haiku-20240307-v1:0", "", "claude-3-haiku", None),
    ("gpt-5", "gpt-5", "gpt-5", None),
    # A Mantle id can have dots in the model name.
    ("openai.gpt-5.5", "", "gpt-5.5", None),
    ("xai.grok-4.3", "", "grok-4.3", None),
]


@pytest.mark.parametrize(("request_model", "response_model", "model", "region"), MODELS)
def test_model_family_and_region(shaped, request_model, response_model, model, region):
    row = shaped([{"request_model": request_model, "response_model": response_model}]).iloc[0]
    assert row.model == model
    assert (None if pd.isna(row.region) else row.region) == region
    assert row.request_model_raw == request_model


@pytest.mark.parametrize("raw", sorted({m for r in MODELS for m in r[:2] if m}))
def test_python_model_family_matches_sql(shaped, raw):
    # The price catalog is keyed with usage._model_family and requests with the
    # SQL; if they disagree, the lookup misses and the call costs $0.
    row = shaped([{"request_model": raw, "response_model": ""}]).iloc[0]
    assert usage._model_family(raw) == row.model


def test_sql_regions_match_python_regions():
    listed = re.search(r"IN \(('[^)]*')\)", usage._SHAPE)
    assert listed is not None
    assert set(re.findall(r"'([^']+)'", listed.group(1))) == usage._REGIONS


NOBODY = {"authenticated_username": None, "authenticated_user_guid": None,
          "authenticated_email": None}
VIEWER = {"viewer_username": "vic", "viewer_user_guid": "guid-vic",
          "viewer_email": "vic@example.com"}
OWNER = {"content_owner_username": "olga", "content_owner_guid": "guid-olga",
         "content_owner_email": "olga@example.com", "content_guid": "content-1"}


@pytest.mark.parametrize(("identities", "kind", "guid", "username", "email"), [
    # The viewer a call was made for wins over its owner.
    ({**NOBODY, **VIEWER, **OWNER}, "viewer", "guid-vic", "vic", "vic@example.com"),
    # A Workbench call: the signed-in user.
    ({}, "authenticated", "guid-alice", "alice", "alice@example.com"),
    # A service-account content call: only the owner is known.
    ({**NOBODY, **OWNER}, "owner", "guid-olga", "olga", "olga@example.com"),
    # The name lookup was off: the guid stands in for the name.
    ({"authenticated_username": None}, "authenticated", "guid-alice", "guid-alice",
     "alice@example.com"),
    # The name follows the identity the guid picked, never another one's.
    ({**VIEWER, "viewer_username": None}, "viewer", "guid-vic", "guid-vic",
     "vic@example.com"),
    # No identity at all.
    (NOBODY, None, None, None, None),
])
def test_principal(shaped, identities, kind, guid, username, email):
    row = shaped([identities]).iloc[0]
    got = [None if pd.isna(v) else v
           for v in (row.user_kind, row.user_guid, row.username, row.user_email)]
    assert got == [kind, guid, username, email]


@pytest.mark.parametrize(("user_agent", "tool"), [
    (None, None),
    ("", None),
    ("PositAssistant/1.2.3 posit_workbench_positron/2026.01.0", "Posit Assistant"),
    # The first match wins: Posit Assistant naming Claude is still Posit Assistant.
    ("PositAssistant/1.2.3 (claude)", "Posit Assistant"),
    ("claude-cli/2.0.14 (external, cli)", "Claude Code"),
    ("GitHubCopilotChat/0.22.4 vscode/1.95.0", "GitHub Copilot"),
    ("codex_cli_rs/0.46.0 (Mac OS 15.1.0; arm64)", "Codex CLI"),
    ("pi-coding-agent/0.50.0", "Pi"),
    ("pi/1.0", "Pi"),
    ("node (pi/1.0)", "Pi"),
    # "pi" inside another word is not Pi.
    ("api-client/1.0", "Other"),
    ("r-ellmer/0.3.2", "R (ellmer)"),
    ("Anthropic/Python 0.39.0", "API SDKs"),
    ("OpenAI/Python 1.54.0", "API SDKs"),
    ("python-httpx/0.27.2", "API SDKs"),
    ("Python/3.11 aiohttp/3.9.5", "API SDKs"),
    ("curl/8.7.1", "API SDKs"),
    ("Mozilla/5.0", "Other"),
])
def test_tool_from_user_agent(shaped, user_agent, tool):
    row = shaped([{"user_agent": user_agent}]).iloc[0]
    assert (None if pd.isna(row.tool) else row.tool) == tool


def test_origin_from_export(shaped):
    df = shaped([{"origin": "local"}, {"origin": None}])
    assert df.source.tolist()[0] == "local"
    assert pd.isna(df.source.tolist()[1])


def test_export_without_origin_or_user_agent(shaped):
    # An older Connect sends neither: content is "connect", the rest "workbench".
    df = shaped([{"content_guid": "content-1"}, {}], drop=("origin", "user_agent"))
    assert df.source.tolist() == ["connect", "workbench"]
    assert df.tool.isna().all()


def test_export_without_request_metadata(shaped):
    df = shaped([{}], drop=("origin", "user_agent", "request_metadata"))
    assert df.request_metadata.isna().all()
    assert df.team.isna().all()


def test_request_metadata_fields(shaped):
    meta = ('{"team":"data-science","cost-center":"cc-1","tier":"gold",'
            '"workbench-session-id":"d1ac60de","workbench-project-name":"Churn"}')
    row = shaped([{"request_metadata": meta}]).iloc[0]
    assert (row.team, row.cost_center, row.tier) == ("data-science", "cc-1", "gold")
    assert row.workbench_session_id == "d1ac60de"
    assert row.workbench_project_name == "Churn"
    assert pd.isna(row.workbench_project_path)


def test_conversation_id_from_older_session_id_column(shaped):
    df = shaped([{"conversation_id": "conv-1"}], rename={"conversation_id": "session_id"})
    assert df.conversation_id.tolist() == ["conv-1"]


def test_window_keeps_start_inclusive_end_exclusive(shaped):
    df = shaped([
        {"start_time": "2026-08-31 23:59:59+00", "response_id": "before"},
        {"start_time": WINDOW_FROM, "response_id": "first"},
        {"start_time": "2026-09-30 23:59:59+00", "response_id": "last"},
        {"start_time": WINDOW_TO, "response_id": "after"},
    ])
    assert df.response_id.tolist() == ["first", "last"]


def test_time_columns_are_utc(shaped):
    # Saturday 2026-09-12, 23:30 UTC: still Saturday and hour 23 in UTC.
    row = shaped([{"start_time": "2026-09-12 23:30:00+00"}]).iloc[0]
    assert str(row.ts) == "2026-09-12 23:30:00"
    assert str(row.day.date() if hasattr(row.day, "date") else row.day) == "2026-09-12"
    assert (row.month, row.hour, bool(row.is_weekend)) == ("2026-09", 23, True)


def test_tokens_and_finish_reason(shaped):
    row = shaped([{
        "input_tokens": 1000, "output_tokens": 50, "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 300, "finish_reasons": ["tool_use", "end_turn"],
    }]).iloc[0]
    assert (row.input_tokens, row.output_tokens) == (1000, 50)
    assert (row.cache_creation_tokens, row.cache_read_tokens) == (200, 300)
    assert row.finish_reason == "tool_use"
