"""SQL filter construction, independent of the Shiny app."""

import json
from datetime import date

import duckdb
import pytest

from filters import sql_str, where


@pytest.fixture
def f() -> dict:
    return {
        "start": date(2026, 9, 1), "end": date(2026, 9, 30),
        "source": "both", "users": [], "all_users": ["O'Brien", "Jane"],
        "contents": [], "all_contents": ["content-1"],
        "sessions": [], "providers": [], "all_providers": ["anthropic"],
        "models": [], "all_models": ["model-1"], "meta": [],
    }


def test_no_selection_and_all_selected_mean_no_narrowing(f):
    base = where(f)
    assert base == ("ts >= TIMESTAMP '2026-09-01 00:00:00' AND "
                    "ts <= TIMESTAMP '2026-09-30 23:59:59.999'")
    f.update(users=["Jane", "O'Brien"], contents=["content-1"],
             providers=["anthropic"], models=["model-1"])
    assert where(f) == base
    assert where(f, " AND status_code = '200'") == base + " AND status_code = '200'"


def test_quoted_values_and_metadata_groups(f):
    f.update(
        users=["O'Brien"], source="Connect", sessions=["a'b"],
        meta=[json.dumps(["team", "O'Brien"]), json.dumps(["team", "Jane"]),
              json.dumps(["env", "prod"]), "not json"],
    )
    predicate = where(f)
    assert "username IN ('O''Brien')" in predicate
    assert "workbench_session_id IN ('a''b')" in predicate
    assert "source = 'Connect'" in predicate
    assert "IN ('O''Brien','Jane')" in predicate
    rows = [
        ("O'Brien", "a'b", '{"team":"O\u0027Brien","env":"prod"}'),
        ("O'Brien", "a'b", '{"team":"Jane","env":"dev"}'),
        ("Jane", "a'b", '{"team":"Jane","env":"prod"}'),
    ]
    values = ",".join(f"({sql_str(u)}, {sql_str(s)}, {sql_str(m)})" for u, s, m in rows)
    with duckdb.connect() as con:
        found = con.sql(f"""SELECT count(*) FROM (
            SELECT DATE '2026-09-10' AS ts, 'Connect' AS source,
                   username, workbench_session_id, request_metadata,
                   NULL AS content_guid, NULL AS provider, NULL AS model
            FROM (VALUES {values}) AS t(username, workbench_session_id, request_metadata)
        ) WHERE {predicate}""").fetchone()
    assert found == (1,)
