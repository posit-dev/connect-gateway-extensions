"""SQL filter construction, independent of the Shiny app."""

import json
from dataclasses import replace
from datetime import date, datetime

import duckdb
import pytest

from filters import Filters, Params


@pytest.fixture
def f() -> Filters:
    return Filters(
        start=date(2026, 9, 1), end=date(2026, 9, 30), all_users=("O'Brien", "Jane"),
        all_contents=("content-1",),
    )


def where(f: Filters, extra: str = "") -> tuple[str, dict]:
    p = Params()
    return f.where(p, extra), p.values


def test_params_name_each_value_once():
    p = Params()
    assert [p.bind("a"), p.bind(2)] == ["$p0", "$p1"]
    assert p.bind_all(["x", "y"]) == "$p2, $p3"
    assert p.values == {"p0": "a", "p1": 2, "p2": "x", "p3": "y"}


def test_no_selection_and_all_selected_mean_no_narrowing(f):
    sql, values = where(f)
    assert sql == "ts >= $p0 AND ts <= $p1"
    # Both edge days are whole: from the first instant to the last millisecond.
    assert values == {"p0": datetime(2026, 9, 1, 0, 0), "p1": datetime(2026, 9, 30, 23, 59, 59, 999000)}
    narrowed = replace(f, users=("Jane", "O'Brien"), contents=("content-1",))
    assert where(narrowed) == (sql, values)
    assert where(f, " AND status_code = 200")[0] == sql + " AND status_code = 200"


def test_picked_values_are_bound_and_never_in_the_sql(f):
    hostile = "x') OR 1=1 --"
    f = replace(f, users=(hostile,), sessions=(hostile,), groups=(hostile,), contents=(hostile,),
                meta=(json.dumps(["k", hostile]), json.dumps([hostile, "v"])))
    sql, values = where(f)
    assert "OR 1=1" not in sql and "x'" not in sql
    assert sum(v == hostile for v in values.values()) == 5  # user, content, session, group, meta
    assert all(f"${name}" in sql for name in values)  # DuckDB rejects an unused value


def test_quoted_values_and_metadata_groups(f):
    f = replace(
        f, users=("O'Brien",), sessions=("a'b",),
        meta=(json.dumps(["team", "O'Brien"]), json.dumps(["team", "Jane"]),
              json.dumps(["env", "prod"]), "not json"),
    )
    predicate, values = where(f)
    rows = [
        ("O'Brien", "a'b", '{"team":"O\u0027Brien","env":"prod"}'),
        ("O'Brien", "a'b", '{"team":"Jane","env":"dev"}'),
        ("Jane", "a'b", '{"team":"Jane","env":"prod"}'),
    ]
    with duckdb.connect() as con:
        con.execute("CREATE TABLE gateway_events(ts DATE, username VARCHAR, "
                    "workbench_session_id VARCHAR, request_metadata VARCHAR, content_guid VARCHAR)")
        con.executemany("INSERT INTO gateway_events VALUES (DATE '2026-09-10', ?, ?, ?, NULL)", rows)
        # Values of one key are alternatives and separate keys all have to match, so
        # of the three rows only the first has team in (O'Brien, Jane) and env = prod.
        found = con.execute(
            f"SELECT count(*) FROM gateway_events WHERE {predicate}", values).fetchall()
    assert found == [(1,)]


def test_groups_narrow_to_their_members(f):
    f = replace(f, groups=("g'1", "g2"))
    predicate, values = where(f)
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE user_groups AS SELECT * FROM (VALUES
            ('u1', 'g''1', 'one'), ('u2', 'g2', 'two'), ('u3', 'g3', 'three'))
            AS t(user_guid, group_guid, group_name)""")
        found = con.execute(f"""SELECT user_guid FROM (
            SELECT DATE '2026-09-10' AS ts, * FROM (VALUES ('u1'), ('u2'), ('u3'), (NULL))
            AS t(user_guid)) WHERE {predicate} ORDER BY 1""", values).fetchall()
    assert found == [("u1",), ("u2",)]
