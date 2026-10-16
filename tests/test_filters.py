"""Tests for SQL filter construction, independent of the Shiny app."""

import json
import unittest
from datetime import date

import duckdb

from filters import sql_str, where


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.f = {
            "start": date(2026, 9, 1), "end": date(2026, 9, 30),
            "source": "both", "users": [], "all_users": ["O'Brien", "Jane"],
            "contents": [], "all_contents": ["content-1"],
            "sessions": [], "providers": [], "all_providers": ["anthropic"],
            "models": [], "all_models": ["model-1"], "meta": [],
        }

    def test_no_selection_and_all_selected_mean_no_narrowing(self):
        base = where(self.f)
        self.assertEqual(base, "ts >= TIMESTAMP '2026-09-01 00:00:00' AND "
                               "ts <= TIMESTAMP '2026-09-30 23:59:59.999'")
        self.f.update(users=["Jane", "O'Brien"], contents=["content-1"],
                      providers=["anthropic"], models=["model-1"])
        self.assertEqual(where(self.f), base)
        self.assertEqual(where(self.f, " AND status_code = '200'"),
                         base + " AND status_code = '200'")

    def test_quoted_values_and_metadata_groups(self):
        self.f.update(
            users=["O'Brien"], source="Connect", sessions=["a'b"],
            meta=[json.dumps(["team", "O'Brien"]), json.dumps(["team", "Jane"]),
                  json.dumps(["env", "prod"]), "not json"],
        )
        predicate = where(self.f)
        self.assertIn("username IN ('O''Brien')", predicate)
        self.assertIn("workbench_session_id IN ('a''b')", predicate)
        self.assertIn("source = 'Connect'", predicate)
        self.assertIn("IN ('O''Brien','Jane')", predicate)
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
        self.assertEqual(found, (1,))


if __name__ == "__main__":
    unittest.main()
