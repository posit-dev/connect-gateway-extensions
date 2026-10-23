"""Incremental-merge edge cases, on exports written by the test.

usage._merge_export replaces the rows ingested at or after a cutoff with those
from a new full-window export, and drops rows that left the window. The main
case, a later Connect export merged into an earlier one, is in
test_connect_export.py; these cover what that pair does not.
"""

import pandas as pd
import pytest

import usage
from conftest import WINDOW_FROM, WINDOW_TO

# The rows an export holds, by response_id: (start_time, ingest_time).
EARLY = ("2026-09-02 10:00:00+00", "2026-09-02 10:00:05+00")
SETTLED = ("2026-09-10 10:00:00+00", "2026-09-10 10:00:05+00")
# Ingested just after the cutoff, so the merge reads it again.
NEAR_CUTOFF = ("2026-09-20 11:55:00+00", "2026-09-20 11:55:05+00")
NEW = ("2026-09-22 15:00:00+00", "2026-09-22 15:00:01+00")

CUTOFF = "2026-09-20T11:50:00Z"


def rows(**named: tuple[str, str]) -> list[dict]:
    return [{"response_id": k, "start_time": s, "ingest_time": i}
            for k, (s, i) in named.items()]


def table(con) -> pd.DataFrame:
    return con.sql("SELECT * FROM gateway_events ORDER BY response_id").df()


def ids(con) -> list[str]:
    return table(con).response_id.tolist()


@pytest.fixture
def full_con():
    con = usage.new_connection()
    yield con
    con.close()


def test_merge_drops_rows_that_left_the_window(export, con, full_con):
    first = export(rows(EARLY=EARLY, SETTLED=SETTLED))
    later = export(rows(EARLY=EARLY, SETTLED=SETTLED, NEW=NEW))
    moved_from = "2026-09-05T00:00:00Z"  # the trailing window moved past EARLY
    usage._read_export(first, WINDOW_FROM, WINDOW_TO, con)
    usage._merge_export(later, moved_from, WINDOW_TO, CUTOFF, con)
    usage._read_export(later, moved_from, WINDOW_TO, full_con)

    pd.testing.assert_frame_equal(table(con), table(full_con))
    assert ids(con) == ["NEW", "SETTLED"]


def test_merge_of_unchanged_export_changes_nothing(export, con):
    path = export(rows(EARLY=EARLY, SETTLED=SETTLED, NEAR_CUTOFF=NEAR_CUTOFF))
    usage._read_export(path, WINDOW_FROM, WINDOW_TO, con)
    before = table(con)
    usage._merge_export(path, WINDOW_FROM, WINDOW_TO, CUTOFF, con)
    pd.testing.assert_frame_equal(table(con), before)
