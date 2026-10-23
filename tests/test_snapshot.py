"""The Parquet snapshot a restarted process starts from.

usage._write_snapshot saves gateway_events with the watermark it was read up
to, when it was last read in full, and the settings it was built with.
usage._read_snapshot loads it back only when it is still worth using; anything
else means a full load, never an error. These tests write snapshots of the real
Connect export to a temporary directory and read them into fresh connections.
"""

import time
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

import usage
from conftest import CONNECT_EXPORT, WINDOW_FROM, WINDOW_TO

AS_OF = datetime(2026, 10, 1, tzinfo=timezone.utc)
STATE = {"as_of": AS_OF, "full_at": AS_OF}
# An hour after the full read: well inside FULL_RELOAD_EVERY.
NOW = AS_OF + timedelta(hours=1)
# A window whose start the snapshot's watermark is far past.
FROM = WINDOW_FROM


def events(con) -> pd.DataFrame:
    return con.sql("SELECT * FROM gateway_events ORDER BY response_id").df()


def has_events(con) -> bool:
    return bool(con.sql(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'gateway_events'"
    ).fetchone()[0])  # pyright: ignore[reportOptionalSubscript]


@pytest.fixture
def source(con) -> duckdb.DuckDBPyConnection:
    """A connection holding the real Connect export, as a load leaves it."""
    usage._read_export(CONNECT_EXPORT, WINDOW_FROM, WINDOW_TO, con)
    return con


@pytest.fixture
def target():
    """A fresh connection, as a restarted process has."""
    c = usage.new_connection()
    yield c
    c.close()


@pytest.fixture
def path(tmp_path):
    # A directory that does not exist yet: the writer creates it.
    return tmp_path / "cache" / "gateway_events.parquet"


def test_round_trip(source, target, path):
    usage._write_snapshot(STATE, con=source, path=path)
    state = usage._read_snapshot(FROM, con=target, path=path, now=NOW)
    assert state == STATE
    pd.testing.assert_frame_equal(events(target), events(source))
    # Neither side leaves its private .write-/.read- file behind.
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_a_later_write_replaces_the_snapshot(source, target, path):
    usage._write_snapshot(STATE, con=source, path=path)
    source.execute("DELETE FROM gateway_events WHERE response_id <> 'old-record'")
    later = {"as_of": AS_OF + timedelta(minutes=5), "full_at": AS_OF}
    usage._write_snapshot(later, con=source, path=path)
    assert usage._read_snapshot(FROM, con=target, path=path, now=NOW) == later
    assert events(target).response_id.tolist() == ["old-record"]


@pytest.mark.parametrize(("now", "window_from", "usable"), [
    # Due a full reload: FULL_RELOAD_EVERY since the last full read.
    (AS_OF + usage.FULL_RELOAD_EVERY - timedelta(seconds=1), FROM, True),
    (AS_OF + usage.FULL_RELOAD_EVERY, FROM, False),
    # The window moved past the watermark less REFETCH_MARGIN: a merge from
    # there would leave a gap at the start of the window.
    (NOW, (AS_OF - usage.REFETCH_MARGIN).strftime(usage._RFC3339), True),
    (NOW, (AS_OF - usage.REFETCH_MARGIN + timedelta(minutes=1)).strftime(usage._RFC3339),
     False),
], ids=["fresh", "due-full-reload", "window-at-margin", "window-past-margin"])
def test_staleness(source, target, path, now, window_from, usable):
    usage._write_snapshot(STATE, con=source, path=path)
    state = usage._read_snapshot(window_from, con=target, path=path, now=now)
    assert (state is not None) == usable
    assert has_events(target) == usable


@pytest.mark.parametrize(("var", "value"), [
    ("CONNECT_SERVER", "https://another-connect.example.com"),
    ("GATEWAY_FROM", "2026-09-15T00:00:00Z"),
    ("GATEWAY_TO", "2026-09-30T00:00:00Z"),
])
def test_other_settings_are_not_reused(source, target, path, monkeypatch, var, value):
    usage._write_snapshot(STATE, con=source, path=path)
    monkeypatch.setenv(var, value)
    assert usage._read_snapshot(FROM, con=target, path=path, now=NOW) is None
    assert not has_events(target)


def test_missing_snapshot(target, path):
    assert usage._read_snapshot(FROM, con=target, path=path, now=NOW) is None


def test_unreadable_snapshot(target, path):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PAR1 not really a parquet file")
    assert usage._read_snapshot(FROM, con=target, path=path, now=NOW) is None
    assert not has_events(target)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_snapshot_without_metadata(target, path):
    # A Parquet file, but not one _write_snapshot wrote: no as_of or full_at.
    path.parent.mkdir(parents=True)
    path.write_bytes(CONNECT_EXPORT.read_bytes())
    assert usage._read_snapshot(FROM, con=target, path=path, now=NOW) is None
    assert not has_events(target)


def test_failed_write_keeps_the_previous_snapshot(source, target, path):
    usage._write_snapshot(STATE, con=source, path=path)
    before = path.read_bytes()
    # Nothing to save: the COPY fails, and the failure is swallowed.
    usage._write_snapshot(STATE, con=target, path=path)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_app_load_leaves_a_snapshot_a_restart_can_use(loaded, target):
    # load() saves the snapshot on a background worker once it has finished,
    # and the rename makes it appear whole. An earlier load's snapshot may be
    # there first, so wait for the one written for the latest load.
    path = loaded.snapshot_path()
    state = None
    for _ in range(100):
        state = usage._read_snapshot(loaded.INFO["window_from"], con=target, path=path)
        if state and state["as_of"].strftime(usage._RFC3339) == loaded.INFO["as_of"]:
            break
        time.sleep(0.1)
    assert state is not None
    assert state["as_of"].strftime(usage._RFC3339) == loaded.INFO["as_of"]
    assert target.sql("SELECT COUNT(*) FROM gateway_events").fetchone() == (
        loaded.INFO["rows"],)


def test_write_to_an_unusable_directory_is_swallowed(source, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    usage._write_snapshot(STATE, con=source, path=blocker / "gateway_events.parquet")
    assert blocker.read_text() == ""
