"""Loading from the fake Connect: refreshes, and paged Connect lists."""

import urllib.parse

import usage


def test_export_url_names_the_window_and_watermark():
    url = usage._export_url("2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z",
                            "2026-09-30T12:00:00Z")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert params == {
        "from": ["2026-09-01T00:00:00Z"], "to": ["2026-10-01T00:00:00Z"],
        "as_of": ["2026-09-30T12:00:00Z"], "format": ["parquet"],
    }


def test_refresh_merges_the_full_window_without_duplicates(loaded):
    before = loaded.INFO["rows"]
    info = loaded.load()
    assert info["mode"] == "incremental"
    # The fake's export never changes, so a merge must add nothing net.
    assert info["rows"] == before


def test_group_members_read_across_capped_pages(loaded):
    # The fake caps each page at 3 rows (see conftest), so every group and its
    # member list take several pages. All of them must arrive.
    groups = loaded.scalar("SELECT COUNT(DISTINCT group_guid) FROM user_groups")
    members = loaded.scalar("SELECT COUNT(*) FROM user_groups")
    assert groups > 3
    assert members == len(loaded.GROUP_MEMBERS)
    assert members > groups * 3


def test_content_names_loaded(loaded):
    assert loaded.CONTENT_NAMES
    assert loaded.scalar("SELECT COUNT(*) FROM content_names") == len(loaded.CONTENT_NAMES)


# ---- full vs. incremental ---------------------------------------------------

from datetime import datetime, timedelta, timezone

import pytest

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
FROM = "2026-09-01T00:00:00Z"


def _state(as_of=NOW - timedelta(minutes=5), full_at=NOW - timedelta(hours=1)):
    return {"as_of": as_of, "full_at": full_at}


def _stamp(t):
    return t.strftime(usage._RFC3339)


def test_window_follows_the_passed_clock():
    window_from, window_to, as_of = usage._window(NOW)
    assert as_of == window_to == _stamp((NOW - usage.AS_OF_LAG).replace(second=0))
    assert usage._parse(window_from) == usage._parse(as_of) - timedelta(days=usage.LOOKBACK_DAYS)


def test_no_state_means_a_full_load():
    assert usage._plan(None, NOW, FROM, "2026-09-30T11:55:00Z") == (False, "2026-09-30T11:55:00Z")


def test_recent_state_is_incremental():
    incremental, _ = usage._plan(_state(), NOW, FROM, _stamp(NOW))
    assert incremental


@pytest.mark.parametrize(("age", "incremental"), [
    (usage.FULL_RELOAD_EVERY - timedelta(seconds=1), True),
    (usage.FULL_RELOAD_EVERY, False),
    (usage.FULL_RELOAD_EVERY + timedelta(hours=1), False),
])
def test_full_reload_after_24h(age, incremental):
    got, _ = usage._plan(_state(full_at=NOW - age), NOW, FROM, _stamp(NOW))
    assert got is incremental


@pytest.mark.parametrize(("window_from", "incremental"), [
    # watermark minus margin is exactly the window start: still covered
    (_stamp(NOW - timedelta(minutes=5) - usage.REFETCH_MARGIN), True),
    # the window has moved one second past it
    ("2026-09-30T11:45:01Z", False),
])
def test_full_when_window_passes_the_watermark_margin(window_from, incremental):
    got, _ = usage._plan(_state(), NOW, window_from, _stamp(NOW))
    assert got is incremental


def test_watermark_never_moves_backwards():
    ahead = NOW + timedelta(minutes=3)
    _, as_of = usage._plan(_state(as_of=ahead), NOW, FROM, _stamp(NOW))
    assert as_of == _stamp(ahead)
    _, as_of = usage._plan(_state(), NOW, FROM, _stamp(NOW))
    assert as_of == _stamp(NOW)


@pytest.fixture
def fresh_state(loaded):
    """Run a test with its own load state; put the shared one back after."""
    saved = loaded._state
    yield loaded
    loaded._state = saved


def test_first_load_without_a_snapshot_is_a_full_export(fresh_state, tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_CACHE_DIR", str(tmp_path))
    fresh_state._state = None
    info = fresh_state._load()
    assert (info["mode"], info["source"]) == ("full", "export")


def test_first_load_takes_the_snapshot(fresh_state, tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_CACHE_DIR", str(tmp_path))
    fresh_state._write_snapshot(dict(fresh_state._state))
    fresh_state._state = None
    info = fresh_state._load()
    assert (info["mode"], info["source"]) == ("incremental", "snapshot")


def test_load_after_24h_rebuilds_in_full(fresh_state):
    later = datetime.now(timezone.utc) + usage.FULL_RELOAD_EVERY + timedelta(minutes=1)
    info = fresh_state._load(now=later)
    assert (info["mode"], info["source"]) == ("full", "export")
    assert info["full_at"] == later
