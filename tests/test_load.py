"""Loading from the fake Connect: the export window, and paged Connect lists."""

import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest
from shiny.session import session_context

import usage
from conftest import stub_session

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def test_export_url_names_the_window_and_watermark():
    url = usage._export_url("2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z",
                            "2026-09-30T12:00:00Z")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert params == {
        "from": ["2026-09-01T00:00:00Z"], "to": ["2026-10-01T00:00:00Z"],
        "as_of": ["2026-09-30T12:00:00Z"], "format": ["parquet"],
    }


def test_window_follows_the_passed_clock():
    def parse(ts):
        return datetime.strptime(ts, usage._RFC3339)

    window_from, window_to, as_of = usage._window(NOW)
    expected = (NOW - usage.AS_OF_LAG).replace(second=0, tzinfo=None)
    assert parse(as_of) == parse(window_to) == expected
    assert parse(window_from) == expected - timedelta(days=usage.LOOKBACK_DAYS)


def test_load_reports_what_it_read(loaded):
    assert loaded.info["rows"] == loaded.scalar("SELECT COUNT(*) FROM gateway_events")
    assert loaded.info["rows"] > 0
    assert loaded.info["priced_models"] > 0


def test_every_request_carries_the_key_the_dataset_was_given(fake_connect, monkeypatch):
    import urllib.request

    sent = []
    real = urllib.request.urlopen

    def spy(req, *args, **kwargs):
        sent.append(req.get_header("Authorization"))
        return real(req, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", spy)
    monkeypatch.setenv("CONNECT_API_KEY", "the owner's key")
    ds = usage.Dataset("visitor")
    try:
        ds.load()
    finally:
        ds.close()
    assert len(sent) > 3  # export, prices, content, groups, ...
    assert set(sent) == {"Key visitor"}


def test_group_members_read_across_capped_pages(loaded):
    # The fake caps each page at 3 rows (see conftest), so every group and its
    # member list take several pages. All of them must arrive.
    groups = loaded.scalar("SELECT COUNT(DISTINCT group_guid) FROM user_groups")
    members = loaded.scalar("SELECT COUNT(*) FROM user_groups")
    assert groups > 3
    assert members == len(loaded.group_members)
    assert members > groups * 3


def test_content_names_loaded(loaded):
    assert loaded.content_names
    assert loaded.scalar("SELECT COUNT(*) FROM content_names") == len(loaded.content_names)


def test_queries_read_the_dataset_of_their_own_session(loaded):
    other, session = usage.Dataset("test"), stub_session("other")
    other.load()
    usage.attach(session, other)
    try:
        other.con.execute("DELETE FROM gateway_events")
        with session_context(session):
            assert usage.scalar("SELECT COUNT(*) FROM gateway_events") == 0
        assert usage.scalar("SELECT COUNT(*) FROM gateway_events") > 0
    finally:
        session.end()
    assert "other" not in usage._datasets


def test_queries_outside_a_session_with_a_dataset_are_refused():
    with session_context(None), pytest.raises(RuntimeError, match="no dataset"):
        usage.q("SELECT 1")
