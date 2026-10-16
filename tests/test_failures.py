"""Failure handling, against a fake Connect told to misbehave.

demo/fake_connect.py has fault switches (POST /__fault__); the `fault` fixture
sets them and clears them after each test. Merge rollback needs no server: it
breaks the local table instead.
"""

import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

import usage
from conftest import WINDOW_FROM, WINDOW_TO

CUTOFF = "2026-09-20T11:50:00Z"


@pytest.fixture
def fault(fake_connect):
    def post(query: str) -> None:
        urllib.request.urlopen(
            urllib.request.Request(f"{fake_connect}/__fault__?{query}", method="POST")
        ).read()

    def set_(route: str, mode: str) -> None:
        post(f"route={route}&mode={mode}")

    yield set_
    post("reset=1")


@pytest.fixture
def isolated(loaded):
    """Run a test with its own load state; put the shared state back after."""
    saved = (usage._state, usage.PRICES, usage.CONTENT_NAMES, usage.GROUP_MEMBERS,
             dict(usage.INFO))
    yield usage
    usage._state, usage.PRICES, usage.CONTENT_NAMES, usage.GROUP_MEMBERS = saved[:4]
    usage.INFO.clear()
    usage.INFO.update(saved[4])


@pytest.fixture
def temp_dir(tmp_path, monkeypatch) -> Path:
    """Make _download's temporary files land somewhere the test can list."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def export_url() -> str:
    return usage._export_url(WINDOW_FROM, WINDOW_TO, WINDOW_TO)


# ---- _download ---------------------------------------------------------------

@pytest.mark.parametrize(("code", "advice"), [
    (401, "CONNECT_API_KEY was refused"),
    (403, "not an administrator key"),
    (404, "does not serve the gateway usage export"),
    (413, "lower LOOKBACK_DAYS"),
    (503, "wait a moment and refresh"),
    (504, "ask for a shorter window"),
    (418, "check CONNECT_SERVER and CONNECT_API_KEY"),  # no specific advice
])
def test_refusals_carry_advice_and_the_servers_reason(fault, temp_dir, code, advice):
    fault("export", f"status:{code}")
    with pytest.raises(RuntimeError) as err:
        usage._download(export_url())
    message = str(err.value)
    assert advice in message
    assert f"{code}: injected {code}" in message
    assert list(temp_dir.iterdir()) == []


def test_html_error_body_is_flattened_and_capped(fault, temp_dir):
    fault("export", "html:502")
    with pytest.raises(RuntimeError) as err:
        usage._download(export_url())
    message = str(err.value)
    detail = message[message.index("(502: "):]
    assert "\n" not in message
    assert "<title>502 Error</title>" in detail
    assert len(detail) <= len("(502: ") + 300 + len(")")
    assert list(temp_dir.iterdir()) == []


def test_connection_cut_mid_download(fault, temp_dir):
    fault("export", "truncate")
    with pytest.raises(RuntimeError, match="cannot read the usage export"):
        usage._download(export_url())
    assert list(temp_dir.iterdir()) == []


def test_unreachable_server(monkeypatch, fake_connect, temp_dir):
    monkeypatch.setenv("CONNECT_SERVER", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="cannot read the usage export"):
        usage._download(export_url())
    assert list(temp_dir.iterdir()) == []


def test_response_that_is_not_parquet_is_a_clear_error_and_leaves_no_file(
    fault, temp_dir, con
):
    fault("export", "garbage")
    with pytest.raises(RuntimeError, match="not a readable Parquet file"):
        usage._ingest(export_url(), lambda p: usage._read_export(p, WINDOW_FROM, WINDOW_TO, con))
    assert list(temp_dir.iterdir()) == []


# ---- _load -------------------------------------------------------------------

def table_state() -> tuple:
    return (usage.scalar("SELECT COUNT(*) FROM gateway_events"),
            usage.scalar("SELECT COUNT(*) FROM content_names"),
            usage.scalar("SELECT COUNT(*) FROM user_groups"))


@pytest.mark.parametrize("mode", ["status:503", "html:502", "garbage", "truncate"])
def test_failed_refresh_keeps_the_loaded_data(isolated, fault, mode):
    before_tables, before_info, before_state = table_state(), dict(usage.INFO), usage._state
    fault("export", mode)
    with pytest.raises(RuntimeError):
        usage.load()
    assert table_state() == before_tables
    assert usage.INFO == before_info
    assert usage._state is before_state


def test_refresh_recovers_once_the_server_does(isolated, fault):
    fault("export", "status:503")
    with pytest.raises(RuntimeError):
        usage.load()
    # The fixture resets after the test, so clear the fault by hand.
    urllib.request.urlopen(urllib.request.Request(
        usage.connect_server() + "/__fault__?reset=1", method="POST")).read()
    assert usage.load()["rows"] > 0


def test_no_price_catalog_on_the_first_load_raises(isolated, fault):
    usage.PRICES = None
    fault("pricing", "status:404")
    with pytest.raises(RuntimeError, match="gateway price catalog"):
        usage._load()
    assert usage.PRICES is None


@pytest.mark.parametrize("mode", ["status:404", "status:500", "garbage"])
def test_failed_price_refresh_keeps_the_last_catalog(isolated, fault, mode):
    before = usage.PRICES
    assert before
    fault("pricing", mode)
    info = usage._load()
    assert usage.PRICES is before
    assert info["priced_models"] > 0


@pytest.mark.parametrize("mode", ["status:500", "html:502", "garbage"])
def test_failed_content_names_keep_the_previous_ones(isolated, fault, mode):
    before = dict(usage.CONTENT_NAMES)
    assert before
    fault("content", mode)
    usage._load()
    assert usage.CONTENT_NAMES == before
    assert usage.scalar("SELECT COUNT(*) FROM content_names") == len(before)


@pytest.mark.parametrize(("route", "mode"), [
    ("groups", "status:500"),
    ("members", "status:500"),  # the group list arrives, then every member list fails
    ("members", "garbage"),
])
def test_failed_groups_keep_the_previous_ones(isolated, fault, route, mode):
    before = list(usage.GROUP_MEMBERS)
    assert before
    fault(route, mode)
    usage._load()
    assert usage.GROUP_MEMBERS == before
    assert usage.scalar("SELECT COUNT(*) FROM user_groups") == len(before)


# ---- merge rollback ----------------------------------------------------------

def test_merge_that_fails_partway_rolls_back(export, con):
    rows = [
        {"response_id": "A", "start_time": "2026-09-10 10:00:00+00",
         "ingest_time": "2026-09-10 10:00:05+00"},
        {"response_id": "B", "start_time": "2026-09-22 15:00:00+00",
         "ingest_time": "2026-09-22 15:00:01+00"},
    ]
    path = export(rows)
    usage._read_export(path, WINDOW_FROM, WINDOW_TO, con)
    # The merge's DELETE removes B (ingested after the cutoff) and then its
    # INSERT cannot name a column the table lost, so the DELETE must be undone.
    con.execute("ALTER TABLE gateway_events DROP COLUMN node_name")
    before = con.sql("SELECT * FROM gateway_events ORDER BY response_id").df()
    assert list(before.response_id) == ["A", "B"]

    with pytest.raises(Exception):
        usage._merge_export(path, WINDOW_FROM, WINDOW_TO, CUTOFF, con)

    after = con.sql("SELECT * FROM gateway_events ORDER BY response_id").df()
    pd.testing.assert_frame_equal(after, before)
    # and the connection is out of the transaction, so it can be used again
    con.execute("BEGIN TRANSACTION")
    con.execute("COMMIT")
