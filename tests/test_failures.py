"""Failure handling, against a fake Connect told to misbehave.

demo/fake_connect.py has fault switches (POST /__fault__); the `fault` fixture
(see conftest.py) sets them and clears them after each test.
"""

import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

import usage
from conftest import WINDOW_FROM, WINDOW_TO
from filters import Query

KEY = "test"


@pytest.fixture
def dataset(fake_connect):
    """A dataset that has not loaded yet."""
    ds = usage.Dataset(KEY)
    yield ds
    ds.close()


@pytest.fixture
def temp_dir(tmp_path, monkeypatch) -> Path:
    """Make _download's temporary files land somewhere the test can list."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def export_url() -> str:
    return usage._export_url(WINDOW_FROM, WINDOW_TO, WINDOW_TO)


# ---- _download ---------------------------------------------------------------

@pytest.mark.parametrize(("code", "advice"), [
    (401, "the API key was refused"),
    (403, "not an administrator key"),
    (404, "does not serve the gateway usage export"),
    (413, "lower LOOKBACK_DAYS"),
    (503, "wait a moment and reload"),
    (504, "ask for a shorter window"),
    (418, "check CONNECT_SERVER and CONNECT_API_KEY"),  # no specific advice
])
def test_refusals_carry_advice_and_the_servers_reason(fault, temp_dir, code, advice):
    fault("export", f"status:{code}")
    with pytest.raises(RuntimeError) as err:
        usage._download(export_url(), KEY)
    message = str(err.value)
    assert advice in message
    assert f"{code}: injected {code}" in message
    assert list(temp_dir.iterdir()) == []


def test_html_error_body_is_flattened_and_capped(fault, temp_dir):
    fault("export", "html:502")
    with pytest.raises(RuntimeError) as err:
        usage._download(export_url(), KEY)
    message = str(err.value)
    detail = message[message.index("(502: "):]
    assert "\n" not in message
    assert "<title>502 Error</title>" in detail
    assert len(detail) <= len("(502: ") + 300 + len(")")
    assert list(temp_dir.iterdir()) == []


def test_connection_cut_mid_download(fault, temp_dir):
    fault("export", "truncate")
    with pytest.raises(RuntimeError, match="cannot read the usage export"):
        usage._download(export_url(), KEY)
    assert list(temp_dir.iterdir()) == []


def test_unreachable_server(monkeypatch, fake_connect, temp_dir):
    monkeypatch.setenv("CONNECT_SERVER", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="cannot read the usage export"):
        usage._download(export_url(), KEY)
    assert list(temp_dir.iterdir()) == []


def test_response_that_is_not_parquet_is_a_clear_error_and_leaves_no_file(
    fault, temp_dir, con
):
    fault("export", "garbage")
    with pytest.raises(RuntimeError, match="not a readable Parquet file"):
        usage._ingest(
            export_url(), KEY, lambda p: usage._read_export(p, WINDOW_FROM, WINDOW_TO, con)
        )
    assert list(temp_dir.iterdir()) == []


# ---- Dataset.load ------------------------------------------------------------

@pytest.mark.parametrize("mode", ["status:503", "html:502", "garbage", "truncate"])
def test_failed_export_fails_the_load_and_sets_nothing(dataset, fault, mode):
    fault("export", mode)
    with pytest.raises(RuntimeError):
        dataset.load()
    assert dataset.info == {}
    assert dataset.prices is None


def test_load_recovers_once_the_server_does(dataset, fault):
    fault("export", "status:503")
    with pytest.raises(RuntimeError):
        dataset.load()
    # The fixture resets after the test, so clear the fault by hand.
    urllib.request.urlopen(urllib.request.Request(
        usage.connect_server() + "/__fault__?reset=1", method="POST")).read()
    assert dataset.load()["rows"] > 0


def test_no_price_catalog_fails_the_load(dataset, fault):
    fault("pricing", "status:404")
    with pytest.raises(RuntimeError, match="gateway price catalog"):
        dataset.load()
    assert dataset.prices is None


@pytest.mark.parametrize("mode", ["status:500", "html:502", "garbage"])
def test_failed_content_names_fall_back_to_guids(dataset, fault, mode):
    fault("content", mode)
    dataset.load()
    assert dataset.content_names == {}
    assert dataset.scalar("SELECT COUNT(*) FROM content_names") == 0


@pytest.mark.parametrize(("route", "mode"), [
    ("groups", "status:500"),
    ("members", "status:500"),  # the group list arrives, then every member list fails
    ("members", "garbage"),
])
def test_failed_groups_load_without_groups(dataset, fault, route, mode):
    fault(route, mode)
    dataset.load()
    assert dataset.group_members == []
    assert dataset.scalar("SELECT COUNT(*) FROM user_groups") == 0


def test_a_load_without_a_key_says_what_to_set(fake_connect):
    ds = usage.Dataset("")
    with pytest.raises(RuntimeError, match="CONNECT_API_KEY is not set"):
        ds.load()
    ds.close()


# The session ends after the export has downloaded, or while Connect's lists
# are still being read.
@pytest.mark.parametrize("step", ["_download", "_fetch_groups"])
def test_a_load_that_outlives_its_session_stops_quietly(dataset, temp_dir, monkeypatch, step):
    real = getattr(usage, step)

    def then_close(*args):
        out = real(*args)
        dataset.close()
        return out

    monkeypatch.setattr(usage, step, then_close)
    assert dataset.load() == {}
    assert dataset.prices is None
    assert dataset.content_names == {} and dataset.group_members == []
    assert list(temp_dir.iterdir()) == []


# ---- the cap on loads at once ------------------------------------------------

@pytest.fixture
def configured(monkeypatch):
    """A server to load from, so load() gets past its configuration check."""
    monkeypatch.setenv("CONNECT_SERVER", "http://connect.invalid")


def test_at_most_max_loads_run_at_once(configured, monkeypatch):
    running, peak, count = 0, 0, threading.Lock()

    def slow_load(self):
        nonlocal running, peak
        with count:
            running += 1
            peak = max(peak, running)
        time.sleep(0.2)
        with count:
            running -= 1
        return {}

    monkeypatch.setattr(usage.Dataset, "_load", slow_load)
    datasets = [usage.Dataset(KEY) for _ in range(usage.MAX_LOADS + 3)]
    with ThreadPoolExecutor(len(datasets)) as pool:
        assert list(pool.map(usage.Dataset.load, datasets)) == [{}] * len(datasets)
    for ds in datasets:
        ds.close()
    assert peak == usage.MAX_LOADS


def test_a_load_that_waits_too_long_for_its_turn_says_so(configured, monkeypatch):
    monkeypatch.setattr(usage, "LOAD_WAIT", 0.05)
    for _ in range(usage.MAX_LOADS):
        usage._loads.acquire()
    ds = usage.Dataset(KEY)
    try:
        with pytest.raises(RuntimeError, match="other sessions were still loading"):
            ds.load()
    finally:
        for _ in range(usage.MAX_LOADS):
            usage._loads.release()
        ds.close()


def test_a_failed_load_gives_its_turn_back(configured, monkeypatch):
    monkeypatch.setattr(usage, "LOAD_WAIT", 0.05)

    def failing_load(self):
        raise RuntimeError("injected")

    monkeypatch.setattr(usage.Dataset, "_load", failing_load)
    ds = usage.Dataset(KEY)
    for _ in range(usage.MAX_LOADS + 1):
        with pytest.raises(RuntimeError, match="injected"):
            ds.load()
    ds.close()


# ---- Dataset.export_csv ------------------------------------------------------

def test_a_failed_csv_export_leaves_no_file(temp_dir):
    ds = usage.Dataset(KEY)
    try:
        with pytest.raises(duckdb.CatalogException):
            ds.export_csv(Query("SELECT * FROM no_such_table", {}))
    finally:
        ds.close()
    assert list(temp_dir.iterdir()) == []


def test_a_csv_export_can_land_in_a_directory_with_a_quote(tmp_path, monkeypatch):
    odd = tmp_path / "o'hare"
    odd.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(odd))
    ds = usage.Dataset(KEY)
    try:
        path = ds.export_csv(Query("SELECT 1 AS one", {}))
    finally:
        ds.close()
    assert path.parent == odd
    assert path.read_text().split() == ["one", "1"]
