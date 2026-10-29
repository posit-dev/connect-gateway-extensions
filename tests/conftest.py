"""Shared fixtures.

Tests give real inputs to real code; nothing here patches the app. Four kinds
of input:

- tests/fixtures/connect_export*.parquet: exports built by Connect's own code
  (`just connect-fixtures`). CONNECT_EXPORT and CONNECT_EXPORT_EARLY name them.
- `export`: a small Parquet file with exactly the real export's schema, holding
  just the rows a test describes. `shaped` loads such a file through the app's
  own `_SHAPE` into a private DuckDB connection.
- `fake_connect`: demo/fake_connect.py, the same fake Connect `just run-local`
  uses, started once per session. Any test that uses it (directly or through
  `loaded`) is marked `connect` automatically, so `just test-fast` skips it.
- `loaded`: the app's own data, loaded from the fake Connect by importing app.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import duckdb
import pandas as pd
import pytest

from filters import Filters, Params

ROOT = Path(__file__).resolve().parents[1]
# The extension the Gallery ships: the app modules, and the directory `shiny run` starts in.
APP = ROOT / "extensions" / "gateway-usage-dashboard"

# The window every export-shaping test reads with: all of September 2026.
WINDOW_FROM = "2026-09-01T00:00:00Z"
WINDOW_TO = "2026-10-01T00:00:00Z"

FIXTURES = ROOT / "tests" / "fixtures"
# Exports Connect built for September 2026, at as_of 2026-10-01 and at
# 2026-09-20T12:00; see tests/fixtures/connect_export_fixture_test.go for the
# records, each named by its response_id.
CONNECT_EXPORT = FIXTURES / "connect_export.parquet"
CONNECT_EXPORT_EARLY = FIXTURES / "connect_export_early.parquet"


def parquet_schema(path: Path) -> dict[str, str]:
    """Column name -> DuckDB type of a Parquet file, in file order."""
    with duckdb.connect() as db:
        return {
            name: str(typ)
            for name, typ, *_ in db.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
        }


# The columns of the Connect usage export and their types, read from the export
# Connect built, so this follows Connect whenever the fixtures are regenerated.
EXPORT_SCHEMA = parquet_schema(CONNECT_EXPORT)

# What a row holds unless the test says otherwise: one successful Workbench
# call by alice on the 10th, ingested a second after it started.
ROW_DEFAULTS = {
    "start_time": "2026-09-10 12:00:00+00",
    "node_name": "node-1",
    "provider": "anthropic",
    "operation": "chat",
    "request_model": "claude-sonnet-4-5",
    "response_model": "claude-sonnet-4-5",
    "duration_ms": 1000,
    "status_code": 200,
    "finish_reasons": ["end_turn"],
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "authenticated_username": "alice",
    "authenticated_user_guid": "guid-alice",
    "authenticated_email": "alice@example.com",
}


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """A private DuckDB connection, set up like the app's."""
    import usage

    c = usage.new_connection()
    yield c
    c.close()


ExportFactory = Callable[..., Path]


@pytest.fixture
def export(tmp_path: Path) -> ExportFactory:
    """Write rows as a usage-export Parquet file and return its path.

    Each row is a dict of export columns; ROW_DEFAULTS fills the rest, and
    ingest_time defaults to one second after start_time. `drop` leaves columns
    out of the file, as an older Connect's export does, and `rename` writes a
    column under an older name ({"conversation_id": "session_id"}).
    """
    count = 0

    def write(
        rows: list[dict], drop: tuple[str, ...] = (), rename: dict[str, str] | None = None,
    ) -> Path:
        nonlocal count
        count += 1
        path = tmp_path / f"export-{count}.parquet"
        with duckdb.connect() as db:
            db.execute("SET TimeZone='UTC'")
            cols = ", ".join(f'"{c}" {t}' for c, t in EXPORT_SCHEMA.items())
            db.execute(f"CREATE TABLE export ({cols})")
            for row in rows:
                full = {**ROW_DEFAULTS, **row}
                unknown = set(full) - set(EXPORT_SCHEMA)
                assert not unknown, f"not export columns: {unknown}"
                names = ", ".join(f'"{c}"' for c in full)
                marks = ", ".join("?" for _ in full)
                db.execute(f"INSERT INTO export ({names}) VALUES ({marks})", list(full.values()))
            db.execute(
                "UPDATE export SET ingest_time = start_time + INTERVAL 1 SECOND "
                "WHERE ingest_time IS NULL"
            )
            keep = "*" + (f" EXCLUDE ({', '.join(drop)})" if drop else "")
            if rename:
                keep += f" RENAME ({', '.join(f'{a} AS {b}' for a, b in rename.items())})"
            db.execute(f"COPY (SELECT {keep} FROM export) TO '{path}' (FORMAT parquet)")
        return path

    return write


@pytest.fixture
def shaped(export: ExportFactory, con: duckdb.DuckDBPyConnection) -> Callable[..., pd.DataFrame]:
    """Shape rows through the app's _SHAPE and return gateway_events.

    Rows come back in start_time order. Arguments are those of `export`.
    """
    import usage

    def run(rows: list[dict], **kwargs) -> pd.DataFrame:
        usage._read_export(export(rows, **kwargs), WINDOW_FROM, WINDOW_TO, con)
        return con.sql("SELECT * FROM gateway_events ORDER BY ts").df()

    return run


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _running_fake(**env: str) -> Iterator[str]:
    """demo/fake_connect.py on a free port, with these extra environment
    variables (FAKE_PROFILE, ...). Yields its URL once it is listening."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "demo" / "fake_connect.py")],
        cwd=ROOT, env={**os.environ, "MOCK_CONNECT_PORT": str(port), **env},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            if proc.poll() is not None:
                assert proc.stderr is not None
                raise RuntimeError(proc.stderr.read().decode())
            try:
                urllib.request.urlopen(f"{url}/", timeout=1)
            except urllib.error.HTTPError:
                break  # any HTTP status: it has built its export and is listening
            except urllib.error.URLError:
                time.sleep(0.1)
        else:
            raise RuntimeError("fake Connect did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if proc.stderr:
            proc.stderr.close()


@contextmanager
def _running_app(env: dict[str, str]) -> Iterator[str]:
    """`shiny run app:app` on a free port with this environment. Yields its URL."""
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", "app:app", "--port", str(port)],
        cwd=APP, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    assert server.stderr is not None
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError(server.stderr.read().decode())
            try:
                urllib.request.urlopen(f"{url}/", timeout=1).close()
                break
            except urllib.error.HTTPError:
                break  # it answered; a failing page is for the test to report
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            raise RuntimeError("app did not start")
        yield url
    finally:
        server.terminate()
        server.wait(timeout=10)
        server.stderr.close()


@pytest.fixture(scope="session")
def fake_connect(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start demo/fake_connect.py and point the app's environment at it.

    The fake caps list pages at 3 rows, so reading groups and their members
    follows several pages, as it can against a real Connect.
    """
    with _running_fake(FAKE_ROWS="1000", MOCK_PAGE_CAP="3") as url, \
            pytest.MonkeyPatch.context() as mp:
        mp.setenv("CONNECT_SERVER", url)
        mp.setenv("CONNECT_API_KEY", "test")
        mp.setenv("GATEWAY_CACHE_DIR", str(tmp_path_factory.mktemp("gateway-cache")))
        mp.delenv("GATEWAY_FROM", raising=False)
        mp.delenv("GATEWAY_TO", raising=False)
        yield url


@pytest.fixture(scope="session")
def loaded(fake_connect: str) -> ModuleType:
    """The usage module with the fake Connect's data loaded, as the app loads it."""
    import app  # noqa: F401  # importing app runs the first usage.load()
    import usage

    return usage


@pytest.fixture
def f(loaded) -> Filters:
    """The filters for the app's default window with nothing narrowed."""
    import app

    c = app.catalog()
    start, end = app.preset_range(app.DEFAULT_PRESET, c["start"], c["end"])
    return Filters(
        start=start, end=end,
        all_users=tuple(c["users"]), all_contents=tuple(c["contents"]),
    )


def viewed(f: Filters) -> tuple[str, dict]:
    """A filter's predicate and the values bound into it, to use in a test's own SQL."""
    p = Params()
    return f.where(p), p.values


@pytest.fixture(scope="session")
def app_server(fake_connect: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """`shiny run app:app` against the fake Connect, started once. Yields its URL.

    Tests open sessions on it with tests/session.py. Marked `app` (see pytest.ini).
    """
    cache = tmp_path_factory.mktemp("app-cache")
    with _running_app({**os.environ, "GATEWAY_CACHE_DIR": str(cache)}) as url:
        yield url


@pytest.fixture(scope="session")
def sparse_app_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The app against a fake Connect in its "sparse" profile: no User-Agent, no
    Gateway-Metadata, almost nothing from content (see demo/fake_connect.py).
    What a real server can look like, and where empty charts and zero-divisions
    show up. Its own fake and app, so the shared ones keep their data.
    """
    with _running_fake(FAKE_ROWS="1500", FAKE_PROFILE="sparse") as fake:
        env = {**os.environ, "CONNECT_SERVER": fake, "CONNECT_API_KEY": "test",
               "GATEWAY_CACHE_DIR": str(tmp_path_factory.mktemp("sparse-cache"))}
        env.pop("GATEWAY_FROM", None)
        env.pop("GATEWAY_TO", None)
        with _running_app(env) as url:
            yield url


@pytest.fixture(scope="session")
def empty_app_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The app against a Connect whose usage export has no rows: a server that
    has the gateway on and has not served a request yet. Its own fake and app,
    so the shared ones keep their data.
    """
    with _running_fake(FAKE_ROWS="0") as fake:
        env = {**os.environ, "CONNECT_SERVER": fake, "CONNECT_API_KEY": "test",
               "GATEWAY_CACHE_DIR": str(tmp_path_factory.mktemp("empty-cache"))}
        env.pop("GATEWAY_FROM", None)
        env.pop("GATEWAY_TO", None)
        with _running_app(env) as url:
            yield url


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test that needs the fake Connect, so -m can select them."""
    for item in items:
        names = getattr(item, "fixturenames", ())
        if "fake_connect" in names:
            item.add_marker(pytest.mark.connect)
        if "app_server" in names:
            item.add_marker(pytest.mark.app)
        if "sparse_app_server" in names or "empty_app_server" in names:
            item.add_marker(pytest.mark.app)
            item.add_marker(pytest.mark.connect)
