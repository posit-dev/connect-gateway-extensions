"""The app entrypoint: its UI, and a real `shiny run` serving a session."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from conftest import APP, _free_port


def test_ui_has_every_page_and_output(loaded):
    import app

    html = str(app.app_ui(None))
    for output_id in (
        "main_tabs", "date_range", "f_users",
        "kpis", "group_by", "grain", "spend_time",
        "adoption_kind", "adoption_kpis", "adoption_lines", "tool_time",
        "model_time", "cache_savings_time", "cache_rate_time",
        "errors_time", "dl_adoption",
        "records_table", "records_title", "records_capped", "dl_records",
        "pf_kind", "pf_pick", "pf_table",
    ):
        assert f'id="{output_id}"' in html
    pages = ["Overview", "Attribution", "Adoption", "Raw Data"]
    positions = [html.index(f'data-value="{p}"') for p in pages]
    assert positions == sorted(positions)


@pytest.mark.app
def test_shiny_run_serves_page_and_session(fake_connect, tmp_path):
    """`shiny run app:app` serves the page, and a session renders every page's outputs."""
    import websockets

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", "app:app", "--port", str(port)],
        cwd=APP, env={**os.environ, "GATEWAY_CACHE_DIR": str(tmp_path)},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    assert server.stderr is not None
    try:
        for _ in range(200):
            if server.poll() is not None:
                pytest.fail(server.stderr.read().decode())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as resp:
                    status, html = resp.status, resp.read().decode()
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            pytest.fail("app did not start")
        assert status == 200
        assert 'id="main_tabs"' in html

        # Every element is reported visible, so no output stays suspended.
        data: dict[str, object] = {f".clientdata_output_{i}_hidden": False
                                   for i in set(re.findall(r'\bid="([^"]+)"', html))}
        data.update(main_tabs="Overview", preset="30d")

        async def session():
            values, errors = set(), {}
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/websocket/", max_size=None
            ) as ws:
                await ws.send(json.dumps({"method": "init", "data": data}))
                try:
                    while True:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                        values.update(msg.get("values") or {})
                        errors.update(msg.get("errors") or {})
                except asyncio.TimeoutError:
                    pass
            return values, errors

        values, errors = asyncio.run(session())
        assert errors == {}
        for output_id in ("topbar_status", "kpis",
                          "pf_header", "adoption_kpis",
                          "records_table", "dl_records"):
            assert output_id in values
    finally:
        server.terminate()
        server.wait(timeout=10)
        server.stderr.close()
