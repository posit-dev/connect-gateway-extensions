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

from conftest import ROOT, _free_port


def test_ui_has_every_page_and_output(loaded):
    import app
    import pricing
    from pages import conversations as conversations_page

    assert app.est_cost is pricing.est_cost
    assert conversations_page.rate_sql is pricing.rate_sql
    html = str(app.app_ui(None))
    for output_id in (
        "main_tabs", "date_range", "f_users",
        "kpis", "origin_tool", "origin_total", "origin_share", "provider_total",
        "provider_share", "grain", "model_time", "model_share",
        "adoption_kpis", "adoption_lines", "tool_time", "usage_heatmap",
        "models_bars", "models_adoption", "cache_savings_time", "latency_time",
        "errors_time", "models_table", "dl_adoption", "dl_models",
        "records_table", "records_title", "records_capped", "dl_records",
        "attr_kind", "attr_order", "attr_kpis", "attr_bars", "attr_rate_bars",
        "attr_scatter", "attr_table", "dl_attr",
        "conv_kpis", "conv_by", "conv_spend_by", "conv_spend_note",
        "conv_detail", "top_sessions", "dl_conversations",
    ):
        assert f'id="{output_id}"' in html
    pages = ["Overview", "Attribution", "Conversations", "Principals",
             "Adoption", "Models", "Raw Data"]
    positions = [html.index(f'data-value="{p}"') for p in pages]
    assert positions == sorted(positions)


@pytest.mark.app
def test_shiny_run_serves_page_and_session(fake_connect, tmp_path):
    """`shiny run app:app` serves the page, and a session renders every page's outputs."""
    import websockets

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", "app:app", "--port", str(port)],
        cwd=ROOT, env={**os.environ, "GATEWAY_CACHE_DIR": str(tmp_path)},
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
        data.update(main_tabs="Overview", preset="30d", dark_mode="dark")

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
        for output_id in ("topbar_status", "kpis", "attr_table", "conv_kpis",
                          "pf_header", "adoption_kpis", "models_table",
                          "records_table", "dl_records"):
            assert output_id in values
    finally:
        server.terminate()
        server.wait(timeout=10)
        server.stderr.close()
