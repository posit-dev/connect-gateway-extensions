"""The app against a server with no gateway usage yet.

An administrator can install the dashboard the day the gateway is switched on,
before any request has gone through it. The export is then empty, and the page
has to load with empty charts rather than fail.
"""

from __future__ import annotations

from session import browser_inputs, fetch_page, open_session, run
from test_app_session import PAGE_OUTPUTS


def test_page_loads_with_no_usage(empty_app_server):
    html = fetch_page(empty_app_server)

    async def go():
        async with open_session(empty_app_server) as session:
            return await session.init(browser_inputs(html))

    reply = run(go())
    assert reply.errors == {}
    for page, outputs in PAGE_OUTPUTS.items():
        for output_id in outputs:
            assert reply.values.get(output_id) is not None, f"{page}: {output_id} has no value"
