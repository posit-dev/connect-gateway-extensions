"""A browser stand-in: a websocket session against a running `shiny run` app.

Shiny's page tells the server which outputs are visible and what the URL is,
then sends input changes; the server answers with output values, errors and
messages for the page. This speaks that protocol, so a test sees what a browser
would be given, without a browser.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import urllib.request
from datetime import date
from html import unescape as html_unescape
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import numpy as np
import websockets

# A reply ends when the app reports idle. Output values can still arrive just
# after that, so the connection is read until it has been quiet for GRACE_SECONDS.
# LIMIT_SECONDS stops a reply that never goes idle from hanging a test.
GRACE_SECONDS = 1.0
LIMIT_SECONDS = 60


@dataclass
class Reply:
    """Everything the server sent since the last call, merged."""

    values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)  # the rest, in order

    def notifications(self) -> list[str]:
        """The HTML of each notification shown."""
        return [
            m["notification"]["message"]["html"]
            for m in self.messages
            if m.get("notification", {}).get("type") == "show"
        ]

    def add(self, later: Reply) -> None:
        """Fold in what the server sent after this reply."""
        self.values.update(later.values)
        self.errors.update(later.errors)
        self.messages += later.messages

    def is_loaded(self) -> bool:
        """Whether the server has said the data is in (or that loading failed)."""
        return any("loaded" in (m.get("custom") or {}) for m in self.messages)

    def query_strings(self) -> list[str]:
        """Each URL the app asked the browser to show, in order."""
        return [m["updateQueryString"]["queryString"] for m in self.messages
                if "updateQueryString" in m]


def figure_traces(reply: Reply, output_id: str) -> list[dict]:
    """The traces of the plotly figure a widget output was given.

    The output's value names a widget model; the figure arrives separately, as
    the state of a shinywidgets comm whose ident is that model's id. Numeric
    arrays travel beside the state as base64 buffers; they are put back as lists.
    """
    model = reply.values[output_id]["model_id"]
    for m in reply.messages:
        opened = (m.get("custom") or {}).get("shinywidgets_comm_open")
        if opened:
            msg = json.loads(opened)
            if msg.get("ident") == f"comm-{model}":
                data = msg["content"]["data"]
                # Each path ends in "buffer", a key of the {dtype, shape} array.
                for path, buf in zip(data["buffer_paths"], msg["buffers"]):
                    node = data["state"]
                    for key in path[:-2]:
                        node = node[key]
                    dtype = node[path[-2]]["dtype"]
                    node[path[-2]] = np.frombuffer(base64.b64decode(buf), dtype).tolist()
                return data["state"]["_data"]
    raise AssertionError(f"no figure sent for {output_id}")


def page_outputs(html: str) -> list[str]:
    """Every element id on the page: the outputs a browser would report visible."""
    return sorted(set(re.findall(r'\bid="([^"]+)"', html)))


def fetch_page(base_url: str, query: str = "") -> str:
    """The page's HTML. A bookmark is restored here, into the inputs' initial values."""
    with urllib.request.urlopen(base_url + "/" + query, timeout=5) as resp:
        return resp.read().decode()


class Session:
    def __init__(self, ws, base_url: str):
        self._ws = ws
        self.base_url = base_url
        self.downloads: dict[str, str] = {}  # download button id -> its URL, once sent

    async def _drain(self) -> Reply:
        reply = Reply()
        idle = False
        deadline = asyncio.get_running_loop().time() + LIMIT_SECONDS
        while True:
            timeout = GRACE_SECONDS if idle else deadline - asyncio.get_running_loop().time()
            try:
                msg = json.loads(await asyncio.wait_for(self._ws.recv(), max(timeout, 0.01)))
            except asyncio.TimeoutError:
                if not idle:
                    raise AssertionError("the app never went idle") from None
                return reply
            reply.values.update(msg.get("values") or {})
            self.downloads.update(
                {k: v for k, v in (msg.get("values") or {}).items() if k.startswith("dl_")}
            )
            reply.errors.update(msg.get("errors") or {})
            idle = idle or msg.get("busy") == "idle"
            rest = {k: v for k, v in msg.items() if k not in ("values", "errors")}
            if rest:
                reply.messages.append(rest)

    async def init(self, inputs: dict, url: str = "") -> Reply:
        """Open the page as a browser does. `url` is its address, query string included."""
        parts = urlsplit(self.base_url + "/" + url.lstrip("/"))
        data = {
            ".clientdata_url_protocol": "http:",
            ".clientdata_url_hostname": parts.hostname,
            ".clientdata_url_port": parts.port,
            ".clientdata_url_pathname": "/",
            ".clientdata_url_search": f"?{parts.query}" if parts.query else "",
            ".clientdata_url_hash_initial": "",
            ".clientdata_url_hash": "",
            **inputs,
        }
        await self._ws.send(json.dumps({"method": "init", "data": data}))
        reply = await self._drain()
        # The app is idle while its data loads in the background. The page is
        # ready when the server says so, and what it draws arrives with that.
        while not reply.is_loaded():
            reply.add(await self._drain())
        return reply

    async def update(self, **inputs) -> Reply:
        await self._ws.send(json.dumps({"method": "update", "data": inputs}))
        return await self._drain()

    def download(self, output_id: str) -> str:
        """The file a download button serves now, as the browser would fetch it.

        The file is built from the filter as the session holds it at this moment.
        """
        url = self.downloads[output_id]
        with urllib.request.urlopen(f"{self.base_url}/{url}", timeout=30) as resp:
            return resp.read().decode()


@asynccontextmanager
async def open_session(base_url: str):
    ws_url = base_url.replace("http://", "ws://") + "/websocket/"
    async with websockets.connect(ws_url, max_size=None) as ws:
        yield Session(ws, base_url)


def browser_inputs(html: str) -> dict[str, object]:
    """What a browser reports on load: each input's default, and every output visible.

    Radio buttons give their checked choice, the date range its initial dates (or
    today's, as a browser fills in a date with none), the checkbox filters their
    ticked boxes, and the selectize filters nothing. Pass the HTML of a bookmarked
    address and these are the restored values. An output reported hidden is not
    rendered, so all of them are reported visible.
    """
    inputs: dict[str, object] = {
        f".clientdata_output_{i}_hidden": False for i in page_outputs(html)
    }
    for name, value in re.findall(
        r'<input type="radio" name="([^"]+)" value="([^"]*)" checked="checked"', html
    ):
        inputs[name] = value
    box = re.search(r'id="date_range".*?</div>', html, re.S)
    assert box, "no date range on the page"
    initial = [re.search(r'data-initial-date="([^"]+)"', tag)
               for tag in re.findall(r"<input[^>]*>", box.group(0))]
    inputs["date_range"] = [m.group(1) if m else date.today().isoformat() for m in initial]
    for name in ("f_users", "f_contents", "f_sessions", "f_groups"):  # selects: the selected
        box = re.search(rf'<select[^>]*id="{name}"[^>]*>(.*?)</select>', html, re.S)
        inputs[name] = [html_unescape(v) for v in re.findall(
            r'<option value="([^"]*)" selected', box.group(1) if box else "")]
    inputs["f_meta"] = [  # a checkbox group: the ticked ones
        html_unescape(v) for v in re.findall(
            r'<input type="checkbox" name="f_meta" value="([^"]*)" checked', html)
    ]
    active = re.search(
        r'data-value="([^"]+)"[^>]*class="nav-link active"[^>]*href="#tab-main_tabs-', html)
    inputs["main_tabs"] = active.group(1) if active else "Overview"
    # The Attribution picker starts empty and the server fills it; a browser
    # reports the empty value, and without it the choices never load.
    inputs["pf_pick"] = ""
    return inputs


def run(coro):
    return asyncio.run(coro)


__all__ = ["Reply", "Session", "browser_inputs", "fetch_page", "figure_traces",
           "open_session", "page_outputs", "run"]

