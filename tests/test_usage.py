"""Run with .venv/bin/python -m unittest discover -s tests."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class UsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "MOCK_CONNECT_PORT": str(port), "FAKE_ROWS": "1000"}
        cls.mock = subprocess.Popen(
            [sys.executable, str(ROOT / "demo" / "fake_connect.py")],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            for _ in range(100):
                if cls.mock.poll() is not None:
                    raise RuntimeError(cls.mock.stderr.read().decode())
                try:
                    # Any HTTP status means the mock has finished building its export.
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                except urllib.error.HTTPError:
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("mock Connect did not start")
            os.environ.update(
                CONNECT_SERVER=f"http://127.0.0.1:{port}", CONNECT_API_KEY="test",
                GATEWAY_CACHE_DIR=cls.cache.name,
            )
            import usage
            cls.usage = usage
        except BaseException:
            cls.mock.terminate()
            cls.mock.wait()
            cls.cache.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.mock.terminate()
        cls.mock.wait(timeout=5)
        cls.mock.stderr.close()
        cls.cache.cleanup()

    def test_refresh_exports_full_start_time_window(self):
        seen = []
        download = self.usage._download

        def recording_download(url):
            seen.append(url)
            return download(url)

        before = self.usage.INFO["rows"]
        with patch.object(self.usage, "_download", recording_download):
            info = self.usage.load()
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(seen[0]).query)
        self.assertEqual(info["mode"], "incremental")
        self.assertEqual(params["from"], [info["window_from"]])
        self.assertEqual(params["to"], [info["window_to"]])
        self.assertEqual(info["rows"], before)

    def test_content_names_across_pages(self):
        items = [{"guid": str(i), "title": f"Title {i}"} for i in range(215)]

        def get(url):
            page = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page_number"][0])
            return items[(page - 1) * 100:page * 100]

        with patch.object(self.usage, "_get_json", get):
            names = self.usage._fetch_content_names()
        self.assertEqual(len(names), 215)
        self.assertEqual(names["214"], "Title 214")

    def test_group_members_across_capped_pages_and_failure(self):
        def get(url):
            page = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page_number"][0])
            path = urllib.parse.urlsplit(url).path
            if path.endswith("/groups"):
                return {"results": [{"guid": "group", "name": "Team"}], "total": 1}
            if page == 2 and self.fail_page:
                return None
            members = [{"guid": str(i)} for i in range(5)]
            return {"results": members[(page - 1) * 2:page * 2],
                    "current_page": page, "total": 5}

        self.fail_page = False
        with patch.object(self.usage, "_get_json", get):
            self.assertEqual(len(self.usage._fetch_groups()), 5)
            self.fail_page = True
            self.assertIsNone(self.usage._fetch_groups())


if __name__ == "__main__":
    unittest.main()
