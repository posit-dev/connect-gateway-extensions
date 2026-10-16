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

    def test_conversation_grouping_without_shiny_session(self):
        import pandas as pd
        import conversations

        df = pd.DataFrame({
            "est": [1.0] * 19 + [10.0],
            "main_model": ["small"] * 19 + ["large"],
            "high": [False] * 19 + [True],
            "calls": list(range(1, 21)),
            "duration_min": [1.0] * 20,
            "cache_hit": [0.0] * 20,
            "med_idle": [None] + [3.0] * 19,
        })
        key, order = conversations.conv_keys(df, "model")
        self.assertEqual(order, ["small", "large"])
        grouped = conversations.conv_groups(df, key, order)
        self.assertEqual(grouped.spend.tolist(), [19.0, 10.0])
        self.assertEqual(grouped.n_hi.tolist(), [0, 1])
        table = conversations.conv_group_table(grouped)
        self.assertEqual(table.pct_of_spend.sum(), 100.0)
        self.assertEqual(len(conversations.conv_spend_fig(grouped, "model", "dark", 90).data), 2)
        pace, pace_order = conversations.conv_keys(df, "pace")
        self.assertEqual(pace.iloc[0], conversations.SINGLE)
        self.assertEqual(pace_order[0], conversations.SINGLE)

    def test_app_entrypoint_uses_pricing_module(self):
        import app
        import pricing

        from pages import conversations as conversations_page

        self.assertIs(app.est_cost, pricing.est_cost)
        self.assertIs(conversations_page.rate_sql, pricing.rate_sql)
        html = str(app.app_ui(None))
        self.assertIn('id="main_tabs"', html)
        self.assertIn('id="date_range"', html)
        self.assertIn('id="f_users"', html)
        for output_id in (
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
            self.assertIn(f'id="{output_id}"', html)
        # Page order is unchanged by the moves: Overview first, then the rest.
        pages = ["Overview", "Attribution", "Conversations", "Principals",
                 "Adoption", "Models", "Raw Data"]
        positions = [html.index(f'data-value="{p}"') for p in pages]
        self.assertEqual(positions, sorted(positions))

    def test_attribution_page_ranking_and_export(self):
        from types import SimpleNamespace

        from shiny import reactive

        import app
        import filters
        import pricing
        from pages import attribution
        from pages.context import PageContext

        c = app.catalog()
        f = {
            "start": c["start"], "end": c["end"], "measure": "spend", "source": "both",
            "dim": "username", "all_providers": c["providers"], "all_models": c["models"],
            "all_users": c["users"], "all_contents": c["contents"], "providers": [],
            "models": [], "users": [], "contents": [], "sessions": [], "meta": [],
            "mode": "dark",
        }

        def attribution_export(attr_order: str):
            """The page's registered exports and its Attribution CSV frame."""
            exports = {}
            ctx = PageContext(
                flt=lambda: f, where=filters.where, cat=app.catalog,
                est_cost=pricing.est_cost,
                register_export=lambda name, fn, filename: exports.update(
                    {name: (fn, filename)}
                ),
                register_sql_export=None, est_cost_uncached=pricing.est_cost_uncached,
                total_bars=None, measure_of=app.measure_of, formatted=app.formatted,
                chart_labels=app.chart_labels, fitted_grid=app.fitted_grid,
                priced_only=lambda: "",
            )
            inputs = SimpleNamespace(attr_kind=lambda: "all", attr_order=lambda: attr_order)
            attribution.server(inputs, ctx, lambda kind, key: None)
            with reactive.isolate():
                return exports, exports["attr"][0]()

        exports, df = attribution_export("spend")
        self.assertEqual(list(exports), ["attr"])
        self.assertEqual(exports["attr"][1], "attribution.csv")
        self.assertEqual(
            df.columns.tolist(),
            ["kind", "id", "name", "detail", "why", "est_spend_usd", "requests",
             "per_request", "x_median", "tokens"],
        )
        self.assertEqual(set(df.kind), {"user", "content", "session"})
        self.assertTrue((df.est_spend_usd > 0).all())
        self.assertEqual(df.est_spend_usd.tolist(), sorted(df.est_spend_usd, reverse=True))
        users = df[df.kind == "user"]
        user_spend = self.usage.scalar(f"""SELECT SUM(s) FROM (
            SELECT SUM({pricing.est_cost()}) s FROM gateway_events
            WHERE {filters.where(f)} AND username IS NOT NULL AND username <> ''
            GROUP BY username HAVING SUM({pricing.est_cost()}) > 0)""")
        self.assertAlmostEqual(users.est_spend_usd.sum(), user_spend)
        self.assertEqual(users.id.tolist(), users.name.tolist())

        _, grouped = attribution_export("type")
        kinds = grouped.kind.drop_duplicates().tolist()
        self.assertEqual(kinds, ["user", "content", "session"])
        self.assertEqual(len(grouped.kind.ne(grouped.kind.shift()).cumsum().unique()), 3)

    def test_conversations_page_export_and_unpriced_exclusion(self):
        from types import SimpleNamespace

        from shiny import reactive

        import app
        import conversations
        import filters
        import pricing
        from pages import conversations as conversations_page
        from pages.context import PageContext

        c = app.catalog()
        f = {
            "start": c["start"], "end": c["end"], "measure": "spend", "source": "both",
            "dim": "username", "all_providers": c["providers"], "all_models": c["models"],
            "all_users": c["users"], "all_contents": c["contents"], "providers": [],
            "models": [], "users": [], "contents": [], "sessions": [], "meta": [],
            "mode": "dark",
        }

        def conversations_export(priced_only: str):
            """The page's registered exports and its conversations CSV frame."""
            exports = {}
            ctx = PageContext(
                flt=lambda: f, where=filters.where, cat=app.catalog,
                est_cost=pricing.est_cost,
                register_export=lambda name, fn, filename: exports.update(
                    {name: (fn, filename)}
                ),
                register_sql_export=None, est_cost_uncached=pricing.est_cost_uncached,
                total_bars=None, measure_of=app.measure_of, formatted=app.formatted,
                chart_labels=app.chart_labels, fitted_grid=app.fitted_grid,
                priced_only=lambda: priced_only,
            )
            conversations_page.server(SimpleNamespace(conv_by=lambda: "model"), ctx)
            with reactive.isolate():
                return exports, exports["conversations"][0]()

        exports, df = conversations_export("")
        self.assertEqual(list(exports), ["conversations"])
        self.assertEqual(exports["conversations"][1], "conversations.csv")
        self.assertFalse(df.empty)
        for col in ("conversation_id", "principal", "calls", "est", "main_model",
                    "high", "line"):
            self.assertIn(col, df.columns)
        for k, *_ in conversations.LEVERS:
            self.assertIn(f"sav_{k}", df.columns)
        self.assertEqual(df.est.tolist(), sorted(df.est, reverse=True))
        # The per-part cost the page sums equals the catalog's own estimate.
        total = self.usage.scalar(f"""SELECT SUM({pricing.est_cost()}) FROM gateway_events
            WHERE {filters.where(f)} AND conversation_id IS NOT NULL""")
        self.assertAlmostEqual(df.est.sum(), total, places=6)
        self.assertEqual(df.calls.sum(), self.usage.scalar(
            f"""SELECT COUNT(*) FROM gateway_events
                WHERE {filters.where(f)} AND conversation_id IS NOT NULL"""))

        # The shared unpriced-model fragment from app.py narrows the page's query.
        top = df.main_model.dropna().iloc[0]
        _, narrowed = conversations_export(
            f" AND (model IS NULL OR model NOT IN ({filters.sql_str(top)}))"
        )
        self.assertNotIn(top, set(narrowed.main_model.dropna()))
        self.assertLess(narrowed.calls.sum(), df.calls.sum())

    def test_principals_page_exports_use_raw_data_sql(self):
        from types import SimpleNamespace

        from shiny import reactive, ui

        import app
        import filters
        import pricing
        import principals
        from pages import raw_data
        from pages.context import PageContext

        c = app.catalog()
        f = {
            "start": c["start"], "end": c["end"], "measure": "spend", "source": "both",
            "dim": "username", "all_providers": c["providers"], "all_models": c["models"],
            "all_users": c["users"], "all_contents": c["contents"], "providers": [],
            "models": [], "users": [], "contents": [], "sessions": [], "meta": [],
            "mode": "dark",
        }
        html = str(ui.TagList(ui.navset_tab(principals.panel(), id="main_tabs")))
        self.assertIn('data-value="Principals"', html)
        for output_id in (
            "pf_kind", "pf_pick", "pf_header", "pf_kpis", "pf_grain", "pf_trend",
            "pf_models", "pf_providers", "pf_origins", "pf_records_title", "pf_records",
            "dl_pf_records", "pf_rel_0_chart", "pf_rel_1", "dl_pf_rel_1",
            "pf_conv_by", "pf_conv_spend", "pf_conv_groups", "dl_pf_conv_groups",
        ):
            self.assertIn(f'id="{output_id}"', html)

        exports, sql_exports = {}, {}
        ctx = PageContext(
            flt=lambda: f, where=filters.where, cat=app.catalog,
            est_cost=pricing.est_cost,
            register_export=lambda name, fn, filename: exports.update(
                {name: (fn, filename)}
            ),
            register_sql_export=lambda name, fn, filename: sql_exports.update(
                {name: (fn, filename)}
            ),
            est_cost_uncached=pricing.est_cost_uncached, total_bars=None,
            measure_of=app.measure_of, formatted=app.formatted,
            chart_labels=app.chart_labels, fitted_grid=app.fitted_grid,
            priced_only=lambda: "",
        )
        peers = principals.peers_df("user", filters.where(f), pricing.est_cost())
        user = peers.k.iloc[0]
        inputs = SimpleNamespace(
            pf_kind=lambda: "user", pf_pick=lambda: user,
            pf_conv_by=lambda: "model", pf_grain=lambda: "day",
        )
        records_sql = raw_data.server(inputs, ctx)
        open_profile = principals.server(inputs, ctx, records_sql)
        self.assertTrue(callable(open_profile))
        self.assertEqual(
            {name: filename for name, (_, filename) in exports.items()},
            {"pf_rel_0": "pf_connections_1.csv", "pf_rel_1": "pf_connections_2.csv",
             "pf_conv_groups": "principal_conversation_groups.csv"},
        )
        self.assertEqual(sql_exports["pf_records"][1], "principal_records.csv")

        with reactive.isolate():
            # The uncapped export is the Raw Data builder over this principal's rows.
            sql = sql_exports["pf_records"][0]()
            self.assertEqual(
                sql, records_sql(f"{filters.where(f)} AND {principals.pred('user', user)}")
            )
            records = self.usage.q(sql)
            groups = exports["pf_conv_groups"][0]()
            content = exports["pf_rel_0"][0]()
        self.assertEqual(len(records), peers.requests.iloc[0])
        self.assertEqual(set(records.username), {user})
        self.assertNotIn("LIMIT", sql)
        self.assertEqual(groups.conversations.sum(), records.conversation_id.nunique())
        self.assertIn("content", content.columns)

    def test_pricing_sql_and_catalog_refresh(self):
        import duckdb
        import pricing

        prices = {
            ("anthropic", "", "o'hare"): (2, 10, 3, 0.5),
            ("aws.bedrock", "", "o'hare"): (4, 12, 5, 1),
            ("aws.bedrock", "us", "o'hare"): (6, 14, 7, 2),
        }
        with patch.object(self.usage, "PRICES", prices):
            cost = pricing.est_cost()
            self.assertIs(cost, pricing.est_cost())
            self.assertEqual(pricing.model_rates("o'hare"), (2e-6, 10e-6, 3e-6, 0.5e-6))
            source = """(VALUES
                ('anthropic', '', 'o''hare', 1000000, 200000, 100000, 200000),
                ('bedrock', 'us', 'o''hare', 1000000, 200000, 100000, 200000),
                ('bedrock', 'eu', 'o''hare', 1000000, 200000, 100000, 200000),
                ('other', '', 'o''hare', 1000000, 200000, 100000, 200000),
                ('other', '', 'unknown', 1000000, 200000, 100000, 200000)
            ) AS events(provider, region, model, input_tokens, output_tokens,
                        cache_creation_tokens, cache_read_tokens)"""
            parts = pricing.rate_sql()
            with duckdb.connect() as con:
                rows = con.sql(f"""SELECT {cost} AS billed,
                    {pricing.est_cost_uncached()} AS uncached,
                    ({parts['i']} * (input_tokens - cache_read_tokens - cache_creation_tokens)
                     + {parts['o']} * output_tokens
                     + {parts['cw']} * cache_creation_tokens
                     + {parts['cr']} * cache_read_tokens) AS parts
                    FROM {source}""").fetchall()
            self.assertEqual([round(r[0], 2) for r in rows], [3.8, 8.1, 5.9, 3.8, 0])
            self.assertEqual([round(r[1], 2) for r in rows], [4.0, 8.8, 6.4, 4.0, 0])
            for billed, _, summed in rows:
                self.assertAlmostEqual(billed, summed)

            with patch.object(self.usage, "PRICES", {("anthropic", "", "o'hare"): (9, 10, 3, 0.5)}):
                self.assertNotEqual(pricing.est_cost(), cost)
                self.assertIsNot(pricing.rate_sql(), parts)

    def test_app_entrypoint_serves_page_and_session(self):
        """`shiny run app:app` serves the page, and a session renders every page's outputs."""
        import asyncio
        import json
        import re

        import websockets

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        server = subprocess.Popen(
            [sys.executable, "-m", "shiny", "run", "app:app", "--port", str(port)],
            cwd=ROOT, env={**os.environ, "GATEWAY_CACHE_DIR": cache.name},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            for _ in range(200):
                if server.poll() is not None:
                    self.fail(server.stderr.read().decode())
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as resp:
                        status, html = resp.status, resp.read().decode()
                    break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
            else:
                self.fail("app did not start")
            self.assertEqual(status, 200)
            self.assertIn('id="main_tabs"', html)

            # Every element is reported visible, so no output stays suspended.
            data = {f".clientdata_output_{i}_hidden": False
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
            self.assertEqual(errors, {})
            for output_id in ("topbar_status", "kpis", "attr_table", "conv_kpis",
                              "pf_header", "adoption_kpis", "models_table",
                              "records_table", "dl_records"):
                self.assertIn(output_id, values)
        finally:
            server.terminate()
            server.wait(timeout=10)
            server.stderr.close()


if __name__ == "__main__":
    unittest.main()
