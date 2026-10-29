# Connect AI Gateway usage dashboard

A Shiny for Python dashboard that reports LLM usage from the Posit Connect AI
Gateway. It shows requests, tokens, and estimated spend, and it groups them by
user, content, model, and provider.

The dashboard reads the gateway usage export. This export is one Parquet file
that covers a time window. The app reads it into a local DuckDB table, and
every chart then queries local data. The export holds metadata only. It never
contains the text of a prompt or a reply.

Connect builds the export on request, and the time that takes grows with the
window: 17-30 s for 1.6M records. Each refresh exports the whole start-time
window, then merges only recently ingested records locally. The full export is
necessary because a call can be ingested long after it started; requesting only
recent start-time partitions would omit it. The app rebuilds its local table
from scratch every 24 hours. A Parquet snapshot in `.gateway-cache/` lets a
restarted process reuse its existing rows, although catching up still requires
a full-window export from Connect.

## Install from the Gallery

On a Connect server with `[Gateway] Enabled = true`, the dashboard appears in
the Gallery under "LLM Gateway Extensions". Only administrators see it. It
needs Connect 2026.11.0 or later.

Connect adds the feed by default. If your `[GalleryFeed]` settings replace the
defaults, add it by hand:

```toml
[GalleryFeed "gateway"]
Name = "LLM Gateway Extensions"
URL = "https://raw.githubusercontent.com/posit-dev/connect-gateway-extensions/refs/heads/main/extensions.json"
```

The app lives in `extensions/gateway-usage-dashboard/`. Its `manifest.json`
carries the Gallery metadata. A version bump there, merged to `main`, publishes
a release.

## Configuration

The app needs two environment variables:

- `CONNECT_SERVER` — the Connect URL, for example `http://localhost:3939`.
- `CONNECT_API_KEY` — an administrator API key. The export needs this role.

Connect sets both variables for content that it runs, so a deployed copy needs
no setup. To run the app on your machine, set both variables yourself.

Two optional variables pin the export window. Without them, the window trails
the clock by 90 days.

- `GATEWAY_FROM` — the start of the window, in RFC 3339 UTC, for example
  `2026-09-10T00:00:00Z`.
- `GATEWAY_TO` — the end of the window, in the same format.

`GATEWAY_CACHE_DIR` moves the snapshot, for example to a shared mount that an
administrator set up and that outlives deploys.

## Run locally

The app has one code path and two possible backends. It behaves the same with
either one; only the server it reads from changes. There is no built-in data
and no built-in price list, so all usage, prices and content names come from
the backend.

```bash
just setup                               # once: .venv + requirements
just run-local                           # against the mock Connect (demo/)
just run <connect-url> <admin-api-key>   # against a real Connect
just run                                 # same, using CONNECT_SERVER / CONNECT_API_KEY
```

Then open http://localhost:8000.

`just run-local` starts `demo/fake_connect.py` on port 3940 and stops it when
the app exits. The mock serves the same four routes the app reads from Connect:

- the usage export, with synthetic data
- the price catalog, as a snapshot of a real one in `demo/pricing.json`
- the price overrides, which are empty
- the content list

`FAKE_PROFILE=sparse just run-local` makes the mock look like a server whose
callers send no User-Agent and no Gateway-Metadata, with almost all traffic
outside content: no harness, no Workbench session, and an origin only on content
rows. Real servers can look like this, and it is the case the origin, harness and
session views have to handle without going blank (`tests/test_sparse.py`).

Like Connect, the mock wants an API key on every request. Models the catalog
does not price cost $0 against either backend; the mock's `gpt-5` traffic is
one such model.

## Backfill a real Connect with demo usage

`demo/backfill_connect.py` fills a real Connect's gateway usage log with synthetic
usage for a demo server. It reads the server's users, content and prices through
the API (`CONNECT_SERVER`, `CONNECT_API_KEY`), so every GUID in a row resolves
when the dashboard calls Connect. It writes JSONL files in the layout Connect
uses for its own usage log, under a node name of its own.

```bash
python demo/backfill_connect.py --out /tmp/backfill --integration <integration-guid>
tar -C /tmp/backfill -cf - . | kubectl -n <ns> exec -i <connect-pod> -c connect -- \
  tar -C /var/lib/rstudio-connect/gateway -xf -
```

To remove the data, delete the `node=<node>` directories (the default node is
`connect-backfill`). The node name is the only mark of the rows.

A running dashboard only merges recently ingested records, and the backfilled
rows have older ingest times. It shows them after its next full reload (every 24
hours), or sooner if you change its `GATEWAY_FROM` or `GATEWAY_TO` setting,
which restarts it with a full reload. A pinned `GATEWAY_FROM` hides rows before
it.

## Develop

```bash
just setup-dev    # once: .venv + requirements + dev-requirements.txt
just test         # every test (pytest); some start their own mock Connect
just test-fast    # only the tests that need no server, in about a second
just typecheck    # pyright, configured in pyrightconfig.json
```

The tests give real inputs to real code rather than mocking it: usage exports
that Connect's own export code built (`tests/fixtures/`), small Parquet exports
written by the test with the same schema (see `tests/conftest.py`), private
DuckDB connections, and `demo/fake_connect.py` for anything that talks to
Connect. A test checks that the fake serves exactly Connect's export schema.

To rebuild the Connect exports after Connect's export changes, run
`just connect-fixtures [path-to-connect-checkout]`. It runs the generator in
`tests/fixtures/connect_export_fixture_test.go` inside a throwaway connect-dev
container, with the checkout mounted read-only.
Importing `usage` does nothing; `app.py` runs the first `usage.load()`.

`dev-requirements.txt` adds pytest, pyright, the pandas type stubs, and
`websockets` for the session smoke test. None of it, `pytest.ini`,
`pyrightconfig.json` or `tests/` is deployed.

## Architecture

The paths below are in `extensions/gateway-usage-dashboard/`. `app:app` is the entrypoint. `app.py` is the shell and the wiring. It builds
the page frame and the filter drawer, keeps the state that every page shares,
and starts each page's server. The shared state is the loaded-data catalog, the
filter dict `flt()`, the data refresh, bookmarks and URL query state, the
topbar, the unpriced-model checks, and the CSV export registration. It also
defines the page order.

Each page lives in its own module and exposes `panel()`, its `nav_panel`, and
`server(input, ctx)`:

- `pages/overview.py`, `pages/adoption.py` and `pages/raw_data.py`.
- `principals.py` is the Attribution page. It takes Raw Data's SQL builder, and
  flags the outliers among each kind (`outliers`).

`pages/context.py` defines `PageContext`, the explicit set of shared services a
page server receives in place of importing `app.py`: `flt`, `cat`, `where`,
`est_cost`, `register_export`, `formatted` and the rest.

The helper modules hold no Shiny session state:

- `usage.py` loads the export into DuckDB and reads Connect's price catalog
  and content list.
- `pricing.py` builds the spend SQL from the price catalog and rebuilds it
  after a refresh.
- `filters.py` turns the filter dict into a SQL `WHERE` clause and quotes
  values. It also holds `TOKEN_COUNT`, the SQL for a row's token count.
- `conversations.py` groups requests into conversations from data frames,
  and holds the higher-cost flag (`HIGH_PCT`, `higher_cost`) that Attribution
  uses.
- `ui_shell.py` holds the navigation rail, the filter drawer and shared page
  components.
- `theme.py` holds the chart styling, palette and stat tiles.

## Deploy by hand

Most servers should install the app from the Gallery (see above). To deploy a
copy yourself, run `just deploy <connect-url> <api-key>`, or call rsconnect
directly. Add `--app-id <content-guid>` to redeploy to an existing content item.

```bash
rsconnect deploy shiny extensions/gateway-usage-dashboard \
  --server <connect-url> --api-key <api-key> --entrypoint app:app
```

