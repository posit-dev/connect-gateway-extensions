# Posit AI Gateway usage dashboard

A Shiny for Python dashboard that reports LLM usage from the Posit Connect AI
Gateway. It shows requests, tokens, and estimated spend, and it groups them by
user, content, model, and provider.

The dashboard reads the gateway usage export. This export is one Parquet file
that covers a time window. The app reads it into a local DuckDB table, and
every chart then queries local data. The export holds metadata only. It never
contains the text of a prompt or a reply.

Connect builds the export on request, and the time that takes grows with the
window: 17-30 s for 1.6M records. Each visitor's session reads the export for
itself, with that visitor's own API key, so the page shows a loading bar for
that long when it opens. The data lives in the session's memory and is dropped
when the session ends. Nothing is shared between visitors or saved to disk, and
nothing is read until an administrator opens the dashboard.

At most four sessions load at once; any others wait their turn. When a load
fails, the page says so, and the reason is in the app's log (on Connect, the
content's Logs tab).

## Install from the Gallery

On a Connect server with `[Gateway] Enabled = true`, the dashboard appears in
the Gallery under "AI Gateway Extensions". Only administrators see it. It
needs Connect 2026.11.0 or later.

Connect adds the feed by default. If your `[GalleryFeed]` settings replace the
defaults, add it by hand:

```toml
[GalleryFeed "gateway"]
Name = "AI Gateway Extensions"
URL = "https://raw.githubusercontent.com/posit-dev/connect-gateway-extensions/refs/heads/main/extensions.json"
```

The app lives in `extensions/gateway-usage-dashboard/`. Its `manifest.json`
carries the Gallery metadata. A version bump there, merged to `main`, publishes
a release.

## Configuration

On Connect, the dashboard reads everything as the person viewing it, so it
needs a Connect API integration (Max Role: Admin). Add one in System →
Integrations, then add it to the content's Runtime settings. A visitor who is
not an administrator, or a deployment with no integration, sees an access
message and nothing is read.

Connect sets `CONNECT_SERVER` for content that it runs. To run the app on your
machine, set two environment variables yourself:

- `CONNECT_SERVER` — the Connect URL, for example `http://localhost:3939`.
- `CONNECT_API_KEY` — an administrator API key. With no visitor to act as, a
  local run reads with this key.

Two optional variables pin the export window. Without them, the window trails
the clock by 90 days.

- `GATEWAY_FROM` — the start of the window, in RFC 3339 UTC, for example
  `2026-09-10T00:00:00Z`.
- `GATEWAY_TO` — the end of the window, in the same format.

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

Each session reads the export when it opens, so the backfilled rows show the
next time the page loads. A pinned `GATEWAY_FROM` hides rows before it.

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
Importing `usage` does nothing; each session's `Dataset.load()` reads the data.

The app's packages are pinned. Edit `requirements.in`, run `just lock` to
regenerate `requirements.txt`, and run the tests on the oldest and newest Python
the app supports before you release. The lock gives Python 3.10 pandas 2 and
duckdb 1, and Python 3.11 and later pandas 3 and duckdb 2.

`dev-requirements.txt` adds pytest, pyright, the pandas type stubs, and
`websockets` for the session smoke test. None of it, `pytest.ini`,
`pyrightconfig.json` or `tests/` is deployed.

## Architecture

The paths below are in `extensions/gateway-usage-dashboard/`. `app:app` is the entrypoint. `app.py` is the shell and the wiring. It builds
the page frame and the filter drawer, keeps the state that every page shares,
and starts each page's server. The shared state is the loaded-data catalog, the
filter dict `flt()`, the data load, bookmarks and URL query state, the
topbar, the unpriced-model checks, and the CSV export registration. It also
defines the page order. The page is built empty and filled in once a session's
data has loaded; everything that reads a filter waits for that.

Each page lives in its own module and exposes `panel()`, its `nav_panel`, and
`server(input, ctx)`:

- `pages/overview.py`, `pages/adoption.py` and `pages/raw_data.py`.
- `principals.py` is the Attribution page. It takes Raw Data's SQL builder, and
  flags the outliers among each kind (`outliers`).

`pages/context.py` defines `PageContext`, the explicit set of shared services a
page server receives in place of importing `app.py`: `flt`, `cat`, `where`,
`est_cost`, `register_export`, `formatted` and the rest.

The helper modules hold no Shiny session state of their own:

- `access.py` checks that the visitor is an administrator, and returns the API
  key to read Connect with as them.
- `usage.py` holds a `Dataset` per session: the visitor's own DuckDB copy of
  the export, with Connect's price catalog, content list and groups, all read
  with the visitor's key. `usage.q` and the other queries find the dataset of
  the session they run in.
- `pricing.py` builds the spend SQL from the session's price catalog.
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

