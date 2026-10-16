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

## Configuration

The app needs two environment variables:

- `CONNECT_SERVER` — the Connect URL, for example `http://localhost:3939`.
- `CONNECT_API_KEY` — an administrator API key. The export needs this role.

Connect sets both variables for content that it runs, so a deployed copy needs
no setup. To run the app on your machine, set both variables yourself.

Two optional variables pin the export window. Without them, the window trails
the clock by 30 days.

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

Like Connect, the mock wants an API key on every request. Models the catalog
does not price cost $0 against either backend; the mock's `gpt-5` traffic is
one such model.

## Deploy to Agave

The app runs on Agave, https://connect.agave.pwb.posit.it. Sign in once with
OAuth. rsconnect opens a browser and stores the token:

```bash
rsconnect login https://connect.agave.pwb.posit.it
```

Then redeploy new code to the same content item. To deploy to any other server
with an API key, run `just deploy <connect-url> <api-key>`.

```bash
rsconnect deploy shiny . --server https://connect.agave.pwb.posit.it \
  --app-id a9aa52e2-3b47-4d39-ad15-e4dcbe509b72 --entrypoint app:app
```

