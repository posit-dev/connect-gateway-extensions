# Connect AI Gateway usage dashboard

A Shiny for Python dashboard that reports LLM usage from the Posit Connect AI
Gateway. It shows requests, tokens, and estimated spend, and it groups them by
user, content, model, and provider.

The dashboard reads the gateway usage export. This export is one Parquet file
that covers a time window. The app reads it once into a local DuckDB table at
start. Every chart then queries local data. Press "Refresh data" to read a newer
window. The export holds metadata only. It never contains the text of a prompt
or a reply.

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

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/shiny run app:app --port 8000
```

Then open http://localhost:8000.

## Deploy to Agave

The app runs on Agave, https://connect.agave.pwb.posit.it. Sign in once with
OAuth. rsconnect opens a browser and stores the token:

```bash
rsconnect login https://connect.agave.pwb.posit.it
```

Then redeploy new code to the same content item:

```bash
rsconnect deploy shiny . --server https://connect.agave.pwb.posit.it \
  --app-id a9aa52e2-3b47-4d39-ad15-e4dcbe509b72 --entrypoint app:app
```

