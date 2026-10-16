# The dashboard has one code path and two possible backends: a real Connect
# server, or the mock in demo/fake_connect.py. Both recipes below start the app
# the same way (see _app); only CONNECT_SERVER and CONNECT_API_KEY differ.

python := ".venv/bin/python"
port := "8000"
mock_port := "3940"

# List the recipes.
default:
    @just --list

# Create .venv and install the app's requirements.
setup:
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

# Run the app against a real Connect server: `just run <url> <api-key>`, or
# `just run` to use CONNECT_SERVER and CONNECT_API_KEY from the environment.
run server=env_var_or_default("CONNECT_SERVER", "") key=env_var_or_default("CONNECT_API_KEY", ""):
    @[ -n "{{server}}" ] && [ -n "{{key}}" ] || { echo "usage: just run <connect-url> <api-key>" >&2; exit 1; }
    @just port={{port}} _app "{{server}}" "{{key}}"

# Run the app against the mock Connect server, started here and stopped on exit.
run-local:
    #!/usr/bin/env bash
    set -euo pipefail
    MOCK_CONNECT_PORT={{mock_port}} {{python}} demo/fake_connect.py &
    mock=$!
    trap 'kill $mock 2>/dev/null' EXIT
    # The mock builds its Parquet before it listens; wait until it answers.
    until curl -s -o /dev/null http://localhost:{{mock_port}}/; do
        kill -0 $mock 2>/dev/null || { echo "mock Connect exited" >&2; exit 1; }
        sleep 0.5
    done
    just port={{port}} _app "http://localhost:{{mock_port}}" "mock-key"

# Run only the mock Connect server.
mock:
    MOCK_CONNECT_PORT={{mock_port}} {{python}} demo/fake_connect.py

# Deploy to a Connect server: `just deploy <url> <api-key>`, defaulting to the environment.
deploy server=env_var_or_default("CONNECT_SERVER", "") key=env_var_or_default("CONNECT_API_KEY", ""):
    @rsconnect deploy shiny . --server "{{server}}" --api-key "{{key}}" \
        --title "Posit Team AI Usage Dashboard" --entrypoint app:app \
        --exclude '.venv/*' --exclude 'demo/*' --exclude '__pycache__/*' \
        --exclude 'rsconnect-python/*' --exclude 'justfile' \
        --exclude '.gateway-cache/*'

# The one way the app starts. Do not add mode-specific settings here.
_app server key:
    @CONNECT_SERVER="{{server}}" CONNECT_API_KEY="{{key}}" \
        .venv/bin/shiny run app:app --port {{port}}
