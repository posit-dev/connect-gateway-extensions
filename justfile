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
setup: (_install "requirements.txt")

# Create .venv with the app's requirements plus the development tools.
setup-dev: (_install "dev-requirements.txt")

# Install a requirements file into .venv, creating it if missing. Uses uv when it
# is on PATH: a venv uv made has no pip, so `.venv/bin/pip` would not exist.
_install file:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v uv >/dev/null; then
        [ -x {{python}} ] || uv venv .venv
        uv pip install --python {{python}} -r {{file}}
    else
        [ -x {{python}} ] || python3 -m venv .venv
        {{python}} -m pip install -r {{file}}
    fi

# Type-check the app, pages, tests and mock with pyright (see pyrightconfig.json).
typecheck:
    PYRIGHT_PYTHON_IGNORE_WARNINGS=1 .venv/bin/pyright

# Run every test with pytest. The ones marked `connect` start their own fake
# Connect; pass pytest arguments through, e.g. `just test -k pricing`.
test *args:
    {{python}} -m pytest {{args}}

# Only the tests that need no fake Connect server and no running app.
test-fast *args:
    {{python}} -m pytest -m "not connect and not app" {{args}}

# Regenerate tests/fixtures/connect_export*.parquet with Connect's own export
# code (see tests/fixtures/connect_export_fixture_test.go). It runs in a
# throwaway connect-dev container against a Connect checkout mounted read-only;
# `go test -overlay` adds the generator to the usagelog package without writing
# to the checkout. Needs Docker and the checkout's ext/ubuntu22 DuckDB libraries.
connect-fixtures connect=(env_var("HOME") / "Documents/connect") image="ghcr.io/posit-dev/connect-dev-ubuntu22:latest":
    docker run --rm \
        -v "{{connect}}":/connect:ro \
        -v "{{connect}}/ext/ubuntu22":/connect/ext:ro \
        -v "{{justfile_directory()}}/tests/fixtures":/fixtures \
        -v ai-usage-dashboard-gomod:/root/go/pkg/mod \
        -v ai-usage-dashboard-gobuild:/root/.cache/go-build \
        -e DASHBOARD_FIXTURE_OUT=/fixtures \
        -w /connect/src/connect \
        {{image}} bash -euo pipefail -c ' \
            lib=/connect/ext/duckdb; \
            export CGO_CFLAGS="-I$lib"; \
            export CGO_LDFLAGS="$(for f in $lib/lib*.a; do n=$(basename $f .a); printf -- "-l%s " ${n#lib}; done) -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic -lm -ldl -L$lib"; \
            echo "{\"Replace\": {\"/connect/src/connect/gateway/usagelog/zz_dashboard_fixture_test.go\": \"/fixtures/connect_export_fixture_test.go\"}}" > /tmp/overlay.json; \
            go test -tags duckdb_use_static_lib -overlay /tmp/overlay.json \
                -run TestDashboardFixture -count=1 -v ./gateway/usagelog/'

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
        --exclude 'dev-requirements.txt' --exclude 'pyrightconfig.json' \
        --exclude 'pytest.ini' --exclude 'tests/*' \
        --exclude '.gateway-cache/*'

# The one way the app starts. Do not add mode-specific settings here.
_app server key:
    @CONNECT_SERVER="{{server}}" CONNECT_API_KEY="{{key}}" \
        .venv/bin/shiny run app:app --port {{port}}
