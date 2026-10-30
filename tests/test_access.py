"""The visitor check that keeps the owner's API key from serving everyone."""

from email.message import Message
import io
import json
import urllib.error
import urllib.parse

import pytest

import access


def _respond(routes):
    def fake(req, timeout=None):
        body = routes[req.full_url.rsplit("/", 1)[-1]]
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(json.dumps(body).encode())
    return fake


@pytest.fixture
def on_connect(monkeypatch):
    monkeypatch.setenv("CONNECT_SERVER", "http://connect")
    monkeypatch.setenv("CONNECT_API_KEY", "owner")
    monkeypatch.setenv("CONNECT_CONTENT_GUID", "abc")


def test_local_run_is_not_checked(monkeypatch):
    monkeypatch.delenv("CONNECT_CONTENT_GUID", raising=False)
    assert access.denial({}) is None


def test_no_session_token_needs_integration(on_connect):
    assert access.denial({}) == access.NEEDS_INTEGRATION


@pytest.mark.parametrize("role,expected", [
    ("administrator", None), ("publisher", access.NOT_ADMIN), ("viewer", access.NOT_ADMIN),
])
def test_role_decides(on_connect, monkeypatch, role, expected):
    monkeypatch.setattr("urllib.request.urlopen", _respond(
        {"credentials": {"access_token": "k"}, "user": {"user_role": role}}))
    assert access.denial({access.SESSION_TOKEN_HEADER: "t"}) == expected


def test_exchange_requests_visitor_api_key(on_connect, monkeypatch):
    requests = []
    respond = _respond({"credentials": {"access_token": "visitor"},
                        "user": {"user_role": "administrator"}})

    def fake(req, timeout=None):
        requests.append(req)
        return respond(req, timeout)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert access.denial({access.SESSION_TOKEN_HEADER: "t"}) is None
    exchange, user = requests
    assert exchange.full_url == "http://connect/__api__/v1/oauth/integrations/credentials"
    assert exchange.get_method() == "POST"
    assert exchange.get_header("Authorization") == "Key owner"
    assert urllib.parse.parse_qs(exchange.data.decode()) == {
        "grant_type": ["urn:ietf:params:oauth:grant-type:token-exchange"],
        "subject_token": ["t"],
        "subject_token_type": ["urn:posit:connect:user-session-token"],
        "requested_token_type": ["urn:posit:connect:api-key"],
    }
    assert user.get_header("Authorization") == "Key visitor"


def test_failed_exchange_needs_integration(on_connect, monkeypatch):
    err = urllib.error.HTTPError("u", 400, "bad", Message(), None)
    monkeypatch.setattr("urllib.request.urlopen", _respond({"credentials": err}))
    assert access.denial({access.SESSION_TOKEN_HEADER: "t"}) == access.NEEDS_INTEGRATION
