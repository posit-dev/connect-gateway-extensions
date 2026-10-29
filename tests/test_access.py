"""The visitor check that keeps the owner's API key from serving everyone."""

import io
import json
import urllib.error

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


def test_failed_exchange_needs_integration(on_connect, monkeypatch):
    err = urllib.error.HTTPError("u", 400, "bad", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", _respond({"credentials": err}))
    assert access.denial({access.SESSION_TOKEN_HEADER: "t"}) == access.NEEDS_INTEGRATION
