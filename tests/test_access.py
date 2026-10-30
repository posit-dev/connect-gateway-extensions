"""The visitor check, and the key the app reads Connect with."""

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


def test_local_run_reads_with_the_configured_key(monkeypatch):
    monkeypatch.delenv("CONNECT_CONTENT_GUID", raising=False)
    monkeypatch.setenv("CONNECT_API_KEY", "mine")
    assert access.visitor_key({}) == "mine"


def test_local_run_without_a_key_has_none(monkeypatch):
    monkeypatch.delenv("CONNECT_CONTENT_GUID", raising=False)
    monkeypatch.delenv("CONNECT_API_KEY", raising=False)
    assert access.visitor_key({}) == ""


def test_no_session_token_needs_integration(on_connect):
    with pytest.raises(access.Denied) as denied:
        access.visitor_key({})
    assert str(denied.value) == access.NEEDS_INTEGRATION


@pytest.mark.parametrize("role,denial", [
    ("administrator", None), ("publisher", access.NOT_ADMIN), ("viewer", access.NOT_ADMIN),
])
def test_role_decides(on_connect, monkeypatch, role, denial):
    monkeypatch.setattr("urllib.request.urlopen", _respond(
        {"credentials": {"access_token": "k"}, "user": {"user_role": role}}))
    headers = {access.SESSION_TOKEN_HEADER: "t"}
    if denial is None:
        assert access.visitor_key(headers) == "k"
    else:
        with pytest.raises(access.Denied) as denied:
            access.visitor_key(headers)
        assert str(denied.value) == denial


def test_exchange_requests_visitor_api_key(on_connect, monkeypatch):
    requests = []
    respond = _respond({"credentials": {"access_token": "visitor"},
                        "user": {"user_role": "administrator"}})

    def fake(req, timeout=None):
        requests.append(req)
        return respond(req, timeout)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert access.visitor_key({access.SESSION_TOKEN_HEADER: "t"}) == "visitor"
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
    with pytest.raises(access.Denied) as denied:
        access.visitor_key({access.SESSION_TOKEN_HEADER: "t"})
    assert str(denied.value) == access.NEEDS_INTEGRATION


@pytest.mark.parametrize("credential,user", [
    (["k"], {"user_role": "administrator"}),          # a list, not an object
    ("k", {"user_role": "administrator"}),            # a bare string
    ({"access_token": ""}, {"user_role": "administrator"}),
    ({"access_token": 7}, {"user_role": "administrator"}),
    ({"access_token": "k"}, [{"user_role": "administrator"}]),
    ({"access_token": "k"}, None),
], ids=["list", "string", "empty-token", "number-token", "user-list", "user-null"])
def test_an_unexpected_answer_needs_integration(on_connect, monkeypatch, credential, user):
    monkeypatch.setattr("urllib.request.urlopen", _respond(
        {"credentials": credential, "user": user}))
    with pytest.raises(access.Denied) as denied:
        access.visitor_key({access.SESSION_TOKEN_HEADER: "t"})
    assert str(denied.value) == access.NEEDS_INTEGRATION


@pytest.mark.parametrize("owner_key", [None, ""])
def test_no_owner_key_on_connect_needs_integration(on_connect, monkeypatch, owner_key):
    if owner_key is None:
        monkeypatch.delenv("CONNECT_API_KEY")
    else:
        monkeypatch.setenv("CONNECT_API_KEY", owner_key)

    def forbidden(*args, **kwargs):
        raise AssertionError("the exchange was attempted without a key")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(access.Denied) as denied:
        access.visitor_key({access.SESSION_TOKEN_HEADER: "t"})
    assert str(denied.value) == access.NEEDS_INTEGRATION


def test_a_denied_visitor_gets_the_denial_page_and_nothing_is_fetched(on_connect, monkeypatch):
    import app

    def forbidden(*args, **kwargs):
        raise AssertionError("a request was made for a visitor who may not view")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    # No session token: Connect would not say who is looking.
    page = str(app.app_ui(None))
    assert access.NEEDS_INTEGRATION in page
    assert "f_users" not in page and "records_table" not in page
