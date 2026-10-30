"""Who may see the dashboard, and the key to read Connect as them.

Every request the app makes to Connect uses the visitor's own API key, so
Connect decides what they can read and nothing is fetched before they arrive.
On Connect each visitor must therefore come with a Visitor API Key integration,
and must be an administrator, the role the usage export requires.

Connect sets CONNECT_CONTENT_GUID for deployed content only. A copy run on a
developer's machine has no visitor, so it reads with CONNECT_API_KEY instead.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import usage

SESSION_TOKEN_HEADER = "posit-connect-user-session-token"

NEEDS_INTEGRATION = (
    "This dashboard needs a Connect API integration (Max Role: Admin) to check who "
    "is viewing it. Ask an administrator to add one in System \u2192 Integrations, "
    "then add it to this content's Runtime settings."
)
NOT_ADMIN = "Only Connect administrators can view this dashboard."


class Denied(Exception):
    """The visitor may not see the dashboard; the message says why."""


def _request(url: str, key: str, data: dict | None = None) -> dict:
    """One Connect API call. Raises ValueError unless it answers with a JSON object."""
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        method="POST" if data else "GET",
        headers={"Authorization": f"Key {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        answer = json.loads(r.read())
    if not isinstance(answer, dict):
        raise ValueError("not a JSON object")
    return answer


def visitor_key(headers) -> str:
    """An API key that acts as this visitor. Raises Denied when they may not view.

    A local run returns CONNECT_API_KEY, or "" when it is unset (loading then
    says what to set).
    """
    if not os.environ.get("CONNECT_CONTENT_GUID"):
        return os.environ.get("CONNECT_API_KEY", "")
    token = headers.get(SESSION_TOKEN_HEADER)
    owner_key = os.environ.get("CONNECT_API_KEY")
    if not token or not owner_key:
        raise Denied(NEEDS_INTEGRATION)
    try:
        # Exchange the visitor's session token for an API key that acts as them.
        # The owner's key only authorises the exchange.
        credential = _request(
            f"{usage.connect_server()}/__api__/v1/oauth/integrations/credentials",
            owner_key,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": token,
                "subject_token_type": "urn:posit:connect:user-session-token",
                "requested_token_type": "urn:posit:connect:api-key",
            },
        )
        key = credential.get("access_token")
        if not isinstance(key, str) or not key:
            raise ValueError("no access token")
        user = _request(f"{usage.connect_server()}/__api__/v1/user", key)
    except (urllib.error.URLError, OSError, ValueError):
        raise Denied(NEEDS_INTEGRATION) from None
    if user.get("user_role") != "administrator":
        raise Denied(NOT_ADMIN)
    return key
