"""Who may see the dashboard.

The app reads the usage export with the owner's CONNECT_API_KEY, so without a
check it would show that data to anyone the content is shared with. On Connect
each visitor must instead come with a Visitor API Key integration, and the
visitor must be an administrator, the role the export itself requires.

Connect sets CONNECT_CONTENT_GUID for deployed content only, so a copy run on a
developer's machine, where there is no visitor, is not checked.
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
    "This dashboard needs a Connect API integration to check who is viewing it. "
    "An administrator can add one on the Integration Settings page, with Max Role "
    "set to Administrator, and then associate it with this content."
)
NOT_ADMIN = "Only Connect administrators can view this dashboard."


def _request(url: str, key: str, data: dict | None = None) -> dict:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        method="POST" if data else "GET",
        headers={"Authorization": f"Key {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def denial(headers) -> str | None:
    """Why this visitor may not see the dashboard, or None when they may."""
    if not os.environ.get("CONNECT_CONTENT_GUID"):
        return None
    token = headers.get(SESSION_TOKEN_HEADER)
    if not token:
        return NEEDS_INTEGRATION
    try:
        # Exchange the visitor's session token for an API key that acts as them.
        credential = _request(
            f"{usage.connect_server()}/__api__/v1/oauth/integrations/credentials",
            usage._api_key() or "",
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": token,
                "subject_token_type": "urn:posit:connect:user-session-token",
            },
        )
        user = _request(
            f"{usage.connect_server()}/__api__/v1/user", credential["access_token"]
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return NEEDS_INTEGRATION
    return None if user.get("user_role") == "administrator" else NOT_ADMIN
