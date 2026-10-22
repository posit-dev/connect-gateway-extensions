"""SQL predicates shared by dashboard pages (no Shiny state or DB access)."""

from __future__ import annotations

import json

# SQL for a row's token count: input plus output.
TOKEN_COUNT = "(input_tokens + output_tokens)"


def sql_str(x: str) -> str:
    return "'" + str(x).replace("'", "''") + "'"


# What a chart calls a row whose origin or harness is NULL. Connect sets origin
# only for content, or for a Posit client that names its host IDE in the
# User-Agent; Claude Code, Copilot and any caller that sends no User-Agent get
# none. Dropping those rows made the origin and harness totals smaller than the
# spend they claimed to break down, so they get a named bucket of their own.
UNIDENTIFIED = "(unidentified)"
NO_USER_AGENT = "(no user agent)"
NULL_LABEL = {"source": UNIDENTIFIED, "tool": NO_USER_AGENT}


def dim_sql(dim: str) -> str:
    """SQL for a chart category: the column, with NULL named where it means something."""
    label = NULL_LABEL.get(dim)
    return f"COALESCE({dim}, {sql_str(label)})" if label else dim


def meta_expr(key: str) -> str:
    """SQL for one metadata key's value, as text."""
    path = '$."' + key.replace('"', '\\"') + '"'
    return f"json_extract_string(request_metadata, {sql_str(path)})"


def sql_in(col: str, vals) -> str:
    return f"{col} IN ({','.join(sql_str(x) for x in vals)})"


def where(f: dict, extra: str = "") -> str:
    """Build the gateway_events predicate from one resolved filter selection."""
    conds = [
        f"ts >= TIMESTAMP '{f['start']} 00:00:00'",
        f"ts <= TIMESTAMP '{f['end']} 23:59:59.999'",
    ]
    if f["source"] != "both":
        conds.append(f"source = {sql_str(f['source'])}")
    if f["users"] and set(f["users"]) != set(f["all_users"]):
        conds.append(sql_in("username", f["users"]))
    # Applies to every origin. A Workbench call has a null content_guid, so
    # picking content naturally leaves Workbench out of the view.
    if f["contents"] and set(f["contents"]) != set(f["all_contents"]):
        conds.append(sql_in("content_guid", f["contents"]))
    if f["sessions"]:
        conds.append(sql_in("workbench_session_id", f["sessions"]))
    if f["providers"] and set(f["providers"]) != set(f["all_providers"]):
        conds.append(sql_in("provider", f["providers"]))
    if f["models"] and set(f["models"]) != set(f["all_models"]):
        conds.append(sql_in("model", f["models"]))
    # Values of one key are alternatives; separate keys all have to match.
    by_key: dict[str, list[str]] = {}
    for pick in f["meta"]:
        try:
            k, v = json.loads(pick)
        except (ValueError, TypeError):
            continue
        by_key.setdefault(k, []).append(v)
    for k, vals in by_key.items():
        conds.append(f"{meta_expr(k)} IN ({','.join(sql_str(v) for v in vals)})")
    return " AND ".join(conds) + extra
