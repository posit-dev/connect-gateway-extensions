"""SQL predicates shared by dashboard pages (no Shiny state or DB access)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

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


@dataclass(frozen=True)
class Filters:
    """One resolved filter selection: the dates, and what is picked in each filter.

    An empty pick means no narrowing. `all_users` and `all_contents` are every
    choice on offer, so picking all of them narrows nothing either.
    """

    start: date
    end: date
    all_users: tuple[str, ...] = ()
    all_contents: tuple[str, ...] = ()
    users: tuple[str, ...] = ()
    contents: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    meta: tuple[str, ...] = ()
    mode: str = "dark"

    def where(self, extra: str = "") -> str:
        """The gateway_events predicate for this selection, then `extra` verbatim."""
        conds = [
            f"ts >= TIMESTAMP '{self.start} 00:00:00'",
            f"ts <= TIMESTAMP '{self.end} 23:59:59.999'",
        ]
        if self.users and set(self.users) != set(self.all_users):
            conds.append(sql_in("username", self.users))
        # Applies to every origin. A Workbench call has a null content_guid, so
        # picking content naturally leaves Workbench out of the view.
        if self.contents and set(self.contents) != set(self.all_contents):
            conds.append(sql_in("content_guid", self.contents))
        if self.sessions:
            conds.append(sql_in("workbench_session_id", self.sessions))
        # A Connect group is its members: a request counts when its user is in
        # any of the picked groups.
        if self.groups:
            conds.append("user_guid IN (SELECT user_guid FROM user_groups WHERE "
                         f"{sql_in('group_guid', self.groups)})")
        # Values of one key are alternatives; separate keys all have to match.
        by_key: dict[str, list[str]] = {}
        for pick in self.meta:
            try:
                k, v = json.loads(pick)
            except (ValueError, TypeError):
                continue
            by_key.setdefault(k, []).append(v)
        for k, vals in by_key.items():
            conds.append(f"{meta_expr(k)} IN ({','.join(sql_str(v) for v in vals)})")
        return " AND ".join(conds) + extra
