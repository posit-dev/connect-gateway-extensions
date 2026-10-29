"""The dashboard's SQL against gateway_events and user_groups, one function per query.

Page code calls these and does the formatting and charting. Functions that
take a `Filters` narrow to the rows in view; the rest read the whole table.

Values (filter picks, principal keys, dates, names read from the data) are bound
as `$name` parameters through `Params`, never written into the SQL text. What
stays text is what a parameter cannot be: column names and time grains, which
callers choose from fixed lists, and the price expression from pricing.py.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

import usage
from filters import TOKEN_COUNT, Filters, Params, Query, dim_sql, sql_str
from pricing import est_cost, est_cost_uncached


# ---- what the loaded data covers ------------------------------------------------
def data_window() -> tuple[date, date]:
    """The first and last day with any record."""
    first = usage.scalar("SELECT MIN(day) FROM gateway_events")
    last = usage.scalar("SELECT MAX(day) FROM gateway_events")
    return first.date(), last.date()


def usernames() -> pd.DataFrame:
    """username, sorted."""
    return usage.q("""SELECT DISTINCT username FROM gateway_events
                      WHERE username IS NOT NULL ORDER BY 1""")


def user_emails() -> pd.DataFrame:
    """username, email."""
    return usage.q("""SELECT username, MAX(user_email) AS email
                      FROM gateway_events WHERE username IS NOT NULL
                      GROUP BY 1 ORDER BY 1""")


def sessions() -> pd.DataFrame:
    """sid, name: every Workbench session."""
    return usage.q("""SELECT workbench_session_id AS sid,
                             MAX(workbench_session_name) AS name
                      FROM gateway_events
                      WHERE workbench_session_id IS NOT NULL
                      GROUP BY 1 ORDER BY 2, 1""")


def content_guids() -> pd.DataFrame:
    """content_guid, sorted."""
    return usage.q("""SELECT DISTINCT content_guid FROM gateway_events
                      WHERE content_guid IS NOT NULL ORDER BY 1""")


def content_owners() -> pd.DataFrame:
    """guid, owner: every content item."""
    return usage.q("""SELECT content_guid AS guid,
                             MAX(content_owner_username) AS owner
                      FROM gateway_events WHERE content_guid IS NOT NULL
                      GROUP BY 1 ORDER BY 1""")


def groups() -> pd.DataFrame:
    """guid, name, members: every Connect group."""
    return usage.q("""SELECT group_guid AS guid, MAX(group_name) AS name,
                             COUNT(DISTINCT user_guid) AS members
                      FROM user_groups GROUP BY 1 ORDER BY 2, 1""")


def status_codes() -> pd.DataFrame:
    """status, as text, sorted."""
    return usage.q("""SELECT DISTINCT CAST(status_code AS VARCHAR) status
                      FROM gateway_events WHERE status_code IS NOT NULL
                      ORDER BY 1""")


def metadata_pairs() -> pd.DataFrame:
    """key, v: every key and value in the callers' own request metadata."""
    return usage.q(
        """SELECT k AS key, json_extract_string(request_metadata, '$."' || k || '"') AS v
           FROM (SELECT request_metadata, unnest(json_keys(request_metadata)) AS k
                 FROM gateway_events WHERE request_metadata IS NOT NULL)
           WHERE NOT starts_with(k, 'workbench-') AND v IS NOT NULL
           GROUP BY 1, 2 ORDER BY 1, 2"""
    )


# ---- the rows in view -------------------------------------------------------------
def record_count(f: Filters) -> int:
    p = Params()
    return int(usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {f.where(p)}", p.values))


def total_spend(f: Filters) -> float:
    p = Params()
    return usage.scalar(
        f"SELECT COALESCE(SUM({est_cost()}),0) FROM gateway_events WHERE {f.where(p)}", p.values
    )


def unpriced_models(f: Filters) -> pd.DataFrame:
    """model, requests: models that carried tokens but cost $0 for want of a price."""
    p = Params()
    return usage.q(f"""SELECT model, COUNT(*) requests
                       FROM gateway_events
                       WHERE {f.where(p)} AND model IS NOT NULL AND model <> ''
                       GROUP BY 1
                       HAVING SUM({est_cost()}) = 0 AND SUM({TOKEN_COUNT}) > 0
                       ORDER BY 2 DESC""", p.values)


# ---- raw records ------------------------------------------------------------------
# The derived principal answers "by whom", and the three raw identities behind
# it are kept beside it: audit evidence must show what the gateway recorded, not
# only what this dashboard concluded from it.
# `status` is a string, not a number: an HTTP code is a label rather than a
# quantity, so sorting or ranging over it ("status > 300") would say nothing.
_RECORD_COLS = """ts, source AS origin, tool, user_agent, username, user_kind AS via,
                     user_email AS email,
                     authenticated_username, viewer_username,
                     content_owner_username, content_guid,
                     (SELECT n.content_name FROM content_names n
                      WHERE n.content_guid = gateway_events.content_guid) AS content_name,
                     job_key,
                     model, provider, integration_auth_type AS auth_type,
                     CAST(status_code AS VARCHAR) AS status,
                     ROUND(duration_ms) AS ms,
                     input_tokens AS tok_in, output_tokens AS tok_out,
                     cache_read_tokens AS tok_cache_read,
                     ROUND(EST_COST_EXPR,4) AS est_cost_usd,
                     finish_reason, response_id, conversation_id,
                     workbench_session_id, workbench_session_name,
                     workbench_project_name, workbench_project_path,
                     team, cost_center, tier, enforced,
                     CAST(request_metadata AS VARCHAR) AS request_metadata"""


def _records(where_sql: str, p: Params, limit: int | None) -> Query:
    cols = _RECORD_COLS.replace("EST_COST_EXPR", est_cost())
    cap = f" LIMIT {limit}" if limit else ""
    return Query(
        f"SELECT {cols} FROM gateway_events WHERE {where_sql} ORDER BY ts DESC{cap}", p.values)


def records_query(f: Filters, limit: int | None = None) -> Query:
    """Every record in view, newest first, or the newest `limit`.

    A Query and not rows, because the CSV export hands it to DuckDB to write the file.
    """
    p = Params()
    return _records(f.where(p), p, limit)


def principal_records_query(f: Filters, kind: str, key: str) -> Query:
    """Every record of one principal in view, uncapped."""
    p = Params()
    return _records(f"{f.where(p)} AND {pred(kind, key, p)}", p, None)


def records(f: Filters, limit: int) -> pd.DataFrame:
    """The newest `limit` records in view."""
    return usage.q(*records_query(f, limit))


# ---- adoption ---------------------------------------------------------------------
def active_principals(f: Filters, col: str, history_from: date) -> pd.DataFrame:
    """day, dau, wau, mau: rolling active counts of `col`, one row per day in range.

    Windows look back to `history_from`, before the start of the range, so a
    30-day chart does not start with an empty MAU. Every other filter applies.
    """
    p = Params()
    return usage.q(f"""
        WITH days AS (
            SELECT CAST(d AS DATE) AS d
            FROM generate_series({p.bind(f.start)}::DATE, {p.bind(f.end)}::DATE,
                                 INTERVAL 1 DAY) t(d)),
        ev AS (SELECT DISTINCT day, {col} AS who FROM gateway_events
               WHERE {replace(f, start=history_from).where(p)} AND {col} IS NOT NULL)
        SELECT days.d AS day,
               COUNT(DISTINCT CASE WHEN ev.day = days.d THEN ev.who END) AS dau,
               COUNT(DISTINCT CASE WHEN ev.day > days.d - INTERVAL 7 DAY
                                   THEN ev.who END) AS wau,
               COUNT(DISTINCT ev.who) AS mau
        FROM days LEFT JOIN ev
          ON ev.day <= days.d AND ev.day > days.d - INTERVAL 30 DAY
        GROUP BY 1 ORDER BY 1""", p.values)


def _active_in_view(f: Filters, col: str, dim: str, p: Params) -> str:
    # Requests with no User-Agent are a line of their own: on a server whose
    # callers send none, leaving them out made the chart look empty.
    return (f"FROM gateway_events WHERE {f.where(p)} AND {col} IS NOT NULL "
            f"AND {dim_sql(dim)} <> ''")


def top_by_active(f: Filters, col: str, dim: str, n: int = 6) -> list[str]:
    """The `n` values of `dim` with the most distinct `col` active in view."""
    k, p = dim_sql(dim), Params()
    return usage.q(f"""SELECT {k} AS k {_active_in_view(f, col, dim, p)} GROUP BY 1
                       ORDER BY COUNT(DISTINCT {col}) DESC LIMIT {n}""", p.values).k.tolist()


def active_by_bucket(f: Filters, col: str, dim: str, grain: str, keep: list[str]) -> pd.DataFrame:
    """b, k, users, requests per `grain` bucket; values of `dim` not in `keep` fold to "Other"."""
    p = Params()
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b,
                              {_folded(dim, keep, p)} AS k,
                              COUNT(DISTINCT {col}) AS users, COUNT(*) AS requests
                       {_active_in_view(f, col, dim, p)} GROUP BY 1, 2 ORDER BY 1""", p.values)


# ---- overview ---------------------------------------------------------------------
# The overview charts take the group as `by`: a column to split on, or "none" for
# a single series named by `label`. Callers pass `by` and `grain` from their own
# fixed choices; they are SQL identifiers and are not escaped.
def overview_totals(f: Filters) -> pd.Series:
    """One row of totals for the rows in view.

    Each average the page shows divides the spend of that kind's own rows (the
    `*_est` columns): all of the spend over two content items read as $3,769 per
    item when nearly all of it came from people outside content.
    """
    p = Params()
    return usage.q(f"""SELECT COALESCE(SUM({est_cost()}),0) est,
                           COUNT(*) requests,
                           COUNT(DISTINCT username) users,
                           COUNT(DISTINCT content_guid) contents,
                           COUNT(DISTINCT workbench_session_id) sessions,
                           COALESCE(SUM({est_cost_uncached()} - {est_cost()}), 0) saved,
                           100.0 * COUNT(*) FILTER (WHERE status_code >= 400)
                               / NULLIF(COUNT(*), 0) AS err,
                           COALESCE(SUM({est_cost()}) FILTER (
                               WHERE username IS NOT NULL), 0) user_est,
                           COALESCE(SUM({est_cost()}) FILTER (
                               WHERE content_guid IS NOT NULL), 0) content_est,
                           COALESCE(SUM({est_cost()}) FILTER (
                               WHERE workbench_session_id IS NOT NULL), 0) session_est
                    FROM gateway_events WHERE {f.where(p)}""", p.values).iloc[0]


def top_values(f: Filters, dim: str, n: int = 6) -> list[str]:
    """The `n` busiest values of `dim` by requests, ignoring rows with none."""
    k, p = dim_sql(dim), Params()
    return usage.q(f"""SELECT {k} AS k FROM gateway_events
                       WHERE {f.where(p)} AND {k} IS NOT NULL AND {k} <> ''
                       GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {n}""", p.values).k.tolist()


def _folded(dim: str, keep: list[str], p: Params) -> str:
    """SQL for the group of a row: the values in `keep`, and "Other" for the rest."""
    k = dim_sql(dim)
    return f"CASE WHEN {k} IN ({p.bind_all(keep or [''])}) THEN {k} ELSE 'Other' END"


def _group(f: Filters, by: str, label: str, p: Params) -> str:
    """SQL for a chart's group: the six busiest values of `by`, the rest "Other".

    A rate cannot be summed after the query, so the tail is folded in SQL.
    """
    return sql_str(label) if by == "none" else _folded(by, top_values(f, by), p)


def _has_group(by: str) -> str:
    """Leaves out the rows with no value for the group, as the spend chart does."""
    k = dim_sql(by)
    return "" if by == "none" else f"AND {k} IS NOT NULL AND {k} <> ''"


def spend_by_period(f: Filters, grain: str, by: str, label: str) -> pd.DataFrame:
    """b, k, v: estimated spend per period and group. Every group is returned, not six."""
    k, p = (sql_str(label) if by == "none" else dim_sql(by)), Params()
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS b, {k} AS k,
                              SUM({est_cost()}) v
                       FROM gateway_events
                       WHERE {f.where(p)} AND {k} IS NOT NULL AND {k} <> ''
                       GROUP BY 1, 2 ORDER BY 1""", p.values)


def cache_savings_by_period(f: Filters, grain: str, by: str, label: str) -> pd.DataFrame:
    """day, k, v: what the prompt cache saved against the uncached price."""
    p = Params()
    k = _group(f, by, label, p)
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, {k} AS k,
                              SUM({est_cost_uncached()} - {est_cost()}) v
                       FROM gateway_events
                       WHERE {f.where(p)} {_has_group(by)}
                       GROUP BY 1, 2 ORDER BY 1""", p.values)


def cache_read_share_by_period(f: Filters, grain: str, by: str, label: str) -> pd.DataFrame:
    """day, k, v: the percent of input tokens read from the cache.

    input_tokens already counts cache writes and reads (see usage._SHAPE).
    """
    p = Params()
    k = _group(f, by, label, p)
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, {k} AS k,
                              100.0 * SUM(COALESCE(cache_read_tokens, 0))
                                  / NULLIF(SUM(COALESCE(input_tokens, 0)), 0) AS v
                       FROM gateway_events WHERE {f.where(p)} {_has_group(by)}
                       GROUP BY 1, 2 ORDER BY 1""", p.values)


def error_rate_by_period(f: Filters, grain: str, by: str, label: str) -> pd.DataFrame:
    """day, k, v: the percent of requests with status 400 or more."""
    p = Params()
    k = _group(f, by, label, p)
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, {k} AS k,
                              100.0 * COUNT(*) FILTER (WHERE status_code >= 400)
                                  / COUNT(*) AS v
                       FROM gateway_events
                       WHERE {f.where(p)} AND status_code IS NOT NULL {_has_group(by)}
                       GROUP BY 1, 2 ORDER BY 1""", p.values)


# ---- principals -------------------------------------------------------------------
# A principal is one user, content item, Workbench session or Connect group. `kind`
# and `key` name one, and `key` is escaped into the SQL.
#
# The column that identifies a principal of each kind, on gateway_events. A group
# is not a column: it is reached through user_groups (see _kind_source).
KEY_COL = {"user": "username", "content": "content_guid", "session": "workbench_session_id"}


def _kind_source(kind: str) -> tuple[str, str]:
    """(FROM clause, key expression) giving one group of rows per principal."""
    if kind == "group":
        return "gateway_events JOIN user_groups USING (user_guid)", "group_guid"
    return "gateway_events", KEY_COL[kind]


def pred(kind: str, key: str, p: Params) -> str:
    """The rows of one principal, as a condition on plain gateway_events."""
    if kind == "group":
        return f"user_guid IN (SELECT user_guid FROM user_groups WHERE group_guid = {p.bind(key)})"
    return f"{KEY_COL[kind]} = {p.bind(key)}"


def peer_spend(f: Filters, kind: str) -> pd.DataFrame:
    """k, spend, requests, tokens, owner, sname (and gname for groups): one row per principal."""
    src, key = _kind_source(kind)
    has_key = f"{key} IS NOT NULL AND CAST({key} AS VARCHAR) <> ''"
    extra = ", MAX(group_name) AS gname" if kind == "group" else ""
    p = Params()
    return usage.q(f"""
        SELECT {key} AS k, COALESCE(SUM({est_cost()}), 0) AS spend, COUNT(*) AS requests,
               COALESCE(SUM({TOKEN_COUNT}), 0) AS tokens,
               MAX(username) AS owner, MAX(workbench_session_name) AS sname{extra}
        FROM {src}
        WHERE {f.where(p)} AND {has_key}
        GROUP BY 1""", p.values)


def peer_top_model(f: Filters, kind: str) -> pd.DataFrame:
    """k, top_model: the model each principal spent most on."""
    src, key = _kind_source(kind)
    has_key = f"{key} IS NOT NULL AND CAST({key} AS VARCHAR) <> ''"
    p = Params()
    return usage.q(f"""
        SELECT k, arg_max(model, v) AS top_model
        FROM (SELECT {key} AS k, model, COALESCE(SUM({est_cost()}), 0) AS v
              FROM {src} WHERE {f.where(p)} AND {has_key} GROUP BY 1, 2)
        GROUP BY k""", p.values)


def principal_spend_by_period(f: Filters, kind: str, key: str, grain: str) -> pd.DataFrame:
    """day, v: one principal's spend per day, week or month."""
    p = Params()
    return usage.q(f"""SELECT date_trunc('{grain}', day)::DATE AS day, SUM({est_cost()}) v
                       FROM gateway_events WHERE {f.where(p)} AND {pred(kind, key, p)}
                       GROUP BY 1 ORDER BY 1""", p.values)


def principal_spend_split(f: Filters, kind: str, key: str, col: str) -> pd.DataFrame:
    """k, v, n: one principal's spend and requests by the values of `col`."""
    k, p = dim_sql(col), Params()
    return usage.q(f"""SELECT {k} AS k, SUM({est_cost()}) v, COUNT(*) n FROM gateway_events
                       WHERE {f.where(p)} AND {pred(kind, key, p)} AND {k} IS NOT NULL AND {k} <> ''
                       GROUP BY 1 ORDER BY v DESC""", p.values)


def user_content(f: Filters, user: str) -> pd.DataFrame:
    """content_guid, via, requests, est_spend_usd: the content a user was charged for."""
    p = Params()
    return usage.q(f"""
        SELECT content_guid, string_agg(DISTINCT user_kind, ', ' ORDER BY user_kind) AS via,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events
        WHERE {f.where(p)} AND {pred('user', user, p)} AND content_guid IS NOT NULL
        GROUP BY 1 ORDER BY est_spend_usd DESC""", p.values)


def user_sessions(f: Filters, user: str) -> pd.DataFrame:
    """session, id, project, first, last, requests, est_spend_usd."""
    p = Params()
    return usage.q(f"""
        SELECT MAX(workbench_session_name) AS session, workbench_session_id AS id,
               MAX(workbench_project_name) AS project,
               strftime(MIN(ts), '%Y-%m-%d') AS first, strftime(MAX(ts), '%Y-%m-%d') AS last,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events
        WHERE {f.where(p)} AND {pred('user', user, p)} AND workbench_session_id IS NOT NULL
        GROUP BY 2 ORDER BY est_spend_usd DESC""", p.values)


def content_users(f: Filters, guid: str) -> pd.DataFrame:
    """user, via, requests, est_spend_usd: who was charged for a content item."""
    p = Params()
    return usage.q(f"""
        SELECT username AS user, string_agg(DISTINCT user_kind, ', ' ORDER BY user_kind) AS via,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events WHERE {f.where(p)} AND {pred('content', guid, p)}
        GROUP BY 1 ORDER BY est_spend_usd DESC""", p.values)


def content_groups(f: Filters, guid: str) -> pd.DataFrame:
    """group, users, requests, est_spend_usd: by the charged users' Connect groups."""
    p = Params()
    return usage.q(f"""
        SELECT group_name AS "group", COUNT(DISTINCT user_guid) users,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events JOIN user_groups USING (user_guid)
        WHERE {f.where(p)} AND {pred('content', guid, p)}
        GROUP BY 1 ORDER BY est_spend_usd DESC""", p.values)


def session_owner(session_id: str) -> str | None:
    """The username that ran a session, over all records and not only those in view."""
    p = Params()
    return usage.scalar(
        f"SELECT MAX(username) FROM gateway_events WHERE {pred('session', session_id, p)}",
        p.values)


def owner_sessions(f: Filters, owner: str) -> pd.DataFrame:
    """session, id, first, last, requests, est_spend_usd: every session of one user."""
    p = Params()
    return usage.q(f"""
        SELECT MAX(workbench_session_name) AS session, workbench_session_id AS id,
               strftime(MIN(ts), '%Y-%m-%d') AS first, strftime(MAX(ts), '%Y-%m-%d') AS last,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events
        WHERE {f.where(p)} AND {pred('user', owner, p)} AND workbench_session_id IS NOT NULL
        GROUP BY 2 ORDER BY est_spend_usd DESC""", p.values)


def owner_projects(f: Filters, owner: str) -> pd.DataFrame:
    """project, sessions, requests, est_spend_usd: a user's Workbench projects."""
    p = Params()
    return usage.q(f"""
        SELECT COALESCE(workbench_project_name, '(no project)') AS project,
               COUNT(DISTINCT workbench_session_id) sessions,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events
        WHERE {f.where(p)} AND {pred('user', owner, p)} AND source = 'workbench'
        GROUP BY 1 ORDER BY est_spend_usd DESC""", p.values)


def group_members(f: Filters, group: str) -> pd.DataFrame:
    """user, other_groups, requests, est_spend_usd: the members of a group by spend."""
    p = Params()
    return usage.q(f"""
        SELECT a.username AS user, o.other_groups, a.requests, a.est_spend_usd
        FROM (SELECT user_guid, MAX(username) username, COUNT(*) requests,
                     SUM({est_cost()}) est_spend_usd
              FROM gateway_events WHERE {f.where(p)} AND {pred('group', group, p)} GROUP BY 1) a
        LEFT JOIN (SELECT user_guid, string_agg(group_name, ', ' ORDER BY group_name) other_groups
                   FROM user_groups WHERE group_guid <> {p.bind(group)} GROUP BY 1) o
        USING (user_guid) ORDER BY a.est_spend_usd DESC""", p.values)


def group_overlap(f: Filters, group: str) -> pd.DataFrame:
    """group, shared_members, requests, est_spend_usd: the other groups its members are in."""
    p = Params()
    return usage.q(f"""
        SELECT group_name AS "group", COUNT(DISTINCT user_guid) shared_members,
               COUNT(*) requests, SUM({est_cost()}) est_spend_usd
        FROM gateway_events JOIN user_groups USING (user_guid)
        WHERE {f.where(p)} AND {pred('group', group, p)} AND group_guid <> {p.bind(group)}
        GROUP BY 1 ORDER BY est_spend_usd DESC""", p.values)


def principal_conversations(f: Filters, kind: str, key: str) -> pd.DataFrame:
    """One row per conversation of a principal, with its cost, cache misses and main model."""
    p = Params()
    return usage.q(f"""
        WITH ev AS (
            SELECT *, {est_cost()} AS cost,
                   LAG(input_tokens) OVER (PARTITION BY conversation_id
                                           ORDER BY ts, response_id) AS prev_input,
                   MAX(cache_read_tokens) OVER (PARTITION BY conversation_id) AS cache_used
            FROM gateway_events
            WHERE {f.where(p)} AND {pred(kind, key, p)} AND conversation_id IS NOT NULL)
        SELECT conversation_id AS conversation, MIN(ts) AS started,
               date_diff('second', MIN(ts), MAX(ts)) / 60.0 AS minutes,
               COUNT(*) AS turns, SUM(cost) AS est,
               MAX(input_tokens) AS peak_context,
               SUM(cache_read_tokens + cache_creation_tokens) > 0 AS uses_cache,
               -- A cache miss: a request that read back under half of what the
               -- one before it sent, in a conversation that uses the cache at
               -- all. The tokens are what had to be read again. Below 1024
               -- tokens nothing is cached, so a short prompt is no miss.
               SUM(CASE WHEN prev_input >= 1024 AND cache_used > 0
                             AND cache_read_tokens < 0.5 * prev_input
                        THEN prev_input - cache_read_tokens ELSE 0 END) AS miss_tokens,
               SUM(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS errors,
               SUM(CASE WHEN finish_reason = 'tool_use' THEN 1 ELSE 0 END) AS tool_calls,
               mode(model) AS model,
               arg_max(model, cost) FILTER (WHERE model <> '') AS main_model,
               arg_max(tool, cost) FILTER (WHERE tool IS NOT NULL) AS main_tool,
               MAX(username) AS who, MAX(content_guid) AS content
        FROM ev
        GROUP BY 1 ORDER BY est DESC""", p.values)
