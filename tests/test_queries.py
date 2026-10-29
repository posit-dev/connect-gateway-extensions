"""queries.py against a few rows written for the purpose.

The page tests run these queries over the fake Connect's data and check what the
pages show. These read the results directly, over rows small enough to count by
hand, so a changed WHERE, join or grouping fails here with the query's name.

The rows (all in September 2026):

    10th  alice  Workbench session s1 "Alpha", project "Proj"   sonnet     200
    11th  alice  viewer of content c1, owned by olga            sonnet     200
    12th  bob    team=x                                         haiku      500
    12th  (none) Workbench session s2, no user at all           sonnet     200
    13th  bob    a model the catalog has no price for           mystery    200
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

import pricing
import principals
import queries
import usage
from conftest import WINDOW_FROM, WINDOW_TO
from filters import Filters

NOBODY = {"authenticated_username": None, "authenticated_user_guid": None,
          "authenticated_email": None}


def meta(**kw) -> str:
    return json.dumps(kw)


ROWS = [
    {"start_time": "2026-09-10 12:00:00+00", "conversation_id": "conv-1", "origin": "workbench",
     "request_metadata": meta(**{"workbench-session-id": "s1", "workbench-session-name": "Alpha",
                                 "workbench-project-name": "Proj"})},
    {"start_time": "2026-09-11 12:00:00+00", "conversation_id": "conv-1",
     "viewer_username": "alice", "viewer_user_guid": "guid-alice",
     "viewer_email": "alice@example.com", "content_owner_username": "olga",
     "content_owner_guid": "guid-olga", "content_owner_email": "olga@example.com",
     "content_guid": "c1", **NOBODY},
    {"start_time": "2026-09-12 12:00:00+00", "authenticated_username": "bob",
     "authenticated_user_guid": "guid-bob", "authenticated_email": "bob@example.com",
     "request_model": "claude-haiku-4-5", "response_model": "claude-haiku-4-5",
     "status_code": 500, "request_metadata": meta(team="x")},
    {"start_time": "2026-09-12 13:00:00+00", **NOBODY,
     "request_metadata": meta(**{"workbench-session-id": "s2"})},
    {"start_time": "2026-09-13 12:00:00+00", "authenticated_username": "bob",
     "authenticated_user_guid": "guid-bob", "authenticated_email": "bob@example.com",
     "request_model": "mystery-1", "response_model": "mystery-1"},
]

ALL = Filters(start=date(2026, 9, 1), end=date(2026, 9, 30))


@pytest.fixture
def world(loaded, export, con, monkeypatch):
    """usage pointed at a private connection holding ROWS, with groups and content names."""
    usage._read_export(export(ROWS), WINDOW_FROM, WINDOW_TO, con)
    con.execute("CREATE TABLE content_names (content_guid VARCHAR, content_name VARCHAR)")
    con.execute("INSERT INTO content_names VALUES ('c1', 'Report')")
    con.execute("CREATE TABLE user_groups (user_guid VARCHAR, group_guid VARCHAR, group_name VARCHAR)")
    con.execute("""INSERT INTO user_groups VALUES ('guid-alice', 'g1', 'Analysts'),
                   ('guid-bob', 'g1', 'Analysts'), ('guid-alice', 'g2', 'Admins')""")
    monkeypatch.setattr(usage, "_con", con)


# ---- what the data covers ---------------------------------------------------------
def test_catalogue_lists(world):
    assert queries.data_window() == (date(2026, 9, 10), date(2026, 9, 13))
    assert queries.usernames().username.tolist() == ["alice", "bob"]
    assert queries.content_guids().content_guid.tolist() == ["c1"]
    assert queries.status_codes().status.tolist() == ["200", "500"]
    emails = queries.user_emails()
    assert dict(zip(emails.username, emails.email)) == {
        "alice": "alice@example.com", "bob": "bob@example.com"}
    assert dict(zip(queries.content_owners().guid, queries.content_owners().owner)) == {"c1": "olga"}
    sessions = queries.sessions()
    assert sessions.sid.tolist() == ["s1", "s2"]
    assert sessions.name[0] == "Alpha" and pd.isna(sessions.name[1])
    groups = queries.groups().set_index("guid")
    assert groups.members.to_dict() == {"g1": 2, "g2": 1}
    assert groups.name.to_dict() == {"g1": "Analysts", "g2": "Admins"}


def test_metadata_pairs_leave_out_workbench_keys(world):
    pairs = queries.metadata_pairs()
    assert list(zip(pairs.key, pairs.v)) == [("team", "x")]


# ---- the rows in view -------------------------------------------------------------
def test_record_count_follows_the_filter(world):
    assert queries.record_count(ALL) == 5
    assert queries.record_count(replace(ALL, users=("bob",))) == 2
    assert queries.record_count(replace(ALL, start=date(2026, 9, 12))) == 3
    assert queries.record_count(replace(ALL, end=date(2026, 9, 9))) == 0
    assert queries.record_count(replace(ALL, sessions=("s1",))) == 1
    assert queries.record_count(replace(ALL, groups=("g2",))) == 2  # alice's two rows
    assert queries.record_count(replace(ALL, meta=(json.dumps(["team", "x"]),))) == 1


def test_total_spend_is_the_sum_of_every_period(world):
    total = queries.total_spend(ALL)
    assert total > 0
    by_period = queries.spend_by_period(ALL, "day", "none", "Estimated spend")
    assert by_period.v.sum() == pytest.approx(total)
    assert set(by_period.k) == {"Estimated spend"}
    assert queries.total_spend(replace(ALL, end=date(2026, 9, 9))) == 0


def test_unpriced_models_lists_only_what_the_catalog_lacks(world):
    unpriced = queries.unpriced_models(ALL)
    assert list(zip(unpriced.model, unpriced.requests)) == [("mystery-1", 1)]
    assert queries.unpriced_models(replace(ALL, end=date(2026, 9, 12))).empty


# ---- overview ---------------------------------------------------------------------
def test_overview_totals(world):
    r = queries.overview_totals(ALL)
    assert (r.requests, r.users, r.contents, r.sessions) == (5, 2, 1, 2)
    assert r.err == pytest.approx(20.0)  # one 500 in five
    assert r.est == pytest.approx(queries.total_spend(ALL))
    assert r.user_est <= r.est and r.saved >= 0
    assert r.session_est > 0


def test_top_values_ranks_by_requests_and_skips_nulls(world):
    assert queries.top_values(ALL, "model")[0] == "claude-sonnet-4-5"
    assert queries.top_values(ALL, "model", n=1) == ["claude-sonnet-4-5"]
    # Origin and harness NULLs get a named bucket instead of being dropped.
    assert "(no user agent)" in queries.top_values(ALL, "tool")


def test_folded_group_names_the_rest_other():
    sql = queries._folded("model", ["a", "it's"])
    assert sql == "CASE WHEN model IN ('a','it''s') THEN model ELSE 'Other' END"
    assert "IN ('')" in queries._folded("model", [])  # an empty list matches nothing


def test_period_queries_group_and_label(world):
    by_model = queries.spend_by_period(ALL, "day", "model", "x")
    assert set(by_model.k) == {"claude-sonnet-4-5", "claude-haiku-4-5", "mystery-1"}
    # One bucket per month at the monthly grain.
    assert len(queries.spend_by_period(ALL, "month", "none", "x")) == 1

    err = queries.error_rate_by_period(ALL, "day", "none", "Error rate").set_index("day").v
    assert err[pd.Timestamp("2026-09-12")] == pytest.approx(50.0)  # 500 and 200 that day
    assert err[pd.Timestamp("2026-09-10")] == 0

    cache = queries.cache_read_share_by_period(ALL, "day", "none", "Cache read share")
    assert (cache.v == 0).all()  # none of the rows read from the cache
    saved = queries.cache_savings_by_period(ALL, "day", "none", "Saved")
    assert (saved.v.fillna(0) >= 0).all() and set(saved.k) == {"Saved"}


# ---- adoption ---------------------------------------------------------------------
def test_active_principals_roll_up_over_windows(world):
    f = replace(ALL, start=date(2026, 9, 12), end=date(2026, 9, 13))
    df = queries.active_principals(f, "username", date(2026, 9, 1)).set_index("day")
    assert df.index.tolist() == [pd.Timestamp("2026-09-12"), pd.Timestamp("2026-09-13")]
    # alice on the 10th and 11th, bob on the 12th and 13th.
    assert df.dau.tolist() == [1, 1]
    assert df.wau.tolist() == [2, 2]
    assert df.mau.tolist() == [2, 2]
    # Without the earlier history the windows only see the range itself.
    lone = queries.active_principals(f, "username", date(2026, 9, 12)).set_index("day")
    assert lone.mau.tolist() == [1, 1]


def test_active_by_bucket_folds_the_tail(world):
    # Each model has one active user, so the order among them is a tie.
    assert set(queries.top_by_active(ALL, "username", "model")) == {
        "claude-sonnet-4-5", "claude-haiku-4-5", "mystery-1"}
    assert len(queries.top_by_active(ALL, "username", "model", n=1)) == 1
    df = queries.active_by_bucket(ALL, "username", "model", "month", ["claude-sonnet-4-5"])
    assert set(df.k) == {"claude-sonnet-4-5", "Other"}
    assert df.set_index("k").requests.to_dict() == {"claude-sonnet-4-5": 2, "Other": 2}


# ---- records ----------------------------------------------------------------------
def test_records_are_newest_first_and_limited(world):
    got = queries.records(ALL, 2)
    assert len(got) == 2
    assert got.ts.is_monotonic_decreasing
    assert got.content_name.dropna().empty  # c1 is on the 11th, outside the newest two
    assert "LIMIT" not in queries.records_sql("TRUE")
    everything = usage.q(queries.records_sql(ALL.where()))
    assert len(everything) == 5
    assert everything.content_name.dropna().tolist() == ["Report"]
    assert everything.est_cost_usd.sum() == pytest.approx(queries.total_spend(ALL), abs=1e-3)


# ---- principals -------------------------------------------------------------------
def test_peer_spend_and_top_model(world):
    users = queries.peer_spend(ALL, "user").set_index("k")
    assert users.requests.to_dict() == {"alice": 2, "bob": 2}
    top = queries.peer_top_model(ALL, "user").set_index("k").top_model
    assert top["alice"] == "claude-sonnet-4-5"

    groups = queries.peer_spend(ALL, "group").set_index("k")
    # Each member counts in each of their groups, so alice is in both.
    assert groups.requests.to_dict() == {"g1": 4, "g2": 2}
    assert groups.gname.to_dict() == {"g1": "Analysts", "g2": "Admins"}

    sessions = queries.peer_spend(ALL, "session").set_index("k")
    assert sessions.sname["s1"] == "Alpha"
    assert pd.isna(sessions.sname["s2"])


def test_pred_selects_one_principal(world):
    def count(kind, key):
        return usage.scalar(f"SELECT COUNT(*) FROM gateway_events WHERE {queries.pred(kind, key)}")

    assert count("user", "alice") == 2
    assert count("content", "c1") == 1
    assert count("session", "s1") == 1
    assert count("group", "g1") == 4
    assert count("user", "x' OR '1'='1") == 0


def test_principal_records_are_that_principals_only(world):
    sql = queries.principal_records_sql(ALL, "user", "alice")
    assert set(usage.q(sql).username) == {"alice"}
    narrowed = queries.principal_records_sql(replace(ALL, end=date(2026, 9, 10)), "user", "alice")
    assert len(usage.q(narrowed)) == 1


def test_principal_spend_by_period_and_split(world):
    per_day = queries.principal_spend_by_period(ALL, "user", "alice", "day")
    assert per_day.day.dt.day.tolist() == [10, 11]
    split = queries.principal_spend_split(ALL, "group", "g1", "model")
    assert split.v.is_monotonic_decreasing
    assert split.n.sum() == 4


def test_user_relations(world):
    content = queries.user_content(ALL, "alice")
    assert list(zip(content.content_guid, content.via, content.requests)) == [("c1", "viewer", 1)]
    sessions = queries.user_sessions(ALL, "alice")
    assert list(zip(sessions.id, sessions.session, sessions.project)) == [("s1", "Alpha", "Proj")]
    assert sessions["first"][0] == sessions["last"][0] == "2026-09-10"


def test_content_relations(world):
    users = queries.content_users(ALL, "c1")
    assert list(zip(users.user, users.via)) == [("alice", "viewer")]
    groups = queries.content_groups(ALL, "c1").set_index("group")
    # alice is in two groups, so the one request counts in each.
    assert groups.requests.to_dict() == {"Analysts": 1, "Admins": 1}


def test_session_relations(world):
    assert queries.session_owner("s1") == "alice"
    assert queries.session_owner("s2") is None
    assert queries.session_owner("missing") is None
    assert queries.owner_sessions(ALL, "alice").id.tolist() == ["s1"]
    projects = queries.owner_projects(ALL, "alice")
    assert list(zip(projects.project, projects.sessions)) == [("Proj", 1)]


def test_a_session_with_no_user_has_no_relations(world):
    tables = principals.relations(ALL, "session", "s2")
    assert [t["df"].empty for t in tables] == [True, True]
    assert all("No user is recorded" in t["empty"] for t in tables)
    # With a user, the same call fills both tables.
    alpha = principals.relations(ALL, "session", "s1")
    assert [len(t["df"]) for t in alpha] == [1, 1]


def test_group_relations(world):
    members = queries.group_members(ALL, "g1").set_index("user")
    assert members.other_groups["alice"] == "Admins"
    assert pd.isna(members.other_groups["bob"])
    overlap = queries.group_overlap(ALL, "g1")
    assert list(zip(overlap.group, overlap.shared_members)) == [("Admins", 1)]


def test_principal_conversations(world):
    df = queries.principal_conversations(ALL, "user", "alice")
    assert df.conversation.tolist() == ["conv-1"]
    row = df.iloc[0]
    assert row.turns == 2 and row.errors == 0
    assert row.who == "alice" and row.content == "c1"
    assert row.est == pytest.approx(
        queries.total_spend(replace(ALL, users=("alice",))))
    assert queries.principal_conversations(ALL, "user", "bob").empty  # no conversation ids


def test_prices_are_current_for_the_test_models(world):
    # The tests above rely on the catalog pricing the sonnet rows and not mystery-1.
    assert pricing.est_cost()
    assert usage.scalar(
        f"SELECT SUM({pricing.est_cost()}) FROM gateway_events WHERE model = 'mystery-1'") == 0
