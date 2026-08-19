"""Guard tables and store methods — unit lane, no Postgres.

testpaths is tests/unit and the unit lane has no database, so this covers the
two things that can be checked without one: that the DDL is present in the
single schema string boot executes, and that every new write path is FAIL-OPEN
when the connection pool is broken. Round-tripping real bytes belongs in the
live smoke (step 15 of Docs/GUARD_INTEGRATION_PLAN.md).
"""

import pytest

from semantic_cache.gateway.store import _SCHEMA_SQL, PostgresGatewayStore


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fragment",
    [
        "ALTER TABLE gw.messages ADD COLUMN IF NOT EXISTS request_id uuid",
        "CREATE TABLE IF NOT EXISTS gw.guard_indexes",
        "CREATE INDEX IF NOT EXISTS guard_indexes_lru",
        "CREATE TABLE IF NOT EXISTS gw.guard_decisions",
        "CREATE INDEX IF NOT EXISTS guard_decisions_scope_time",
    ],
)
def test_the_guard_ddl_ships_in_the_one_schema_string(fragment) -> None:
    assert fragment in _SCHEMA_SQL


def test_every_guard_ddl_statement_is_idempotent() -> None:
    # ensure_schema() runs this on every boot, against a database that may
    # already have it. Nothing here may fail the second time.
    for statement in _SCHEMA_SQL.split(";"):
        stripped = statement.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if stripped.upper().startswith(("CREATE TABLE", "CREATE INDEX")):
            assert "IF NOT EXISTS" in stripped, stripped[:60]
        if stripped.upper().startswith("ALTER TABLE"):
            assert "IF NOT EXISTS" in stripped, stripped[:60]


def _squashed(text: str) -> str:
    """Whitespace-insensitive view, so formatting edits don't break these."""
    return " ".join(text.split())


def _table_body(name: str) -> str:
    marker = f"CREATE TABLE IF NOT EXISTS {name}"
    body = _SCHEMA_SQL.split(marker, 1)[1]
    return _squashed(body.split(");", 1)[0])


def test_vectors_are_bytea_not_pgvector() -> None:
    # pgvector cannot be assumed on a customer's database, and per-client
    # embedders mean per-row dimensions anyway.
    body = _table_body("gw.guard_indexes")
    assert "vectors bytea NOT NULL" in body
    assert "sentinel bytea NOT NULL" in body
    assert "dim integer NOT NULL" in body
    # No pgvector column anywhere in the DDL itself (comments may name it).
    ddl_only = "\n".join(
        line for line in _SCHEMA_SQL.splitlines()
        if not line.strip().startswith("--")
    )
    assert "vector(" not in ddl_only
    assert "CREATE EXTENSION" not in ddl_only


def test_guard_action_has_no_check_constraint() -> None:
    # It must carry block/allow/flag/degraded/unavailable; a CHECK on the
    # customer's database would make any new value a deploy-order landmine.
    body = _table_body("gw.guard_decisions")
    assert "action text NOT NULL" in body
    assert "CHECK" not in body


# --------------------------------------------------------------------------- #
# Fail-open behaviour
# --------------------------------------------------------------------------- #


class BrokenPool:
    """A connection pool whose every use raises."""

    def connection(self):
        raise RuntimeError("database is down")


@pytest.fixture
def broken_store() -> PostgresGatewayStore:
    store = PostgresGatewayStore.__new__(PostgresGatewayStore)
    store.pool = BrokenPool()
    store.guard_persist_failures = 0
    return store


def test_saving_an_index_fails_open(broken_store) -> None:
    assert broken_store.save_guard_index(
        "k", "m", "p", 8, 2, b"\x00" * 64, b"\x00" * 32, {"a": 1}
    ) is False
    assert broken_store.guard_persist_failures == 1


def test_loading_an_index_fails_open_to_none(broken_store) -> None:
    # "Build it", never "fail the request".
    assert broken_store.load_guard_index("k") is None
    assert broken_store.guard_persist_failures == 1


def test_touch_delete_and_sweep_fail_open(broken_store) -> None:
    assert broken_store.touch_guard_index("k") is False
    assert broken_store.delete_guard_index("k") == 0
    assert broken_store.sweep_guard_indexes(30) == 0


def test_the_decision_log_fails_open(broken_store) -> None:
    rows = [{"scope": "s", "policy_hash": "p", "action": "block"}]
    assert broken_store.log_guard_decisions(rows) == 0


def test_an_empty_decision_batch_opens_no_connection(broken_store) -> None:
    # BrokenPool raises on any use, so a non-zero result here would mean we
    # took a connection to write nothing.
    assert broken_store.log_guard_decisions([]) == 0


def test_reads_that_serve_an_api_still_raise(broken_store) -> None:
    # list_guard_decisions backs GET /v1/guard/decisions. A caller asking for
    # data must get an error, not a silent empty list — unlike a log WRITE,
    # where losing a row must never fail the request that produced it.
    with pytest.raises(RuntimeError):
        broken_store.list_guard_decisions("scope")


# --------------------------------------------------------------------------- #
# Signature compatibility
# --------------------------------------------------------------------------- #


def test_log_message_request_id_is_optional_and_last() -> None:
    import inspect

    parameters = list(
        inspect.signature(PostgresGatewayStore.log_message).parameters
    )
    assert parameters[-1] == "request_id"
    default = inspect.signature(
        PostgresGatewayStore.log_message
    ).parameters["request_id"].default
    assert default is None
