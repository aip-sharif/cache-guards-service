"""Postgres persistence for the OpenAI gateway.

Two concerns under the `gw` schema:
  * messages       — a log row per chat completion served (question, answer,
                     hit/miss), keyed by cache scope
  * cache_entries  — a DURABLE BACKUP of every Redis cache entry (including
                     its embedding vector). Redis stays the serving cache;
                     these rows exist only so Redis can be rebuilt on boot
                     after a wipe (see gateway/backup.py).

There is NO auth state here: the APP mints the client keys and validates them
(the gateway presents each key to the APP's config endpoint). Model configs
are not stored here either — the APP owns them (see gateway/app_config.py).

The Postgres instance is expected to already exist (SC_PG_DSN points at it);
the store only bootstraps its own `gw` schema. It is sync (psycopg3 +
connection pool); async callers wrap calls in asyncio.to_thread. All writes
are FAIL-OPEN: losing a log row or a backup row must never fail the request
that produced it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS gw;

CREATE TABLE IF NOT EXISTS gw.messages (
    id            bigserial PRIMARY KEY,
    scope         text NOT NULL,
    model         text NOT NULL,
    cache_hit     boolean NOT NULL,
    similarity    double precision,
    query_text    text NOT NULL,
    response      text NOT NULL,
    finish_reason text,
    usage         jsonb,
    latency_ms    integer,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_scope_time
    ON gw.messages (scope, created_at);

CREATE TABLE IF NOT EXISTS gw.cache_entries (
    redis_key   text PRIMARY KEY,
    key_prefix  text NOT NULL,
    scope       text NOT NULL,
    fields      jsonb NOT NULL,
    vector      bytea NOT NULL,
    expires_at  timestamptz,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS cache_entries_expiry
    ON gw.cache_entries (expires_at);

-- Join key onto gw.guard_decisions, so a refusal can be traced to the message
-- log row it produced. Added late, hence an ALTER rather than a column above.
ALTER TABLE gw.messages ADD COLUMN IF NOT EXISTS request_id uuid;

-- Guard exemplar matrices. bytea + an integer `dim`, NOT pgvector: compose
-- ships postgres:16-alpine and production points at the CUSTOMER's own
-- database, so the extension cannot be assumed — and since every client may
-- use a different embedding model, the dimension varies per row, which a fixed
-- vector(N) column cannot express. gw.cache_entries.vector is the precedent.
CREATE TABLE IF NOT EXISTS gw.guard_indexes (
    index_key     text PRIMARY KEY,
    embed_model   text NOT NULL,
    policy_hash   text NOT NULL,
    dim           integer NOT NULL,
    n             integer NOT NULL,
    vectors       bytea NOT NULL,
    sentinel      bytea NOT NULL,
    meta          jsonb NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_used_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS guard_indexes_lru
    ON gw.guard_indexes (last_used_at);

-- One row per guard decision. No CHECK on `action`: it carries
-- block/allow/flag/degraded/unavailable, and a CHECK constraint on a
-- customer's database makes any future action value a deploy-order landmine.
-- turn_text stays NULL unless the client set guard.log_turn_text.
CREATE TABLE IF NOT EXISTS gw.guard_decisions (
    id               bigserial PRIMARY KEY,
    request_id       uuid,
    scope            text NOT NULL,
    policy_hash      text NOT NULL,
    action           text NOT NULL,
    matched_category text,
    reason           text,
    embedding_score  double precision,
    judge_score      double precision,
    judge_invoked    boolean NOT NULL DEFAULT false,
    top_matches      jsonb,
    turn_text        text,
    latency_ms       integer,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS guard_decisions_scope_time
    ON gw.guard_decisions (scope, created_at);
"""


class PostgresGatewayStore:
    """All gateway state that must survive a restart or a Redis wipe."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        self.pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=True
        )
        #: Guard index reads/writes that failed. Surfaced because a silent
        #: persist failure degrades exemplar embedding from once-per-policy to
        #: once-per-process, with no other symptom than a bigger bill.
        self.guard_persist_failures = 0

    def close(self) -> None:
        self.pool.close()

    def ping(self, timeout: float = 2.0) -> bool:
        """Round-trips a trivial query. For readiness probes.

        Bounded by `timeout` so a Postgres that accepts connections and then
        stalls reports down instead of hanging the probe — the pool's own wait
        is otherwise unbounded, which would turn a slow database into a stuck
        readiness endpoint."""
        try:
            with self.pool.connection(timeout=timeout) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 — any failure is "not ready"
            return False

    def ensure_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(_SCHEMA_SQL)

    # -- messages: one row per served completion ---------------------------- #

    def log_message(
        self,
        scope: str,
        model: str,
        cache_hit: bool,
        query_text: str,
        response: str,
        similarity: Optional[float] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Logs one served chat completion. FAIL-OPEN: returns False on error
        instead of raising — losing a log row must never fail the request.

        ``request_id`` joins this row to its gw.guard_decisions row. Optional
        and last, so existing callers and test doubles keep working."""
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "INSERT INTO gw.messages (scope, model, cache_hit,"
                    " similarity, query_text, response, finish_reason, usage,"
                    " latency_ms, request_id)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        scope, model, cache_hit, similarity, query_text,
                        response, finish_reason,
                        Json(usage) if usage is not None else None, latency_ms,
                        request_id,
                    ),
                )
            return True
        except Exception as e:
            logger.error("Message log write failed: %s", e)
            return False

    def list_messages(
        self, scope: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Most recent messages for a scope (newest first)."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT model, cache_hit, similarity, query_text, response,"
                " finish_reason, usage, latency_ms, created_at"
                " FROM gw.messages WHERE scope = %s"
                " ORDER BY id DESC LIMIT %s",
                (scope, limit),
            ).fetchall()
        return [
            {
                "model": r[0],
                "cache_hit": r[1],
                "similarity": r[2],
                "query_text": r[3],
                "response": r[4],
                "finish_reason": r[5],
                "usage": r[6],
                "latency_ms": r[7],
                "created_at": r[8].isoformat(),
            }
            for r in rows
        ]

    # -- cache backup: durable mirror of Redis entries ----------------------- #

    def save_cache_entry(
        self,
        redis_key: str,
        key_prefix: str,
        scope: str,
        fields: Dict[str, Any],
        vector: bytes,
        expires_at: Optional[Any] = None,  # datetime | None (None = forever)
    ) -> bool:
        """Upserts one cache entry's backup row. FAIL-OPEN: the backup must
        never break the cache write it mirrors."""
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "INSERT INTO gw.cache_entries"
                    " (redis_key, key_prefix, scope, fields, vector,"
                    "  expires_at, updated_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, now())"
                    " ON CONFLICT (redis_key) DO UPDATE SET"
                    "  fields = EXCLUDED.fields, vector = EXCLUDED.vector,"
                    "  expires_at = EXCLUDED.expires_at, updated_at = now()",
                    (redis_key, key_prefix, scope, Json(fields), vector,
                     expires_at),
                )
            return True
        except Exception as e:
            logger.error("Cache backup write failed: %s", e)
            return False

    def touch_cache_entry(self, redis_key: str, expires_at: Optional[Any]) -> bool:
        """Refreshes a backup row's expiry after Redis extended/removed the
        TTL on a hit. Without this the backup expiry freezes at write time and
        rebuild-on-boot drops exactly the entries that get used most."""
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE gw.cache_entries SET expires_at = %s, updated_at = now()"
                    " WHERE redis_key = %s",
                    (expires_at, redis_key),
                )
            return True
        except Exception as e:
            logger.error("Cache backup touch failed: %s", e)
            return False

    def delete_cache_entries(self, redis_keys: List[str]) -> int:
        """Removes backup rows for purged entries so a later rebuild cannot
        resurrect deleted data. Returns rows removed (0 on error)."""
        if not redis_keys:
            return 0
        try:
            with self.pool.connection() as conn:
                result = conn.execute(
                    "DELETE FROM gw.cache_entries WHERE redis_key = ANY(%s)",
                    (list(redis_keys),),
                )
            return result.rowcount
        except Exception as e:
            logger.error("Cache backup delete failed: %s", e)
            return 0

    def load_cache_entries(self, batch_size: int = 500):
        """Yields live (non-expired) backup rows for rebuild-on-boot, pruning
        expired rows as it goes. Each row:
        (redis_key, fields, vector, expires_at)."""
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM gw.cache_entries WHERE expires_at <= now()")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT redis_key, fields, vector, expires_at"
                    " FROM gw.cache_entries"
                    " WHERE expires_at IS NULL OR expires_at > now()"
                )
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    yield from rows

    def count_cache_entries(self) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) FROM gw.cache_entries"
            ).fetchone()
        return int(row[0])

    # -- guard: exemplar matrices -------------------------------------------- #

    def save_guard_index(
        self,
        index_key: str,
        embed_model: str,
        policy_hash: str,
        dim: int,
        n: int,
        vectors: bytes,
        sentinel: bytes,
        meta: Dict[str, Any],
    ) -> bool:
        """Upserts one policy's exemplar matrix. FAIL-OPEN.

        ON CONFLICT makes re-ingest idempotent BY CONSTRUCTION — the index key
        is a content hash, so the same policy on the same embedder is the same
        row. (GaaS's ingest had no conflict clause and no natural key, so
        re-running it silently doubled every exemplar.)

        A failure here is not fatal but it is not free either: it degrades the
        build to once per process rather than once per policy, so it is logged
        at ERROR with a counter rather than swallowed quietly.
        """
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "INSERT INTO gw.guard_indexes"
                    " (index_key, embed_model, policy_hash, dim, n, vectors,"
                    "  sentinel, meta, last_used_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())"
                    " ON CONFLICT (index_key) DO UPDATE SET"
                    "  embed_model = EXCLUDED.embed_model,"
                    "  policy_hash = EXCLUDED.policy_hash,"
                    "  dim = EXCLUDED.dim, n = EXCLUDED.n,"
                    "  vectors = EXCLUDED.vectors,"
                    "  sentinel = EXCLUDED.sentinel, meta = EXCLUDED.meta,"
                    "  last_used_at = now()",
                    (index_key, embed_model, policy_hash, dim, n, vectors,
                     sentinel, Json(meta)),
                )
            return True
        except Exception as e:
            self.guard_persist_failures += 1
            logger.error("Guard index persist failed (%s): %s", index_key[:12], e)
            return False

    def load_guard_index(self, index_key: str) -> Optional[Dict[str, Any]]:
        """Returns a stored matrix, or None if absent or unreadable.

        FAIL-OPEN: a read failure means "build it", never "fail the request".
        """
        try:
            with self.pool.connection() as conn:
                row = conn.execute(
                    "SELECT embed_model, policy_hash, dim, n, vectors, sentinel,"
                    " meta FROM gw.guard_indexes WHERE index_key = %s",
                    (index_key,),
                ).fetchone()
        except Exception as e:
            self.guard_persist_failures += 1
            logger.error("Guard index load failed (%s): %s", index_key[:12], e)
            return None
        if row is None:
            return None
        return {
            "embed_model": row[0],
            "policy_hash": row[1],
            "dim": int(row[2]),
            "n": int(row[3]),
            "vectors": bytes(row[4]),
            "sentinel": bytes(row[5]),
            "meta": row[6],
        }

    def touch_guard_index(self, index_key: str) -> bool:
        """Marks an index as recently used, so the sweeper keeps it."""
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE gw.guard_indexes SET last_used_at = now()"
                    " WHERE index_key = %s",
                    (index_key,),
                )
            return True
        except Exception as e:
            logger.error("Guard index touch failed: %s", e)
            return False

    def delete_guard_index(self, index_key: str) -> int:
        """Drops one stored matrix — used when its sentinel no longer verifies."""
        try:
            with self.pool.connection() as conn:
                result = conn.execute(
                    "DELETE FROM gw.guard_indexes WHERE index_key = %s",
                    (index_key,),
                )
            return result.rowcount
        except Exception as e:
            logger.error("Guard index delete failed: %s", e)
            return 0

    def sweep_guard_indexes(self, older_than_days: int = 30) -> int:
        """Removes matrices unused for N days.

        Every policy edit strands its predecessor's row, on the CUSTOMER's
        Postgres. Without this they accumulate forever.
        """
        try:
            with self.pool.connection() as conn:
                result = conn.execute(
                    "DELETE FROM gw.guard_indexes"
                    " WHERE last_used_at < now() - make_interval(days => %s)",
                    (older_than_days,),
                )
            if result.rowcount:
                logger.info(
                    "Swept %d guard index row(s) unused for %d days.",
                    result.rowcount, older_than_days,
                )
            return result.rowcount
        except Exception as e:
            logger.error("Guard index sweep failed: %s", e)
            return 0

    # -- guard: decision log -------------------------------------------------- #

    def log_guard_decisions(self, rows: List[Dict[str, Any]]) -> int:
        """Batch-inserts guard decisions. FAIL-OPEN, like every other log here.

        Batched via executemany because there is one decision per request and
        the pool has only max_size connections; a per-request round trip would
        contend with the message log and the cache backup for the same handful.
        """
        if not rows:
            return 0
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO gw.guard_decisions"
                        " (request_id, scope, policy_hash, action,"
                        "  matched_category, reason, embedding_score,"
                        "  judge_score, judge_invoked, top_matches, turn_text,"
                        "  latency_ms)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (
                                r.get("request_id"), r["scope"], r["policy_hash"],
                                r["action"], r.get("matched_category"),
                                r.get("reason"), r.get("embedding_score"),
                                r.get("judge_score"),
                                bool(r.get("judge_invoked", False)),
                                Json(r["top_matches"])
                                if r.get("top_matches") is not None else None,
                                r.get("turn_text"), r.get("latency_ms"),
                            )
                            for r in rows
                        ],
                    )
            return len(rows)
        except Exception as e:
            logger.error("Guard decision log write failed (%d rows): %s",
                         len(rows), e)
            return 0

    def list_guard_decisions(
        self, scope: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Most recent guard decisions for a scope (newest first)."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT request_id, policy_hash, action, matched_category,"
                " reason, embedding_score, judge_score, judge_invoked,"
                " top_matches, turn_text, latency_ms, created_at"
                " FROM gw.guard_decisions WHERE scope = %s"
                " ORDER BY id DESC LIMIT %s",
                (scope, limit),
            ).fetchall()
        return [
            {
                "request_id": str(r[0]) if r[0] else None,
                "policy": (r[1] or "")[:12],
                "action": r[2],
                "matched_category": r[3],
                "reason": r[4],
                "embedding_score": r[5],
                "judge_score": r[6],
                "judge_invoked": r[7],
                "top_matches": r[8],
                "turn_text": r[9],
                "latency_ms": r[10],
                "created_at": r[11].isoformat(),
            }
            for r in rows
        ]
