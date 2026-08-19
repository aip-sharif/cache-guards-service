"""Durable cache backup — write-through hook + rebuild-on-boot. Pure unit:
fake store, fake Redis, no network."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from semantic_cache.gateway.backup import (
    make_delete_through,
    make_touch_through,
    make_write_through,
    rebuild_redis_from_postgres,
)

KEY = "scache:gw:bge-m3:" + "a" * 64


class FakeBackupStore:
    def __init__(self) -> None:
        self.saved: List[Dict[str, Any]] = []
        self.rows: List[tuple] = []   # (redis_key, fields, vector, expires_at)
        self.touched: List[tuple] = []
        self.deleted: List[str] = []
        self.boom = False

    def save_cache_entry(self, redis_key, key_prefix, scope, fields, vector,
                         expires_at=None) -> bool:
        self.saved.append({
            "redis_key": redis_key, "key_prefix": key_prefix, "scope": scope,
            "fields": fields, "vector": vector, "expires_at": expires_at,
        })
        return True

    def touch_cache_entry(self, redis_key, expires_at) -> bool:
        self.touched.append((redis_key, expires_at))
        return True

    def delete_cache_entries(self, redis_keys) -> int:
        self.deleted.extend(redis_keys)
        return len(redis_keys)

    def load_cache_entries(self, batch_size: int = 500):
        if self.boom:
            raise RuntimeError("pg down")
        yield from self.rows


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: Dict[str, Dict[str, Any]] = {}
        self.ttls: Dict[str, Optional[int]] = {}

    def exists(self, key) -> bool:
        return key in self.hashes

    def hset(self, key, mapping) -> None:
        self.hashes[key] = dict(mapping)

    def expire(self, key, seconds) -> None:
        self.ttls[key] = seconds

    def persist(self, key) -> None:
        self.ttls[key] = None

    def delete(self, key) -> None:
        self.hashes.pop(key, None)
        self.ttls.pop(key, None)


MAPPING = {
    "text": "capital of france",
    "response": "Paris",
    "vector": b"\x00\x01\x02\x03",
    "hits": 1,
    "timestamp": 1700000000,
    "metadata": "{}",
    "entities": "[]",
    "entity_sig": "",
    "domain": "general",
    "scope": "proj1",
}


# -- write-through hook -------------------------------------------------------


def test_hook_mirrors_entry_with_vector_and_prefix() -> None:
    store = FakeBackupStore()
    make_write_through(store)(KEY, MAPPING, ttl=3600)
    row = store.saved[0]
    assert row["redis_key"] == KEY
    assert row["key_prefix"] == "scache:gw:bge-m3:"
    assert row["scope"] == "proj1"
    assert row["vector"] == b"\x00\x01\x02\x03"
    assert "vector" not in row["fields"]            # bytes never go into jsonb
    assert row["fields"]["response"] == "Paris"
    remaining = row["expires_at"] - datetime.now(timezone.utc)
    assert timedelta(minutes=55) < remaining <= timedelta(hours=1)


def test_hook_keep_forever_has_no_expiry() -> None:
    store = FakeBackupStore()
    make_write_through(store)(KEY, MAPPING, ttl=None)
    assert store.saved[0]["expires_at"] is None


# -- touch hook: keep the backup's expiry in step with Redis ------------------


def test_touch_refreshes_expiry_so_hot_entries_survive() -> None:
    # Without this the backup expiry freezes at write time and rebuild-on-boot
    # reaps exactly the entries that get used most.
    store = FakeBackupStore()
    make_touch_through(store)(KEY, 86400)
    key, expires_at = store.touched[0]
    assert key == KEY
    assert expires_at - datetime.now(timezone.utc) > timedelta(hours=23)


def test_touch_with_no_ttl_marks_entry_permanent() -> None:
    store = FakeBackupStore()
    make_touch_through(store)(KEY, None)      # LFU promotion to permanent
    assert store.touched[0] == (KEY, None)


# -- delete hook: purged data must not come back ------------------------------


def test_delete_removes_backup_rows() -> None:
    store = FakeBackupStore()
    make_delete_through(store)([KEY, "scache:gw:x:" + "b" * 64])
    assert store.deleted == [KEY, "scache:gw:x:" + "b" * 64]


# -- rebuild-on-boot ----------------------------------------------------------


def _fields():
    return {k: v for k, v in MAPPING.items() if not isinstance(v, bytes)}


def test_rebuild_restores_missing_entries_with_remaining_ttl() -> None:
    store, redis = FakeBackupStore(), FakeRedis()
    later = datetime.now(timezone.utc) + timedelta(seconds=500)
    store.rows = [(KEY, _fields(), memoryview(b"\x09\x08"), later)]
    assert rebuild_redis_from_postgres(store, redis) == 1
    assert redis.hashes[KEY]["response"] == "Paris"
    assert redis.hashes[KEY]["vector"] == b"\x09\x08"   # memoryview → bytes
    assert 0 < redis.ttls[KEY] <= 500


def test_rebuild_keep_forever_persists() -> None:
    store, redis = FakeBackupStore(), FakeRedis()
    store.rows = [(KEY, _fields(), b"\x01", None)]
    rebuild_redis_from_postgres(store, redis)
    assert redis.ttls[KEY] is None
    assert KEY in redis.hashes


def test_rebuild_skips_entries_redis_already_has() -> None:
    store, redis = FakeBackupStore(), FakeRedis()
    redis.hashes[KEY] = {"response": "newer-answer"}
    store.rows = [(KEY, _fields(), b"\x01", None)]
    assert rebuild_redis_from_postgres(store, redis) == 0
    assert redis.hashes[KEY] == {"response": "newer-answer"}  # untouched


def test_rebuild_drops_rows_expired_between_query_and_restore() -> None:
    store, redis = FakeBackupStore(), FakeRedis()
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    store.rows = [(KEY, _fields(), b"\x01", past)]
    assert rebuild_redis_from_postgres(store, redis) == 0
    assert KEY not in redis.hashes


def test_rebuild_never_raises_when_store_fails() -> None:
    store, redis = FakeBackupStore(), FakeRedis()
    store.boom = True
    assert rebuild_redis_from_postgres(store, redis) == 0
