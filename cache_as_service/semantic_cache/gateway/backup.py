"""Durable cache backup: Postgres mirror of the Redis cache.

Redis stays the SERVING cache (all lookups are vector KNN in RediSearch).
Postgres (`gw.cache_entries`) is a write-through backup of every entry —
including its embedding vector — so a wiped/replaced Redis can be rebuilt on
boot without re-calling the embedding API.

Two pieces:
  * make_write_through(store)  — the manager persist_hook: mirrors each
    successful Redis write into Postgres (fail-open).
  * rebuild_redis_from_postgres(store, redis_client) — on boot, restores any
    backed-up entry Redis no longer has, honoring the remaining TTL. The
    RediSearch indexes are created lazily by the managers (FT.CREATE scans
    existing keys), so restored hashes are picked up automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _split_prefix(redis_key: str) -> str:
    """The key's prefix (everything before the trailing sha256 hex digest)."""
    return redis_key[:-64] if len(redis_key) > 64 else redis_key


def make_write_through(store) -> Callable[[str, Dict[str, Any], Optional[int]], None]:
    """A SemanticCacheManager.persist_hook that mirrors writes to Postgres."""

    def hook(redis_key: str, mapping: Dict[str, Any], ttl: Optional[int]) -> None:
        fields = {k: v for k, v in mapping.items() if not isinstance(v, bytes)}
        vector = mapping.get("vector") or b""
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl)
            if ttl is not None
            else None
        )
        store.save_cache_entry(
            redis_key=redis_key,
            key_prefix=_split_prefix(redis_key),
            scope=str(mapping.get("scope") or ""),
            fields=fields,
            vector=vector,
            expires_at=expires_at,
        )

    return hook


def make_touch_through(store) -> Callable[[str, Optional[int]], None]:
    """A SemanticCacheManager.touch_hook that keeps the backup's expiry in
    step with Redis when a hit refreshes or removes the TTL."""

    def hook(redis_key: str, ttl: Optional[int]) -> None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl)
            if ttl is not None
            else None
        )
        store.touch_cache_entry(redis_key, expires_at)

    return hook


def make_delete_through(store) -> Callable[[List[str]], None]:
    """A SemanticCacheManager.delete_hook that drops backup rows for purged
    entries, so rebuild-on-boot cannot resurrect deleted data."""

    def hook(redis_keys: List[str]) -> None:
        store.delete_cache_entries(list(redis_keys))

    return hook


def rebuild_redis_from_postgres(store, redis_client) -> int:
    """Restores backed-up cache entries that Redis no longer holds.

    Returns the number of entries restored. Never raises — a failed rebuild
    must not stop the service from booting (the cache refills from traffic)."""
    restored = 0
    try:
        now = datetime.now(timezone.utc)
        for redis_key, fields, vector, expires_at in store.load_cache_entries():
            try:
                if redis_client.exists(redis_key):
                    continue  # Redis already has a (possibly newer) entry
                mapping = dict(fields)
                mapping["vector"] = bytes(vector)  # psycopg returns memoryview
                redis_client.hset(redis_key, mapping=mapping)
                if expires_at is not None:
                    remaining = int((expires_at - now).total_seconds())
                    if remaining <= 0:
                        redis_client.delete(redis_key)
                        continue
                    redis_client.expire(redis_key, remaining)
                else:
                    redis_client.persist(redis_key)
                restored += 1
            except Exception as e:  # noqa: BLE001 — one bad row must not stop the rest
                logger.error("Restore failed for %s: %s", redis_key, e)
    except Exception as e:  # noqa: BLE001
        logger.error("Cache rebuild from Postgres failed: %s", e)
    if restored:
        logger.info("Rebuilt %d cache entries from Postgres into Redis.", restored)
    return restored
