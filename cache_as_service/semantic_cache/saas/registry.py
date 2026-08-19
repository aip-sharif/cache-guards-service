"""Tenant / API-key / cache registry for the multi-tenant SaaS layer.

Everything lives in plain Redis structures under the ``sc_saas:`` prefix —
deliberately OUTSIDE the ``scache:`` prefix that the RediSearch index covers,
so registry records never pollute the vector index.

Data model:
    sc_saas:apikey:{sha256(key)}        HASH  tenant_id, created_at
    sc_saas:tenantkeys:{tenant_id}      SET   key hashes (for rotation/revoke)
    sc_saas:tenant:{tenant_id}          HASH  name, created_at
    sc_saas:cache:{tenant_id}:{cache_id} HASH name, config_json, created_at
    sc_saas:caches:{tenant_id}          SET   cache_ids
    sc_saas:cachekey:{sha256(key)}      HASH  tenant_id, cache_id, created_at
    sc_saas:cachekeys:{tenant_id}:{cache_id} SET  key hashes (for rotation/revoke)

API keys are ``sc-<48 hex chars>``, returned exactly once at creation; only
the sha256 digest is stored.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from semantic_cache.core.config import SemanticCacheConfig

_PREFIX = "sc_saas:"

# Settings a tenant may customize per cache. Everything else (embedding model,
# index name, Redis connection, ...) is global infrastructure: per-cache
# embedding models would fragment the shared vector space.
PER_CACHE_FIELDS = frozenset(
    {
        "cache_mode",
        "lexical_scorer",
        "lexical_min_score",
        "fuzzy_distance",
        "semantic",
        "bm25",
        "fuzzy",
        "similarity_threshold",
        "entity_threshold",
        "default_ttl",
        "permanent_hit_threshold",
        "exact_tier",
        "normalize_aggressive",
        "fail_open",
    }
)


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class SaaSRegistry:
    """CRUD over tenants, API keys, and per-tenant named caches."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    # -- tenants / keys ---------------------------------------------------- #

    def create_tenant(self, name: str) -> Tuple[str, str]:
        """Creates a tenant; returns ``(tenant_id, plaintext_api_key)``.

        The plaintext key is shown exactly once — only its hash is stored."""
        tenant_id = uuid.uuid4().hex
        api_key = f"sc-{secrets.token_hex(24)}"
        now = str(int(time.time()))
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(f"{_PREFIX}tenant:{tenant_id}", mapping={"name": name, "created_at": now})
        pipe.hset(
            f"{_PREFIX}apikey:{_hash_key(api_key)}",
            mapping={"tenant_id": tenant_id, "created_at": now},
        )
        pipe.sadd(f"{_PREFIX}tenantkeys:{tenant_id}", _hash_key(api_key))
        pipe.execute()
        return tenant_id, api_key

    def get_or_create_tenant_by_external(self, external_id: str) -> str:
        """Maps an SSO identity (``"{org}|{user}"``) to a stable tenant_id,
        creating the tenant record on first sight (auto-provision).

        tenant_id = sha256(external_id) → 64 hex chars, a fixed point of the
        scope sanitizer, so no random key or admin bootstrap is needed."""
        tenant_id = hashlib.sha256(external_id.encode("utf-8")).hexdigest()
        tkey = f"{_PREFIX}tenant:{tenant_id}"
        if not self.redis.exists(tkey):
            self.redis.hset(
                tkey,
                mapping={"name": external_id, "created_at": str(int(time.time()))},
            )
        return tenant_id

    def resolve_api_key(self, api_key: str) -> Optional[str]:
        """Returns the tenant_id owning ``api_key``, or None."""
        if not api_key:
            return None
        tenant_id = self.redis.hget(f"{_PREFIX}apikey:{_hash_key(api_key)}", "tenant_id")
        return _decode(tenant_id) if tenant_id else None

    def rotate_tenant_key(self, tenant_id: str) -> str:
        """Issues a new tenant API key and revokes ALL previous ones."""
        new_key = f"sc-{secrets.token_hex(24)}"
        keyset = f"{_PREFIX}tenantkeys:{tenant_id}"
        old_hashes = [_decode(h) for h in self.redis.smembers(keyset)]
        pipe = self.redis.pipeline(transaction=False)
        for h in old_hashes:
            pipe.delete(f"{_PREFIX}apikey:{h}")
            pipe.srem(keyset, h)
        pipe.hset(
            f"{_PREFIX}apikey:{_hash_key(new_key)}",
            mapping={"tenant_id": tenant_id, "created_at": str(int(time.time()))},
        )
        pipe.sadd(keyset, _hash_key(new_key))
        pipe.execute()
        return new_key

    # -- per-cache keys ------------------------------------------------------ #

    def issue_cache_key(self, tenant_id: str, cache_id: str) -> str:
        """Issues a data-plane key scoped to ONE cache; plaintext shown once."""
        api_key = f"sc-{secrets.token_hex(24)}"
        h = _hash_key(api_key)
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(
            f"{_PREFIX}cachekey:{h}",
            mapping={
                "tenant_id": tenant_id,
                "cache_id": cache_id,
                "created_at": str(int(time.time())),
            },
        )
        pipe.sadd(f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}", h)
        pipe.execute()
        return api_key

    def resolve_cache_key(self, api_key: str) -> Optional[Tuple[str, str]]:
        """Returns ``(tenant_id, cache_id)`` for a per-cache key, or None."""
        if not api_key:
            return None
        raw = self.redis.hgetall(f"{_PREFIX}cachekey:{_hash_key(api_key)}")
        if not raw:
            return None
        data = {_decode(k): _decode(v) for k, v in raw.items()}
        return data["tenant_id"], data["cache_id"]

    def revoke_cache_keys(self, tenant_id: str, cache_id: str) -> int:
        """Revokes every key issued for this cache; returns how many."""
        keyset = f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}"
        hashes = [_decode(h) for h in self.redis.smembers(keyset)]
        if not hashes:
            return 0
        pipe = self.redis.pipeline(transaction=False)
        for h in hashes:
            pipe.delete(f"{_PREFIX}cachekey:{h}")
        pipe.delete(keyset)
        pipe.execute()
        return len(hashes)

    def rotate_cache_key(self, tenant_id: str, cache_id: str) -> str:
        """Revokes all existing keys for the cache and issues a fresh one."""
        self.revoke_cache_keys(tenant_id, cache_id)
        return self.issue_cache_key(tenant_id, cache_id)

    # -- caches ------------------------------------------------------------ #

    @staticmethod
    def data_scope(tenant_id: str, cache_id: str) -> str:
        """Composite data-isolation scope for one tenant-cache.

        Both ids are uuid4 hex, so the token is already a fixed point of
        SemanticCacheManager._scope_tag — no sanitization drift, no collisions."""
        return f"t{tenant_id}__c{cache_id}"

    @staticmethod
    def _validate_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(overrides) - PER_CACHE_FIELDS
        if unknown:
            raise ValueError(
                f"Fields not customizable per cache: {sorted(unknown)}. "
                f"Allowed: {sorted(PER_CACHE_FIELDS)}"
            )
        # Round-trip through the pydantic model so range/type errors surface
        # at create time, not at first request.
        try:
            validated = SemanticCacheConfig.model_validate(overrides)
        except Exception as e:
            raise ValueError(f"Invalid cache config: {e}") from e
        return {k: getattr(validated, k) for k in overrides}

    def create_cache(
        self, tenant_id: str, name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """Registers a new named cache for ``tenant_id``; returns its cache_id."""
        config = self._validate_overrides(overrides or {})
        cache_id = uuid.uuid4().hex
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(
            f"{_PREFIX}cache:{tenant_id}:{cache_id}",
            mapping={
                "name": name,
                "config_json": json.dumps(config),
                "created_at": str(int(time.time())),
            },
        )
        pipe.sadd(f"{_PREFIX}caches:{tenant_id}", cache_id)
        pipe.execute()
        return cache_id

    def get_cache(self, tenant_id: str, cache_id: str) -> Optional[Dict[str, Any]]:
        raw = self.redis.hgetall(f"{_PREFIX}cache:{tenant_id}:{cache_id}")
        if not raw:
            return None
        data = {_decode(k): _decode(v) for k, v in raw.items()}
        return {
            "cache_id": cache_id,
            "name": data.get("name", ""),
            "config": json.loads(data.get("config_json", "{}")),
            "config_json": data.get("config_json", "{}"),
            "created_at": int(data.get("created_at", "0")),
        }

    def list_caches(self, tenant_id: str) -> List[Dict[str, Any]]:
        cache_ids = sorted(_decode(c) for c in self.redis.smembers(f"{_PREFIX}caches:{tenant_id}"))
        caches = (self.get_cache(tenant_id, cid) for cid in cache_ids)
        return [c for c in caches if c is not None]

    def update_cache(
        self, tenant_id: str, cache_id: str, overrides: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Replaces the cache's config overrides; returns the updated record."""
        current = self.get_cache(tenant_id, cache_id)
        if current is None:
            return None
        config = self._validate_overrides(overrides)
        self.redis.hset(
            f"{_PREFIX}cache:{tenant_id}:{cache_id}",
            "config_json",
            json.dumps(config),
        )
        return self.get_cache(tenant_id, cache_id)

    def delete_cache(self, tenant_id: str, cache_id: str) -> bool:
        """Removes the registry record (NOT the cached data — callers purge
        the data scope first so a crash leaves a re-deletable record, never
        orphaned data)."""
        pipe = self.redis.pipeline(transaction=False)
        pipe.delete(f"{_PREFIX}cache:{tenant_id}:{cache_id}")
        pipe.srem(f"{_PREFIX}caches:{tenant_id}", cache_id)
        deleted, _ = pipe.execute()
        return bool(deleted)
