"""Tenant / API-key / cache registry for the multi-tenant SaaS layer.

Everything lives in plain Redis structures under the ``sc_saas:`` prefix —
deliberately OUTSIDE the ``scache:`` prefix that the RediSearch index covers,
so registry records never pollute the vector index.

Data model:
    sc_saas:apikey:{fp(key)}            HASH  tenant_id, key_id, prefix,
                                              created_at, expires_at, last_used_at
    sc_saas:tenantkeys:{tenant_id}      SET   fingerprints (rotation/revoke)
    sc_saas:tenant:{tenant_id}          HASH  name, created_at
    sc_saas:cache:{tenant_id}:{cache_id} HASH name, config_json, created_at
    sc_saas:caches:{tenant_id}          SET   cache_ids
    sc_saas:cachekey:{fp(key)}          HASH  tenant_id, cache_id, key_id,
                                              prefix, created_at, expires_at,
                                              last_used_at
    sc_saas:cachekeys:{tenant_id}:{cache_id} SET  fingerprints

API keys are ``sc-<48 hex chars>``, returned exactly once at creation; only a
FINGERPRINT is stored.

THE FINGERPRINT
---------------
fp(key) is HMAC-SHA256 under a server-side pepper (SC_API_KEY_PEPPER), not a
bare sha256. A bare digest of a key is not a secret held only by us: anyone who
reads the Redis keyspace — a backup, a snapshot, a support dump, an instance
running with protected-mode off — can confirm a candidate key offline by
hashing it, and can precompute over any structure the key format has. With a
pepper they cannot check a guess without also stealing the pepper, which lives
in the app environment rather than in the data store.

Peppering is BACKWARD COMPATIBLE. Bare-sha256 records predate this and are
still live credentials, so a lookup falls back to the legacy digest and, on a
hit, rewrites the record under the peppered fingerprint. Keys migrate as they
are used; nobody is logged out by a deploy. With no pepper configured the
registry behaves exactly as before and says so at boot.

KEY LIFECYCLE
-------------
Every record carries a key_id and a display prefix, so a key can be listed,
audited and revoked INDIVIDUALLY by someone who does not hold the key itself.
Records may carry expires_at; an expired key resolves to None and is deleted on
sight. Rotation takes a grace period so the old key keeps working while callers
redeploy — an atomic rotate that instantly revokes every existing key is a
self-service outage button.

Never log a key OR its fingerprint: the fingerprint is a verifier for the key.
key_id and prefix exist to be logged instead.
"""

from __future__ import annotations

import hashlib
import hmac
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


def _legacy_hash(api_key: str) -> str:
    """The un-peppered digest this registry used before. Retained ONLY so
    existing keys keep working; nothing new is written under it."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_key() -> str:
    return f"sc-{secrets.token_hex(24)}"


def _prefix_of(api_key: str) -> str:
    """The displayable head of a key — enough to tell two keys apart in an
    audit line, far too little to authenticate with."""
    return api_key[:11]


def _now() -> int:
    return int(time.time())


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class SaaSRegistry:
    """CRUD over tenants, API keys, and per-tenant named caches."""

    def __init__(self, redis_client: Any, pepper: Optional[str] = None) -> None:
        self.redis = redis_client
        self._pepper = pepper.encode("utf-8") if pepper else None

    # -- fingerprints ------------------------------------------------------ #

    def _fingerprint(self, api_key: str) -> str:
        """The stored lookup value for a key.

        Falls back to the bare digest when no pepper is configured, so a
        deployment that has not set SC_API_KEY_PEPPER keeps working unchanged
        rather than logging every tenant out."""
        if self._pepper is None:
            return _legacy_hash(api_key)
        return hmac.new(
            self._pepper, api_key.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _candidates(self, api_key: str) -> List[str]:
        """Fingerprints to try, current first. The legacy digest is second and
        only while a pepper is configured — that is the migration path."""
        current = self._fingerprint(api_key)
        if self._pepper is None:
            return [current]
        return [current, _legacy_hash(api_key)]

    def _lookup(
        self, namespace: str, api_key: str
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        """(fingerprint, record) for a LIVE key, or None.

        Three things happen here because they are all the same read: try the
        peppered fingerprint then the legacy one, migrate a legacy hit onto the
        peppered fingerprint, and delete an expired record rather than return
        it."""
        if not api_key:
            return None
        for index, fp in enumerate(self._candidates(api_key)):
            raw = self.redis.hgetall(f"{_PREFIX}{namespace}:{fp}")
            if not raw:
                continue
            record = {_decode(k): _decode(v) for k, v in raw.items()}

            expires_at = record.get("expires_at")
            if expires_at and int(expires_at) <= _now():
                # Expiry is enforced HERE rather than left to a Redis TTL: the
                # fingerprint also lives in a tenantkeys/cachekeys SET, and a
                # TTL would silently leave that set pointing at nothing.
                self._forget(namespace, record, fp)
                return None

            if index > 0:
                fp = self._migrate(namespace, record, fp, api_key)
            self._touch(namespace, fp, record)
            return fp, record
        return None

    def _migrate(
        self, namespace: str, record: Dict[str, str], legacy_fp: str, api_key: str
    ) -> str:
        """Rewrites a legacy-digest record under the peppered fingerprint.

        Best effort on purpose: if it fails the caller still authenticated, and
        the next request tries again. An authentication path is the wrong place
        to turn a housekeeping failure into a 500."""
        new_fp = self._fingerprint(api_key)
        try:
            record.setdefault("key_id", uuid.uuid4().hex)
            record.setdefault("prefix", _prefix_of(api_key))
            pipe = self.redis.pipeline(transaction=False)
            pipe.hset(f"{_PREFIX}{namespace}:{new_fp}", mapping=dict(record))
            pipe.delete(f"{_PREFIX}{namespace}:{legacy_fp}")
            for set_key in self._key_sets(namespace, record):
                pipe.srem(set_key, legacy_fp)
                pipe.sadd(set_key, new_fp)
            pipe.execute()
            return new_fp
        except Exception:  # noqa: BLE001 — see docstring
            return legacy_fp

    def _forget(self, namespace: str, record: Dict[str, str], fp: str) -> None:
        try:
            pipe = self.redis.pipeline(transaction=False)
            pipe.delete(f"{_PREFIX}{namespace}:{fp}")
            for set_key in self._key_sets(namespace, record):
                pipe.srem(set_key, fp)
            pipe.execute()
        except Exception:  # noqa: BLE001 — the key is already refused
            pass

    def _touch(self, namespace: str, fp: str, record: Dict[str, str]) -> None:
        """Records last use, at ONE-MINUTE granularity.

        Writing on every request would turn every authenticated read into a
        write against the control plane — a throughput ceiling bought for a
        field nobody reads to the second."""
        now = _now()
        try:
            last = int(record.get("last_used_at") or 0)
        except ValueError:
            last = 0
        if now - last < 60:
            return
        try:
            self.redis.hset(f"{_PREFIX}{namespace}:{fp}", "last_used_at", str(now))
        except Exception:  # noqa: BLE001 — bookkeeping, never the request
            pass

    @staticmethod
    def _key_sets(namespace: str, record: Dict[str, str]) -> List[str]:
        """The SET(s) that index this record's fingerprint."""
        tenant_id = record.get("tenant_id")
        if not tenant_id:
            return []
        if namespace == "apikey":
            return [f"{_PREFIX}tenantkeys:{tenant_id}"]
        cache_id = record.get("cache_id")
        if not cache_id:
            return []
        return [f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}"]

    # -- tenants / keys ---------------------------------------------------- #

    def create_tenant(self, name: str) -> Tuple[str, str]:
        """Creates a tenant; returns ``(tenant_id, plaintext_api_key)``.

        The plaintext key is shown exactly once — only its hash is stored."""
        tenant_id = uuid.uuid4().hex
        api_key = _new_key()
        now = str(_now())
        fp = self._fingerprint(api_key)
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(
            f"{_PREFIX}tenant:{tenant_id}",
            mapping={"name": name, "created_at": now},
        )
        pipe.hset(
            f"{_PREFIX}apikey:{fp}",
            mapping={
                "tenant_id": tenant_id,
                "key_id": uuid.uuid4().hex,
                "prefix": _prefix_of(api_key),
                "created_at": now,
            },
        )
        pipe.sadd(f"{_PREFIX}tenantkeys:{tenant_id}", fp)
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
        """Returns the tenant_id owning ``api_key``, or None.

        None covers unknown, revoked AND expired — a caller must not be able to
        tell those apart, or the endpoint becomes an oracle for probing which
        keys once existed."""
        found = self._lookup("apikey", api_key)
        return found[1].get("tenant_id") if found else None

    def rotate_tenant_key(self, tenant_id: str, grace: int = 0) -> str:
        """Issues a new tenant API key.

        ``grace`` is seconds the OLD keys keep working. The default of 0 is the
        historical behaviour — instant revocation — but a caller that cannot
        redeploy in zero seconds should pass a real grace period: a rotate that
        breaks production the moment it is clicked is a rotate nobody clicks,
        and unrotated keys are the actual risk this endpoint exists to reduce.
        """
        new_key = _new_key()
        keyset = f"{_PREFIX}tenantkeys:{tenant_id}"
        old_fps = [_decode(h) for h in self.redis.smembers(keyset)]
        deadline = str(_now() + grace)
        now = str(_now())
        pipe = self.redis.pipeline(transaction=False)
        for fp in old_fps:
            if grace > 0:
                # Expiry, not deletion: the key stays resolvable until the
                # deadline and _lookup removes it on first use after that.
                pipe.hset(f"{_PREFIX}apikey:{fp}", "expires_at", deadline)
            else:
                pipe.delete(f"{_PREFIX}apikey:{fp}")
                pipe.srem(keyset, fp)
        fp = self._fingerprint(new_key)
        pipe.hset(
            f"{_PREFIX}apikey:{fp}",
            mapping={
                "tenant_id": tenant_id,
                "key_id": uuid.uuid4().hex,
                "prefix": _prefix_of(new_key),
                "created_at": now,
            },
        )
        pipe.sadd(keyset, fp)
        pipe.execute()
        return new_key

    def list_tenant_keys(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Every live key for a tenant, by key_id and prefix.

        Returns no fingerprint. A fingerprint verifies a key; handing it to a
        UI or a log would put a credential-checking value somewhere it is not
        guarded like one."""
        return self._list_keys("apikey", f"{_PREFIX}tenantkeys:{tenant_id}")

    def revoke_key(self, tenant_id: str, key_id: str) -> bool:
        """Revokes ONE tenant key by its key_id. True if it existed.

        Selective revocation is the point: before this, losing one key meant
        rotating every key the tenant had, which punishes the incident."""
        return self._revoke_by_id(
            "apikey", f"{_PREFIX}tenantkeys:{tenant_id}", key_id
        )

    # -- shared key-listing / revocation ------------------------------------ #

    def _list_keys(self, namespace: str, keyset: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        now = _now()
        for fp in sorted(_decode(h) for h in self.redis.smembers(keyset)):
            raw = self.redis.hgetall(f"{_PREFIX}{namespace}:{fp}")
            if not raw:
                continue
            record = {_decode(k): _decode(v) for k, v in raw.items()}
            expires_at = record.get("expires_at")
            if expires_at and int(expires_at) <= now:
                continue
            out.append(
                {
                    "key_id": record.get("key_id"),
                    "prefix": record.get("prefix"),
                    "created_at": record.get("created_at"),
                    "expires_at": expires_at,
                    "last_used_at": record.get("last_used_at"),
                }
            )
        return out

    def _revoke_by_id(self, namespace: str, keyset: str, key_id: str) -> bool:
        for fp in (_decode(h) for h in self.redis.smembers(keyset)):
            raw = self.redis.hgetall(f"{_PREFIX}{namespace}:{fp}")
            if not raw:
                continue
            record = {_decode(k): _decode(v) for k, v in raw.items()}
            if record.get("key_id") != key_id:
                continue
            pipe = self.redis.pipeline(transaction=False)
            pipe.delete(f"{_PREFIX}{namespace}:{fp}")
            pipe.srem(keyset, fp)
            pipe.execute()
            return True
        return False

    # -- per-cache keys ------------------------------------------------------ #

    def issue_cache_key(
        self, tenant_id: str, cache_id: str, expires_in: Optional[int] = None
    ) -> str:
        """Issues a data-plane key scoped to ONE cache; plaintext shown once.

        ``expires_in`` is seconds. Unset means a key that never expires, which
        is the historical behaviour and is why keys accumulate."""
        api_key = _new_key()
        fp = self._fingerprint(api_key)
        mapping = {
            "tenant_id": tenant_id,
            "cache_id": cache_id,
            "key_id": uuid.uuid4().hex,
            "prefix": _prefix_of(api_key),
            "created_at": str(_now()),
        }
        if expires_in:
            mapping["expires_at"] = str(_now() + int(expires_in))
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(f"{_PREFIX}cachekey:{fp}", mapping=mapping)
        pipe.sadd(f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}", fp)
        pipe.execute()
        return api_key

    def list_cache_keys(self, tenant_id: str, cache_id: str) -> List[Dict[str, Any]]:
        """Live keys for one cache, by key_id and prefix. No fingerprints."""
        return self._list_keys(
            "cachekey", f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}"
        )

    def revoke_cache_key(self, tenant_id: str, cache_id: str, key_id: str) -> bool:
        """Revokes ONE per-cache key by key_id. True if it existed."""
        return self._revoke_by_id(
            "cachekey", f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}", key_id
        )

    def resolve_cache_key(self, api_key: str) -> Optional[Tuple[str, str]]:
        """Returns ``(tenant_id, cache_id)`` for a per-cache key, or None.

        As with ``resolve_api_key``, None covers unknown, revoked and expired
        alike."""
        found = self._lookup("cachekey", api_key)
        if not found:
            return None
        record = found[1]
        return record["tenant_id"], record["cache_id"]

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

    def rotate_cache_key(
        self, tenant_id: str, cache_id: str, grace: int = 0
    ) -> str:
        """Issues a fresh key for the cache and retires the existing ones.

        ``grace`` seconds keeps the old keys working so callers can redeploy;
        0 (the default, and the historical behaviour) revokes immediately."""
        if grace > 0:
            keyset = f"{_PREFIX}cachekeys:{tenant_id}:{cache_id}"
            deadline = str(_now() + grace)
            pipe = self.redis.pipeline(transaction=False)
            for fp in (_decode(h) for h in self.redis.smembers(keyset)):
                pipe.hset(f"{_PREFIX}cachekey:{fp}", "expires_at", deadline)
            pipe.execute()
        else:
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
