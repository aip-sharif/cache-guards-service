"""API-key storage and lifecycle in the SaaS registry.

There were no registry tests at all, which is why key handling drifted without
anything noticing. These cover the properties that matter for a credential
store: what is written down, whether a stolen keyspace lets you check a guess,
whether revocation is selective, whether expiry is honoured, and whether an
existing key survives the change that introduced the pepper.

A dict-backed fake Redis stands in for the real one — the point is the
registry's logic, not redis-py.
"""

from typing import Dict, List, Set

import pytest

from semantic_cache.saas.registry import SaaSRegistry, _legacy_hash

PEPPER = "a-long-random-server-side-pepper"


class FakeRedis:
    """Just the hash/set surface the registry uses, plus a no-op pipeline."""

    def __init__(self) -> None:
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.sets: Dict[str, Set[str]] = {}

    # -- hashes -------------------------------------------------------------
    def hset(self, key: str, field: str = None, value: str = None, mapping=None):
        target = self.hashes.setdefault(key, {})
        if mapping:
            target.update({str(k): str(v) for k, v in mapping.items()})
        if field is not None:
            target[str(field)] = str(value)

    def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0

    def delete(self, key: str) -> None:
        self.hashes.pop(key, None)
        self.sets.pop(key, None)

    # -- sets ---------------------------------------------------------------
    def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key: str, member: str) -> None:
        self.sets.get(key, set()).discard(member)

    def smembers(self, key: str) -> Set[str]:
        return set(self.sets.get(key, set()))

    # -- pipeline: the registry only ever batches, never transacts ----------
    def pipeline(self, transaction: bool = False) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: List[tuple] = []

    def __getattr__(self, name: str):
        def queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self
        return queue

    def execute(self) -> None:
        for name, args, kwargs in self._ops:
            getattr(self._redis, name)(*args, **kwargs)
        self._ops.clear()


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


def _peppered(redis: FakeRedis) -> SaaSRegistry:
    return SaaSRegistry(redis, pepper=PEPPER)


# --------------------------------------------------------------------------- #
# What gets written down
# --------------------------------------------------------------------------- #


def test_the_key_itself_is_never_stored(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    _, api_key = registry.create_tenant("acme")
    blob = repr(redis.hashes) + repr(redis.sets)
    assert api_key not in blob


def test_a_stolen_keyspace_cannot_confirm_a_guess(redis: FakeRedis) -> None:
    """The point of the pepper. A bare sha256 lets anyone holding a backup
    check a candidate key offline; an HMAC needs the pepper too."""
    registry = _peppered(redis)
    _, api_key = registry.create_tenant("acme")
    assert not any(_legacy_hash(api_key) in k for k in redis.hashes)


def test_without_a_pepper_the_legacy_layout_is_kept(redis: FakeRedis) -> None:
    """An existing deployment that has not set the pepper must keep working
    exactly as before, not lose every tenant on deploy."""
    registry = SaaSRegistry(redis)
    _, api_key = registry.create_tenant("acme")
    assert any(_legacy_hash(api_key) in k for k in redis.hashes)


def test_a_record_carries_an_id_and_a_prefix(redis: FakeRedis) -> None:
    """So a key can be listed, audited and revoked by someone who does not
    hold it."""
    registry = _peppered(redis)
    tenant_id, api_key = registry.create_tenant("acme")
    (record,) = registry.list_tenant_keys(tenant_id)
    assert record["key_id"]
    assert record["prefix"] == api_key[:11]
    assert "fingerprint" not in record  # a fingerprint verifies a key


# --------------------------------------------------------------------------- #
# Migration: existing keys must not stop working
# --------------------------------------------------------------------------- #


def test_a_legacy_key_still_authenticates_after_the_pepper_arrives(
    redis: FakeRedis,
) -> None:
    legacy = SaaSRegistry(redis)  # pre-pepper deployment
    tenant_id, api_key = legacy.create_tenant("acme")

    peppered = _peppered(redis)  # after the config change
    assert peppered.resolve_api_key(api_key) == tenant_id


def test_a_legacy_key_migrates_to_the_peppered_fingerprint_on_use(
    redis: FakeRedis,
) -> None:
    legacy = SaaSRegistry(redis)
    tenant_id, api_key = legacy.create_tenant("acme")
    assert any(_legacy_hash(api_key) in k for k in redis.hashes)

    peppered = _peppered(redis)
    peppered.resolve_api_key(api_key)

    assert not any(_legacy_hash(api_key) in k for k in redis.hashes)
    assert peppered.resolve_api_key(api_key) == tenant_id


def test_migration_keeps_the_tenant_key_set_consistent(redis: FakeRedis) -> None:
    """A migration that moved the record but not the set entry would leave
    rotation and listing pointing at nothing."""
    legacy = SaaSRegistry(redis)
    tenant_id, api_key = legacy.create_tenant("acme")

    peppered = _peppered(redis)
    peppered.resolve_api_key(api_key)

    assert len(peppered.list_tenant_keys(tenant_id)) == 1


# --------------------------------------------------------------------------- #
# Revocation and expiry
# --------------------------------------------------------------------------- #


def test_revoking_one_key_leaves_the_others_working(redis: FakeRedis) -> None:
    """Selective revocation is the point: losing one key should not mean
    rotating every key the tenant has."""
    registry = _peppered(redis)
    tenant_id, first = registry.create_tenant("acme")
    second = registry.rotate_tenant_key(tenant_id, grace=3600)

    (a, b) = registry.list_tenant_keys(tenant_id)
    leaked = a if registry.resolve_api_key(first) and a["prefix"] == first[:11] else b
    assert registry.revoke_key(tenant_id, leaked["key_id"]) is True

    assert registry.resolve_api_key(first) is None
    assert registry.resolve_api_key(second) == tenant_id


def test_revoking_an_unknown_key_id_is_false_not_an_error(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    tenant_id, _ = registry.create_tenant("acme")
    assert registry.revoke_key(tenant_id, "no-such-id") is False


def test_rotation_with_a_grace_keeps_the_old_key_alive(redis: FakeRedis) -> None:
    """A rotate that breaks production the instant it is clicked is a rotate
    nobody clicks — and unrotated keys are the real risk."""
    registry = _peppered(redis)
    tenant_id, old = registry.create_tenant("acme")
    new = registry.rotate_tenant_key(tenant_id, grace=3600)

    assert registry.resolve_api_key(old) == tenant_id
    assert registry.resolve_api_key(new) == tenant_id


def test_rotation_without_a_grace_revokes_immediately(redis: FakeRedis) -> None:
    """The historical behaviour, still the default."""
    registry = _peppered(redis)
    tenant_id, old = registry.create_tenant("acme")
    new = registry.rotate_tenant_key(tenant_id)

    assert registry.resolve_api_key(old) is None
    assert registry.resolve_api_key(new) == tenant_id


def test_an_expired_key_stops_resolving(redis: FakeRedis, monkeypatch) -> None:
    registry = _peppered(redis)
    tenant_id, old = registry.create_tenant("acme")
    registry.rotate_tenant_key(tenant_id, grace=60)
    assert registry.resolve_api_key(old) == tenant_id

    import semantic_cache.saas.registry as reg
    monkeypatch.setattr(reg, "_now", lambda: reg.time.time() + 3600)
    assert registry.resolve_api_key(old) is None


def test_an_expired_key_is_deleted_on_sight(redis: FakeRedis, monkeypatch) -> None:
    """Expiry is enforced in the lookup rather than by a Redis TTL, because the
    fingerprint also lives in a SET that a TTL would leave dangling."""
    registry = _peppered(redis)
    tenant_id, old = registry.create_tenant("acme")
    registry.rotate_tenant_key(tenant_id, grace=60)

    import semantic_cache.saas.registry as reg
    monkeypatch.setattr(reg, "_now", lambda: reg.time.time() + 3600)
    registry.resolve_api_key(old)

    assert len(registry.list_tenant_keys(tenant_id)) == 1


def test_an_expired_key_is_indistinguishable_from_an_unknown_one(
    redis: FakeRedis, monkeypatch
) -> None:
    """Otherwise the endpoint is an oracle for which keys once existed."""
    registry = _peppered(redis)
    tenant_id, old = registry.create_tenant("acme")
    registry.rotate_tenant_key(tenant_id, grace=60)

    import semantic_cache.saas.registry as reg
    monkeypatch.setattr(reg, "_now", lambda: reg.time.time() + 3600)
    assert registry.resolve_api_key(old) is registry.resolve_api_key("sc-nonsense")


# --------------------------------------------------------------------------- #
# Per-cache keys carry the same lifecycle
# --------------------------------------------------------------------------- #


def test_a_cache_key_resolves_to_its_tenant_and_cache(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    key = registry.issue_cache_key("t1", "c1")
    assert registry.resolve_cache_key(key) == ("t1", "c1")


def test_a_cache_key_can_expire(redis: FakeRedis, monkeypatch) -> None:
    registry = _peppered(redis)
    key = registry.issue_cache_key("t1", "c1", expires_in=60)
    assert registry.resolve_cache_key(key) == ("t1", "c1")

    import semantic_cache.saas.registry as reg
    monkeypatch.setattr(reg, "_now", lambda: reg.time.time() + 3600)
    assert registry.resolve_cache_key(key) is None


def test_one_cache_key_can_be_revoked_by_id(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    first = registry.issue_cache_key("t1", "c1")
    second = registry.issue_cache_key("t1", "c1")
    listed = registry.list_cache_keys("t1", "c1")
    target = next(r for r in listed if r["prefix"] == first[:11])

    assert registry.revoke_cache_key("t1", "c1", target["key_id"]) is True
    assert registry.resolve_cache_key(first) is None
    assert registry.resolve_cache_key(second) == ("t1", "c1")


def test_a_legacy_cache_key_migrates_too(redis: FakeRedis) -> None:
    legacy = SaaSRegistry(redis)
    key = legacy.issue_cache_key("t1", "c1")

    peppered = _peppered(redis)
    assert peppered.resolve_cache_key(key) == ("t1", "c1")
    assert not any(_legacy_hash(key) in k for k in redis.hashes)


# --------------------------------------------------------------------------- #
# Last-used bookkeeping
# --------------------------------------------------------------------------- #


def test_last_used_is_recorded(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    tenant_id, api_key = registry.create_tenant("acme")
    registry.resolve_api_key(api_key)
    (record,) = registry.list_tenant_keys(tenant_id)
    assert record["last_used_at"]


def test_last_used_is_not_rewritten_on_every_request(redis: FakeRedis) -> None:
    """Writing per request turns every authenticated READ into a control-plane
    WRITE — a throughput ceiling bought for a field nobody reads to the
    second."""
    registry = _peppered(redis)
    tenant_id, api_key = registry.create_tenant("acme")
    registry.resolve_api_key(api_key)
    (first,) = registry.list_tenant_keys(tenant_id)

    for _ in range(50):
        registry.resolve_api_key(api_key)
    (after,) = registry.list_tenant_keys(tenant_id)
    assert after["last_used_at"] == first["last_used_at"]


def test_an_empty_or_unknown_key_resolves_to_none(redis: FakeRedis) -> None:
    registry = _peppered(redis)
    assert registry.resolve_api_key("") is None
    assert registry.resolve_api_key("sc-not-a-real-key") is None
    assert registry.resolve_cache_key("") is None
