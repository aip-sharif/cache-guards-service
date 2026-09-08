"""Authorization tests for the multi-tenant SaaS router, over real HTTP.

`test_jwt_auth_unit` proves the verifier rejects forgeries. This file proves
the ROUTER actually consults it — the gap that let an unsigned payload
auto-provision a tenant was never in the crypto, it was in the wiring, so the
regression has to be pinned at the surface a caller can reach.

Everything is faked at the seams (a dict-backed registry, no manager pool), so
this runs with no Redis and no network.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.adapters.saas_router import (
    get_admin_key,
    get_jwt_verifier,
    get_manager_pool,
    get_redis,
    get_registry,
    health_router,
    saas_router,
)
from semantic_cache.saas.jwt_auth import JwtVerifier

SECRET = "shared-secret-for-tests"
TENANT_KEY = "sc-tenant-minted-key"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeRegistry:
    """Just enough SaaSRegistry to answer the auth dependencies.

    `provisioned` is the point of the whole file: the JWT branch CREATES
    tenants, so a forged token that reaches it is not merely authenticated as
    someone else — it manufactures a new one."""

    def __init__(self) -> None:
        self.provisioned: list = []
        self.revoked: list = []
        self.grace: list = []
        self.expires_in: list = []
        self.caches: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def resolve_api_key(self, token: str) -> Optional[str]:
        return "tenant-minted" if token == TENANT_KEY else None

    def resolve_cache_key(self, token: str) -> Optional[Tuple[str, str]]:
        return None

    def get_or_create_tenant_by_external(self, external_id: str) -> str:
        self.provisioned.append(external_id)
        return hashlib.sha256(external_id.encode()).hexdigest()

    def list_caches(self, tenant_id: str) -> list:
        return []

    def get_cache(self, tenant_id: str, cache_id: str) -> Optional[Dict[str, Any]]:
        return self.caches.get((tenant_id, cache_id))

    # -- key lifecycle -----------------------------------------------------
    def list_tenant_keys(self, tenant_id: str) -> list:
        return [
            {"key_id": "k1", "prefix": "sc-aaaaaaaa", "created_at": "1",
             "expires_at": None, "last_used_at": "2"}
        ]

    def revoke_key(self, tenant_id: str, key_id: str) -> bool:
        self.revoked.append((tenant_id, key_id))
        return key_id == "k1"

    def list_cache_keys(self, tenant_id: str, cache_id: str) -> list:
        return [
            {"key_id": "c1", "prefix": "sc-bbbbbbbb", "created_at": "1",
             "expires_at": None, "last_used_at": None}
        ]

    def revoke_cache_key(self, tenant_id: str, cache_id: str, key_id: str) -> bool:
        self.revoked.append((tenant_id, cache_id, key_id))
        return key_id == "c1"

    def rotate_tenant_key(self, tenant_id: str, grace: int = 0) -> str:
        self.grace.append(grace)
        return "sc-rotated"

    def rotate_cache_key(self, tenant_id: str, cache_id: str, grace: int = 0) -> str:
        self.grace.append(grace)
        return "sc-rotated-cache"

    def issue_cache_key(self, tenant_id, cache_id, expires_in=None) -> str:
        self.expires_in.append(expires_in)
        return "sc-issued"


def _app(verifier: JwtVerifier) -> Tuple[TestClient, FakeRegistry]:
    registry = FakeRegistry()
    app = FastAPI()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_manager_pool] = lambda: None
    app.dependency_overrides[get_admin_key] = lambda: None
    app.dependency_overrides[get_redis] = lambda: None
    app.dependency_overrides[get_jwt_verifier] = lambda: verifier
    app.include_router(saas_router)
    app.include_router(health_router)
    return TestClient(app), registry


def _seg(obj: dict) -> str:
    raw = json.dumps(obj).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _signed(claims: dict, *, secret: str = SECRET) -> str:
    payload = {"exp": time.time() + 3600, **claims}
    signing_input = f'{_seg({"alg": "HS256", "typ": "JWT"})}.{_seg(payload)}'
    sig = hmac.new(secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f'{signing_input}.{base64.urlsafe_b64encode(sig).rstrip(b"=").decode()}'


def _unsigned(claims: dict) -> str:
    payload = {"exp": time.time() + 3600, **claims}
    return f'{_seg({"alg": "none", "typ": "JWT"})}.{_seg(payload)}.sig'


def _hs256() -> JwtVerifier:
    return JwtVerifier(shared_secret=SECRET, algorithms=["HS256"])


def _auth(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# The regression
# --------------------------------------------------------------------------- #


def test_an_unsigned_jwt_cannot_provision_a_tenant() -> None:
    """The reported hole: three dots and a JSON payload used to be a tenant."""
    client, registry = _app(_hs256())
    r = client.get("/v1/caches", headers=_auth(_unsigned({"owner": "victim", "name": "root"})))
    assert r.status_code == 401
    assert registry.provisioned == []


def test_a_forged_signature_cannot_provision_a_tenant() -> None:
    client, registry = _app(_hs256())
    r = client.get(
        "/v1/caches",
        headers=_auth(_signed({"owner": "victim", "name": "root"}, secret="guess")),
    )
    assert r.status_code == 401
    assert registry.provisioned == []


def test_an_expired_jwt_is_refused_at_the_route() -> None:
    client, registry = _app(_hs256())
    r = client.get(
        "/v1/caches",
        headers=_auth(_signed({"owner": "acme", "name": "alice", "exp": time.time() - 60})),
    )
    assert r.status_code == 401
    assert registry.provisioned == []


def test_a_verified_jwt_still_works() -> None:
    """The fix must not close the door on the legitimate SSO path."""
    client, registry = _app(_hs256())
    r = client.get("/v1/caches", headers=_auth(_signed({"owner": "acme", "name": "alice"})))
    assert r.status_code == 200, r.text
    assert registry.provisioned == ["acme|alice"]


def test_a_deployment_without_sso_config_rejects_every_jwt() -> None:
    """Fail closed: no SC_SSO_* set → the JWT path is shut, not wide open."""
    client, registry = _app(JwtVerifier())
    r = client.get("/v1/caches", headers=_auth(_signed({"owner": "acme", "name": "alice"})))
    assert r.status_code == 401
    assert registry.provisioned == []


def test_a_minted_tenant_key_is_unaffected_by_sso_config() -> None:
    """The sc-... path must not depend on SSO being configured either way."""
    for verifier in (_hs256(), JwtVerifier()):
        client, _ = _app(verifier)
        assert client.get("/v1/caches", headers=_auth(TENANT_KEY)).status_code == 200


@pytest.mark.parametrize("header", [{}, {"Authorization": "Bearer"}, {"Authorization": "Basic x"}])
def test_a_missing_or_non_bearer_credential_is_401(header: dict) -> None:
    client, _ = _app(_hs256())
    assert client.get("/v1/caches", headers=header).status_code == 401


# --------------------------------------------------------------------------- #
# Key lifecycle over HTTP
#
# The registry gained per-key listing, expiry and selective revocation; these
# pin that the API actually exposes them, because a capability with no route is
# a capability nobody has.
# --------------------------------------------------------------------------- #


def _with_cache() -> Tuple[TestClient, FakeRegistry]:
    client, registry = _app(_hs256())
    registry.caches[("tenant-minted", "c1")] = {
        "cache_id": "c1", "name": "prod", "config": {}, "created_at": "1"
    }
    return client, registry


def test_listing_keys_returns_no_key_material() -> None:
    """A fingerprint verifies a key. Putting one in a response moves a
    credential-checking value somewhere it is not guarded like one."""
    client, _ = _app(_hs256())
    r = client.get("/v1/tenant/keys", headers=_auth(TENANT_KEY))
    assert r.status_code == 200
    (row,) = r.json()
    assert row["key_id"] == "k1"
    assert row["prefix"] == "sc-aaaaaaaa"
    assert "api_key" not in row and "fingerprint" not in row


def test_one_tenant_key_can_be_revoked_by_id() -> None:
    client, registry = _app(_hs256())
    r = client.delete("/v1/tenant/keys/k1", headers=_auth(TENANT_KEY))
    assert r.status_code == 200
    assert r.json() == {"revoked": True}
    assert registry.revoked == [("tenant-minted", "k1")]


def test_revoking_an_unknown_key_is_404() -> None:
    client, _ = _app(_hs256())
    r = client.delete("/v1/tenant/keys/nope", headers=_auth(TENANT_KEY))
    assert r.status_code == 404


def test_rotation_passes_the_grace_period_through() -> None:
    """Rotation with no grace revokes instantly, which is why nobody rotates."""
    client, registry = _app(_hs256())
    client.post("/v1/tenant/rotate-key?grace=3600", headers=_auth(TENANT_KEY))
    assert registry.grace == [3600]


def test_rotation_defaults_to_immediate() -> None:
    client, registry = _app(_hs256())
    client.post("/v1/tenant/rotate-key", headers=_auth(TENANT_KEY))
    assert registry.grace == [0]


def test_a_cache_key_can_be_issued_with_an_expiry() -> None:
    client, registry = _with_cache()
    r = client.post("/v1/caches/c1/keys?expires_in=86400", headers=_auth(TENANT_KEY))
    assert r.status_code == 201
    assert registry.expires_in == [86400]


def test_a_cache_key_without_an_expiry_never_expires() -> None:
    """The historical behaviour, and why keys accumulate. Kept as the default
    so existing callers are not broken by the change."""
    client, registry = _with_cache()
    client.post("/v1/caches/c1/keys", headers=_auth(TENANT_KEY))
    assert registry.expires_in == [None]


def test_one_cache_key_can_be_revoked_by_id() -> None:
    client, registry = _with_cache()
    r = client.delete("/v1/caches/c1/keys/c1", headers=_auth(TENANT_KEY))
    assert r.status_code == 200
    assert registry.revoked == [("tenant-minted", "c1", "c1")]


def test_key_management_needs_the_tenant_key_not_a_forged_jwt() -> None:
    """The whole file's premise: these are tenant-scoped management routes."""
    client, registry = _app(_hs256())
    forged = _auth(_unsigned({"owner": "victim", "name": "root"}))
    assert client.get("/v1/tenant/keys", headers=forged).status_code == 401
    assert client.delete("/v1/tenant/keys/k1", headers=forged).status_code == 401
    assert registry.revoked == []


def test_key_routes_for_another_tenants_cache_are_404() -> None:
    """Ownership: the registry lookup embeds tenant_id, so another tenant's
    cache simply does not exist under this one."""
    client, _ = _app(_hs256())
    assert client.get(
        "/v1/caches/not-mine/keys", headers=_auth(TENANT_KEY)
    ).status_code == 404
