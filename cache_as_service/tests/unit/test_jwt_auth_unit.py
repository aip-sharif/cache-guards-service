"""Pure unit tests for SSO JWT decoding — no Redis, no embedding model.

These exercise `semantic_cache.saas.jwt_auth` in isolation so they run in the
fast CI lane and pin the exact identity-resolution contract the router relies
on.
"""

import base64
import json

import pytest

from semantic_cache.saas.jwt_auth import decode_claims, external_identity


def _jwt(claims: dict, *, header: dict | None = None, sig: str = "sig") -> str:
    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    head = seg(header or {"alg": "none", "typ": "JWT"})
    return f"{head}.{seg(claims)}.{sig}"


# -- decode_claims ---------------------------------------------------------- #


def test_decode_claims_reads_payload() -> None:
    claims = decode_claims(_jwt({"organization": "acme", "user_id": "alice", "n": 3}))
    assert claims == {"organization": "acme", "user_id": "alice", "n": 3}


def test_decode_claims_ignores_signature() -> None:
    # Signature segment is never validated — any garbage there still decodes.
    assert decode_claims(_jwt({"user_id": "x"}, sig="not-a-real-signature")) == {
        "user_id": "x"
    }


def test_decode_claims_handles_missing_base64_padding() -> None:
    # A payload whose base64 length is not a multiple of 4 must still decode.
    claims = decode_claims(_jwt({"organization": "a", "user_id": "b"}))
    assert claims["organization"] == "a"


@pytest.mark.parametrize(
    "token",
    [
        "",
        "garbage",
        "only.two",
        "a.b.c.d",
        "header.%%%notbase64%%%.sig",
        _jwt(["not", "a", "dict"]),  # payload is a JSON array, not an object
    ],
)
def test_decode_claims_rejects_malformed(token: str) -> None:
    assert decode_claims(token) is None


# -- external_identity ------------------------------------------------------ #


def test_external_identity_combines_org_and_user() -> None:
    assert external_identity(_jwt({"organization": "acme", "user_id": "alice"})) == "acme|alice"


def test_external_identity_accepts_claim_aliases() -> None:
    assert external_identity(_jwt({"org": "acme", "sub": "u42"})) == "acme|u42"
    assert external_identity(_jwt({"organization_name": "beta", "uid": "9"})) == "beta|9"


def test_external_identity_coerces_numeric_user_id() -> None:
    assert external_identity(_jwt({"organization": "acme", "user_id": 12345})) == "acme|12345"


def test_external_identity_prefers_primary_over_alias() -> None:
    # 'organization' outranks 'org'; 'user_id' outranks 'sub'.
    tok = _jwt({"organization": "prim", "org": "alias", "user_id": "u", "sub": "s"})
    assert external_identity(tok) == "prim|u"


@pytest.mark.parametrize(
    "claims",
    [
        {"organization": "acme"},          # no user
        {"user_id": "alice"},              # no org
        {"foo": "bar"},                    # neither
        {"organization": "", "user_id": "x"},   # blank org
        {"organization": "x", "user_id": "  "},  # whitespace user
    ],
)
def test_external_identity_requires_both_halves(claims: dict) -> None:
    assert external_identity(_jwt(claims)) is None


def test_external_identity_none_on_malformed_token() -> None:
    assert external_identity("garbage") is None
    assert external_identity("") is None


def test_casdoor_token_uses_owner_and_name() -> None:
    """Real Casdoor/Casbin SSO token: org=`owner`, username=`name`."""
    tok = _jwt(
        {
            "owner": "organization_sharif",
            "name": "adminc2level",
            "sub": "79a1d03b-cf04-481b-956b-9ad62710978a",
            "email": "adminc2level@sharif.edu",
        }
    )
    assert external_identity(tok) == "organization_sharif|adminc2level"


def test_owner_outranks_generic_org_aliases() -> None:
    tok = _jwt({"owner": "org_a", "organization": "org_b", "name": "u"})
    assert external_identity(tok) == "org_a|u"


def test_name_outranks_sub() -> None:
    tok = _jwt({"owner": "o", "name": "alice", "sub": "uuid-123"})
    assert external_identity(tok) == "o|alice"


def test_identity_is_stable_across_calls() -> None:
    a = external_identity(_jwt({"organization": "acme", "user_id": "alice"}))
    b = external_identity(_jwt({"organization": "acme", "user_id": "alice"}))
    assert a == b == "acme|alice"
