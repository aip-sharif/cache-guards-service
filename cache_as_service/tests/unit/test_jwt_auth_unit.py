"""Pure unit tests for SSO JWT verification — no Redis, no embedding model.

These exercise `semantic_cache.saas.jwt_auth` in isolation so they run in the
fast CI lane and pin two contracts the router relies on: the exact identity
resolution, and — since a verified JWT AUTO-PROVISIONS a tenant — that nothing
unverified ever reaches it.

Everything here signs with HS256, which the module verifies in-process with
stdlib hmac. That is deliberate: the real verification path runs on a bare
checkout, with no PyJWT and no key fixtures to drift out of date.
"""

import base64
import hashlib
import hmac
import json
import time

import pytest

from semantic_cache.saas.jwt_auth import (
    JwtConfigError,
    JwtVerifier,
    external_identity,
)

SECRET = "correct-horse-battery-staple"


def _seg(obj: dict) -> str:
    raw = json.dumps(obj).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _signed(claims: dict, *, secret: str = SECRET, alg: str = "HS256") -> str:
    """A genuinely signed HS* token, with an hour of life unless overridden."""
    payload = {"exp": time.time() + 3600, **claims}
    signing_input = f'{_seg({"alg": alg, "typ": "JWT"})}.{_seg(payload)}'
    digest = {"HS256": hashlib.sha256, "HS384": hashlib.sha384,
              "HS512": hashlib.sha512}[alg]
    sig = hmac.new(secret.encode(), signing_input.encode("ascii"), digest).digest()
    return f'{signing_input}.{base64.urlsafe_b64encode(sig).rstrip(b"=").decode()}'


def _unsigned(claims: dict, *, alg: str = "none", sig: str = "sig") -> str:
    """The forgery the old module accepted: a payload and a made-up signature."""
    payload = {"exp": time.time() + 3600, **claims}
    return f'{_seg({"alg": alg, "typ": "JWT"})}.{_seg(payload)}.{sig}'


def _verifier(**kw) -> JwtVerifier:
    kw.setdefault("shared_secret", SECRET)
    kw.setdefault("algorithms", ["HS256"])
    return JwtVerifier(**kw)


# -- the vulnerability this module exists to close -------------------------- #


def test_an_unsigned_token_is_rejected() -> None:
    """The old `decode_claims` returned these claims and the router turned them
    into a tenant. alg=none must be unreachable."""
    assert _verifier().verify(_unsigned({"owner": "acme", "name": "alice"})) is None


def test_a_forged_signature_is_rejected() -> None:
    assert _verifier().verify(
        _signed({"owner": "acme", "name": "alice"}, secret="attacker-guess")
    ) is None


def test_a_tampered_payload_is_rejected() -> None:
    """Take a real token for one org, swap the payload for another. The
    signature no longer covers it."""
    good = _signed({"owner": "acme", "name": "alice"})
    head, _, sig = good.split(".")
    forged = f'{head}.{_seg({"owner": "victim-corp", "name": "root", "exp": time.time() + 3600})}.{sig}'
    assert _verifier().verify(forged) is None


def test_an_unconfigured_verifier_accepts_nothing() -> None:
    """No key material → the SSO path is closed, not open. This is the state a
    deployment that never set SC_SSO_* boots into."""
    bare = JwtVerifier()
    assert not bare.configured
    assert bare.verify(_signed({"owner": "acme", "name": "alice"})) is None


def test_alg_confusion_is_rejected() -> None:
    """An RS256 deployment must not verify an HS256 MAC, even one computed with
    a secret the attacker knows."""
    rsa_only = JwtVerifier(jwks_url="https://sso.example/jwks", algorithms=["RS256"])
    assert rsa_only.verify(_signed({"owner": "acme", "name": "alice"})) is None


def test_an_algorithm_outside_the_allowlist_is_rejected() -> None:
    hs256_only = _verifier(algorithms=["HS256"])
    assert hs256_only.verify(
        _signed({"owner": "acme", "name": "alice"}, alg="HS512")
    ) is None


# -- claim validation ------------------------------------------------------- #


def test_an_expired_token_is_rejected() -> None:
    assert _verifier().verify(
        _signed({"owner": "acme", "name": "alice", "exp": time.time() - 3600})
    ) is None


def test_a_token_with_no_expiry_is_rejected() -> None:
    """A bearer credential that never dies is not one we accept."""
    unsigned_forever = _signed({"owner": "acme", "name": "alice", "exp": None})
    assert _verifier().verify(unsigned_forever) is None


def test_expiry_inside_the_leeway_is_accepted() -> None:
    tok = _signed({"owner": "acme", "name": "alice", "exp": time.time() - 5})
    assert _verifier(leeway=60).verify(tok) is not None
    assert _verifier(leeway=0).verify(tok) is None


def test_a_not_yet_valid_token_is_rejected() -> None:
    assert _verifier(leeway=0).verify(
        _signed({"owner": "acme", "name": "alice", "nbf": time.time() + 3600})
    ) is None


def test_the_issuer_is_enforced_when_configured() -> None:
    v = _verifier(issuer="https://sso.example")
    assert v.verify(_signed({"owner": "a", "name": "b", "iss": "https://evil"})) is None
    assert v.verify(
        _signed({"owner": "a", "name": "b", "iss": "https://sso.example"})
    ) is not None


def test_the_audience_is_enforced_when_configured() -> None:
    """A token the same SSO minted for a DIFFERENT service must not work here."""
    v = _verifier(audience="semantic-cache")
    assert v.verify(_signed({"owner": "a", "name": "b", "aud": "other-app"})) is None
    assert v.verify(_signed({"owner": "a", "name": "b"})) is None
    assert v.verify(
        _signed({"owner": "a", "name": "b", "aud": ["other-app", "semantic-cache"]})
    ) is not None


# -- malformed input -------------------------------------------------------- #


@pytest.mark.parametrize(
    "token",
    ["", "garbage", "a.b", "a.b.c.d", "..", ".payload.sig", "not!base64.x.y"],
)
def test_malformed_tokens_are_rejected(token: str) -> None:
    assert _verifier().verify(token) is None


# -- configuration errors are fatal, not silent ----------------------------- #


def test_an_hmac_algorithm_without_a_secret_refuses_to_construct() -> None:
    with pytest.raises(JwtConfigError):
        JwtVerifier(jwks_url="https://sso.example/jwks", algorithms=["HS256"])


def test_an_asymmetric_algorithm_without_a_jwks_url_refuses_to_construct() -> None:
    with pytest.raises(JwtConfigError):
        JwtVerifier(shared_secret=SECRET, algorithms=["RS256"])


def test_an_unknown_algorithm_refuses_to_construct() -> None:
    with pytest.raises(JwtConfigError):
        JwtVerifier(shared_secret=SECRET, algorithms=["none"])


def test_no_key_material_is_disabled_not_misconfigured() -> None:
    """A deployment that never turned SSO on is not broken — it is off."""
    assert JwtVerifier(algorithms=["RS256"]).configured is False


# -- identity resolution (unchanged contract, now on verified claims) ------- #


def test_external_identity_combines_org_and_user() -> None:
    assert external_identity({"organization": "acme", "user_id": "alice"}) == "acme|alice"


def test_external_identity_accepts_claim_aliases() -> None:
    assert external_identity({"org": "acme", "sub": "u42"}) == "acme|u42"
    assert external_identity({"organization_name": "beta", "uid": "9"}) == "beta|9"


def test_external_identity_coerces_numeric_user_id() -> None:
    assert external_identity({"organization": "acme", "user_id": 12345}) == "acme|12345"


def test_casdoor_claims_use_owner_and_name() -> None:
    """Real Casdoor/Casbin SSO token: org=`owner`, username=`name`."""
    assert external_identity(
        {
            "owner": "organization_sharif",
            "name": "adminc2level",
            "sub": "79a1d03b-cf04-481b-956b-9ad62710978a",
            "email": "adminc2level@sharif.edu",
        }
    ) == "organization_sharif|adminc2level"


def test_owner_outranks_generic_org_aliases() -> None:
    assert external_identity(
        {"owner": "org_a", "organization": "org_b", "name": "u"}
    ) == "org_a|u"


def test_name_outranks_sub() -> None:
    assert external_identity({"owner": "o", "name": "alice", "sub": "uuid-123"}) == "o|alice"


@pytest.mark.parametrize(
    "claims",
    [
        {"organization": "acme"},
        {"user_id": "alice"},
        {},
        {"organization": "", "user_id": "alice"},
        {"organization": "acme", "user_id": "   "},
    ],
)
def test_external_identity_requires_both_halves(claims: dict) -> None:
    assert external_identity(claims) is None


def test_external_identity_rejects_a_non_dict() -> None:
    assert external_identity("acme|alice") is None  # type: ignore[arg-type]


# -- the whole path, end to end --------------------------------------------- #


def test_a_verified_token_yields_the_identity() -> None:
    claims = _verifier().verify(_signed({"owner": "acme", "name": "alice"}))
    assert claims is not None
    assert external_identity(claims) == "acme|alice"


def test_the_identity_is_stable_across_calls() -> None:
    v = _verifier()
    a = external_identity(v.verify(_signed({"organization": "acme", "user_id": "alice"})))
    b = external_identity(v.verify(_signed({"organization": "acme", "user_id": "alice"})))
    assert a == b == "acme|alice"
