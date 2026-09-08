"""SSO JWT -> tenant identity, with SIGNATURE VERIFICATION.

The JWT is issued by a trusted upstream SSO (Casdoor in our deployment). A
tenant is the combination of *organization* + *user id*, so the same person in
the same organization always resolves to the same tenant — and therefore the
same set of caches.

WHY THIS MODULE IS PARANOID
---------------------------
This module used to base64-decode the payload and trust it. Because
``saas_router._tenant_from_token`` feeds the resulting identity straight into
``get_or_create_tenant_by_external``, anyone who could type three dots could
mint a payload for another organization — or auto-provision a brand new tenant
— with no signature at all. That is the whole authorization model of the SaaS
API, so verification is not optional here.

FAIL CLOSED
-----------
``JwtVerifier`` with no key material verifies NOTHING and accepts NOTHING: an
unconfigured deployment loses the SSO path entirely and keeps only the minted
``sc-...`` API keys. That is deliberate. A verifier that degrades to "trust the
payload" when its config is missing is the bug this module exists to remove,
and a misconfigured deployment must lock people out rather than let everyone
in.

TWO SIGNING FAMILIES
--------------------
HS256/384/512  Shared secret. Verified HERE with stdlib hmac — no dependency,
               so the unit suite exercises the real verification path on a bare
               checkout.
RS/ES/PS/Ed*   Public key from the SSO's JWKS. Delegated to PyJWT
               (``pip install '.[sso]'``), because hand-rolling RSA/ECDSA
               verification is exactly the wrong kind of clever.

The algorithm allowlist is enforced BEFORE any key is chosen, so a token cannot
talk us into verifying an RS256-configured deployment with an HS256 MAC over
the public key (the classic alg-confusion attack), and ``alg: none`` is never
reachable.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Claim names we accept for each half of the identity, in priority order.
# The identity is f"{org}|{user}"; both must be present.
#
# `owner` / `name` are Casdoor's fields (owner = organization, name = username)
# and are the primary picks for our SSO; the rest are generic fallbacks so the
# same code also handles other providers.
_ORG_CLAIMS = ("owner", "organization", "organization_name", "org_name", "org", "tenant")
_USER_CLAIMS = ("name", "user_id", "userId", "uid", "user", "sub")

#: HMAC algorithms we verify in-process, and the digest each one names.
_HMAC_ALGS = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

#: Asymmetric families we accept, delegated to PyJWT. "none" is absent on
#: purpose and there is no branch that could add it.
_ASYMMETRIC_PREFIXES = ("RS", "ES", "PS", "Ed")

DEFAULT_ALGORITHMS = ("RS256",)


class JwtConfigError(Exception):
    """The verifier was asked for something its configuration cannot do.

    Raised at CONSTRUCTION, never per request: a deployment that names RS256
    without a JWKS URL is broken, and it should fail to boot rather than
    silently reject every user at 3am."""


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _is_asymmetric(alg: str) -> bool:
    return alg.startswith(_ASYMMETRIC_PREFIXES)


def _split(token: str) -> Optional[tuple]:
    """``(header, payload, signature, signing_input)``, or None.

    None means "not a well-formed JWT" — never "trust it anyway"."""
    if not token or token.count(".") != 2:
        return None
    header_b64, payload_b64, signature_b64 = token.split(".", 2)
    if not header_b64 or not payload_b64:
        return None
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(signature_b64)
    except (ValueError, binascii.Error, json.JSONDecodeError):
        return None
    if not isinstance(header, dict) or not isinstance(payload, dict):
        return None
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    return header, payload, signature, signing_input


class JwtVerifier:
    """Verifies SSO JWTs and returns their claims. Fails closed."""

    def __init__(
        self,
        *,
        shared_secret: Optional[str] = None,
        jwks_url: Optional[str] = None,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        algorithms: Optional[Sequence[str]] = None,
        leeway: float = 60.0,
        jwks_ttl: float = 300.0,
    ) -> None:
        algorithms = tuple(algorithms or DEFAULT_ALGORITHMS)
        unknown = [
            a for a in algorithms if a not in _HMAC_ALGS and not _is_asymmetric(a)
        ]
        if unknown:
            raise JwtConfigError(
                f"Unsupported SSO JWT algorithm(s): {unknown}. "
                f"Supported: {sorted(_HMAC_ALGS)} and RS*/ES*/PS*/Ed* via JWKS."
            )

        self._algorithms = algorithms
        self._shared_secret = shared_secret.encode("utf-8") if shared_secret else None
        self._jwks_url = jwks_url or None
        self._issuer = issuer or None
        self._audience = audience or None
        self._leeway = float(leeway)
        self._jwks_ttl = float(jwks_ttl)
        self._jwk_client: Any = None

        # With NO key material at all the verifier is simply disabled and the
        # checks below do not apply — a deployment that never turned SSO on is
        # not misconfigured. Once any key IS present, an algorithm it has no
        # key for accepts nothing, so say so at boot, not one login at a time.
        if not self.configured:
            return
        if any(a in _HMAC_ALGS for a in algorithms) and self._shared_secret is None:
            raise JwtConfigError(
                "SC_SSO_ALGORITHMS names an HS* algorithm but "
                "SC_SSO_SHARED_SECRET is unset."
            )
        if any(_is_asymmetric(a) for a in algorithms) and self._jwks_url is None:
            raise JwtConfigError(
                "SC_SSO_ALGORITHMS names an asymmetric algorithm but "
                "SC_SSO_JWKS_URL is unset."
            )

    @property
    def configured(self) -> bool:
        """False -> this verifier rejects every token.

        ``create_app`` logs a warning on False so an operator who expected SSO
        to work finds out at boot instead of from a user."""
        return bool(self._shared_secret or self._jwks_url)

    def verify(self, token: str) -> Optional[Dict[str, Any]]:
        """Verified claims, or None if the token is not trustworthy.

        Every rejection returns None; the reason is logged at debug, never
        returned, because a caller learning *why* their forgery failed is a
        free oracle."""
        if not self.configured:
            return None
        parts = _split(token)
        if parts is None:
            return None
        header, payload, signature, signing_input = parts

        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in self._algorithms:
            logger.debug("SSO JWT rejected: alg %r not allowed.", alg)
            return None

        if alg in _HMAC_ALGS:
            if not self._verify_hmac(alg, signature, signing_input):
                logger.debug("SSO JWT rejected: bad HMAC signature.")
                return None
            if not self._validate_claims(payload):
                return None
            return payload

        return self._verify_asymmetric(token, alg)

    # -- HS*: stdlib ------------------------------------------------------- #

    def _verify_hmac(self, alg: str, signature: bytes, signing_input: bytes) -> bool:
        assert self._shared_secret is not None  # guaranteed by __init__
        expected = hmac.new(
            self._shared_secret, signing_input, _HMAC_ALGS[alg]
        ).digest()
        return hmac.compare_digest(expected, signature)

    def _validate_claims(self, payload: Dict[str, Any]) -> bool:
        """exp / nbf / iss / aud. Applies to the HS path only — PyJWT does its
        own on the asymmetric one, and two implementations of one rule is how
        they drift apart."""
        now = time.time()

        exp = payload.get("exp")
        if not isinstance(exp, (int, float)) or isinstance(exp, bool):
            # An SSO token with no expiry is a bearer credential that never
            # dies. Refuse it rather than inherit it.
            logger.debug("SSO JWT rejected: missing or non-numeric exp.")
            return False
        if now > float(exp) + self._leeway:
            logger.debug("SSO JWT rejected: expired.")
            return False

        nbf = payload.get("nbf")
        if isinstance(nbf, (int, float)) and not isinstance(nbf, bool):
            if now < float(nbf) - self._leeway:
                logger.debug("SSO JWT rejected: not yet valid.")
                return False

        if self._issuer is not None and payload.get("iss") != self._issuer:
            logger.debug("SSO JWT rejected: issuer mismatch.")
            return False

        if self._audience is not None and not _audience_matches(
            payload.get("aud"), self._audience
        ):
            logger.debug("SSO JWT rejected: audience mismatch.")
            return False

        return True

    # -- RS/ES/PS/Ed*: PyJWT ----------------------------------------------- #

    def _verify_asymmetric(self, token: str, alg: str) -> Optional[Dict[str, Any]]:
        try:
            import jwt as pyjwt  # noqa: PLC0415 — optional [sso] extra
        except ImportError:
            # Not a silent reject: an operator who configured a JWKS URL and
            # forgot the extra would otherwise see "invalid credentials" with
            # nothing in the log to explain it.
            logger.error(
                "SSO JWT uses %s but PyJWT is not installed. "
                "Install the [sso] extra to enable JWKS verification.",
                alg,
            )
            return None

        try:
            key = self._signing_key(pyjwt, token)
            return pyjwt.decode(
                token,
                key,
                algorithms=[a for a in self._algorithms if _is_asymmetric(a)],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={
                    "require": ["exp"],
                    "verify_aud": self._audience is not None,
                    "verify_iss": self._issuer is not None,
                },
            )
        except Exception as e:  # noqa: BLE001 — every failure is one rejection
            logger.debug("SSO JWT rejected: %s: %s", type(e).__name__, e)
            return None

    def _signing_key(self, pyjwt: Any, token: str) -> Any:
        if self._jwk_client is None:
            # PyJWKClient fetches over the network with urllib and caches.
            # Safe to block: FastAPI runs `def` dependencies in a threadpool,
            # so this never sits on the event loop.
            self._jwk_client = pyjwt.PyJWKClient(
                self._jwks_url, cache_keys=True, lifespan=self._jwks_ttl
            )
        return self._jwk_client.get_signing_key_from_jwt(token).key


def _audience_matches(claim: Any, expected: str) -> bool:
    if isinstance(claim, str):
        return claim == expected
    if isinstance(claim, (list, tuple)):
        return expected in claim
    return False


def _first_claim(claims: Dict[str, Any], names: Iterable[str]) -> Optional[str]:
    for n in names:
        v = claims.get(n)
        if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip():
            return str(v).strip()
    return None


def external_identity(claims: Dict[str, Any]) -> Optional[str]:
    """The stable external id ``"{org}|{user}"`` from ALREADY-VERIFIED claims.

    Takes a claims dict, not a token, so there is no way to reach this function
    without having gone through ``JwtVerifier.verify`` first."""
    if not isinstance(claims, dict):
        return None
    org = _first_claim(claims, _ORG_CLAIMS)
    user = _first_claim(claims, _USER_CLAIMS)
    if org is None or user is None:
        return None
    return f"{org}|{user}"


__all__ = [
    "DEFAULT_ALGORITHMS",
    "JwtConfigError",
    "JwtVerifier",
    "external_identity",
]
