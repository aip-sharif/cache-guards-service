"""SSO JWT → tenant identity.

The JWT is issued by a trusted upstream SSO. We do NOT verify the signature
here (no strategy / no shared secret in scope) — we decode the payload and read
its claims. The tenant is the combination of *organization name* + *user id*,
so the same person in the same organization always resolves to the same tenant
(and therefore the same set of caches).

To harden later, verify the signature in `decode_claims` (JWKS or shared secret)
before trusting the claims — nothing downstream changes.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

# Claim names we accept for each half of the identity, in priority order.
# The identity is f"{org}|{user}"; both must be present.
#
# `owner` / `name` are Casdoor's fields (owner = organization, name = username)
# and are the primary picks for our SSO; the rest are generic fallbacks so the
# same code also handles other providers.
_ORG_CLAIMS = ("owner", "organization", "organization_name", "org_name", "org", "tenant")
_USER_CLAIMS = ("name", "user_id", "userId", "uid", "user", "sub")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def decode_claims(token: str) -> Optional[Dict[str, Any]]:
    """Decodes a JWT payload WITHOUT verifying the signature.

    Returns the claims dict, or None if the token is not a well-formed JWT."""
    if not token or token.count(".") != 2:
        return None
    try:
        payload = _b64url_decode(token.split(".", 2)[1])
        claims = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        return None
    return claims if isinstance(claims, dict) else None


def _first_claim(claims: Dict[str, Any], names: tuple) -> Optional[str]:
    for n in names:
        v = claims.get(n)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return None


def external_identity(token: str) -> Optional[str]:
    """Returns the stable external id ``"{org}|{user}"`` from a JWT, or None
    if the token is malformed or missing either identity half."""
    claims = decode_claims(token)
    if claims is None:
        return None
    org = _first_claim(claims, _ORG_CLAIMS)
    user = _first_claim(claims, _USER_CLAIMS)
    if org is None or user is None:
        return None
    return f"{org}|{user}"
