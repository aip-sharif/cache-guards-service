"""Centralized secret redaction for anything on its way to a log or Sentry.

WHY IT IS CENTRAL AND NOT PER-CALL-SITE
---------------------------------------
Every call site that logs a secret was written by someone who did not think
they were logging a secret. `gateway/upstream.py` is the clearest case: it
logged the provider's raw error body, under a comment that itself noted
providers echo API keys back in error messages. Knowing the risk did not
prevent it, because the knowledge lived in a comment and the code did not act
on it. So redaction goes in ONE place, applied by the log formatter to every
record, and call sites are not asked to remember.

WHAT IT CATCHES
---------------
Bearer/Basic authorization values, `sk-`/`sc-`-style key literals, JSON and
query-string fields whose NAME says secret (api_key, token, password, secret,
authorization, dsn...), JWTs, and DSN passwords.

WHAT IT DOES NOT CATCH
----------------------
An arbitrary opaque credential with no marker in its name or shape. A denylist
of patterns is a mitigation, not a boundary — the boundary is not logging
untrusted bodies at all, which is why `upstream` now logs a redacted TRUNCATION
rather than the whole body. Treat this as defense in depth.
"""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"

#: Field names whose VALUE is a secret, wherever they appear — JSON, query
#: strings, `key=value` log text.
_SECRET_NAMES = (
    "api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|id[_-]?token"
    r"|token|secret|password|passwd|pwd|authorization|auth|credential"
    r"|client[_-]?secret|private[_-]?key|dsn|shared[_-]?secret"
    # The APP service credential, in every spelling it travels in: the
    # X-Service-Key header, the SC_APP_SERVICE_KEY env var, and the config
    # field. A secret masked under only one of its names is not masked.
    r"|x[_-]?service[_-]?key|service[_-]?key|pepper"
)

_PATTERNS = (
    # Authorization: Bearer xxx / Basic xxx  (header or log text)
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"), r"\1 " + MASK),
    # JSON: "api_key": "xxx"
    (
        re.compile(rf'(?i)("(?:{_SECRET_NAMES})"\s*:\s*)"[^"]*"'),
        r"\1" + f'"{MASK}"',
    ),
    # Query string / form / log text / env line: api_key=xxx
    #
    # The boundary is a negative lookbehind, NOT \b. `_` is a word character,
    # so \b does not match between "SC_APP_" and "SERVICE_KEY" — which meant
    # every SC_*_KEY env var slipped past this pattern and was then eaten by
    # the provider-literal rule below as if the NAME were the secret, masking
    # the name and printing the value. Exactly backwards.
    (
        re.compile(rf"(?i)(?<![A-Za-z0-9])({_SECRET_NAMES})\s*[=:]\s*[^\s,;&'\"}}\]]+"),
        r"\1=" + MASK,
    ),
    # DSN credentials: postgresql://user:pass@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^:/@\s]+:[^@\s]+@"), r"\1" + MASK + "@"),
    # JWTs, which carry identity claims even unsigned.
    (
        re.compile(r"\beyJ[A-Za-z0-9._\-]{10,}\.[A-Za-z0-9._\-]+\.[A-Za-z0-9._\-]*"),
        MASK,
    ),
    # Provider key literals: sk-..., sc-..., sk_live_..., ghp_...
    #
    # The lookahead excludes SCREAMING_SNAKE tails, because "SC_APP_SERVICE_KEY"
    # is an env var NAME that reads exactly like an "sc_" key literal. Without
    # it this rule masked the name and left the value in the log. Real keys
    # ("sc-proj-3f8a1c", "sk_live_9f8e") have lowercase or mixed tails, so this
    # costs nothing; a genuinely all-uppercase key is still caught by the
    # name=value rule above whenever it appears next to its field name.
    #
    # The lookahead is scoped CASE-SENSITIVE with (?-i:…). Under the pattern's
    # own (?i) flag, [A-Z_] also matches lowercase, so the exclusion swallowed
    # every real key as well — the fix has to be case-sensitive to mean
    # "SCREAMING_SNAKE" rather than "any letters".
    (
        re.compile(
            r"(?i)\b(sk|sc|pk|rk|ghp|gho|xox[abps])[-_]"
            r"(?-i:(?![A-Z_]+\b))[A-Za-z0-9._\-]{8,}"
        ),
        MASK,
    ),
)


def redact(value: Any) -> Any:
    """Masks anything that looks like a credential in `value`.

    Non-strings are returned unchanged — this is a text scrubber, and silently
    stringifying a dict here would change what the caller logs."""
    if not isinstance(value, str) or not value:
        return value
    out = value
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact_body(body: Any, limit: int = 500) -> str:
    """Redacts an upstream body AND truncates it.

    Truncation is not cosmetic. An upstream body is attacker-influenced content
    of unbounded length; logging it whole means a peer can choose how much of
    our log budget to spend and what ends up in our log store. 500 characters
    is enough to identify an error and not enough to smuggle a payload."""
    if isinstance(body, bytes):
        text = body.decode("utf-8", "replace")
    else:
        text = str(body)
    text = redact(text)
    if len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text)} chars]"
    return text


__all__ = ["MASK", "redact", "redact_body"]
