"""Secret redaction on the path to the logs.

The concrete incident these pin: `gateway/upstream.py` logged the provider's
raw error body under a comment noting that providers echo API keys in it. So
the tests here are less about the regexes than about the two guarantees —
redaction happens in the FORMATTER (no call site has to remember) and bodies
are truncated as well as scrubbed.
"""

import json
import logging

import pytest

from semantic_cache.observability import JsonLogFormatter
from semantic_cache.redaction import MASK, redact, redact_body


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer sk-abc123def456ghi789",
        "authorization: basic YWxhZGRpbjpvcGVuc2VzYW1l",
        '{"error": {"message": "invalid key", "api_key": "sk-live-9f8e7d6c5b4a"}}',
        "api_key=sk-live-9f8e7d6c5b4a&model=gpt",
        "connecting to postgresql://scuser:hunter2@db:5432/sccache",
        "token: eyJhbGciOiJIUzI1NiJ9.eyJvd25lciI6ImFjbWUifQ.c2ln",
        "your key sk-proj-AAAAAAAAAAAAAAAA was rejected",
        '{"password": "hunter2"}',
        "client_secret=abcdefgh12345678",
    ],
)
def test_credentials_do_not_survive_redaction(text: str) -> None:
    out = redact(text)
    assert MASK in out
    for leak in (
        "sk-abc123def456ghi789", "YWxhZGRpbjpvcGVuc2VzYW1l",
        "sk-live-9f8e7d6c5b4a", "hunter2", "sk-proj-AAAAAAAAAAAAAAAA",
        "abcdefgh12345678", "eyJvd25lciI6ImFjbWUifQ",
    ):
        assert leak not in out, f"{leak!r} survived in {out!r}"


def test_redaction_keeps_the_diagnosable_part() -> None:
    """A scrubber that eats the error message trades one blindness for
    another."""
    out = redact('{"error": {"message": "model not found", "api_key": "sk-xyz12345"}}')
    assert "model not found" in out
    assert "sk-xyz12345" not in out


def test_ordinary_text_is_untouched() -> None:
    text = "Upstream error 502: gateway timeout after 30s"
    assert redact(text) == text


def test_non_strings_pass_through_unchanged() -> None:
    """This is a text scrubber; stringifying here would change what the caller
    asked to log."""
    for value in (None, 42, 3.5, True, ["a"], {"b": 1}):
        assert redact(value) == value


# -- bodies are truncated as well as scrubbed ------------------------------- #


def test_a_body_is_truncated() -> None:
    """An upstream body is attacker-influenced content of unbounded length —
    otherwise a peer picks how much of our log budget to spend."""
    out = redact_body("x" * 5000, limit=100)
    assert len(out) < 200
    assert "truncated" in out


def test_a_short_body_is_not_annotated() -> None:
    assert redact_body("boom") == "boom"


def test_bytes_are_decoded_not_repr_ed() -> None:
    assert redact_body(b'{"error": "nope"}') == '{"error": "nope"}'


def test_undecodable_bytes_do_not_raise() -> None:
    assert isinstance(redact_body(b"\xff\xfe binary"), str)


# -- the formatter is the guarantee ----------------------------------------- #


def _format(msg: str, **extra) -> dict:
    record = logging.LogRecord(
        "test", logging.ERROR, __file__, 1, msg, (), None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return json.loads(JsonLogFormatter().format(record))


def test_the_formatter_redacts_the_message() -> None:
    """No call site has to remember — that is the whole point."""
    out = _format("Upstream error 401: {'api_key': 'sk-live-abcdefgh'}")
    assert "sk-live-abcdefgh" not in json.dumps(out)


def test_the_formatter_redacts_extra_fields() -> None:
    out = _format("request failed", upstream_body="api_key=sk-live-abcdefgh")
    assert "sk-live-abcdefgh" not in json.dumps(out)
    assert MASK in out["upstream_body"]


def test_the_formatter_redacts_the_traceback() -> None:
    """Exception text carries whatever was interpolated into it."""
    try:
        raise RuntimeError("auth failed for Bearer sk-live-abcdefgh")
    except RuntimeError:
        import sys
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "boom", (), sys.exc_info()
        )
        out = json.loads(JsonLogFormatter().format(record))
    assert "sk-live-abcdefgh" not in json.dumps(out)


def test_the_formatter_still_emits_one_json_object() -> None:
    out = _format("plain message", request_id="abc123")
    assert out["message"] == "plain message"
    assert out["request_id"] == "abc123"
    assert out["level"] == "ERROR"


# --------------------------------------------------------------------------- #
# The SC_* env-var bug
#
# `_` is a word character, so \b never matched between "SC_APP_" and
# "SERVICE_KEY". Every SC_*_KEY line slipped past the name=value rule and was
# then eaten by the provider-literal rule as though the NAME were the secret —
# masking the name and printing the value. Exactly backwards, and it would have
# put real keys in the log.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line, secret",
    [
        ("SC_APP_SERVICE_KEY=svc-abcdef123456", "svc-abcdef123456"),
        ("SC_API_KEY_PEPPER=pep-abcdef123456", "pep-abcdef123456"),
        ("SC_ADMIN_API_KEY=sc-realadminkey123", "sc-realadminkey123"),
        ("SC_SSO_SHARED_SECRET=shh-abcdef123456", "shh-abcdef123456"),
        ("REDIS_PASSWORD=hunter2isthepassword", "hunter2isthepassword"),
        ("POSTGRES_PASSWORD=hunter2isthepassword", "hunter2isthepassword"),
    ],
)
def test_an_env_var_line_masks_the_value_not_the_name(line: str, secret: str) -> None:
    out = redact(line)
    assert secret not in out, f"leaked in {out!r}"
    assert MASK in out
    # The NAME must survive: a log line that hides which variable was involved
    # is not a redacted diagnostic, it is a deleted one.
    assert line.split("=")[0] in out


def test_the_service_key_header_is_masked() -> None:
    assert "svc-abcdef123456" not in redact("X-Service-Key: svc-abcdef123456")


def test_a_real_key_literal_is_still_masked() -> None:
    """The env-var fix must not have blunted the literal rule."""
    out = redact("using sc-proj-3f8a1cdeadbeef for this request")
    assert "sc-proj-3f8a1cdeadbeef" not in out
