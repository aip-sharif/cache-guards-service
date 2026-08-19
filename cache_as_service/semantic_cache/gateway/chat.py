"""Pure helpers for the OpenAI-compatible chat surface.

No I/O here: message parsing, cached-response shaping (JSON and SSE), and an
accumulator that reconstructs the assistant message from an upstream SSE
stream so it can be cached after the stream completes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# The cache key is the LAST user message: highest hit rate, and the right
# granularity for FAQ-style traffic. Multi-turn context is deliberately not
# part of the key (decided v1 behavior).


def _content_to_text(content: Any) -> str:
    """Flattens OpenAI message content (str or typed-part list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(t for t in parts if t)
    return ""


def extract_cache_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """The text to embed/cache for this request: the last user message.

    Returns None when there is no non-blank user message (nothing sensible to
    cache — the caller should pass the request straight through)."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _content_to_text(msg.get("content")).strip()
            return text or None
    return None


#: Sentinel returned by extract_guard_segments when a message in scope carries
#: content the guard cannot read (an image, audio, a part type we don't know).
#: Distinct from an empty list, which means "there was genuinely nothing to
#: check" — treating the two the same would serve unreadable content unguarded.
UNGUARDABLE = object()

#: What the guard checks when the APP does not say. Mirrors
#: guard_config.DEFAULT_CHECK_ROLES.
_DEFAULT_GUARD_ROLES = ("user", "system", "tool")


def _guard_content_to_text(content: Any) -> Tuple[str, bool]:
    """Like :func:`_content_to_text`, but for guarding rather than caching.

    Returns ``(text, had_unextractable)``. Two differences, both deliberate:

    * it takes ``text`` from EVERY dict part regardless of ``type``, so a part
      type we have not seen (``input_text``, a vendor extension) is still
      checked rather than silently skipped;
    * it reports when content existed but produced no text, so an image-only
      turn becomes a refusal instead of an unguarded pass.
    """
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text")
                if isinstance(value, str) and value:
                    parts.append(value)
        text = " ".join(parts)
        return text, bool(content) and not text.strip()
    if content in (None, ""):
        return "", False
    return "", True


def extract_guard_segments(
    messages: List[Dict[str, Any]],
    check_roles: Sequence[str] = _DEFAULT_GUARD_ROLES,
):
    """Every message the guard must score, across the WHOLE array.

    NOT :func:`extract_cache_text`. That returns the last user message, which
    is the right cache key and a fatal guard: this service is stateless and the
    CLIENT authors the entire array, so

        [user: "<payload>", assistant: "Sure.", user: "continue"]

    would be scored as ``"continue"`` while the router forwards the array
    verbatim to the model. One request, no race, complete bypass.

    Returns a list of :class:`Segment` (possibly empty), or :data:`UNGUARDABLE`.
    """
    from semantic_cache.gateway.guard_logic import Segment

    roles = set(check_roles or _DEFAULT_GUARD_ROLES)
    segments: List[Any] = []
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") not in roles:
            continue
        text, unextractable = _guard_content_to_text(message.get("content"))
        if unextractable:
            return UNGUARDABLE
        text = text.strip()
        if text:
            segments.append(Segment(role=str(message.get("role")), text=text,
                                    index=index))
    return segments


def _completion_id() -> str:
    return f"chatcmpl-cache-{uuid.uuid4().hex[:24]}"


def build_cached_completion(
    model: str, content: str, similarity: Optional[float] = None
) -> Dict[str, Any]:
    """An OpenAI chat.completion response served from cache.

    Usage is all-zeros — no upstream tokens were spent; the extra
    `semantic_cache` field marks the hit for observability (harmless to
    standard OpenAI clients)."""
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "semantic_cache": {"hit": True, "similarity": similarity},
    }


def build_cached_sse(
    model: str, content: str, include_usage: bool = False
) -> Iterator[str]:
    """Replays a cache hit as an OpenAI-compatible SSE stream.

    Shape mirrors real streaming: a role chunk, one content chunk, a finish
    chunk, then — when the client asked for stream_options.include_usage —
    a final zero-usage chunk with empty choices, then the [DONE] sentinel."""
    cid = _completion_id()
    created = int(time.time())

    def envelope(**extra: Any) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
        return envelope(
            choices=[{"index": 0, "delta": delta, "finish_reason": finish}]
        )

    yield chunk({"role": "assistant"})
    yield chunk({"content": content})
    yield chunk({}, finish="stop")
    if include_usage:
        yield envelope(
            choices=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    yield "data: [DONE]\n\n"


def _guard_id() -> str:
    return f"chatcmpl-guard-{uuid.uuid4().hex[:24]}"


def guardrail_extra(
    action: str,
    matched_category: Optional[str] = None,
    score: Optional[float] = None,
    judge_invoked: bool = False,
    policy: Optional[str] = None,
    request_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``guardrail`` field, mirroring the cache's ``semantic_cache`` one."""
    extra: Dict[str, Any] = {"action": action}
    if matched_category:
        extra["category"] = matched_category
    if score is not None:
        extra["score"] = round(float(score), 4)
    extra["judge_invoked"] = bool(judge_invoked)
    if policy:
        extra["policy"] = policy[:12]
    if request_id:
        extra["request_id"] = request_id
    if reason:
        extra["reason"] = reason
    return extra


def build_guard_completion(
    model: str, content: str, guardrail: Dict[str, Any]
) -> Dict[str, Any]:
    """A refusal, shaped as an ordinary chat.completion.

    ``finish_reason`` is ``content_filter`` — the OpenAI-spec value for exactly
    this event — and never ``stop``, which would affirmatively claim the model
    completed normally. Usage is zeroed: no upstream tokens were spent.
    """
    return {
        "id": _guard_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "guardrail": guardrail,
    }


def build_guard_sse(
    model: str, content: str, guardrail: Dict[str, Any],
    include_usage: bool = False,
) -> Iterator[str]:
    """The same refusal as an SSE stream, shaped like ``build_cached_sse``."""
    cid = _guard_id()
    created = int(time.time())

    def envelope(**extra: Any) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # The guardrail object rides the FIRST chunk, so a client that stops
    # reading early still learns why.
    yield envelope(
        choices=[{"index": 0, "delta": {"role": "assistant"},
                  "finish_reason": None}],
        guardrail=guardrail,
    )
    yield envelope(
        choices=[{"index": 0, "delta": {"content": content},
                  "finish_reason": None}]
    )
    yield envelope(
        choices=[{"index": 0, "delta": {},
                  "finish_reason": "content_filter"}]
    )
    if include_usage:
        yield envelope(
            choices=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    yield "data: [DONE]\n\n"


class SSEAccumulator:
    """Reconstructs the assistant message from upstream OpenAI SSE lines.

    Feed each line (with or without trailing newlines). Malformed or non-data
    lines are ignored — a broken chunk must never break the passthrough."""

    def __init__(self) -> None:
        self.content: str = ""
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.done: bool = False

    def feed(self, line: str) -> None:
        line = (line or "").strip()
        if not line.startswith("data:"):
            return
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        for choice in payload.get("choices") or []:
            # Only choice 0 is reconstructed: with n>1 the choices interleave
            # and merging them would cache garbage. (n>1 requests are not
            # cached by the router anyway — this is defense in depth.)
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                self.content += piece
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    @property
    def cacheable(self) -> bool:
        """Cache only complete, non-empty answers: content + finish 'stop'."""
        return bool(self.content.strip()) and self.finish_reason == "stop"
