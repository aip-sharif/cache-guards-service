"""Pure unit tests for the OpenAI-gateway chat helpers — no backend, no network."""

import json


from semantic_cache.gateway.chat import (
    SSEAccumulator,
    build_cached_completion,
    build_cached_sse,
    extract_cache_text,
)


# -- extract_cache_text: the cache key is the LAST user message -------------- #


def test_last_user_message_single_turn() -> None:
    msgs = [{"role": "user", "content": "what is aspirin?"}]
    assert extract_cache_text(msgs) == "what is aspirin?"


def test_last_user_message_multi_turn() -> None:
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    assert extract_cache_text(msgs) == "second question"


def test_trailing_assistant_message_still_finds_last_user() -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert extract_cache_text(msgs) == "q"


def test_content_parts_array_joined() -> None:
    # OpenAI content can be a list of typed parts; text parts are joined.
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
                {"type": "text", "text": "part two"},
            ],
        }
    ]
    assert extract_cache_text(msgs) == "part one part two"


def test_no_user_message_returns_none() -> None:
    assert extract_cache_text([{"role": "system", "content": "x"}]) is None
    assert extract_cache_text([]) is None


def test_blank_user_message_returns_none() -> None:
    assert extract_cache_text([{"role": "user", "content": "   "}]) is None


# -- build_cached_completion: OpenAI-shaped hit response ---------------------- #


def test_cached_completion_shape() -> None:
    body = build_cached_completion("my-model", "Paris", similarity=0.97)
    assert body["object"] == "chat.completion"
    assert body["model"] == "my-model"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Paris"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert body["semantic_cache"]["hit"] is True
    assert body["semantic_cache"]["similarity"] == 0.97
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)


# -- build_cached_sse: hit replayed as an OpenAI SSE stream ------------------- #


def test_cached_sse_stream_is_valid_openai_sse() -> None:
    lines = list(build_cached_sse("my-model", "Paris"))
    assert lines[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(l[len("data: "):]) for l in lines[:-1]]
    # First chunk carries the role, some chunk carries the content,
    # last data chunk carries finish_reason=stop.
    assert payloads[0]["choices"][0]["delta"].get("role") == "assistant"
    contents = [
        p["choices"][0]["delta"].get("content")
        for p in payloads
        if p["choices"][0]["delta"].get("content")
    ]
    assert "".join(contents) == "Paris"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    assert all(p["model"] == "my-model" for p in payloads)


# -- SSEAccumulator: collect content from an upstream OpenAI stream ---------- #


def _chunk(content=None, role=None, finish=None, index=0) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "up-model",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}"


def test_accumulator_collects_full_content() -> None:
    acc = SSEAccumulator()
    for line in (_chunk(role="assistant"), _chunk("Pa"), _chunk("ris"),
                 _chunk(finish="stop"), "data: [DONE]"):
        acc.feed(line)
    assert acc.content == "Paris"
    assert acc.finish_reason == "stop"
    assert acc.done is True


def test_accumulator_ignores_noise_lines() -> None:
    acc = SSEAccumulator()
    acc.feed("")                      # keep-alive blank
    acc.feed(": comment")             # SSE comment
    acc.feed("data: not-json{{{")     # malformed → ignored, not raised
    acc.feed(_chunk("ok"))
    assert acc.content == "ok"


def test_accumulator_captures_usage_when_streamed() -> None:
    acc = SSEAccumulator()
    acc.feed(_chunk("hi"))
    usage_payload = {
        "id": "chatcmpl-x", "object": "chat.completion.chunk", "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    acc.feed(f"data: {json.dumps(usage_payload)}")
    acc.feed("data: [DONE]")
    assert acc.usage == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def test_accumulator_not_cacheable_on_length_finish() -> None:
    acc = SSEAccumulator()
    acc.feed(_chunk("partial answ"))
    acc.feed(_chunk(finish="length"))
    acc.feed("data: [DONE]")
    assert acc.finish_reason == "length"
    assert acc.cacheable is False


def test_accumulator_tracks_only_choice_zero() -> None:
    # n>1 streams interleave choices; merging them would cache garbage.
    acc = SSEAccumulator()
    acc.feed(_chunk("Pa", index=0))
    acc.feed(_chunk("London", index=1))
    acc.feed(_chunk("ris", index=0))
    acc.feed(_chunk(finish="length", index=1))   # other choice truncates...
    acc.feed(_chunk(finish="stop", index=0))     # ...choice 0 finishes clean
    acc.feed("data: [DONE]")
    assert acc.content == "Paris"
    assert acc.finish_reason == "stop"


def test_cached_sse_include_usage_emits_final_zero_usage_chunk() -> None:
    lines = list(build_cached_sse("m", "Paris", include_usage=True))
    assert lines[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(l[len("data: "):]) for l in lines[:-1]]
    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"] == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
    # default: no usage chunk at all
    default = [json.loads(l[6:]) for l in build_cached_sse("m", "x") if "[DONE]" not in l]
    assert all("usage" not in p for p in default)


def test_accumulator_cacheable_only_with_content_and_stop() -> None:
    acc = SSEAccumulator()
    acc.feed(_chunk(finish="stop"))
    acc.feed("data: [DONE]")
    assert acc.cacheable is False  # no content
    acc2 = SSEAccumulator()
    acc2.feed(_chunk("full answer"))
    acc2.feed(_chunk(finish="stop"))
    acc2.feed("data: [DONE]")
    assert acc2.cacheable is True
