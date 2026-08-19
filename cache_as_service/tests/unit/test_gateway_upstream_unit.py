"""Upstream LLM client — tested against httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from semantic_cache.gateway.upstream import UpstreamClient, UpstreamError

UPSTREAM_ANSWER = {
    "id": "chatcmpl-real",
    "object": "chat.completion",
    "created": 1,
    "model": "up-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
}


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_complete_posts_to_chat_completions_with_auth() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=UPSTREAM_ANSWER)

    up = UpstreamClient(_mock_client(handler))
    result = await up.complete(
        "https://llm.example.com", "sk-up", {"model": "up-model", "messages": []}
    )
    assert seen["url"] == "https://llm.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-up"
    assert seen["body"]["model"] == "up-model"
    assert result["choices"][0]["message"]["content"] == "Paris"


async def test_complete_base_url_with_v1_not_doubled() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=UPSTREAM_ANSWER)

    up = UpstreamClient(_mock_client(handler))
    await up.complete("https://llm.example.com/v1/", "k", {"messages": []})
    assert seen["url"] == "https://llm.example.com/v1/chat/completions"


async def test_complete_upstream_error_raises_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    up = UpstreamClient(_mock_client(handler))
    with pytest.raises(UpstreamError) as e:
        await up.complete("https://llm.example.com", "k", {"messages": []})
    assert e.value.status_code == 429
    assert e.value.detail == "rate limited"      # kept for the log
    assert "rate limited" not in str(e.value)    # but NOT for the client


async def test_upstream_error_never_leaks_the_provider_key() -> None:
    # Providers echo the presented key in auth errors; that key belongs to the
    # APP, not the caller, so the client-facing message must not carry it.
    leak = "Incorrect API key provided: sk-secret-abc123"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": leak}})

    up = UpstreamClient(_mock_client(handler))
    with pytest.raises(UpstreamError) as e:
        await up.complete("https://llm.example.com", "sk-secret-abc123",
                          {"messages": []})
    assert e.value.status_code == 401
    assert "sk-secret" not in str(e.value)
    assert e.value.detail == leak


async def test_stream_yields_sse_lines() -> None:
    sse_body = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, content=sse_body.encode(), headers={"content-type": "text/event-stream"}
        )

    up = UpstreamClient(_mock_client(handler))
    lines = []
    async for line in up.stream("https://llm.example.com", "k", {"stream": True, "messages": []}):
        lines.append(line)
    data_lines = [l for l in lines if l.startswith("data:")]
    assert data_lines[-1].strip() == "data: [DONE]"
    assert any("Paris" in l for l in data_lines)


async def test_stream_upstream_error_raises_before_yielding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    up = UpstreamClient(_mock_client(handler))
    with pytest.raises(UpstreamError) as e:
        async for _ in up.stream("https://llm.example.com", "k", {"messages": []}):
            pass
    assert e.value.status_code == 500
