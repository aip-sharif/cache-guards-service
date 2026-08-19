"""AppConfigClient — fetch + TTL-cache a key's model config from the APP.

httpx.MockTransport stands in for the APP; no network, CI-safe. In the
key-only model the client sends the caller's own key as the bearer to a single
fixed URL — there is no project id to substitute and no service token.
"""

from typing import List

import httpx
import pytest

from semantic_cache.gateway.app_config import AppConfigClient, AppConfigError

APP_URL = "https://app.example.com/api/gateway-config"

GOOD = {
    "model": "gpt-x",
    "model_api_key": "sk-llm",
    "embed_model": "bge-m3",
    "embed_api_key": "sk-embed",
    "extractor_model": None,
    "extractor_api_key": None,
}


class AppSpy:
    def __init__(self, status=200, body=None, text=None) -> None:
        self.status = status
        self.body = GOOD if body is None else body
        self.text = text
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.text is not None:
            return httpx.Response(self.status, text=self.text)
        return httpx.Response(self.status, json=self.body)


def _client(spy, ttl=60.0, url=APP_URL):
    http = httpx.AsyncClient(transport=httpx.MockTransport(spy))
    return AppConfigClient(http, url=url, ttl=ttl)


@pytest.mark.asyncio
async def test_url_used_verbatim_and_key_is_the_bearer() -> None:
    spy = AppSpy()
    await _client(spy, url="http://localhost:8000/cache").resolve("sc-proj-abc")
    request = spy.requests[0]
    # No id substitution/appending — the APP identifies the client from the key.
    assert str(request.url) == "http://localhost:8000/cache"
    assert request.headers["authorization"] == "Bearer sc-proj-abc"
    assert request.headers["accept"] == "application/json"


@pytest.mark.asyncio
async def test_returns_normalized_config() -> None:
    spy = AppSpy()
    config = await _client(spy).resolve("sc-proj-abc")
    assert config["model"] == "gpt-x"
    assert config["embed_api_key"] == "sk-embed"
    assert config["extractor_model"] is None
    assert config["cache_config"] == {}


@pytest.mark.asyncio
async def test_config_is_cached_per_key_within_ttl() -> None:
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("k1")
    await client.resolve("k1")
    assert len(spy.requests) == 1          # second resolve served from memory
    await client.resolve("k2")             # different key → own fetch
    assert len(spy.requests) == 2


@pytest.mark.asyncio
async def test_invalidate_forces_refetch() -> None:
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("k1")
    client.invalidate("k1")
    await client.resolve("k1")
    assert len(spy.requests) == 2


@pytest.mark.asyncio
async def test_unknown_key_maps_to_403() -> None:
    for status in (401, 403, 404):
        with pytest.raises(AppConfigError) as e:
            await _client(AppSpy(status=status)).resolve("bad")
        assert e.value.status_code == 403


@pytest.mark.asyncio
async def test_app_5xx_and_bad_json_map_to_502() -> None:
    with pytest.raises(AppConfigError) as e:
        await _client(AppSpy(status=500)).resolve("k")
    assert e.value.status_code == 502
    with pytest.raises(AppConfigError) as e2:
        await _client(AppSpy(text="<html>oops</html>")).resolve("k")
    assert e2.value.status_code == 502


@pytest.mark.asyncio
async def test_missing_required_fields_map_to_502() -> None:
    body = {k: v for k, v in GOOD.items() if k != "embed_api_key"}
    with pytest.raises(AppConfigError) as e:
        await _client(AppSpy(body=body)).resolve("k")
    assert e.value.status_code == 502
    # reported under the APP's own field name
    assert "embedd_key" in str(e.value)


@pytest.mark.asyncio
async def test_app_unreachable_maps_to_502() -> None:
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    with pytest.raises(AppConfigError) as e:
        await AppConfigClient(http, url=APP_URL).resolve("k")
    assert e.value.status_code == 502


# -- the APP's real payload shape --------------------------------------------

APP_REAL = {
    "id": "18188ae7-c928-4614-ae19-3208e2cd6f3c",
    "id_user": "id",
    "name": "string",
    "embedd_model": "bge-m3",
    "embedd_key": "sk-embed-real",
    "llm_model": "gpt-4o-mini",
    "llm_key": "sk-llm-real",
    "api_key": "sc-proj-9f2a92a37e9b3dae42eebec4168ba5df827978e7d1dbe6b2",
    "project_id": "71ae3f7f1e564f238d5c45cde004cd9c",
    "extaractor": None,
    "extaractor_key": None,
    "created_at": "2026-07-19T13:02:09.360834",
}


@pytest.mark.asyncio
async def test_reads_the_apps_actual_field_names() -> None:
    spy = AppSpy(body=APP_REAL)
    config = await _client(spy).resolve("sc-proj-abc")
    assert config["model"] == "gpt-4o-mini"          # from llm_model
    assert config["model_api_key"] == "sk-llm-real"  # from llm_key
    assert config["embed_model"] == "bge-m3"         # from embedd_model
    assert config["embed_api_key"] == "sk-embed-real"
    assert config["extractor_model"] is None
    # project_id is kept for the router to use as the cache scope.
    assert config["project_id"] == "71ae3f7f1e564f238d5c45cde004cd9c"


@pytest.mark.asyncio
async def test_reads_the_apps_extractor_spelling() -> None:
    body = dict(APP_REAL, extaractor="gpt-mini", extaractor_key="sk-ex")
    config = await _client(AppSpy(body=body)).resolve("k")
    assert config["extractor_model"] == "gpt-mini"
    assert config["extractor_api_key"] == "sk-ex"


@pytest.mark.asyncio
async def test_missing_app_fields_named_in_the_error() -> None:
    body = {k: v for k, v in APP_REAL.items() if k != "embedd_key"}
    with pytest.raises(AppConfigError) as e:
        await _client(AppSpy(body=body)).resolve("k")
    assert "embedd_key" in str(e.value)


@pytest.mark.asyncio
async def test_errors_are_not_cached() -> None:
    spy = AppSpy(status=500)
    client = _client(spy)
    with pytest.raises(AppConfigError):
        await client.resolve("k")
    spy.status = 200                        # APP recovers
    config = await client.resolve("k")
    assert config["model"] == "gpt-x"
