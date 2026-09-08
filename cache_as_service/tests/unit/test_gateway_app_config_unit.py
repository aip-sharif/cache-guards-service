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


def _client(spy, ttl=60.0, url=APP_URL, max_entries=10_000, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(spy))
    return AppConfigClient(
        http, url=url, ttl=ttl, max_entries=max_entries, **kw
    )


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


# --------------------------------------------------------------------------- #
# The cache is BOUNDED and keyed by a fingerprint
#
# It is keyed by the caller's own key, so an unbounded dict is sized by our
# callers rather than by us — every wrong bearer included — and each entry
# holds live provider credentials.
# --------------------------------------------------------------------------- #


async def test_the_cache_is_bounded() -> None:
    spy = AppSpy()
    client = _client(spy, max_entries=3)
    for i in range(50):
        await client.resolve(f"key-{i}")
    assert len(client) == 3


async def test_eviction_is_least_recently_used_not_oldest_first() -> None:
    """A burst of unknown keys must not flush the working set: the live key
    keeps being used, so it must be the one that survives."""
    spy = AppSpy()
    client = _client(spy, max_entries=3)

    await client.resolve("live-key")
    for i in range(10):
        await client.resolve(f"noise-{i}")
        await client.resolve("live-key")  # keeps it recent

    before = len(spy.requests)
    await client.resolve("live-key")
    assert len(spy.requests) == before, "the live key was evicted and refetched"


async def test_the_raw_key_is_never_a_cache_key() -> None:
    """A heap dump of this dict must not hand over the bearer tokens."""
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("sk-super-secret-key")
    assert "sk-super-secret-key" not in client._cache


async def test_invalidate_drops_the_fingerprinted_entry() -> None:
    """Rotation and revocation have to take effect now, not in up to a TTL."""
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("k")
    await client.resolve("k")
    assert len(spy.requests) == 1

    client.invalidate("k")
    await client.resolve("k")
    assert len(spy.requests) == 2


async def test_invalidate_all_clears_everything() -> None:
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("a")
    await client.resolve("b")
    client.invalidate()
    assert len(client) == 0


async def test_an_expired_entry_is_dropped_not_merely_ignored() -> None:
    """Expired credentials should leave memory when they expire, rather than
    linger until something else needs the space."""
    spy = AppSpy()
    client = _client(spy, ttl=0.0)
    await client.resolve("k")
    await client.resolve("k")
    assert len(client) == 1
    assert len(spy.requests) == 2


# --------------------------------------------------------------------------- #
# The SERVICE credential
#
# Two credentials answering two different questions: the bearer says WHICH
# CLIENT this config is for, the service key says IT IS THE CACHE SERVICE
# asking. Without the second, a leaked client key is enough to pull that
# client's model-provider credentials straight out of the APP.
# --------------------------------------------------------------------------- #

SERVICE_KEY = "svc-shared-between-app-and-us"


async def test_the_service_key_is_sent_in_its_own_header() -> None:
    spy = AppSpy()
    client = _client(spy, service_key=SERVICE_KEY)
    await client.resolve("sc-proj-client")

    (request,) = spy.requests
    assert request.headers["X-Service-Key"] == SERVICE_KEY
    # And NOT collapsed into Authorization, which carries the client's key.
    assert request.headers["Authorization"] == "Bearer sc-proj-client"


async def test_the_header_name_is_configurable() -> None:
    """So the APP team can choose it without a code change here."""
    spy = AppSpy()
    client = _client(
        spy, service_key=SERVICE_KEY, service_key_header="X-Cache-Service-Token"
    )
    await client.resolve("sc-proj-client")

    (request,) = spy.requests
    assert request.headers["X-Cache-Service-Token"] == SERVICE_KEY
    assert "x-service-key" not in request.headers


async def test_no_service_key_sends_no_header() -> None:
    """Unset must leave the contract exactly as it was, so the two sides can be
    deployed in either order rather than in lockstep."""
    spy = AppSpy()
    client = _client(spy)
    await client.resolve("sc-proj-client")

    (request,) = spy.requests
    assert "x-service-key" not in request.headers
    assert client.authenticates_as_a_service is False


async def test_the_client_reports_whether_it_authenticates_as_a_service() -> None:
    """create_app logs a warning on False — an operator should not have to read
    a header dump to find out they are unauthenticated to the APP."""
    assert _client(AppSpy(), service_key=SERVICE_KEY).authenticates_as_a_service
    assert not _client(AppSpy(), service_key="").authenticates_as_a_service
    assert not _client(AppSpy(), service_key=None).authenticates_as_a_service


async def test_the_service_key_is_sent_on_every_uncached_call() -> None:
    """It is not a handshake — there is no session, so every request carries
    it. A cached config makes no request at all."""
    spy = AppSpy()
    client = _client(spy, service_key=SERVICE_KEY)
    await client.resolve("key-a")
    await client.resolve("key-b")
    await client.resolve("key-a")  # cached: no third request

    assert len(spy.requests) == 2
    assert all(r.headers["X-Service-Key"] == SERVICE_KEY for r in spy.requests)


async def test_an_empty_header_name_falls_back_to_the_default() -> None:
    """A blank SC_APP_SERVICE_KEY_HEADER must not produce a header with no
    name, which httpx would reject at request time."""
    spy = AppSpy()
    client = _client(spy, service_key=SERVICE_KEY, service_key_header="")
    await client.resolve("sc-proj-client")
    assert spy.requests[0].headers["X-Service-Key"] == SERVICE_KEY
