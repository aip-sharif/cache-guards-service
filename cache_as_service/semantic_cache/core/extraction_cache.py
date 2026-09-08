"""Composable cache layer in front of any `BaseEntityExtractor`.

Why this matters for cost:
    Every entity-aware lookup costs one LLM round trip. For traffic with
    repeat queries, that round trip is pure waste — the entity set is a
    deterministic function of the normalized text. We cache it in Redis
    keyed by `sha256(text)` with a configurable TTL.

    Hit rate on a steady-state production query mix tends to be high (>70%
    for FAQ-style traffic), turning the average extraction cost from
    `1 * llm_call` to `(1 - hit_rate) * llm_call`.

Design:
    `CachedEntityExtractor` is a transparent wrapper. It implements the same
    `BaseEntityExtractor` ABC so the cache manager treats it identically to
    the underlying LLM extractor. Compose it like:

        inner = LLMEntityExtractor(...)
        outer = CachedEntityExtractor(inner, redis_client, ttl_seconds=3600)
        manager = SemanticCacheManager(..., entity_extractor=outer)

    On cache miss, errors from the inner extractor are propagated unchanged
    so the manager's fail-soft logic still triggers (cache MISS, never HIT).

    A failed extraction is NOT cached — we don't want to remember the
    failure and serve it as a recurring MISS for an hour. A successful
    extraction with an empty entity list IS cached (it's a legitimate
    deterministic result).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Dict, List

import redis

from semantic_cache.core.entity_extractor import BaseEntityExtractor
from semantic_cache.core.entity_extractor_async import BaseAsyncEntityExtractor
from semantic_cache.core import metrics

logger = logging.getLogger(__name__)


class CachedEntityExtractor(BaseEntityExtractor):
    """Wraps another extractor with a Redis-backed result cache."""

    # All extraction-cache keys live under this prefix so they are easy to
    # spot in the keyspace and easy to mass-evict if the prompt ever changes.
    _PREFIX = "scache:entcache:"

    def __init__(
        self,
        inner: BaseEntityExtractor,
        redis_client: redis.Redis,
        ttl_seconds: int = 3600,
    ):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0 (0 disables caching).")
        self.inner = inner
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def _key(cls, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{cls._PREFIX}{digest}"

    def extract(self, text: str) -> List[Dict[str, str]]:
        # Caching disabled — just delegate.
        if self.ttl_seconds == 0:
            return self.inner.extract(text)

        key = self._key(text)
        try:
            cached = self.redis.get(key)
        except Exception as e:  # noqa: BLE001
            # Redis hiccup → fall through to inner extractor; don't fail.
            logger.warning("Extraction cache GET failed (%s); bypassing cache.", e)
            cached = None

        if cached is not None:
            metrics.record_extraction_cache("hit")
            try:
                parsed = json.loads(cached)
                if isinstance(parsed, list):
                    return parsed
                logger.warning("Extraction cache hit was not a list; ignoring.")
            except json.JSONDecodeError:
                logger.warning("Extraction cache hit was malformed JSON; ignoring.")

        metrics.record_extraction_cache("miss")

        # Cache MISS: call the inner extractor. Errors propagate unchanged
        # so the manager's fail-soft logic still turns it into a MISS.
        result = self.inner.extract(text)

        # Cache the result, including empty lists (deterministic outcome).
        # Never cache failures — they're not cached because the call raised
        # before we got here.
        try:
            self.redis.set(key, json.dumps(result), ex=self.ttl_seconds)
        except Exception as e:  # noqa: BLE001
            logger.warning("Extraction cache SET failed (%s); continuing.", e)
        return result


class AsyncCachedEntityExtractor(BaseAsyncEntityExtractor):
    """Async counterpart of :class:`CachedEntityExtractor`.

    Wraps an async extractor with the same Redis-backed result cache. The
    Redis client is the (sync) client the manager already holds, so its calls
    are offloaded to a worker thread to keep the event loop free.
    """

    _PREFIX = "scache:entcache:"

    def __init__(
        self,
        inner: BaseAsyncEntityExtractor,
        redis_client: redis.Redis,
        ttl_seconds: int = 3600,
    ):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0 (0 disables caching).")
        self.inner = inner
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def _key(cls, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{cls._PREFIX}{digest}"

    async def extract(self, text: str) -> List[Dict[str, str]]:
        if self.ttl_seconds == 0:
            return await self.inner.extract(text)

        key = self._key(text)
        try:
            cached = await asyncio.to_thread(self.redis.get, key)
        except Exception as e:  # noqa: BLE001
            logger.warning("Extraction cache GET failed (%s); bypassing cache.", e)
            cached = None

        if cached is not None:
            metrics.record_extraction_cache("hit")
            try:
                parsed = json.loads(cached)
                if isinstance(parsed, list):
                    return parsed
                logger.warning("Extraction cache hit was not a list; ignoring.")
            except json.JSONDecodeError:
                logger.warning("Extraction cache hit was malformed JSON; ignoring.")

        metrics.record_extraction_cache("miss")

        # MISS: call the inner extractor. Errors propagate unchanged so the
        # manager's fail-soft logic still turns it into a cache MISS. A failed
        # extraction is never cached.
        result = await self.inner.extract(text)
        try:
            await asyncio.to_thread(
                self.redis.set, key, json.dumps(result), ex=self.ttl_seconds
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Extraction cache SET failed (%s); continuing.", e)
        return result
