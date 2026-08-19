"""Per-model cache managers for the gateway.

Every gateway model config gets its OWN SemanticCacheManager with its own
RediSearch index, its own key prefix, and its own API embedding backend —
different embedding models produce different vector spaces (and dimensions),
so they must never share an index. Project isolation WITHIN a model is by
scope tag, exactly like the SaaS layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from pydantic import SecretStr

from semantic_cache.core.cache_manager import SemanticCacheManager
from semantic_cache.core.config import (
    CacheDomain,
    EmbeddingProvider,
    SemanticCacheConfig,
)
from semantic_cache.core.embedding_manager import (
    AsyncEmbeddingManagerFactory,
    EmbeddingManagerFactory,
)
from semantic_cache.core.entity_extractor import EntityExtractorFactory
from semantic_cache.core.entity_extractor_async import AsyncEntityExtractorFactory
from semantic_cache.saas.registry import PER_CACHE_FIELDS

logger = logging.getLogger(__name__)

_EXTRACTOR_DOMAINS = {CacheDomain.MEDICAL, CacheDomain.LEGAL}


def sanitize_model_slug(model: str) -> str:
    """Model names may contain '/', ':' etc. — fold to index/key-safe chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", model)


def _validated_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    unknown = set(overrides) - PER_CACHE_FIELDS
    if unknown:
        raise ValueError(
            f"Fields not customizable per model: {sorted(unknown)}. "
            f"Allowed: {sorted(PER_CACHE_FIELDS)}"
        )
    try:
        validated = SemanticCacheConfig.model_validate(overrides)
    except Exception as e:
        raise ValueError(f"Invalid cache_config: {e}") from e
    return {k: getattr(validated, k) for k in overrides}


def derive_cache_config(
    base: SemanticCacheConfig, model_row: Dict[str, Any]
) -> SemanticCacheConfig:
    """Builds the cache config for one gateway model from its registry row.

    Raises ValueError on an invalid row (unknown/out-of-range cache_config
    fields, extractor without a supported domain).

    The index/key-prefix slug is the EMBEDDING model, not the chat model:
    the vector space is defined by the embedder, so projects sharing an
    embedding model share one index (isolated by scope), while different
    embedding models never mix."""
    slug = sanitize_model_slug(model_row["embed_model"])
    update: Dict[str, Any] = {
        "cache_index_name": f"gw__{slug}",
        "key_prefix": f"scache:gw:{slug}:",
        "embedding_provider": EmbeddingProvider.CUSTOM_API,
        "embedding_base_url": model_row["embed_base_url"],
        "embedding_api_key": SecretStr(model_row["embed_api_key"] or ""),
        "embedding_model": model_row["embed_model"],
        "embedding_dim": None,  # auto-probe per model
    }

    has_extractor = bool(
        model_row.get("extractor_base_url") and model_row.get("extractor_model")
    )
    if has_extractor:
        raw_domain = model_row.get("extractor_domain")
        try:
            domain = CacheDomain(raw_domain) if raw_domain else None
        except ValueError:
            domain = None
        if domain not in _EXTRACTOR_DOMAINS:
            # Entity extraction only has prompts for these domains. An APP
            # that sends an extractor without a usable domain gets plain
            # semantic caching rather than a failed request.
            logger.warning(
                "Ignoring extractor %r: extractor_domain is %r, expected one "
                "of %s. Entity-aware caching stays OFF.",
                model_row.get("extractor_model"), raw_domain,
                sorted(d.value for d in _EXTRACTOR_DOMAINS),
            )
            update["entity_aware"] = False
            update.update(_validated_overrides(model_row.get("cache_config") or {}))
            return base.model_copy(update=update)
        update.update(
            entity_aware=True,
            domain=domain,
            entity_model=model_row["extractor_model"],
            entity_llm_base_url=model_row["extractor_base_url"],
            entity_llm_api_key=SecretStr(model_row.get("extractor_api_key") or ""),
        )
    else:
        update["entity_aware"] = False

    update.update(_validated_overrides(model_row.get("cache_config") or {}))
    return base.model_copy(update=update)


class GatewayModelPool:
    """LRU pool of per-model managers (own index + own embedder each)."""

    def __init__(
        self,
        base_config: SemanticCacheConfig,
        redis_client: Any,
        max_size: int = 64,
        persist_hook: Optional[Any] = None,
        touch_hook: Optional[Any] = None,
        delete_hook: Optional[Any] = None,
    ) -> None:
        self.base_config = base_config
        self.redis = redis_client
        self.max_size = max_size
        # Durable-backup hooks (see gateway/backup.py). Installed on every
        # manager this pool builds.
        self.persist_hook = persist_hook
        self.touch_hook = touch_hook
        self.delete_hook = delete_hook
        self._lock = threading.Lock()
        self._pool: "OrderedDict[Tuple[str, str], SemanticCacheManager]" = OrderedDict()

    def get(self, model_row: Dict[str, Any]) -> SemanticCacheManager:
        """The manager for this model config, built on first use. The config
        hash is part of the pool key, so a changed config from the APP yields
        a fresh manager (and, if the embed model changed, a fresh index) on
        next use."""
        row_hash = hashlib.sha256(
            json.dumps(model_row, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        key = (model_row["embed_model"], row_hash)

        # Fast path under the lock — cheap dict access only.
        with self._lock:
            manager = self._pool.get(key)
            if manager is not None:
                self._pool.move_to_end(key)
                return manager

        # Slow path OUTSIDE the lock: building a manager bootstraps the Redis
        # index and probes the embedding dimension over HTTP. Holding the lock
        # across that would stall every other project's requests (and hang the
        # whole gateway if the embed endpoint is slow). Two threads racing the
        # same new config may both build; the loser's manager is discarded.
        config = derive_cache_config(self.base_config, model_row)
        manager = SemanticCacheManager(
            config=config,
            redis_client=self.redis,
            embedding_manager=EmbeddingManagerFactory.create(config),
            async_embedding_manager=AsyncEmbeddingManagerFactory.create(config),
            entity_extractor=EntityExtractorFactory.create(config),
            async_entity_extractor=AsyncEntityExtractorFactory.create(config),
        )
        manager.persist_hook = self.persist_hook
        manager.touch_hook = self.touch_hook
        manager.delete_hook = self.delete_hook

        with self._lock:
            existing = self._pool.get(key)
            if existing is not None:  # lost the race — reuse the winner
                self._pool.move_to_end(key)
                return existing
            self._pool[key] = manager
            while len(self._pool) > self.max_size:
                self._pool.popitem(last=False)
            return manager
