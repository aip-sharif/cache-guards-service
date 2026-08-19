"""Lazy LRU pool of per-cache SemanticCacheManager instances.

A full manager per (tenant, cache) is cheap — construction is attribute
setup plus one FT.INFO round-trip, and every manager shares the SAME Redis
client, embedding backend, and RediSearch index (isolation is by scope TAG,
not by index). The config hash is part of the pool key, so updating a
cache's settings naturally yields a fresh manager on next use.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

from semantic_cache.core.cache_manager import SemanticCacheManager
from semantic_cache.core.config import SemanticCacheConfig


class ManagerPool:
    def __init__(
        self,
        base_config: SemanticCacheConfig,
        redis_client: Any,
        embedding_manager: Any,
        async_embedding_manager: Any = None,
        max_size: int = 256,
    ) -> None:
        self.base_config = base_config
        self.redis = redis_client
        self.embedding = embedding_manager
        self.async_embedding = async_embedding_manager
        self.max_size = max_size
        self._lock = threading.Lock()
        self._pool: "OrderedDict[Tuple[str, str, str], SemanticCacheManager]" = OrderedDict()

    def get(
        self, tenant_id: str, cache_id: str, config_json: Optional[str] = None
    ) -> SemanticCacheManager:
        """Returns the manager for this cache, building it on first use."""
        config_json = config_json or "{}"
        cfg_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]
        key = (tenant_id, cache_id, cfg_hash)
        with self._lock:
            manager = self._pool.get(key)
            if manager is not None:
                self._pool.move_to_end(key)
                return manager
            config = self.base_config.model_copy(update=json.loads(config_json))
            manager = SemanticCacheManager(
                config=config,
                redis_client=self.redis,
                embedding_manager=self.embedding,
                async_embedding_manager=self.async_embedding,
            )
            self._pool[key] = manager
            while len(self._pool) > self.max_size:
                self._pool.popitem(last=False)
            return manager
