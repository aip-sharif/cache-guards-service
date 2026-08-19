"""Semantic Cache — vector-based LLM response cache with an optional
entity-aware safety layer for high-stakes domains (medical, legal).

The public API is re-exported here so integrators can simply::

    from semantic_cache import SemanticCacheManager, SemanticCacheConfig

Optional-dependency adapters (FastAPI, LangChain) are intentionally NOT
imported at top level — import them explicitly from
``semantic_cache.adapters.*`` so the base package stays import-light.
"""

__version__ = "0.2.0"

from semantic_cache.core.cache_manager import (
    SemanticCacheManager,
    versioned_index_name,
)
from semantic_cache.core.config import (
    CacheDomain,
    CacheMode,
    EmbeddingProvider,
    SemanticCacheConfig,
)
from semantic_cache.core.embedding_manager import (
    APIEmbeddingManager,
    AsyncAPIEmbeddingManager,
    AsyncEmbeddingManagerFactory,
    BaseAsyncEmbeddingManager,
    BaseEmbeddingManager,
    EmbeddingManagerFactory,
    HuggingFaceEmbeddingManager,
)
from semantic_cache.core.entity_extractor import (
    BaseEntityExtractor,
    EntityExtractorFactory,
    LLMEntityExtractor,
)
from semantic_cache.core.entity_extractor_async import (
    AsyncEntityExtractorFactory,
    AsyncLLMEntityExtractor,
    BaseAsyncEntityExtractor,
)
from semantic_cache.core.exceptions import (
    CacheOperationError,
    ConfigurationError,
    EmbeddingGenerationError,
    EntityExtractionError,
    RedisConnectionError,
    SemanticCacheError,
)
from semantic_cache.core.extraction_cache import (
    AsyncCachedEntityExtractor,
    CachedEntityExtractor,
)
from semantic_cache.infrastructure.redis_client import RedisClientFactory

__all__ = [
    "__version__",
    # Core
    "SemanticCacheManager",
    "SemanticCacheConfig",
    "CacheDomain",
    "CacheMode",
    "EmbeddingProvider",
    "versioned_index_name",
    # Embeddings
    "BaseEmbeddingManager",
    "HuggingFaceEmbeddingManager",
    "APIEmbeddingManager",
    "EmbeddingManagerFactory",
    "BaseAsyncEmbeddingManager",
    "AsyncAPIEmbeddingManager",
    "AsyncEmbeddingManagerFactory",
    # Entity extraction
    "BaseEntityExtractor",
    "LLMEntityExtractor",
    "EntityExtractorFactory",
    "BaseAsyncEntityExtractor",
    "AsyncLLMEntityExtractor",
    "AsyncEntityExtractorFactory",
    "CachedEntityExtractor",
    "AsyncCachedEntityExtractor",
    # Infrastructure
    "RedisClientFactory",
    # Exceptions
    "SemanticCacheError",
    "ConfigurationError",
    "RedisConnectionError",
    "EmbeddingGenerationError",
    "CacheOperationError",
    "EntityExtractionError",
]
