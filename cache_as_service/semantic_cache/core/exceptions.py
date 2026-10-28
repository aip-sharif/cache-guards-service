"""Custom exceptions for the Semantic Caching system."""

class SemanticCacheError(Exception):
    """Base exception for all semantic cache errors."""
    pass

class ConfigurationError(SemanticCacheError):
    """Raised when there is an invalid configuration."""
    pass

class RedisConnectionError(SemanticCacheError):
    """Raised when a connection to Redis fails.

    This exception is used to gracefully degrade the system when
    the backend cache store is unreachable.
    """
    pass

class EmbeddingGenerationError(SemanticCacheError):
    """Raised when generating embeddings fails.

    This can occur either due to local HuggingFace issues or
    errors with third-party API endpoints, such as timeouts or
    invalid API keys.
    """
    pass

class CacheOperationError(SemanticCacheError):
    """Raised when a specific cache operation (set/get/purge) fails."""
    pass

class EntityExtractionError(SemanticCacheError):
    """Raised when entity extraction fails.

    Callers should treat this as a fall-through to cache MISS — never a HIT —
    to avoid serving wrong answers in high-stakes domains.
    """
    pass
