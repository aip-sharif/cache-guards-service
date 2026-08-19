"""The public API must be importable straight from the top-level package."""

import semantic_cache


def test_public_api_is_exported():
    expected = [
        "SemanticCacheManager",
        "SemanticCacheConfig",
        "CacheDomain",
        "EmbeddingProvider",
        "versioned_index_name",
        "EmbeddingManagerFactory",
        "AsyncEmbeddingManagerFactory",
        "APIEmbeddingManager",
        "AsyncAPIEmbeddingManager",
        "EntityExtractorFactory",
        "AsyncEntityExtractorFactory",
        "CachedEntityExtractor",
        "AsyncCachedEntityExtractor",
        "RedisClientFactory",
        "SemanticCacheError",
        "CacheOperationError",
        "EntityExtractionError",
        "EmbeddingGenerationError",
    ]
    missing = [name for name in expected if not hasattr(semantic_cache, name)]
    assert not missing, f"Not exported from semantic_cache: {missing}"


def test_one_line_import_works():
    from semantic_cache import SemanticCacheConfig, SemanticCacheManager

    assert SemanticCacheManager.__name__ == "SemanticCacheManager"
    # Config is constructible without args (no network needed).
    assert 0.0 <= SemanticCacheConfig().similarity_threshold <= 1.0
