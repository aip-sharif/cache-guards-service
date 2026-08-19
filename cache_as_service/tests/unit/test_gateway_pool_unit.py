"""Config derivation for per-model gateway caches — pure unit."""

import pytest

from semantic_cache.core.config import (
    CacheDomain,
    CacheMode,
    EmbeddingProvider,
    SemanticCacheConfig,
)
from semantic_cache.gateway.pool import derive_cache_config, sanitize_model_slug

BASE = SemanticCacheConfig(similarity_threshold=0.85)


def _row(**overrides):
    row = {
        "model": "acme-chat",
        "llm_base_url": "https://up.example.com",
        "llm_api_key": "sk-up",
        "llm_model": "gpt-x",
        "embed_base_url": "https://embed.example.com",
        "embed_api_key": "sk-embed",
        "embed_model": "text-embedding-3-small",
        "extractor_base_url": None,
        "extractor_api_key": None,
        "extractor_model": None,
        "extractor_domain": None,
        "cache_config": {},
    }
    row.update(overrides)
    return row


def test_slug_sanitizes_special_chars() -> None:
    assert sanitize_model_slug("openai/gpt-4o:latest") == "openai_gpt-4o_latest"
    assert sanitize_model_slug("simple-name") == "simple-name"


def test_derived_config_isolates_index_and_prefix_per_embed_model() -> None:
    a = derive_cache_config(BASE, _row(embed_model="embed-a"))
    b = derive_cache_config(BASE, _row(embed_model="embed-b"))
    assert a.cache_index_name != b.cache_index_name
    assert a.key_prefix != b.key_prefix
    assert a.key_prefix.startswith("scache:gw:")
    # Different embed models may have different dims — indexes MUST not share
    # a key prefix, or each index would index the other's vectors.
    assert not a.key_prefix.startswith(b.key_prefix)
    assert not b.key_prefix.startswith(a.key_prefix)


def test_same_embed_model_shares_one_index() -> None:
    # The vector space belongs to the EMBEDDER — two chat models on the same
    # embedding model share the index (projects isolate by scope tag).
    a = derive_cache_config(BASE, _row(model="chat-a"))
    b = derive_cache_config(BASE, _row(model="chat-b"))
    assert a.cache_index_name == b.cache_index_name
    assert a.key_prefix == b.key_prefix


def test_derived_config_uses_api_embeddings_from_row() -> None:
    cfg = derive_cache_config(BASE, _row())
    assert cfg.embedding_provider == EmbeddingProvider.CUSTOM_API
    assert cfg.embedding_base_url == "https://embed.example.com"
    assert cfg.embedding_api_key.get_secret_value() == "sk-embed"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_no_extractor_means_entity_aware_off() -> None:
    cfg = derive_cache_config(BASE, _row())
    assert cfg.entity_aware is False


def test_extractor_present_enables_entity_aware() -> None:
    cfg = derive_cache_config(
        BASE,
        _row(
            extractor_base_url="https://ex.example.com",
            extractor_api_key="sk-ex",
            extractor_model="gpt-mini",
            extractor_domain="medical",
        ),
    )
    assert cfg.entity_aware is True
    assert cfg.domain == CacheDomain.MEDICAL
    assert cfg.entity_model == "gpt-mini"
    assert cfg.entity_llm_base_url == "https://ex.example.com"
    assert cfg.entity_llm_api_key.get_secret_value() == "sk-ex"


def test_extractor_without_usable_domain_degrades_to_plain_caching() -> None:
    # The APP sends no domain field, so an extractor must NOT fail the
    # request — entity-aware caching just stays off.
    for domain in (None, "general"):
        cfg = derive_cache_config(
            BASE,
            _row(
                extractor_base_url="https://ex.example.com",
                extractor_model="gpt-mini",
                extractor_domain=domain,
            ),
        )
        assert cfg.entity_aware is False
        # …and the rest of the config is still honored
        assert cfg.embedding_model == "text-embedding-3-small"


def test_extractor_without_domain_still_applies_cache_overrides() -> None:
    cfg = derive_cache_config(
        BASE,
        _row(
            extractor_base_url="https://ex.example.com",
            extractor_model="gpt-mini",
            extractor_domain=None,
            cache_config={"similarity_threshold": 0.93},
        ),
    )
    assert cfg.entity_aware is False
    assert cfg.similarity_threshold == 0.93


def test_cache_config_overrides_applied_and_validated() -> None:
    cfg = derive_cache_config(
        BASE, _row(cache_config={"similarity_threshold": 0.95, "default_ttl": 60})
    )
    assert cfg.similarity_threshold == 0.95
    assert cfg.default_ttl == 60
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"similarity_threshold": 5.0}))
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"embedding_model": "sneaky"}))


def test_default_cache_mode_is_semantic() -> None:
    cfg = derive_cache_config(BASE, _row())
    assert cfg.cache_mode == CacheMode.SEMANTIC


def test_app_can_select_cache_mode_via_cache_config() -> None:
    # The APP chooses the search method per client through cache_config.
    for value, expected in [
        ("exact", CacheMode.EXACT),
        ("bm25", CacheMode.BM25),
        ("off", CacheMode.OFF),
        ("semantic", CacheMode.SEMANTIC),
    ]:
        cfg = derive_cache_config(BASE, _row(cache_config={"cache_mode": value}))
        assert cfg.cache_mode == expected


def test_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"cache_mode": "telepathy"}))


def test_app_can_select_lexical_scorer_and_min_score() -> None:
    cfg = derive_cache_config(
        BASE,
        _row(cache_config={
            "cache_mode": "bm25",
            "lexical_scorer": "tfidf",     # case-insensitive
            "lexical_min_score": 2.5,
        }),
    )
    assert cfg.cache_mode == CacheMode.BM25
    assert cfg.lexical_scorer == "TFIDF"
    assert cfg.lexical_min_score == 2.5


def test_unknown_lexical_scorer_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_cache_config(
            BASE, _row(cache_config={"lexical_scorer": "magic"})
        )


def test_app_can_select_a_cascade_via_cache_mode_list() -> None:
    # Same field (cache_mode) — a list means an ordered cascade.
    cfg = derive_cache_config(
        BASE, _row(cache_config={"cache_mode": ["exact", "bm25", "semantic"]})
    )
    assert cfg.retrieval_cascade() == [
        CacheMode.EXACT, CacheMode.BM25, CacheMode.SEMANTIC
    ]


def test_cache_mode_list_with_off_or_bad_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"cache_mode": ["exact", "off"]}))
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"cache_mode": ["magic"]}))
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"cache_mode": ["off"]}))


def test_app_can_send_per_method_hyperparameter_blocks() -> None:
    cfg = derive_cache_config(BASE, _row(cache_config={
        "cache_mode": ["bm25", "fuzzy", "semantic"],
        "semantic": {"similarity_threshold": 0.9},
        "bm25": {"scorer": "tfidf", "min_score": 1.0},
        "fuzzy": {"distance": 2, "min_score": 0.5},
    }))
    assert cfg.eff_similarity_threshold() == 0.9
    assert cfg.eff_lexical_scorer(CacheMode.BM25) == "TFIDF"   # normalized
    assert cfg.eff_lexical_min_score(CacheMode.BM25) == 1.0
    assert cfg.eff_fuzzy_distance() == 2
    assert cfg.eff_lexical_min_score(CacheMode.FUZZY) == 0.5


def test_bad_per_method_block_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"semantic": {"similarity_threshold": 2.0}}))
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"fuzzy": {"distance": 9}}))
    with pytest.raises(ValueError):
        derive_cache_config(BASE, _row(cache_config={"bm25": {"bogus": 1}}))


def test_base_config_object_is_not_mutated() -> None:
    before = BASE.model_dump()
    derive_cache_config(BASE, _row(cache_config={"similarity_threshold": 0.99}))
    assert BASE.model_dump() == before
