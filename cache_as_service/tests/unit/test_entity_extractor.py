"""Unit tests for the entity extractor and factory.

No Redis, no real embeddings — just HTTP mocking. These tests verify that
parsing, error handling, and factory branching all behave correctly.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from semantic_cache.core.config import (
    CacheDomain,
    EmbeddingProvider,
    SemanticCacheConfig,
)
from semantic_cache.core.entity_extractor import (
    EntityExtractorFactory,
    LLMEntityExtractor,
)
from semantic_cache.core.exceptions import EntityExtractionError


def _mock_response(content: str, status_code: int = 200) -> MagicMock:
    """Build a fake `requests.Response` returning the given chat-completion content."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


# -- LLMEntityExtractor ------------------------------------------------------


def test_extractor_parses_well_formed_response():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="sk-test",
        model_name="gpt-4o-mini",
        domain=CacheDomain.MEDICAL,
    )
    canned = json.dumps({
        "entities": [
            {"text": "Vitamin B12", "type": "DRUG", "identifier": "Vitamin B12"},
            {"text": "type 2 diabetes", "type": "CONDITION", "identifier": "Type 2 Diabetes"},
        ]
    })

    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(canned)) as mock_post:
        out = extractor.extract("Vitamin B12 in type 2 diabetes")

    assert out == [
        {"text": "Vitamin B12", "type": "DRUG", "identifier": "Vitamin B12"},
        {"text": "type 2 diabetes", "type": "CONDITION", "identifier": "Type 2 Diabetes"},
    ]

    # Verify the call shape.
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    payload = kwargs["json"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    # The system prompt is the medical one.
    assert "medical entities" in payload["messages"][0]["content"].lower()


def test_extractor_handles_empty_entities():
    """An empty entity list is valid output — must not raise."""
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(json.dumps({"entities": []}))):
        out = extractor.extract("hello")
    assert out == []


def test_extractor_skips_malformed_items_but_keeps_valid_ones():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
    )
    canned = json.dumps({
        "entities": [
            "not a dict",  # dropped
            {"identifier": "Aspirin"},  # dropped: missing type
            {"type": "DRUG"},  # dropped: missing identifier and text
            {"text": "Ibuprofen", "type": "DRUG", "identifier": "Ibuprofen"},  # kept
        ]
    })
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(canned)):
        out = extractor.extract("painkillers")
    assert out == [{"text": "Ibuprofen", "type": "DRUG", "identifier": "Ibuprofen"}]


def test_extractor_falls_back_text_to_identifier_when_missing():
    """If `text` is missing but `identifier` is present, identifier doubles as text."""
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
    )
    canned = json.dumps({"entities": [{"type": "DRUG", "identifier": "Metformin"}]})
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(canned)):
        out = extractor.extract("metformin")
    assert out == [{"text": "Metformin", "type": "DRUG", "identifier": "Metformin"}]


def test_extractor_raises_on_http_error():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    bad = MagicMock()
    bad.raise_for_status = MagicMock(side_effect=Exception("500 Server Error"))
    with patch("semantic_cache.core.entity_extractor.requests.post", return_value=bad):
        with pytest.raises(EntityExtractionError):
            extractor.extract("anything")


def test_extractor_raises_on_malformed_json():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response("not json at all {")):
        with pytest.raises(EntityExtractionError):
            extractor.extract("anything")


def test_extractor_raises_when_entities_field_wrong_type():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    canned = json.dumps({"entities": "should-be-a-list"})
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(canned)):
        with pytest.raises(EntityExtractionError):
            extractor.extract("anything")


def test_extractor_raises_on_network_timeout():
    import requests as _req
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               side_effect=_req.exceptions.Timeout("timed out")):
        with pytest.raises(EntityExtractionError):
            extractor.extract("anything")


def test_legal_domain_uses_legal_prompt():
    extractor = LLMEntityExtractor(
        base_url="https://fake.example.com",
        api_key="k", model_name="m", domain=CacheDomain.LEGAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(json.dumps({"entities": []}))) as mock_post:
        extractor.extract("any text")
    payload = mock_post.call_args.kwargs["json"]
    assert "legal entities" in payload["messages"][0]["content"].lower()


def test_endpoint_url_construction():
    """When base URL is just the API host, /v1/chat/completions is appended."""
    extractor = LLMEntityExtractor(
        base_url="https://api.openai.com/",  # trailing slash should be trimmed
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(json.dumps({"entities": []}))) as mock_post:
        extractor.extract("x")
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://api.openai.com/v1/chat/completions"


def test_endpoint_url_respects_full_path_override():
    """If the caller already pointed at /chat/completions, don't append again."""
    extractor = LLMEntityExtractor(
        base_url="https://gateway.local/chat/completions",
        api_key="k", model_name="m", domain=CacheDomain.MEDICAL,
    )
    with patch("semantic_cache.core.entity_extractor.requests.post",
               return_value=_mock_response(json.dumps({"entities": []}))) as mock_post:
        extractor.extract("x")
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://gateway.local/chat/completions"


# -- EntityExtractorFactory --------------------------------------------------


def _base_config(**overrides) -> SemanticCacheConfig:
    """Build a config with sane defaults; overrides patch specific fields."""
    base = dict(
        redis_host="localhost",
        redis_port=6379,
        embedding_provider=EmbeddingProvider.HUGGINGFACE,
        embedding_model="x",
    )
    base.update(overrides)
    return SemanticCacheConfig(**base)


def test_factory_returns_none_when_disabled():
    cfg = _base_config(entity_aware=False)
    assert EntityExtractorFactory.create(cfg) is None


def test_factory_rejects_general_domain_when_aware():
    cfg = _base_config(
        entity_aware=True,
        domain=CacheDomain.GENERAL,
        entity_llm_base_url="https://x",
        entity_llm_api_key=SecretStr("k"),
    )
    with pytest.raises(EntityExtractionError):
        EntityExtractorFactory.create(cfg)


def test_factory_requires_base_url_either_dedicated_or_embedding():
    cfg = _base_config(
        entity_aware=True,
        domain=CacheDomain.MEDICAL,
        # Neither entity_llm_base_url nor embedding_base_url set.
    )
    with pytest.raises(EntityExtractionError):
        EntityExtractorFactory.create(cfg)


def test_factory_falls_back_to_embedding_base_url():
    cfg = _base_config(
        entity_aware=True,
        domain=CacheDomain.MEDICAL,
        embedding_base_url="https://api.example.com",
        embedding_api_key=SecretStr("sk-embed"),
    )
    out = EntityExtractorFactory.create(cfg)
    assert isinstance(out, LLMEntityExtractor)
    assert out.base_url == "https://api.example.com"
    assert out.api_key == "sk-embed"
    assert out.domain == CacheDomain.MEDICAL


def test_factory_prefers_dedicated_entity_url_over_embedding():
    cfg = _base_config(
        entity_aware=True,
        domain=CacheDomain.LEGAL,
        embedding_base_url="https://embed.example.com",
        embedding_api_key=SecretStr("sk-embed"),
        entity_llm_base_url="https://llm.example.com",
        entity_llm_api_key=SecretStr("sk-llm"),
        entity_model="gpt-4o-mini",
    )
    out = EntityExtractorFactory.create(cfg)
    assert isinstance(out, LLMEntityExtractor)
    assert out.base_url == "https://llm.example.com"
    assert out.api_key == "sk-llm"
    assert out.model_name == "gpt-4o-mini"
    assert out.domain == CacheDomain.LEGAL
