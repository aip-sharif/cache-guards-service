"""Tests for the tolerant JSON parser used when JSON-mode is unavailable."""

import json
from unittest.mock import MagicMock, patch

import pytest

from semantic_cache.core.config import CacheDomain
from semantic_cache.core.entity_extractor import (
    LLMEntityExtractor,
    _tolerant_json_extract,
)
from semantic_cache.core.exceptions import EntityExtractionError


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# -- Parser unit tests --


def test_parses_plain_json_object():
    assert _tolerant_json_extract('{"a": 1}') == {"a": 1}


def test_parses_markdown_fenced_json():
    raw = '```json\n{"entities": []}\n```'
    assert _tolerant_json_extract(raw) == {"entities": []}


def test_parses_unlabelled_fence():
    raw = '```\n{"x": 2}\n```'
    assert _tolerant_json_extract(raw) == {"x": 2}


def test_extracts_json_from_prose_preamble():
    raw = 'Here is the result you asked for:\n{"entities": [{"type": "DRUG", "identifier": "Ibuprofen"}]}\nThanks!'
    out = _tolerant_json_extract(raw)
    assert out["entities"][0]["identifier"] == "Ibuprofen"


def test_raises_on_empty():
    with pytest.raises(EntityExtractionError):
        _tolerant_json_extract("")


def test_raises_when_no_json_present():
    with pytest.raises(EntityExtractionError):
        _tolerant_json_extract("definitely not json at all")


def test_raises_on_malformed_json_inside_fence():
    with pytest.raises(EntityExtractionError):
        _tolerant_json_extract("```json\n{not real json}\n```")


# -- Integration with LLMEntityExtractor: json_mode toggle --


def test_extractor_omits_response_format_when_json_mode_off():
    extractor = LLMEntityExtractor(
        base_url="https://x",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
        use_json_mode=False,
    )
    with patch(
        "semantic_cache.core.entity_extractor.requests.post",
        return_value=_mock_response(json.dumps({"entities": []})),
    ) as post:
        extractor.extract("anything")

    payload = post.call_args.kwargs["json"]
    assert "response_format" not in payload, (
        "json_mode=False must not send the OpenAI-only response_format field."
    )


def test_extractor_sends_response_format_when_json_mode_on():
    extractor = LLMEntityExtractor(
        base_url="https://x",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
        use_json_mode=True,
    )
    with patch(
        "semantic_cache.core.entity_extractor.requests.post",
        return_value=_mock_response(json.dumps({"entities": []})),
    ) as post:
        extractor.extract("anything")
    assert post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}


def test_extractor_parses_markdown_fenced_response_in_non_json_mode():
    """Realistic gateway scenario: vLLM with no response_format, model
    returns markdown-fenced JSON."""
    extractor = LLMEntityExtractor(
        base_url="https://x",
        api_key="k",
        model_name="m",
        domain=CacheDomain.MEDICAL,
        use_json_mode=False,
    )
    canned = '```json\n{"entities": [{"text": "Vitamin B12", "type": "DRUG", "identifier": "Vitamin B12"}]}\n```'
    with patch(
        "semantic_cache.core.entity_extractor.requests.post",
        return_value=_mock_response(canned),
    ):
        out = extractor.extract("vitamin b12")
    assert out == [{"text": "Vitamin B12", "type": "DRUG", "identifier": "Vitamin B12"}]
