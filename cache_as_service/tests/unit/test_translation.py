"""Tests for the optional translation layer in TextNormalizer."""

import importlib

import pytest

from semantic_cache.core import normalization


@pytest.fixture(autouse=True)
def reset_translator_state():
    """Force the lazy translator resolver to re-run for each test."""
    normalization._translator_impl = None
    normalization._translator_resolved = False
    normalization._cached_translate.cache_clear()
    yield
    normalization._translator_impl = None
    normalization._translator_resolved = False
    normalization._cached_translate.cache_clear()


def test_disabled_translation_is_passthrough():
    out = normalization.TextNormalizer.clean_text(
        "Hola mundo", enable_translation=False, target_lang="en"
    )
    assert out == "Hola mundo"


def test_missing_dep_falls_back_silently(monkeypatch, caplog):
    """If deep_translator is not importable, translation becomes a no-op
    with a single WARNING log — not a crash."""
    # Force ImportError when deep_translator is imported.
    real_import = __import__

    def fake_import(name, *a, **kw):
        if name == "deep_translator":
            raise ImportError("forced missing dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with caplog.at_level("WARNING", logger="semantic_cache.core.normalization"):
        out = normalization.TextNormalizer.clean_text(
            "Hola mundo", enable_translation=True, target_lang="en"
        )

    assert out == "Hola mundo", "Should pass through when translator unavailable."
    assert any("deep-translator" in r.message for r in caplog.records), (
        "Should warn once about the missing optional dep."
    )


def test_translation_failure_returns_original(monkeypatch):
    """Network error inside the translator must not propagate; original text wins."""
    def broken_translate(text, target_lang):
        raise RuntimeError("network unreachable")

    # Inject a broken backend directly into the resolver cache.
    normalization._translator_impl = broken_translate
    normalization._translator_resolved = True

    out = normalization.TextNormalizer.clean_text(
        "Bonjour", enable_translation=True, target_lang="en"
    )
    assert out == "Bonjour"


def test_translation_results_are_cached():
    """LRU cache should mean the same input is only translated once even
    across many calls."""
    calls = {"n": 0}

    def fake_translate(text, target_lang):
        calls["n"] += 1
        return f"<{text}>"

    normalization._translator_impl = fake_translate
    normalization._translator_resolved = True

    for _ in range(5):
        normalization.TextNormalizer.clean_text(
            "Hola", enable_translation=True, target_lang="en"
        )

    assert calls["n"] == 1, "Repeated identical input must not re-call the translator."
