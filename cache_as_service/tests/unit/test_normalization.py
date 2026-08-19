"""Unit tests for TextNormalizer (no Redis / no network needed)."""

from semantic_cache.core.normalization import TextNormalizer


def test_default_only_collapses_whitespace() -> None:
    # Non-aggressive default is unchanged: whitespace collapse + strip only.
    assert TextNormalizer.clean_text("  Hello   World  ") == "Hello World"
    assert TextNormalizer.clean_text("What is Diabetes?") == "What is Diabetes?"


def test_aggressive_lowercases_and_strips_punctuation() -> None:
    assert TextNormalizer.clean_text("What is Diabetes?", aggressive=True) == "diabetes"


def test_aggressive_collapses_phrasing_variants() -> None:
    a = TextNormalizer.clean_text("what is diabetes", aggressive=True)
    b = TextNormalizer.clean_text("what's diabetes?", aggressive=True)
    c = TextNormalizer.clean_text("tell me about diabetes", aggressive=True)
    assert a == b == c == "diabetes"


def test_aggressive_folds_persian_digits() -> None:
    assert TextNormalizer.clean_text("type ۲ diabetes", aggressive=True) == \
        TextNormalizer.clean_text("type 2 diabetes", aggressive=True)


def test_aggressive_hyphen_and_case_collapse() -> None:
    assert TextNormalizer.clean_text("COVID-19", aggressive=True) == \
        TextNormalizer.clean_text("covid 19", aggressive=True)


def test_aggressive_preserves_negation() -> None:
    # "not"/"cannot" are NOT stopwords — dropping them would serve the opposite answer.
    assert "not" in TextNormalizer.clean_text("is it not safe", aggressive=True).split()


def test_aggressive_preserves_word_order() -> None:
    # Comparative questions must keep distinct keys (order preserved, not sorted).
    x = TextNormalizer.clean_text("is aspirin better than ibuprofen", aggressive=True)
    y = TextNormalizer.clean_text("is ibuprofen better than aspirin", aggressive=True)
    assert x != y


def test_aggressive_all_stopwords_keeps_tokens() -> None:
    # A query that is entirely stopwords must not collapse to empty.
    assert TextNormalizer.clean_text("what is it", aggressive=True) != ""


def test_normalize_query_threads_aggressive() -> None:
    out = TextNormalizer.normalize_query("What is Diabetes?", aggressive=True)
    assert out["text"] == "diabetes"


def test_empty_text() -> None:
    assert TextNormalizer.clean_text("", aggressive=True) == ""
