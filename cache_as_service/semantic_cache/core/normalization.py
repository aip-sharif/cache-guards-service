"""Core text normalization and preprocessing module.

Includes an optional translation layer for cross-lingual cache hits. The
translation backend is `deep-translator` (Google Translate by default) and
is only imported on demand — the package install footprint stays minimal if
translation is never used.

Translations are memoized in-process via an LRU cache so identical queries
do not hit the translation API repeatedly. The cache is bounded; see
`_TRANSLATION_CACHE_SIZE` below.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)

_TRANSLATION_CACHE_SIZE = 1024

# --- Aggressive canonicalization tables (see clean_text(aggressive=True)) --- #
# Persian (۰-۹) and Arabic-Indic (٠-٩) digits → ASCII so "type ۲" == "type 2".
_DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"
)
# Apostrophes are DELETED (not spaced) so contractions join into one token:
# "what's"→"whats" (stopword), "can't"→"cant" (negation stays distinct from "can").
_APOSTROPHE_TABLE = str.maketrans({c: "" for c in "'`’‘"})
# Punctuation → space (no semantic load at cache granularity). Hyphen included so
# "covid-19" == "covid 19".
_STRIP_TABLE = str.maketrans({c: " " for c in ".,;:!?\"()[]{}<>«»“”—–-/\\|"})

# English function words dropped so phrasing-only variants collapse. Conservative:
# articles/copulas/modals/wh-words/prepositions/pronouns/request-verbs. EXCLUDES
# every negation (not/no/never/without/cannot/nor/neither) and any clinical term —
# dropping a negation would serve the opposite answer. Word ORDER is preserved.
_DEFAULT_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "the",
        "is", "are", "am", "was", "were", "be", "been", "being",
        "do", "does", "did",
        "what", "whats", "which", "who", "whom", "whose", "when", "where",
        "why", "how",
        "of", "for", "to", "in", "on", "at", "by", "with", "about", "as",
        "into", "from", "and", "or", "so",
        "i", "me", "my", "we", "our", "us", "you", "your", "it", "its",
        "this", "that", "these", "those",
        "can", "could", "would", "should", "may", "might", "will", "shall",
        "tell", "explain", "describe", "give", "want", "know", "please", "help",
    }
)

# Resolved lazily on first use. Holds a callable `(text, target_lang) -> str`
# or `None` if no backend is available.
_translator_impl: Optional[Callable[[str, str], str]] = None
_translator_resolved: bool = False


def _resolve_translator() -> Optional[Callable[[str, str], str]]:
    """Lazily import and configure a translation backend.

    Returns a callable on first successful resolution; subsequent calls reuse
    the cached result. Returns `None` (logged once at WARNING) when the
    optional `deep-translator` dep is missing.
    """
    global _translator_impl, _translator_resolved
    if _translator_resolved:
        return _translator_impl
    _translator_resolved = True
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        def _translate(text: str, target_lang: str) -> str:
            return GoogleTranslator(source="auto", target=target_lang).translate(text)

        _translator_impl = _translate
        logger.info("Translation backend ready: deep-translator (Google).")
    except ImportError:
        _translator_impl = None
        logger.warning(
            "enable_translation=True but `deep-translator` is not installed; "
            "queries will pass through unchanged. Install with: "
            "pip install 'semantic-cache[translate]'"
        )
    return _translator_impl


@lru_cache(maxsize=_TRANSLATION_CACHE_SIZE)
def _cached_translate(text: str, target_lang: str) -> str:
    """LRU-cached translation; falls back to the input on any error."""
    impl = _resolve_translator()
    if impl is None:
        return text
    try:
        translated = impl(text, target_lang)
        # Translator can return None on empty input; guard for that.
        return translated if isinstance(translated, str) and translated else text
    except Exception as e:  # noqa: BLE001
        # Never break the cache lookup on a translation failure — just log
        # and return the original text. The embedding model will still map
        # cross-lingual paraphrases reasonably for many language pairs.
        logger.warning("Translation failed (%s); using original text.", e)
        return text


class TextNormalizer:
    """Handles text preprocessing and normalization before embedding generation.

    Ensures input queries are sanitized to produce consistent embeddings
    and improve cache hit rates. Supports cross-lingual mapping via either
    a translation pre-step or reliance on a multilingual embedding model.
    """

    # Exposed for tests so they can flush the LRU between runs.
    _translation_cache = _cached_translate

    @staticmethod
    def _standardize_language(text: str, target_lang: str) -> str:
        """Translate `text` into `target_lang` using the configured backend.

        The result is memoized so repeated queries do not re-hit the API. If
        the optional dep is not installed or the call fails, the input is
        returned unchanged (logged) — the cache stays correct, just less
        cross-lingual.
        """
        if not text:
            return text
        return _cached_translate(text, target_lang)

    @staticmethod
    def _canonicalize(text: str, stopwords: FrozenSet[str] = _DEFAULT_STOPWORDS) -> str:
        """Aggressive, order-preserving canonicalization (see clean_text).

        NFKC → digit-fold → lowercase → delete apostrophes → punctuation→space →
        collapse whitespace → drop stop-words. If a query is ENTIRELY stop-words
        ("what is it"), the full token list is kept (else all such queries would
        collapse to one empty key).
        """
        norm = unicodedata.normalize("NFKC", text)
        norm = norm.translate(_DIGIT_MAP)
        norm = norm.lower()
        norm = norm.translate(_APOSTROPHE_TABLE)
        norm = norm.translate(_STRIP_TABLE)
        tokens = norm.split()
        content = [t for t in tokens if t not in stopwords]
        return " ".join(content or tokens)

    @staticmethod
    def clean_text(
        text: str,
        enable_translation: bool = False,
        target_lang: str = "en",
        aggressive: bool = False,
    ) -> str:
        """Sanitizes and normalizes input text.

        - Strips leading/trailing whitespace.
        - Collapses multiple spaces/newlines/tabs into a single space.
        - Optionally translates to a canonical language.
        - Optionally (``aggressive``) applies NFKC / digit-fold / lowercase /
          apostrophe-delete / punctuation-strip / stop-word drop so phrasing-only
          variants collapse to one cache entry.

        Args:
            text: The raw input string.
            enable_translation: Whether to apply language translation to a base language.
            target_lang: The target language code for translation.
            aggressive: Apply the canonicalization pipeline (see above).

        Returns:
            The sanitized string.
        """
        if not text:
            return ""

        # Collapse excessive whitespace and control characters
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Translate to the canonical language FIRST (so aggressive canonicalization
        # then runs on the canonical-language text).
        if enable_translation:
            text = TextNormalizer._standardize_language(text, target_lang)

        if aggressive:
            text = TextNormalizer._canonicalize(text)

        return text

    @staticmethod
    def normalize_query(
        query: str,
        enable_translation: bool = False,
        target_lang: str = "en",
        aggressive: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Normalizes a query and packages it with optional metadata.

        Args:
            query: The raw input string.
            enable_translation: Whether to map the text to a base language first.
            target_lang: Base language code.
            aggressive: Apply aggressive canonicalization (see clean_text).
            **kwargs: Optional metadata mapping (e.g., user_id, module, language).

        Returns:
            A dictionary containing the cleaned text and attached metadata.
        """
        return {
            "text": TextNormalizer.clean_text(query, enable_translation, target_lang, aggressive),
            "metadata": kwargs,
        }
