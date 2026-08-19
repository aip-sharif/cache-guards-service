"""Entity extraction abstractions for domain-aware caching.

Calls an OpenAI-compatible chat completion endpoint with a domain-specific
prompt and parses the structured JSON response.

The extractor is intentionally fail-soft: any error — network, timeout,
malformed JSON, missing fields — surfaces as `EntityExtractionError`.
Callers must treat this as a cache MISS, never a HIT, so high-stakes
domains never serve a false match when extraction breaks down.

Portability notes:
* OpenAI's `response_format={"type": "json_object"}` is supported by OpenAI
  and Azure OpenAI but NOT by every OpenAI-compatible gateway (vLLM, some
  Anthropic-compat shims). Toggle off via `config.entity_use_json_mode=False`.
* When JSON mode is off, the tolerant parser extracts the first balanced
  JSON object from the response, so markdown fences and prose preambles do
  not break extraction.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from semantic_cache.core.config import CacheDomain, SemanticCacheConfig
from semantic_cache.core.exceptions import EntityExtractionError
from semantic_cache.core.retry import call_with_retries

logger = logging.getLogger(__name__)


# Domain-specific extraction prompts. Each prompt instructs the LLM to return
# a strict JSON object so we can parse without regex heuristics.
# The `identifier` is hashed to compare entity sets across queries, so the LLM is
# asked to emit a CANONICAL lowercase concept name — collapsing synonyms, brand
# names, and abbreviations onto one form (using its own medical knowledge), which
# is what lets paraphrases like "high blood pressure" and "hypertension" match.
_MEDICAL_PROMPT = """You extract medical entities from a user query.

Allowed entity types: DRUG, CONDITION, PROCEDURE, ANATOMY, DOSAGE.

ALWAYS extract every specific named drug, condition, procedure, and body part
mentioned, even when the query is written in another language (translate it).

CANONICAL IDENTIFIER — the most important rule. The `identifier` is hashed to
decide whether two queries are about the same thing, so for EACH entity output
the SINGLE most standard biomedical name for that concept, in lowercase, and
collapse EVERY way of referring to the same concept onto that one identifier
(use your medical knowledge — do not just copy the surface words):
- Synonyms / lay terms -> standard medical name:
  "high blood pressure" / "elevated blood pressure" / "htn" -> "hypertension";
  "heart attack" -> "myocardial infarction"; "sugar disease" / "مرض قند" ->
  "diabetes"; "water pills" -> "diuretic".
- Brand name -> generic drug: "tylenol" / "panadol" -> "acetaminophen";
  "glucophage" -> "metformin"; "advil" -> "ibuprofen".
- Abbreviation -> full standard term: "t2dm" -> "type 2 diabetes";
  "copd" -> "chronic obstructive pulmonary disease"; "uti" ->
  "urinary tract infection".
- Lowercase, singular, and drop non-essential qualifiers.
- BUT keep clinically essential distinctions and NEVER merge opposites or
  different specificities: "type 1 diabetes" != "type 2 diabetes";
  "hypertension" != "hypotension"; "hyperthyroidism" != "hypothyroidism".

Only emit a DOSAGE entity when a concrete quantity AND unit appear together
(e.g. "500 mg", "1000 mcg"). NEVER extract bare generic words such as "dose",
"dosage", "amount", "treatment", "medication", or "symptom".

Be deterministic: paraphrases of the same question must yield the same set of
canonical identifiers. Output ONLY valid JSON (no prose, no markdown fences):
{"entities": [{"text": "<as written>", "type": "<allowed type>", "identifier": "<canonical lowercase name>"}]}

Examples:
Query: "What helps with high blood pressure?"
{"entities": [{"text": "high blood pressure", "type": "CONDITION", "identifier": "hypertension"}]}

Query: "راه های درمان پرفشاری خون چیست؟"
{"entities": [{"text": "پرفشاری خون", "type": "CONDITION", "identifier": "hypertension"}]}

Query: "Is Tylenol safe to take with T2DM?"
{"entities": [{"text": "Tylenol", "type": "DRUG", "identifier": "acetaminophen"}, {"text": "T2DM", "type": "CONDITION", "identifier": "type 2 diabetes"}]}

Query: "Metformin 500 mg dosing"
{"entities": [{"text": "Metformin", "type": "DRUG", "identifier": "metformin"}, {"text": "500 mg", "type": "DOSAGE", "identifier": "500 mg"}]}

Query:
"""

_LEGAL_PROMPT = """You extract legal entities from a user query.

Allowed entity types: STATUTE, CASE, JURISDICTION, PARTY, DATE.

ALWAYS extract every specific named statute, case, jurisdiction, party, and
date mentioned. Preserve the FULL identifier: "42 U.S.C. § 1983" not "1983",
"Brown v. Board of Education" not "Brown". Use the canonical English form even
if the query is in another language.

NEVER extract bare generic words such as "law", "statute", "case", "court",
"claim", "lawsuit", or "rights" unless they are part of a specific named
citation, case name, or court name.

Be deterministic: paraphrases of the same question must yield the same set.
Output ONLY valid JSON (no prose, no markdown fences) in this schema:
{"entities": [{"text": "<as written>", "type": "<allowed type>", "identifier": "<canonical full form>"}]}

Examples:
Query: "What is the standing requirement under 42 U.S.C. § 1983?"
{"entities": [{"text": "42 U.S.C. § 1983", "type": "STATUTE", "identifier": "42 U.S.C. § 1983"}]}

Query: "Did Miranda v. Arizona apply in California?"
{"entities": [{"text": "Miranda v. Arizona", "type": "CASE", "identifier": "Miranda v. Arizona"}, {"text": "California", "type": "JURISDICTION", "identifier": "California"}]}

Query:
"""

_DOMAIN_PROMPTS: Dict[CacheDomain, str] = {
    CacheDomain.MEDICAL: _MEDICAL_PROMPT,
    CacheDomain.LEGAL: _LEGAL_PROMPT,
}


# Matches the FIRST balanced JSON object in a string.
# Greedy but stops at the matching `}`; works for the small flat structure
# our prompt requests (no deeply nested arrays of objects beyond one level).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _tolerant_json_extract(content: str) -> dict:
    """Parses LLM output that may be wrapped in markdown fences or prose.

    Strategy:
        1. Strip surrounding whitespace.
        2. If the content is itself a JSON document, return it directly.
        3. Otherwise locate the first `{...}` span via regex and parse it.

    Raises `EntityExtractionError` if no valid JSON object can be recovered.
    """
    if not content:
        raise EntityExtractionError("LLM returned empty content.")
    stripped = content.strip()
    # Strip common markdown fences (```json ... ``` or ``` ... ```).
    if stripped.startswith("```"):
        # Remove the opening fence and an optional language tag.
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[: -3].rstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(stripped)
    if not match:
        raise EntityExtractionError(
            "Could not locate a JSON object in the LLM response."
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise EntityExtractionError(f"Malformed JSON in LLM response: {e}") from e


def _coerce_entities(parsed: Any) -> List[Dict[str, str]]:
    """Validates and normalizes the `entities` field of the parsed payload."""
    if not isinstance(parsed, dict):
        raise EntityExtractionError("LLM payload was not a JSON object.")
    raw_entities = parsed.get("entities", [])
    if not isinstance(raw_entities, list):
        raise EntityExtractionError("'entities' is not a list.")

    entities: List[Dict[str, str]] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        identifier = item.get("identifier") or item.get("text")
        etype = item.get("type")
        etext = item.get("text", identifier)
        if not identifier or not etype:
            continue
        entities.append({
            "text": str(etext),
            "type": str(etype),
            "identifier": str(identifier),
        })
    return entities


class BaseEntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    def extract(self, text: str) -> List[Dict[str, str]]:
        """Extracts a list of entities from the given text.

        Args:
            text: The (normalized) input query.

        Returns:
            A list of `{text, type, identifier}` dicts. Empty list is valid
            and means "no entities found" — distinct from extraction failure.

        Raises:
            EntityExtractionError: On any failure. Callers must treat as MISS.
        """
        pass


def _resolve_prompt(domain: CacheDomain) -> str:
    """Common prompt-lookup helper, used by both sync and async extractors."""
    if domain not in _DOMAIN_PROMPTS:
        raise EntityExtractionError(
            f"No extraction prompt registered for domain '{domain}'."
        )
    return _DOMAIN_PROMPTS[domain]


def _build_payload(
    model_name: str,
    system_prompt: str,
    user_text: str,
    use_json_mode: bool,
) -> dict:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _resolve_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("chat/completions") else f"{base}/v1/chat/completions"


class LLMEntityExtractor(BaseEntityExtractor):
    """Calls an OpenAI-compatible chat completion endpoint for entity extraction."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        domain: CacheDomain,
        timeout: int = 10,
        use_json_mode: bool = True,
        max_retries: int = 2,
        backoff_base: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.domain = domain
        self.timeout = timeout
        self.use_json_mode = use_json_mode
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._prompt = _resolve_prompt(domain)

    def extract(self, text: str) -> List[Dict[str, str]]:
        endpoint = _resolve_endpoint(self.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = _build_payload(self.model_name, self._prompt, text, self.use_json_mode)

        def _do() -> str:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        try:
            content = call_with_retries(
                _do, retries=self.max_retries, backoff_base=self.backoff_base
            )
        except Exception as e:
            raise EntityExtractionError(f"LLM call failed: {e}") from e

        parsed = _tolerant_json_extract(content)
        return _coerce_entities(parsed)


class EntityExtractorFactory:
    """Factory to instantiate the configured entity extractor."""

    @staticmethod
    def create(config: SemanticCacheConfig) -> Optional[BaseEntityExtractor]:
        """Builds an extractor based on config, or returns None when disabled.

        Returns None when `entity_aware` is False so the cache manager can
        short-circuit cleanly.
        """
        if not config.entity_aware:
            return None

        if config.domain == CacheDomain.GENERAL:
            # General domain has no entity prompt — entity-aware mode requires
            # a specialized domain to be meaningful.
            raise EntityExtractionError(
                "entity_aware=True requires domain to be 'medical' or 'legal'."
            )

        base_url = config.entity_llm_base_url or config.embedding_base_url
        if not base_url:
            raise EntityExtractionError(
                "entity_llm_base_url (or embedding_base_url as fallback) must be set."
            )

        api_key_secret = config.entity_llm_api_key or config.embedding_api_key
        api_key = api_key_secret.get_secret_value() if api_key_secret else ""

        return LLMEntityExtractor(
            base_url=base_url,
            api_key=api_key,
            model_name=config.entity_model,
            domain=config.domain,
            timeout=config.entity_llm_timeout,
            use_json_mode=config.entity_use_json_mode,
            max_retries=config.http_max_retries,
            backoff_base=config.http_backoff_base,
        )
