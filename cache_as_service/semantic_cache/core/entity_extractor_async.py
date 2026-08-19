"""Async variants of the entity extractor.

Used by `SemanticCacheManager.asearch` / `.aset` and by the async FastAPI
endpoints to avoid blocking the event loop on the LLM round trip.

`httpx` is an OPTIONAL dependency — install with `pip install
'semantic-cache[async]'` or `pip install httpx`. If it is missing the
factory below raises a clear error at construction; the sync extractor
path continues to work.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from semantic_cache.core.config import CacheDomain, SemanticCacheConfig
from semantic_cache.core.entity_extractor import (
    _build_payload,
    _coerce_entities,
    _resolve_endpoint,
    _resolve_prompt,
    _tolerant_json_extract,
)
from semantic_cache.core.exceptions import EntityExtractionError
from semantic_cache.core.retry import acall_with_retries

logger = logging.getLogger(__name__)

try:
    import httpx  # type: ignore
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class BaseAsyncEntityExtractor(ABC):
    """Async counterpart to `BaseEntityExtractor`."""

    @abstractmethod
    async def extract(self, text: str) -> List[Dict[str, str]]:
        """Async extract — same contract as the sync version.

        Raises:
            EntityExtractionError: on any failure. Callers must treat as MISS.
        """
        raise NotImplementedError


class AsyncLLMEntityExtractor(BaseAsyncEntityExtractor):
    """Calls an OpenAI-compatible chat completion endpoint using `httpx.AsyncClient`.

    The client is created lazily on first use so this class is cheap to
    construct and safe to import even when httpx is missing (you only crash
    if you actually try to call `extract`).
    """

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
        if not _HTTPX_AVAILABLE:
            raise EntityExtractionError(
                "AsyncLLMEntityExtractor requires httpx. "
                "Install with: pip install 'semantic-cache[async]'"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.domain = domain
        self.timeout = timeout
        self.use_json_mode = use_json_mode
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._prompt = _resolve_prompt(domain)
        self._client: Optional["httpx.AsyncClient"] = None

    async def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client; safe to call multiple times."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def extract(self, text: str) -> List[Dict[str, str]]:
        endpoint = _resolve_endpoint(self.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = _build_payload(self.model_name, self._prompt, text, self.use_json_mode)

        client = await self._get_client()

        async def _do() -> str:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        try:
            content = await acall_with_retries(
                _do, retries=self.max_retries, backoff_base=self.backoff_base
            )
        except Exception as e:  # noqa: BLE001
            raise EntityExtractionError(f"Async LLM call failed: {e}") from e

        parsed = _tolerant_json_extract(content)
        return _coerce_entities(parsed)


class AsyncEntityExtractorFactory:
    """Factory for async extractors. Mirrors `EntityExtractorFactory` shape."""

    @staticmethod
    def create(config: SemanticCacheConfig) -> Optional[BaseAsyncEntityExtractor]:
        if not config.entity_aware:
            return None

        if config.domain == CacheDomain.GENERAL:
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

        return AsyncLLMEntityExtractor(
            base_url=base_url,
            api_key=api_key,
            model_name=config.entity_model,
            domain=config.domain,
            timeout=config.entity_llm_timeout,
            use_json_mode=config.entity_use_json_mode,
            max_retries=config.http_max_retries,
            backoff_base=config.http_backoff_base,
        )
