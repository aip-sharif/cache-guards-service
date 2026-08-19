"""Embedding generation abstractions and factories.

Supports local HuggingFace models and remote API providers, sync and async.

Dimension handling: the API backend no longer hardcodes 1536. If
``dimension`` is not supplied it is **auto-detected** from the provider on
first use (one probe embedding), so the Redis vector index is always created
with the dimension the model actually returns — no more reaching into a
private ``_dimension`` from the outside.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from semantic_cache.core.config import EmbeddingProvider, SemanticCacheConfig
from semantic_cache.core.exceptions import EmbeddingGenerationError
from semantic_cache.core.retry import acall_with_retries, call_with_retries

logger = logging.getLogger(__name__)

try:
    import httpx  # type: ignore

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False


def _embeddings_endpoint(base_url: str) -> str:
    """Resolve the OpenAI-compatible embeddings endpoint.

    Appends ``/v1/embeddings`` unless the URL already ends with ``embeddings``
    (so callers can pass the full endpoint to avoid a ``/v1/v1`` duplication).
    """
    base = base_url.rstrip("/")
    return base if base.endswith("embeddings") else f"{base}/v1/embeddings"


class BaseEmbeddingManager(ABC):
    """Abstract base class for all (sync) embedding managers."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate a dense vector embedding for the given text.

        Raises:
            EmbeddingGenerationError: If the embedding cannot be generated.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the generated embeddings."""


class HuggingFaceEmbeddingManager(BaseEmbeddingManager):
    """Generates embeddings locally using sentence-transformers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy loads the model to save memory if the cache is never missed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local HF model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                dummy = self._model.encode("test")
                self._dimension = len(dummy)
            except ImportError as e:
                raise EmbeddingGenerationError(
                    "sentence-transformers is not installed. Install it to use "
                    "HuggingFace models (pip install 'semantic-cache[embeddings]')."
                ) from e
            except Exception as e:
                raise EmbeddingGenerationError(f"Failed to load HuggingFace model: {e}") from e

    def get_embedding(self, text: str) -> List[float]:
        self._load_model()
        try:
            vector = self._model.encode(text)
            return vector.tolist()
        except Exception as e:
            raise EmbeddingGenerationError(f"HF Encoding failed: {e}") from e

    def get_dimension(self) -> int:
        self._load_model()
        return self._dimension


class APIEmbeddingManager(BaseEmbeddingManager):
    """Generates embeddings via an OpenAI-compatible REST API.

    ``dimension`` may be left ``None`` to auto-detect from the provider on
    first use; pass it explicitly to skip the probe call.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        dimension: Optional[int] = None,
        timeout: int = 10,
        max_retries: int = 2,
        backoff_base: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # None → auto-detect lazily; an int → trust the caller, skip the probe.
        self._dimension: Optional[int] = dimension

    def get_embedding(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"input": text, "model": self.model_name}
        endpoint = _embeddings_endpoint(self.base_url)

        def _do() -> List[float]:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        try:
            return call_with_retries(
                _do, retries=self.max_retries, backoff_base=self.backoff_base
            )
        except Exception as e:
            raise EmbeddingGenerationError(f"API Embedding failed: {e}") from e

    def get_dimension(self) -> int:
        if self._dimension is None:
            # One-time probe so the vector index matches the model's real DIM.
            vector = self.get_embedding("dimension probe")
            self._dimension = len(vector)
            logger.info("Auto-detected embedding dimension: %d", self._dimension)
        return self._dimension


class BaseAsyncEmbeddingManager(ABC):
    """Async embedding backend used by the manager's ``asearch``/``aset``.

    Only ``get_embedding`` is async — the index dimension is taken from the
    (always-present) sync embedding manager at index-creation time.
    """

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Async embedding. Raises EmbeddingGenerationError on failure."""


class AsyncAPIEmbeddingManager(BaseAsyncEmbeddingManager):
    """Async embeddings via an OpenAI-compatible REST API (httpx)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 10,
        max_retries: int = 2,
        backoff_base: float = 0.5,
    ):
        if not _HTTPX_AVAILABLE:
            raise EmbeddingGenerationError(
                "AsyncAPIEmbeddingManager requires httpx. "
                "Install with: pip install 'semantic-cache[async]'"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client: Optional["httpx.AsyncClient"] = None

    async def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_embedding(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"input": text, "model": self.model_name}
        endpoint = _embeddings_endpoint(self.base_url)
        client = await self._get_client()

        async def _do() -> List[float]:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        try:
            return await acall_with_retries(
                _do, retries=self.max_retries, backoff_base=self.backoff_base
            )
        except Exception as e:
            raise EmbeddingGenerationError(f"Async API Embedding failed: {e}") from e


class EmbeddingManagerFactory:
    """Factory for the (sync) embedding backend based on config."""

    @staticmethod
    def create(config: SemanticCacheConfig) -> BaseEmbeddingManager:
        if config.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddingManager(model_name=config.embedding_model)

        if not config.embedding_base_url:
            raise EmbeddingGenerationError(
                f"base_url must be provided for {config.embedding_provider} provider."
            )
        api_key = (
            config.embedding_api_key.get_secret_value()
            if config.embedding_api_key
            else ""
        )
        return APIEmbeddingManager(
            base_url=config.embedding_base_url,
            api_key=api_key,
            model_name=config.embedding_model,
            dimension=config.embedding_dim,
            timeout=config.embedding_timeout,
            max_retries=config.http_max_retries,
            backoff_base=config.http_backoff_base,
        )


class AsyncEmbeddingManagerFactory:
    """Factory for the optional async embedding backend.

    Returns ``None`` for the local HuggingFace provider (no async story — the
    manager falls back to running the sync embedder in a thread).
    """

    @staticmethod
    def create(config: SemanticCacheConfig) -> Optional[BaseAsyncEmbeddingManager]:
        if config.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            return None
        if not config.embedding_base_url:
            return None
        api_key = (
            config.embedding_api_key.get_secret_value()
            if config.embedding_api_key
            else ""
        )
        return AsyncAPIEmbeddingManager(
            base_url=config.embedding_base_url,
            api_key=api_key,
            model_name=config.embedding_model,
            timeout=config.embedding_timeout,
            max_retries=config.http_max_retries,
            backoff_base=config.http_backoff_base,
        )
