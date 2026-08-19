"""FastAPI HTTP wrapper for Semantic Caching.

Exposes REST endpoints to interact with the Semantic Cache directly,
allowing non-Python services to leverage the vector cache.

Endpoints are `async def` and call the manager's `asearch` / `aset` methods
so the event loop is not blocked while waiting on the embedding model,
Redis, or the entity-extraction LLM. When the manager has an async
extractor wired in, the LLM round trip is awaited natively; otherwise the
sync path is wrapped via `asyncio.to_thread` (see `SemanticCacheManager`).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from semantic_cache.core import metrics
from semantic_cache.core.cache_manager import SemanticCacheManager


def get_cache_manager() -> SemanticCacheManager:
    """Dependency override point for FastAPI.

    Must be set in main.py:
    app.dependency_overrides[get_cache_manager] = lambda: my_active_cache_manager
    """
    raise NotImplementedError("Dependency get_cache_manager must be overridden.")


router = APIRouter(prefix="/cache", tags=["Semantic Cache"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="The input text to search the cache for.")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata to filter by or track."
    )


class SearchResponse(BaseModel):
    hit: bool
    response: Optional[str] = None
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SetRequest(BaseModel):
    query: str = Field(..., description="The trigger question or input text.")
    response: str = Field(..., description="The target LLM response to cache.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata tags (e.g. model version, user ID)."
    )
    ttl: Optional[int] = Field(
        default=None,
        description="Override the default Time-to-Live (in seconds) for this specific entry.",
    )
    keep_forever: bool = Field(
        default=False, description="Whether to store this entry permanently (no expiration)."
    )


class GenericResponse(BaseModel):
    status: str
    message: str


@router.post("/search", response_model=SearchResponse)
async def search_cache(
    request: SearchRequest,
    manager: SemanticCacheManager = Depends(get_cache_manager),
):
    """Query the semantic cache for a vector match."""
    try:
        filters = request.metadata_filters or {}
        result = await manager.asearch(request.query, **filters)

        if result:
            return SearchResponse(
                hit=True,
                response=result["response"],
                similarity=result["similarity"],
                metadata=result["metadata"],
            )

        return SearchResponse(hit=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache search failed: {str(e)}")


@router.post("/set", response_model=GenericResponse)
async def set_cache(
    request: SetRequest,
    manager: SemanticCacheManager = Depends(get_cache_manager),
):
    """Manually insert an entry into the semantic cache."""
    try:
        await manager.aset(
            query=request.query,
            response=request.response,
            metadata=request.metadata,
            ttl=request.ttl,
            keep_forever=request.keep_forever,
        )
        return GenericResponse(status="success", message="Entry cached successfully.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache set failed: {str(e)}")


@router.delete("/purge", response_model=GenericResponse)
async def purge_cache(manager: SemanticCacheManager = Depends(get_cache_manager)):
    """Clear all entries from the semantic cache."""
    try:
        manager.purge()
        return GenericResponse(status="success", message="Cache fully purged.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache purge failed: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
def get_metrics(manager: SemanticCacheManager = Depends(get_cache_manager)):
    """Retrieve cache health metrics.

    See also `/cache/prometheus` for the standard Prometheus text-exposition
    endpoint suitable for scraping.
    """
    try:
        start = manager.redis.time()
        manager.redis.ping()
        latency = manager.redis.time()[0] - start[0]

        def _f(d: Dict[str, Any], key: str) -> float:
            try:
                return float(d.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        try:
            info = manager.redis.ft(manager.active_index_name).info()
            num_docs = info.get("num_docs", "unknown")
            # RediSearch reports vector storage under these keys (the older
            # inverted_sz_mb / vector_space_sz_mb names were always ~0 here).
            size_mb = (
                _f(info, "vector_index_sz_mb")
                + _f(info, "offset_vectors_sz_mb")
                + _f(info, "inverted_sz_mb")
            )
        except Exception:
            num_docs = "Index not yet created or inaccessible"
            size_mb = 0

        cfg = manager.config
        active_threshold = (
            cfg.entity_threshold if cfg.entity_aware else cfg.similarity_threshold
        )
        return {
            "status": "healthy",
            "redis_latency_seconds": round(latency, 4),
            "documents_cached": num_docs,
            "estimated_size_mb": round(size_mb, 2),
            "active_similarity_threshold": active_threshold,
            "similarity_threshold": cfg.similarity_threshold,
            "entity_threshold": cfg.entity_threshold,
            "embedding_provider": cfg.embedding_provider.value,
            "entity_aware": cfg.entity_aware,
            "domain": cfg.domain.value,
            "schema_index": manager.active_index_name,
            "prometheus_available": metrics.is_available(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.get("/prometheus")
def prometheus_metrics():
    """Standard Prometheus text-exposition endpoint.

    Returns a stub document when `prometheus_client` is not installed so
    scraper configs don't 500 the world when the optional dep is absent.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore

        media_type = CONTENT_TYPE_LATEST
    except ImportError:
        media_type = "text/plain; version=0.0.4; charset=utf-8"
    return Response(content=metrics.render_latest(), media_type=media_type)
