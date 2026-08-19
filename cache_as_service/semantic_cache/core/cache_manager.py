"""Core semantic cache orchestrator.

Handles interaction with RedisSearch, managing the lifecycle of cached responses,
and applying configuration logic such as similarity thresholds and TTL/LFU policies.

When entity-aware mode is enabled (config.entity_aware=True), lookup is layered:
    1. Extract entities from the incoming query via the injected extractor.
    2. Filter cache candidates by EXACT entity-set match (and domain tag).
    3. Only over that filtered set, apply the stricter `entity_threshold`
       cosine-similarity check.
Extraction failure (LLM error, timeout, malformed JSON) is always treated as a
cache MISS — never a HIT — to avoid serving the wrong answer in high-stakes
domains.

Operational features:
    - Versioned index name (`<base>:v<N>`) so future schema migrations don't
      collide with the live index. Existing data keys (`scache:<hash>`) are
      automatically picked up by the new index because RediSearch indexes by
      key prefix, not by per-document attachment.
    - `migrate_legacy_entries()` backfills `entity_sig` / `domain` defaults
      onto pre-feature entries so they remain discoverable.
    - LFU hit accounting batches the HINCRBY+TTL reads into a single Redis
      pipeline (1 RT) and only sends the EXPIRE/PERSIST follow-up if needed.
      Any race between concurrent hits crossing the promotion threshold is
      benign (PERSIST is idempotent; an EXPIRE racing PERSIST loses cleanly
      on the next hit).
    - Prometheus metrics distinguish miss reasons (no_candidate /
      extractor_error / below_threshold) so oncall can page on extractor
      degradation without false alarms from cold caches.
    - `asearch` / `aset` wrap the sync path in `asyncio.to_thread` so async
      FastAPI endpoints do not block the event loop. When an
      `AsyncLLMEntityExtractor` is wired in, the manager awaits it directly
      for true async I/O on the LLM call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField

try:
    # redis-py >= 5: snake_case path
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:  # pragma: no cover
    # Legacy camelCase fallback for older redis-py
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from semantic_cache.core import metrics
from semantic_cache.core.config import CacheMode, SemanticCacheConfig
from semantic_cache.core.embedding_manager import (
    BaseAsyncEmbeddingManager,
    BaseEmbeddingManager,
)
from semantic_cache.core.entity_extractor import BaseEntityExtractor
from semantic_cache.core.entity_extractor_async import BaseAsyncEntityExtractor
from semantic_cache.core.exceptions import CacheOperationError, EntityExtractionError
from semantic_cache.core.normalization import TextNormalizer

logger = logging.getLogger(__name__)

# Schema version. Bump whenever the RediSearch field set changes. The
# versioned index name becomes `<config.cache_index_name>:v<N>`; old indexes
# remain untouched and can be migrated explicitly via `migrate_legacy_entries`.
# v3 adds the indexed `scope` TAG field (server-side isolation, e.g. per
# pipeline-mode / tenant). Old v2 indexes remain untouched; migrate_legacy_entries
# backfills the default scope so existing keys stay visible under v3.
SCHEMA_VERSION = 3

# Scope tag written when a consumer does not pass a scope. A search WITHOUT a
# scope does not filter (matches all); a search WITH a scope only matches entries
# stored under that same scope.
_DEFAULT_SCOPE = "_default"

# Sentinel signature used when a query produces an empty entity set.
# Cached entries with no entities still need a deterministic tag so they can
# exact-match each other under entity-aware mode.
_EMPTY_ENTITY_SIG = "none"

# Max KNN candidates considered for entity-OVERLAP matching. The exact
# entity-signature prefilter was removed (the LLM extractor is non-deterministic
# — e.g. 'ear pain' vs 'ear' — so an exact set match almost never held and the
# cache effectively never hit). Instead we pull the K nearest entries in the
# domain and keep the best one that shares >=1 canonical entity AND clears the
# cosine floor. Domain + entity-overlap + threshold remain the safety guards.
_ENTITY_OVERLAP_K = 10


def versioned_index_name(base: str, version: int = SCHEMA_VERSION) -> str:
    """Returns the versioned RediSearch index name for a given base.

    Exposed for ops scripts and migration tooling.
    """
    return f"{base}:v{version}"


class SemanticCacheManager:
    """Orchestrator for vector-based semantic caching."""

    def __init__(
        self,
        config: SemanticCacheConfig,
        redis_client: redis.Redis,
        embedding_manager: BaseEmbeddingManager,
        entity_extractor: Optional[BaseEntityExtractor] = None,
        async_entity_extractor: Optional[BaseAsyncEntityExtractor] = None,
        async_embedding_manager: Optional[BaseAsyncEmbeddingManager] = None,
    ):
        """Initializes the Cache Manager.

        Args:
            config: Dependency-injected configuration.
            redis_client: Active, tested Redis client instance.
            embedding_manager: Strategy-injected (sync) embedding backend. Also
                supplies the index dimension at creation time.
            entity_extractor: Sync entity extractor. Required when
                config.entity_aware is True (unless async_entity_extractor
                is also provided and only the async path is used).
            async_entity_extractor: Optional async extractor. When present,
                `asearch`/`aset` use it directly for native async I/O. When
                absent, `asearch`/`aset` fall back to running the sync
                extractor via `asyncio.to_thread`.
            async_embedding_manager: Optional async embedding backend. When
                present, `asearch`/`aset` embed natively (no thread offload);
                otherwise the sync embedder is run via `asyncio.to_thread`.
        """
        self.config = config
        self.redis = redis_client
        self.embedding = embedding_manager
        self.async_embedding = async_embedding_manager
        self.entity_extractor = entity_extractor
        self.async_entity_extractor = async_entity_extractor

        if self.config.entity_aware and (
            self.entity_extractor is None and self.async_entity_extractor is None
        ):
            raise CacheOperationError(
                "entity_aware=True but no entity_extractor was injected."
            )

        # Transparently cache deterministic entity extractions in Redis so
        # repeat queries don't re-pay the LLM round trip. Skip if the injected
        # extractor is already a cached wrapper (idempotent composition).
        ttl = self.config.entity_extraction_cache_ttl
        if self.config.entity_aware and ttl and ttl > 0:
            from semantic_cache.core.extraction_cache import (  # noqa: PLC0415
                AsyncCachedEntityExtractor,
                CachedEntityExtractor,
            )

            if self.entity_extractor is not None and not isinstance(
                self.entity_extractor, CachedEntityExtractor
            ):
                self.entity_extractor = CachedEntityExtractor(
                    self.entity_extractor, redis_client, ttl_seconds=ttl
                )
            if self.async_entity_extractor is not None and not isinstance(
                self.async_entity_extractor, AsyncCachedEntityExtractor
            ):
                self.async_entity_extractor = AsyncCachedEntityExtractor(
                    self.async_entity_extractor, redis_client, ttl_seconds=ttl
                )

        # RedisSearch works with distance metrics.
        # Cosine Distance = 1 - Cosine Similarity.
        self.max_distance = 1.0 - self.config.similarity_threshold
        self.entity_max_distance = 1.0 - self.config.entity_threshold

        self.prefix = config.key_prefix or "scache:"

        # Optional durable-backup hooks (see gateway/backup.py). All three are
        # best-effort: an error inside a hook is logged and swallowed, never
        # allowed to break the cache operation that triggered it.
        #   persist_hook(redis_key, mapping, ttl_or_None)  after a write
        #   touch_hook(redis_key, ttl_or_None)             after an LFU
        #       refresh/promotion, so the backup's expiry tracks Redis instead
        #       of freezing at write time (otherwise the most-used entries are
        #       the ones the backup loses)
        #   delete_hook(redis_keys)                        after a purge, so
        #       deleted data cannot be resurrected on the next rebuild
        self.persist_hook: Optional[
            Callable[[str, Dict[str, Any], Optional[int]], None]
        ] = None
        self.touch_hook: Optional[Callable[[str, Optional[int]], None]] = None
        self.delete_hook: Optional[Callable[[List[str]], None]] = None

        # The active index name is always versioned. Callers configure
        # `cache_index_name` to a base string; the actual RediSearch index is
        # `<base>:v<SCHEMA_VERSION>`.
        self.active_index_name = versioned_index_name(self.config.cache_index_name)

        self._ensure_index()

    def _run_hook(self, hook: Optional[Callable], *args: Any) -> None:
        """Best-effort hook invocation: the durable backup must never break
        the cache operation it mirrors."""
        if hook is None:
            return
        try:
            hook(*args)
        except Exception as e:  # noqa: BLE001
            logger.error("Cache backup hook failed (ignored): %s", e)

    # ------------------------------------------------------------------ #
    # Index lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        """Ensures the active (versioned) RediSearch vector index exists."""
        try:
            self.redis.ft(self.active_index_name).info()
            logger.info("Index %s already exists.", self.active_index_name)
        except ResponseError as e:
            # RediSearch has used several phrasings for the "missing index"
            # error across versions: "Unknown Index name", "No such index", …
            msg = str(e).lower()
            if "unknown index" in msg or "no such index" in msg:
                self._create_index()
            else:
                logger.error("Error checking index: %s", e)
                raise

    def _create_index(self) -> None:
        """Defines the schema and creates the Redis HASH index.

        The `entity_sig` and `domain` TAG fields are always part of the schema
        so the index remains schema-compatible regardless of whether
        entity-aware mode is currently on. When disabled, all entries are
        written with `entity_sig=_EMPTY_ENTITY_SIG` and the tag is never
        used for filtering — behavior is unchanged.
        """
        dimension = self.embedding.get_dimension()

        schema = (
            TextField("text"),
            TextField("response"),
            TextField("entities"),
            TagField("entity_sig"),
            TagField("domain"),
            TagField("scope"),
            NumericField("hits"),
            NumericField("timestamp"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )

        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)

        try:
            self.redis.ft(self.active_index_name).create_index(
                fields=schema, definition=definition
            )
            logger.info("Successfully created vector index %s.", self.active_index_name)
        except Exception as e:
            logger.error("Failed to create index: %s", e)
            raise CacheOperationError(f"Index creation failed: {e}") from e

    def migrate_legacy_entries(self) -> int:
        """Backfills `entity_sig`/`domain`/`entities` on pre-versioning keys.

        Walks every `scache:*` key and adds the new TAG fields if missing so
        the entry becomes visible to the v2 index and to non-entity-aware
        lookups under entity-aware schemas. Returns the number of entries
        updated. Safe to run online and re-run idempotently.

        Use this once after upgrading a deployment that already has data.
        """
        updated = 0
        for raw_key in self.redis.scan_iter(match=f"{self.prefix}*", count=500):
            # In decode_responses mode keys come back as str. Be tolerant of bytes.
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key

            # Skip non-document keys (e.g. extraction-cache, meta).
            if key.startswith(f"{self.prefix}entcache:"):
                continue

            fields = self.redis.hkeys(key)
            field_set = {
                f.decode("utf-8") if isinstance(f, bytes) else f for f in fields
            }
            patch: Dict[str, str] = {}
            if "entity_sig" not in field_set:
                patch["entity_sig"] = _EMPTY_ENTITY_SIG
            if "domain" not in field_set:
                patch["domain"] = self.config.domain.value
            if "entities" not in field_set:
                patch["entities"] = "[]"
            if "scope" not in field_set:
                patch["scope"] = _DEFAULT_SCOPE

            if patch:
                self.redis.hset(key, mapping=patch)
                updated += 1

        logger.info("migrate_legacy_entries: updated %d entries.", updated)
        return updated

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _generate_key(self, text: str, scope: Optional[str] = None) -> str:
        """Deterministic storage key for normalized ``text`` within ``scope``.

        The scope tag is part of the key so the SAME query stored under two
        different scopes gets two distinct entries (both cached + isolated)
        instead of colliding on one key. A consumer that never passes a scope
        gets a single stable ``_default``-scoped key per query."""
        payload = f"{self._scope_tag(scope)}\x00{text}"
        hashed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.prefix}{hashed}"

    @staticmethod
    def _scope_tag(scope: Optional[str]) -> str:
        """A RediSearch-TAG-safe token for ``scope``.

        Sanitized identically on store and filter so the two always match without
        TAG-escaping headaches. ``None``/empty → the default scope tag."""
        if not scope:
            return _DEFAULT_SCOPE
        cleaned = re.sub(r"[^0-9a-z_]+", "_", scope.strip().lower())
        return cleaned or _DEFAULT_SCOPE

    @staticmethod
    def _entity_signature(entities: List[Dict[str, str]]) -> str:
        """Computes a deterministic signature of the entity set.

        Uses the sorted set of `identifier` strings so two queries with the
        same entities in different order / casing collide to the same tag.
        Empty entity sets collapse to `_EMPTY_ENTITY_SIG`.
        """
        if not entities:
            return _EMPTY_ENTITY_SIG
        identifiers = sorted(
            {e["identifier"].strip().lower() for e in entities if e.get("identifier")}
        )
        if not identifiers:
            return _EMPTY_ENTITY_SIG
        joined = "|".join(identifiers)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _extract_entities_or_miss(self, text: str) -> Optional[List[Dict[str, str]]]:
        """Runs the entity extractor; returns None on failure (→ caller MISSes)."""
        if self.entity_extractor is None:
            metrics.record_extractor(metrics.EXTRACTOR_ERROR)
            logger.warning("No sync entity_extractor wired; treating as MISS.")
            return None
        try:
            with metrics.time_extractor():
                result = self.entity_extractor.extract(text)
            metrics.record_extractor(metrics.EXTRACTOR_OK)
            return result
        except EntityExtractionError as e:
            metrics.record_extractor(metrics.EXTRACTOR_ERROR)
            logger.warning("Entity extraction failed, treating as cache MISS: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            metrics.record_extractor(metrics.EXTRACTOR_ERROR)
            logger.warning("Unexpected entity extraction error, treating as MISS: %s", e)
            return None

    async def _aextract_entities_or_miss(
        self, text: str
    ) -> Optional[List[Dict[str, str]]]:
        """Async counterpart. Uses the async extractor if provided, else
        delegates to the sync one via `asyncio.to_thread`."""
        if self.async_entity_extractor is not None:
            try:
                with metrics.time_extractor():
                    result = await self.async_entity_extractor.extract(text)
                metrics.record_extractor(metrics.EXTRACTOR_OK)
                return result
            except EntityExtractionError as e:
                metrics.record_extractor(metrics.EXTRACTOR_ERROR)
                logger.warning("Async entity extraction failed: %s", e)
                return None
            except Exception as e:  # noqa: BLE001
                metrics.record_extractor(metrics.EXTRACTOR_ERROR)
                logger.warning("Unexpected async entity extraction error: %s", e)
                return None

        # Fall back to running the sync extractor off-thread.
        return await asyncio.to_thread(self._extract_entities_or_miss, text)

    def _apply_lfu(self, doc_id: str) -> None:
        """Per-hit bookkeeping: increment hits, refresh TTL, promote to permanent.

        Implementation notes:
          - HINCRBY + TTL are batched into a single non-transactional pipeline
            (1 round trip) since we need both results to decide the third op.
          - EXPIRE vs PERSIST is then sent as a single follow-up command.
          - Total: 2 round trips per hit, vs 3 with naive sync calls.
          - Best-effort: any failure is logged but never aborts the hit.

        Concurrency: two simultaneous hits crossing the promotion threshold
        is benign. PERSIST is idempotent. An EXPIRE that races a PERSIST will
        either lose to the PERSIST (correct) or win briefly and be promoted
        on the very next hit.
        """
        try:
            pipe = self.redis.pipeline(transaction=False)
            pipe.hincrby(doc_id, "hits", 1)
            pipe.ttl(doc_id)
            hits, ttl = pipe.execute()

            # Key already has no expiration → nothing to refresh or promote.
            if ttl == -1:
                return

            threshold = self.config.permanent_hit_threshold
            if threshold > 0 and hits >= threshold:
                self.redis.persist(doc_id)
                logger.info(
                    "Promoted entry %s to PERMANENT (reached %d hits).", doc_id, hits
                )
                self._run_hook(self.touch_hook, doc_id, None)
            else:
                self.redis.expire(doc_id, self.config.default_ttl)
                self._run_hook(self.touch_hook, doc_id, self.config.default_ttl)
        except Exception as e:  # noqa: BLE001
            # LFU bookkeeping is best-effort — never fail a hit because of it.
            logger.warning("LFU bookkeeping failed for %s: %s", doc_id, e)

    # ------------------------------------------------------------------ #
    # Public sync API
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Shared building blocks (single source of truth for both sync + async)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_bytes(vector: List[float]) -> bytes:
        return np.array(vector, dtype=np.float32).tobytes()

    def _entity_filter(
        self, entities: List[Dict[str, str]]
    ) -> Optional[Tuple[str, str]]:
        """`(entity_sig, domain)` pre-filter when entity-aware, else None."""
        if not self.config.entity_aware:
            return None
        return (self._entity_signature(entities), self.config.domain.value)

    def _active_max_distance(self) -> float:
        return (
            self.entity_max_distance if self.config.entity_aware else self.max_distance
        )

    def _eff_semantic_threshold(self) -> float:
        """Effective similarity floor for the semantic tier, honoring any
        per-method ``semantic`` block (entity_threshold when entity-aware)."""
        return (
            self.config.eff_entity_threshold()
            if self.config.entity_aware
            else self.config.eff_similarity_threshold()
        )

    @staticmethod
    def _entity_ids(entities: List[Dict[str, str]]) -> set:
        """Canonical identifier set (lowercased, stripped) for overlap tests."""
        return {
            e["identifier"].strip().lower()
            for e in entities
            if e.get("identifier")
        }

    def _entities_overlap(self, query_ids: set, doc_entities_json: str) -> bool:
        """True if a stored entry shares >=1 canonical entity with the query.

        Two entity-LESS sets also match (preserves the old empty-signature
        behaviour); a non-empty query never matches an entity-less entry and
        vice-versa, so a generic query can't borrow a specific entry's answer.
        """
        try:
            doc_entities = json.loads(doc_entities_json or "[]")
        except (TypeError, ValueError):
            doc_entities = []
        doc_ids = self._entity_ids(doc_entities)
        if not query_ids and not doc_ids:
            return True
        return bool(query_ids & doc_ids)

    def _exact_lookup(
        self, clean_text: str, scope: Optional[str], record_miss: bool = False
    ) -> Optional[Dict[str, Any]]:
        """L0 exact-match fast path. Returns the entry stored under the EXACT same
        normalized text + scope with a single HGET — no embedding, KNN, or entity
        extraction — or None if there is no exact repeat.

        Works because the storage key IS ``sha256(scope\\x00text)``, so an exact
        repeat maps to the very same key. ``record_miss`` records a miss metric
        only when this is the terminal tier of the cascade (an intermediate exact
        pre-check that misses is silent — a later tier may still hit)."""
        key = self._generate_key(clean_text, scope)
        resp, meta, ents, dom = self.redis.hmget(
            key, "response", "metadata", "entities", "domain"
        )
        if resp is None:
            if record_miss:
                metrics.record_miss(metrics.MISS_NO_CANDIDATE)
            return None
        self._apply_lfu(key)  # keep LFU/TTL accurate for the exact hit
        metrics.record_hit()
        logger.debug("Cache hit (L0 exact).")
        return {
            "response": resp,
            "metadata": json.loads(meta or "{}"),
            "similarity": 1.0,
            "entities": json.loads(ents or "[]"),
            "domain": dom or self.config.domain.value,
            "exact": True,
        }

    def _vector_search(
        self, vector_bytes: bytes, entities: List[Dict[str, str]],
        scope: Optional[str] = None, record_miss: bool = True,
        max_distance: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Domain-prefiltered KNN-K -> entity-overlap -> similarity-floor select.

        Shared by the sync and async search paths. In entity-aware mode the KNN
        is pre-filtered by DOMAIN only (not by an exact entity signature) and
        returns the K nearest entries; we then return the best candidate that
        shares at least one canonical entity with the query AND is within
        `entity_threshold`. In non-entity-aware mode this degrades to the
        original KNN-1 + similarity behaviour.
        """
        entity_aware = self.config.entity_aware
        if max_distance is not None:
            active_max_distance = max_distance
        else:
            active_max_distance = (
                self.entity_max_distance if entity_aware else self.max_distance
            )
        # Build the pre-filter: DOMAIN (entity-aware only) + SCOPE (when a scope is
        # requested — server-side isolation so a hit from another scope is never
        # returned). No scope requested → no scope filter (matches all, backward
        # compatible).
        filters: List[str] = []
        if entity_aware:
            filters.append(f"@domain:{{{self.config.domain.value}}}")
            k = _ENTITY_OVERLAP_K
        else:
            k = 1
        if scope is not None:
            filters.append(f"@scope:{{{self._scope_tag(scope)}}}")
        base_filter = f"({' '.join(filters)})" if filters else "*"

        q = (
            Query(f"{base_filter}=>[KNN {k} @vector $vec_param AS vector_score]")
            .sort_by("vector_score")
            .return_fields(
                "response", "vector_score", "hits", "metadata", "entities", "domain"
            )
            .dialect(2)
        )
        try:
            results = self.redis.ft(self.active_index_name).search(
                q, query_params={"vec_param": vector_bytes}
            )
        except Exception as e:
            logger.error("Cache search failed: %s", e)
            raise CacheOperationError(f"Search failed: {e}") from e

        query_ids = self._entity_ids(entities)
        ids = sorted(query_ids)

        if not results.docs:
            if entity_aware:
                logger.info(
                    "Cache MISS (no candidate) — entities=%s domain=%s",
                    ids, self.config.domain.value,
                )
            else:
                logger.debug("Cache miss (no candidate).")
            if record_miss:
                metrics.record_miss(metrics.MISS_NO_CANDIDATE)
            return None

        # Docs come back sorted by ascending distance (closest first). Return the
        # best one within threshold that shares an entity; stop scanning once a
        # doc passes the similarity floor (all later ones are farther still).
        for doc in results.docs:
            distance = float(doc.vector_score)
            if distance > active_max_distance:
                break
            if entity_aware and not self._entities_overlap(
                query_ids, getattr(doc, "entities", "[]")
            ):
                continue
            self._apply_lfu(doc.id)
            metrics.record_hit()
            if entity_aware:
                logger.info(
                    "Cache HIT (entity-overlap) — similarity=%.4f entities=%s domain=%s",
                    1.0 - distance, ids, self.config.domain.value,
                )
            else:
                logger.debug("Cache hit! similarity=%.4f", 1.0 - distance)
            return {
                "response": doc.response,
                "metadata": json.loads(getattr(doc, "metadata", "{}")),
                "similarity": 1.0 - distance,
                "entities": json.loads(getattr(doc, "entities", "[]")),
                "domain": getattr(doc, "domain", self.config.domain.value),
            }

        # Candidates existed but none qualified (below floor or no shared entity).
        if entity_aware:
            logger.info(
                "Cache MISS (no entity-overlap within threshold) — entities=%s domain=%s",
                ids, self.config.domain.value,
            )
        else:
            logger.debug("Cache miss (below threshold).")
        if record_miss:
            metrics.record_miss(metrics.MISS_BELOW_THRESHOLD)
        return None

    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        """Unicode word tokens for a full-text query (keeps Persian/Arabic, drops
        punctuation). Tokens are alphanumeric/underscore only, so they are safe to
        drop into a RediSearch query with no escaping."""
        return re.findall(r"\w+", text, flags=re.UNICODE)

    def _fulltext_value(self, clean_text: str) -> str:
        """The value stored in the indexed ``text`` field for lexical (bm25) search.

        Space-joined ``\\w+`` tokens, so RediSearch tokenizes the stored text the
        SAME way our query tokenizer does — RediSearch's built-in tokenizer only
        splits on Latin punctuation, so Persian/Arabic marks (؟ ، ؛, ZWNJ) would
        otherwise stay glued to a word at index time and never match the query
        term. Used only for full-text matching; the exact key and the embedding
        are computed from the full ``clean_text``, so this does not affect them."""
        tokens = self._lexical_tokens(clean_text)
        return " ".join(tokens) if tokens else clean_text

    def _lexical_search(
        self, clean_text: str, scope: Optional[str] = None,
        record_miss: bool = True, fuzzy_distance: int = 0,
        method_label: str = "bm25", scorer: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Full-text retrieval — the non-vector methods (``bm25`` and ``fuzzy``).

        Matches the query's tokens (RediSearch intersection: a candidate must
        contain them all) against stored question text and returns the
        best-scoring entry at or above the min score. Computes NO embedding. When
        ``fuzzy_distance`` > 0 each token is matched within that Levenshtein
        distance (``%term%``), so typos still hit. ``scorer`` / ``min_score``
        default to the flat config when not given (the cascade passes the
        method's effective values). ``record_miss`` records a miss metric only
        when this is the cascade's terminal tier."""
        scorer = scorer or self.config.lexical_scorer
        min_score = self.config.lexical_min_score if min_score is None else min_score
        tokens = self._lexical_tokens(clean_text)
        if not tokens:
            if record_miss:
                metrics.record_miss(metrics.MISS_NO_CANDIDATE)
            return None

        if fuzzy_distance > 0:
            pct = "%" * fuzzy_distance
            body = " ".join(f"{pct}{t}{pct}" for t in tokens)
        else:
            body = " ".join(tokens)
        text_expr = "@text:(" + body + ")"
        query_str = (
            f"(@scope:{{{self._scope_tag(scope)}}} {text_expr})"
            if scope is not None
            else text_expr
        )
        q = (
            Query(query_str)
            .scorer(scorer)
            .with_scores()
            .paging(0, 1)
            .return_fields("response", "metadata", "entities", "domain")
            .dialect(2)
        )
        try:
            results = self.redis.ft(self.active_index_name).search(q)
        except Exception as e:
            logger.error("Lexical cache search failed: %s", e)
            raise CacheOperationError(f"Search failed: {e}") from e

        if not results.docs:
            if record_miss:
                metrics.record_miss(metrics.MISS_NO_CANDIDATE)
            logger.debug("Cache miss (lexical, no candidate).")
            return None

        doc = results.docs[0]
        score = float(getattr(doc, "score", 0.0) or 0.0)
        if score < min_score:
            if record_miss:
                metrics.record_miss(metrics.MISS_BELOW_THRESHOLD)
            logger.info(
                "Cache MISS (lexical below score) — score=%.4f min=%.4f",
                score, min_score,
            )
            return None

        self._apply_lfu(doc.id)
        metrics.record_hit()
        logger.debug("Cache hit (lexical %s) score=%.4f", scorer, score)
        return {
            "response": doc.response,
            "metadata": json.loads(getattr(doc, "metadata", "{}")),
            "similarity": None,
            "score": score,
            "entities": json.loads(getattr(doc, "entities", "[]")),
            "domain": getattr(doc, "domain", self.config.domain.value),
            "method": method_label,
        }

    def _knn_search(
        self, vector_bytes: bytes, entity_filter: Optional[Tuple[str, str]]
    ):
        """Run the RediSearch KNN-1 query, optionally entity/domain pre-filtered.

        Hybrid KNN queries require the pre-filter expression in parentheses when
        it is anything richer than the wildcard `*`.
        """
        base_filter = (
            f"(@entity_sig:{{{entity_filter[0]}}} @domain:{{{entity_filter[1]}}})"
            if entity_filter is not None
            else "*"
        )
        q = (
            Query(f"{base_filter}=>[KNN 1 @vector $vec_param AS vector_score]")
            .sort_by("vector_score")
            .return_fields("response", "vector_score", "hits", "metadata", "entities", "domain")
            .dialect(2)
        )
        try:
            return self.redis.ft(self.active_index_name).search(
                q, query_params={"vec_param": vector_bytes}
            )
        except Exception as e:
            logger.error("Cache search failed: %s", e)
            raise CacheOperationError(f"Search failed: {e}") from e

    def _finalize_results(
        self,
        results,
        entities: List[Dict[str, str]],
        entity_filter: Optional[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate KNN results vs the active threshold; log + LFU + return.

        Single source of truth for hit/miss decisioning and logging — shared by
        the sync and async search paths so they cannot drift.
        """
        active_max_distance = self._active_max_distance()
        ids = [e.get("identifier") for e in entities]
        if results.docs:
            doc = results.docs[0]
            distance = float(doc.vector_score)
            if distance <= active_max_distance:
                self._apply_lfu(doc.id)
                metrics.record_hit()
                if self.config.entity_aware:
                    logger.info(
                        "Cache HIT (entity-aware) — similarity=%.4f entities=%s domain=%s",
                        1.0 - distance, ids, self.config.domain.value,
                    )
                else:
                    logger.debug("Cache hit! similarity=%.4f", 1.0 - distance)
                return {
                    "response": doc.response,
                    "metadata": json.loads(getattr(doc, "metadata", "{}")),
                    "similarity": 1.0 - distance,
                    "entities": json.loads(getattr(doc, "entities", "[]")),
                    "domain": getattr(doc, "domain", self.config.domain.value),
                }
            # Candidate existed but lost on the similarity floor — log the
            # near-miss so a too-strict threshold is visible, not just a metric.
            logger.info(
                "Cache MISS (below threshold) — similarity=%.4f threshold=%.4f "
                "entities=%s domain=%s",
                1.0 - distance, 1.0 - active_max_distance, ids,
                self.config.domain.value,
            )
            metrics.record_miss(metrics.MISS_BELOW_THRESHOLD)
        else:
            if self.config.entity_aware:
                logger.info(
                    "Cache MISS (no candidate) — entities=%s domain=%s",
                    ids, self.config.domain.value,
                )
            else:
                logger.debug("Cache miss (no candidate).")
            metrics.record_miss(metrics.MISS_NO_CANDIDATE)
        return None

    def _persist(
        self,
        clean_text: str,
        response: str,
        metadata: Optional[Dict[str, Any]],
        ttl: Optional[int],
        keep_forever: bool,
        entities: List[Dict[str, str]],
        vector_bytes: Optional[bytes],
        scope: Optional[str] = None,
    ) -> None:
        """Write a cache entry. Shared by sync `set` and async `aset`.

        ``vector_bytes`` is ``None`` in exact-only mode: the entry is stored and
        found purely by its normalized-text (+scope) key, so no embedding is
        computed and no vector field is written (the entry is simply invisible to
        KNN — correct, since an exact-mode client never runs a vector search)."""
        key = self._generate_key(clean_text, scope)
        mapping = {
            "text": self._fulltext_value(clean_text),
            "response": response,
            "hits": 1,
            "timestamp": int(time.time()),
            "metadata": json.dumps(metadata or {}),
            "entities": json.dumps(entities),
            "entity_sig": self._entity_signature(entities),
            "domain": self.config.domain.value,
            "scope": self._scope_tag(scope),
        }
        if vector_bytes is not None:
            mapping["vector"] = vector_bytes
        try:
            self.redis.hset(key, mapping=mapping)
            if not keep_forever:
                expiration = ttl if ttl is not None else self.config.default_ttl
                self.redis.expire(key, expiration)
            else:
                self.redis.persist(key)
            metrics.record_write(metrics.WRITE_OK)
            if self.persist_hook is not None:
                effective_ttl = (
                    None if keep_forever
                    else (ttl if ttl is not None else self.config.default_ttl)
                )
                self._run_hook(self.persist_hook, key, mapping, effective_ttl)
            if self.config.entity_aware:
                logger.info(
                    "Cached entry %s (entities=%s domain=%s)",
                    key, [e.get("identifier") for e in entities],
                    self.config.domain.value,
                )
            else:
                logger.debug("Cached entry %s", key)
        except Exception as e:
            logger.error("Failed to set cache entry: %s", e)
            raise CacheOperationError(f"Set failed: {e}") from e

    async def _aembed_bytes(self, clean_text: str) -> bytes:
        """Embed natively async when an async embedder is wired; else offload."""
        if self.async_embedding is not None:
            vector = await self.async_embedding.get_embedding(clean_text)
        else:
            vector = await asyncio.to_thread(self.embedding.get_embedding, clean_text)
        return self._to_bytes(vector)

    def _extract_for_search(self, clean_text: str):
        """Sync extraction → (entities, ok). ok=False signals an extractor-error MISS."""
        if not self.config.entity_aware:
            return [], True
        extracted = self._extract_entities_or_miss(clean_text)
        if extracted is None:
            metrics.record_miss(metrics.MISS_EXTRACTOR_ERROR)
            return [], False
        return extracted, True

    # ------------------------------------------------------------------ #
    # Public sync API
    # ------------------------------------------------------------------ #

    def _fail_open_read(self, exc: Exception) -> None:
        """Handle a backend failure on a READ (search): log, and either swallow
        (return None — a miss) when ``config.fail_open`` or re-raise otherwise."""
        logger.error("Cache search failed (fail_open=%s): %s", self.config.fail_open, exc)
        if not self.config.fail_open:
            raise exc
        return None

    def _fail_open_write(self, exc: Exception) -> None:
        """Handle a backend failure on a WRITE (set): log, and either swallow
        (no-op) when ``config.fail_open`` or re-raise otherwise."""
        logger.error("Cache set failed (fail_open=%s): %s", self.config.fail_open, exc)
        if not self.config.fail_open:
            raise exc
        return None

    def search(
        self, query: str, scope: Optional[str] = None, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Searches the cache for a semantic match to the query.

        ``scope`` (optional) restricts the search to entries stored under the same
        scope — a server-side isolation boundary (e.g. per pipeline-mode or tenant)
        so a hit from a different scope is never returned. Omit it to match all.

        Fail-open (default): a backend failure (Redis outage, embedding/extraction
        error) is swallowed and returns None (a miss). Set config.fail_open=False
        to re-raise."""
        try:
            with metrics.time_search():
                clean_text = TextNormalizer.normalize_query(
                    query,
                    enable_translation=self.config.enable_translation,
                    target_lang=self.config.translation_target_language,
                    aggressive=self.config.normalize_aggressive,
                    **kwargs,
                )["text"]

                # Retrieval CASCADE: try each configured method in order; the
                # first hit wins. Embedding/extraction are computed lazily — only
                # if the semantic tier is actually reached. ``off`` → empty cascade.
                cascade = self.config.retrieval_cascade()
                last = len(cascade) - 1
                for i, method in enumerate(cascade):
                    is_last = i == last
                    if method == CacheMode.EXACT:
                        hit = self._exact_lookup(
                            clean_text, scope, record_miss=is_last
                        )
                    elif method == CacheMode.BM25:
                        hit = self._lexical_search(
                            clean_text, scope, record_miss=is_last,
                            scorer=self.config.eff_lexical_scorer(method),
                            min_score=self.config.eff_lexical_min_score(method),
                        )
                    elif method == CacheMode.FUZZY:
                        hit = self._lexical_search(
                            clean_text, scope, record_miss=is_last,
                            fuzzy_distance=self.config.eff_fuzzy_distance(),
                            method_label="fuzzy",
                            scorer=self.config.eff_lexical_scorer(method),
                            min_score=self.config.eff_lexical_min_score(method),
                        )
                    else:  # SEMANTIC
                        entities, ok = self._extract_for_search(clean_text)
                        if not ok:
                            continue  # extractor error already recorded a miss
                        vector_bytes = self._to_bytes(
                            self.embedding.get_embedding(clean_text)
                        )
                        hit = self._vector_search(
                            vector_bytes, entities, scope=scope, record_miss=is_last,
                            max_distance=1.0 - self._eff_semantic_threshold(),
                        )
                    if hit is not None:
                        return hit
                return None
        except Exception as e:  # noqa: BLE001 — fail-open: a cache error is a miss
            return self._fail_open_read(e)

    def set(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        keep_forever: bool = False,
        scope: Optional[str] = None,
    ) -> None:
        """Sets a new entry in the semantic cache. ``scope`` tags the entry for
        server-side isolation (see :meth:`search`). Fail-open (default): a backend
        failure becomes a no-op. Set config.fail_open=False to re-raise."""
        try:
            cascade = self.config.retrieval_cascade()
            if not cascade:  # off — nothing to store
                return

            clean_text = TextNormalizer.normalize_query(
                query,
                enable_translation=self.config.enable_translation,
                target_lang=self.config.translation_target_language,
                aggressive=self.config.normalize_aggressive,
            )["text"]

            # One write serves every tier: a semantic entry (text + vector) is
            # also found by exact (same key) and bm25 (the text field). So we
            # only pay embedding when the cascade actually includes semantic;
            # a purely exact/bm25 cascade stores text-only (no embedding).
            if CacheMode.SEMANTIC not in cascade:
                self._persist(
                    clean_text, response, metadata, ttl, keep_forever, [],
                    None, scope=scope,
                )
                return

            entities: List[Dict[str, str]] = []
            if self.config.entity_aware:
                extracted = self._extract_entities_or_miss(clean_text)
                if extracted is None:
                    metrics.record_write(metrics.WRITE_SKIPPED_EXTRACTOR_ERROR)
                    logger.warning(
                        "Skipping cache.set: entity extraction failed in entity-aware mode."
                    )
                    return
                entities = extracted

            vector_bytes = self._to_bytes(self.embedding.get_embedding(clean_text))
            self._persist(
                clean_text, response, metadata, ttl, keep_forever, entities,
                vector_bytes, scope=scope,
            )
        except Exception as e:  # noqa: BLE001 — fail-open: a cache error is a no-op
            self._fail_open_write(e)

    def purge(self) -> None:
        """Clears all entries from the semantic cache."""
        try:
            # Collect keys BEFORE dropping so the durable backup can be
            # cleared too — otherwise rebuild-on-boot resurrects everything.
            purged_keys = (
                [
                    k.decode("utf-8") if isinstance(k, bytes) else k
                    for k in self.redis.scan_iter(match=f"{self.prefix}*", count=500)
                ]
                if self.delete_hook is not None
                else []
            )
            self.redis.ft(self.active_index_name).dropindex(delete_documents=True)
            self._create_index()
            if purged_keys:
                self._run_hook(self.delete_hook, purged_keys)
            logger.info("Cache successfully purged.")
        except Exception as e:
            logger.error("Failed to purge cache: %s", e)
            raise CacheOperationError(f"Purge failed: {e}") from e

    def purge_scope(self, scope: str) -> int:
        """Deletes ONLY the entries stored under ``scope``; returns the count.

        The multi-tenant deletion primitive: removing one tenant/cache's data
        must never touch anyone else's. Unlike search/set this is NOT
        fail-open — deletion has to be reliable — so backend errors raise
        :class:`CacheOperationError`.
        """
        if not scope:
            raise ValueError("purge_scope requires a non-empty scope.")
        tag = self._scope_tag(scope)
        deleted = 0
        try:
            try:
                while True:
                    q = Query(f"@scope:{{{tag}}}").no_content().paging(0, 500)
                    docs = self.redis.ft(self.active_index_name).search(q).docs
                    if not docs:
                        break
                    pipe = self.redis.pipeline(transaction=False)
                    for doc in docs:
                        pipe.delete(doc.id)
                    deleted += sum(1 for n in pipe.execute() if n)
                    # Drop the durable backup too, or the next rebuild-on-boot
                    # would resurrect exactly what was just deleted.
                    self._run_hook(self.delete_hook, [doc.id for doc in docs])
            except ResponseError:
                # Index unavailable — fall back to a full keyspace scan.
                for raw_key in self.redis.scan_iter(match=f"{self.prefix}*", count=500):
                    key = (
                        raw_key.decode("utf-8")
                        if isinstance(raw_key, bytes)
                        else raw_key
                    )
                    entry_scope = self.redis.hget(key, "scope")
                    if isinstance(entry_scope, bytes):
                        entry_scope = entry_scope.decode("utf-8")
                    if entry_scope == tag and self.redis.delete(key):
                        deleted += 1
                        self._run_hook(self.delete_hook, [key])
            logger.info("purge_scope(%s): deleted %d entries.", tag, deleted)
            return deleted
        except Exception as e:
            logger.error("Failed to purge scope %s: %s", tag, e)
            raise CacheOperationError(f"Scope purge failed: {e}") from e

    async def apurge_scope(self, scope: str) -> int:
        """Async :meth:`purge_scope` (runs in a worker thread)."""
        return await asyncio.to_thread(self.purge_scope, scope)

    # ------------------------------------------------------------------ #
    # Public async API
    # ------------------------------------------------------------------ #

    async def asearch(
        self, query: str, scope: Optional[str] = None, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Async search. ``scope`` restricts to entries under the same scope
        (see :meth:`search`).

        Entity extraction and embedding run natively async when an async
        extractor / embedder is wired in; otherwise they are offloaded to a
        worker thread. The (blocking) Redis KNN + LFU bookkeeping always runs
        in a thread so the event loop is never pinned.
        """
        try:
            with metrics.time_search():
                clean_text = TextNormalizer.normalize_query(
                    query,
                    enable_translation=self.config.enable_translation,
                    target_lang=self.config.translation_target_language,
                    aggressive=self.config.normalize_aggressive,
                    **kwargs,
                )["text"]

                # Retrieval CASCADE (async): try each method in order, first hit
                # wins. Extraction/embedding run natively async when wired in and
                # only if the semantic tier is reached; the blocking Redis calls
                # are offloaded to a thread. ``off`` → empty cascade.
                cascade = self.config.retrieval_cascade()
                last = len(cascade) - 1
                for i, method in enumerate(cascade):
                    is_last = i == last
                    if method == CacheMode.EXACT:
                        hit = await asyncio.to_thread(
                            self._exact_lookup, clean_text, scope, is_last
                        )
                    elif method == CacheMode.BM25:
                        hit = await asyncio.to_thread(
                            self._lexical_search, clean_text, scope, is_last,
                            0, "bm25",
                            self.config.eff_lexical_scorer(method),
                            self.config.eff_lexical_min_score(method),
                        )
                    elif method == CacheMode.FUZZY:
                        hit = await asyncio.to_thread(
                            self._lexical_search, clean_text, scope, is_last,
                            self.config.eff_fuzzy_distance(), "fuzzy",
                            self.config.eff_lexical_scorer(method),
                            self.config.eff_lexical_min_score(method),
                        )
                    else:  # SEMANTIC
                        entities: List[Dict[str, str]] = []
                        if self.config.entity_aware:
                            extracted = await self._aextract_entities_or_miss(
                                clean_text
                            )
                            if extracted is None:
                                metrics.record_miss(metrics.MISS_EXTRACTOR_ERROR)
                                continue
                            entities = extracted
                        vector_bytes = await self._aembed_bytes(clean_text)
                        hit = await asyncio.to_thread(
                            self._vector_search, vector_bytes, entities, scope,
                            is_last, 1.0 - self._eff_semantic_threshold(),
                        )
                    if hit is not None:
                        return hit
                return None
        except Exception as e:  # noqa: BLE001 — fail-open: a cache error is a miss
            return self._fail_open_read(e)

    async def aset(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        keep_forever: bool = False,
        scope: Optional[str] = None,
    ) -> None:
        """Async set. ``scope`` tags the entry for isolation (see :meth:`search`).
        Entity extraction + embedding run natively async when wired in; the Redis
        write is always offloaded to a worker thread. Fail-open (default): a backend
        failure becomes a no-op."""
        try:
            cascade = self.config.retrieval_cascade()
            if not cascade:  # off — nothing to store
                return

            clean_text = TextNormalizer.normalize_query(
                query,
                enable_translation=self.config.enable_translation,
                target_lang=self.config.translation_target_language,
                aggressive=self.config.normalize_aggressive,
            )["text"]

            # One write serves every tier (see :meth:`set`): pay embedding only
            # when the cascade includes semantic; otherwise store text-only.
            if CacheMode.SEMANTIC not in cascade:
                await asyncio.to_thread(
                    self._persist,
                    clean_text, response, metadata, ttl, keep_forever, [],
                    None, scope,
                )
                return

            entities: List[Dict[str, str]] = []
            if self.config.entity_aware:
                extracted = await self._aextract_entities_or_miss(clean_text)
                if extracted is None:
                    metrics.record_write(metrics.WRITE_SKIPPED_EXTRACTOR_ERROR)
                    logger.warning("Skipping cache.aset: entity extraction failed.")
                    return
                entities = extracted

            vector_bytes = await self._aembed_bytes(clean_text)
            await asyncio.to_thread(
                self._persist,
                clean_text, response, metadata, ttl, keep_forever, entities,
                vector_bytes, scope,
            )
        except Exception as e:  # noqa: BLE001 — fail-open: a cache error is a no-op
            self._fail_open_write(e)
