"""Per-client guard state: resolved configs, exemplar indexes, memos, breaker.

Deliberately a SEPARATE class from :class:`GatewayModelPool` with the inverted
exception policy. That pool's only caller sets ``manager = None`` and serves
uncached on ANY build exception (router.py) — correct for a cache, and the
exact opposite of what a fail-closed guard needs. Routing the guard through it
would silently turn "refuse when unsure" into "serve when broken". If you
change one thing in this file, do not change that.

Four pieces of state, each with a reason:

``resolve`` memo
    Parsing a policy costs a YAML parse and a hash. The APP config is
    TTL-cached upstream, so without a memo we would re-parse the same document
    on every request. Failures are cached too: a broken policy must cost one
    parse per TTL window, not one per request.

index LRU
    Bounded by entries AND by bytes, because one client's policy can be
    hundreds of times larger than another's. The entry owns its build task, so
    evicting it reaps the task rather than leaking one per policy edit.

segment memo
    Keyed by ``(index_key, params_hash, sha256(text))`` — policy, embedder and
    every decision parameter — so any config change invalidates it by
    construction. Short TTL on top. This is what makes checking EVERY message
    of a long conversation affordable, and what stops a client's judge key from
    being an economic DoS target.

circuit breaker
    Does NOT change any outcome. A dead embedding endpoint still refuses (or
    still degrades, for clients who chose that); the breaker only stops each
    request paying the full deadline to discover it again.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from semantic_cache.gateway.guard_config import (
    GuardConfigError,
    GuardLimits,
    ResolvedGuard,
    derive_guard_config,
)
from semantic_cache.gateway.guard_logic import Neighbor, classify, separability_report
from semantic_cache.gateway.guard_vectors import (
    SENTINEL_TEXT,
    GuardIndex,
    GuardEmbedder,
)

logger = logging.getLogger(__name__)

_LOO_CHUNK = 256


class GuardBuildError(Exception):
    """An exemplar index could not be built or loaded."""


@dataclass
class _IndexEntry:
    """One policy's matrix, plus the task that is (or was) building it."""

    task: Optional["asyncio.Future"] = None
    index: Optional[GuardIndex] = None
    nbytes: int = 0


@dataclass
class _Breaker:
    failures: int = 0
    window_started: float = 0.0
    open_until: float = 0.0


@dataclass
class SegmentOutcome:
    """What a single checked segment produced, cached for a short while."""

    score: float
    category: Optional[str]
    judge_score: Optional[float]
    judge_rationale: Optional[str]
    top_matches: List[Neighbor] = field(default_factory=list)


class GuardPool:
    """Everything the guard remembers between requests."""

    def __init__(
        self,
        store: Any,
        *,
        embed_base_url: str,
        limits: Optional[GuardLimits] = None,
        build_timeout: float = 60.0,
        cache_max_bytes: int = 268_435_456,
        max_indexes: int = 32,
        segment_memo_size: int = 50_000,
        segment_memo_ttl: float = 600.0,
        resolve_cache_size: int = 256,
        breaker_threshold: int = 10,
        breaker_window: float = 60.0,
        breaker_cooldown: float = 30.0,
        clock: Any = time.monotonic,
    ) -> None:
        self._store = store
        self._embed_base_url = embed_base_url
        self._limits = limits or GuardLimits()
        self._build_timeout = build_timeout
        self._cache_max_bytes = cache_max_bytes
        self._max_indexes = max_indexes
        self._segment_memo_size = segment_memo_size
        self._segment_memo_ttl = segment_memo_ttl
        self._resolve_cache_size = resolve_cache_size
        self._breaker_threshold = breaker_threshold
        self._breaker_window = breaker_window
        self._breaker_cooldown = breaker_cooldown
        self._clock = clock

        self._resolve_lock = threading.Lock()
        self._resolved: "OrderedDict[str, Any]" = OrderedDict()

        # Touched only from the event loop, so no lock. asyncio gives us
        # atomicity between awaits, and every mutation below is synchronous.
        self._indexes: "OrderedDict[str, _IndexEntry]" = OrderedDict()
        self._index_bytes = 0
        self._segments: "OrderedDict[Tuple[str, str, str], Tuple[float, SegmentOutcome]]" = (
            OrderedDict()
        )
        self._breakers: Dict[str, _Breaker] = {}

        self.builds = 0
        self.blob_hits = 0

    # ------------------------------------------------------------------ #
    # 1. Resolve
    # ------------------------------------------------------------------ #

    def resolve(self, config: Mapping[str, Any]) -> Optional[ResolvedGuard]:
        """The client's ResolvedGuard, or None when the guard is off.

        Synchronous and memoised on the hash the config client precomputed.
        Raises :class:`GuardConfigError` for a guard block that cannot be used
        — the router turns that into a 502.
        """
        key = config.get("guard_row_hash")
        if key:
            with self._resolve_lock:
                if key in self._resolved:
                    self._resolved.move_to_end(key)
                    cached = self._resolved[key]
                    if isinstance(cached, Exception):
                        raise cached
                    return cached

        try:
            resolved = derive_guard_config(
                config, embed_base_url=self._embed_base_url, limits=self._limits
            )
        except GuardConfigError as e:
            if key:
                self._remember_resolution(key, e)
            raise

        if key:
            self._remember_resolution(key, resolved)
        return resolved

    def _remember_resolution(self, key: str, value: Any) -> None:
        with self._resolve_lock:
            self._resolved[key] = value
            self._resolved.move_to_end(key)
            while len(self._resolved) > self._resolve_cache_size:
                self._resolved.popitem(last=False)

    # ------------------------------------------------------------------ #
    # 2. Index
    # ------------------------------------------------------------------ #

    async def get_index(
        self, resolved: ResolvedGuard, embedder: GuardEmbedder, *,
        budget: Optional[float] = None,
    ) -> GuardIndex:
        """The exemplar matrix for this policy, building it at most once.

        The build is OWNED BY THE POOL, not by the request that triggered it.
        Callers await a shielded view of the task, so a caller that runs out of
        budget leaves the build running and the next request finds it finished.
        Without the shield, every request would cancel the build it is waiting
        on and the next would start from zero — a policy that takes longer than
        one request's deadline could never finish building at all.
        """
        entry = self._indexes.get(resolved.index_key)
        if entry is not None:
            self._indexes.move_to_end(resolved.index_key)
            if entry.index is not None:
                return entry.index
        else:
            entry = _IndexEntry()
            self._indexes[resolved.index_key] = entry

        if entry.task is None or (entry.task.done() and entry.task.exception()):
            entry.task = asyncio.ensure_future(self._build(resolved, embedder))

        timeout = budget if budget is not None else self._build_timeout
        try:
            index = await asyncio.wait_for(asyncio.shield(entry.task), timeout)
        except asyncio.TimeoutError:
            raise GuardBuildError(
                "the guard's exemplar index is still building; the build "
                "continues in the background"
            ) from None

        entry.index = index
        entry.nbytes = index.nbytes
        self._index_bytes += entry.nbytes
        self._evict_indexes()
        return index

    async def _build(
        self, resolved: ResolvedGuard, embedder: GuardEmbedder
    ) -> GuardIndex:
        """Loads or embeds one policy's matrix. Never publishes a partial one."""
        exemplars = resolved.policy.exemplars
        labels = tuple(e.label for e in exemplars)
        categories = tuple(e.category_id for e in exemplars)
        texts = tuple(e.text for e in exemplars)

        index = await self._load_from_store(resolved, labels, categories, texts,
                                            embedder)
        if index is None:
            started = self._clock()
            # The BUILD budget, not the request's. The embedder handed to us
            # was built for the request path, whose timeout is a fraction of
            # one decision deadline; embedding a whole policy is exactly the
            # slow thing SC_GUARD_BUILD_TIMEOUT exists to allow. The build is
            # shielded and pool-owned, so nothing is waiting on this call.
            matrix = await embedder.embed(
                list(texts), input_type="document", timeout=self._build_timeout
            )
            sentinel = await embedder.embed(
                [SENTINEL_TEXT], input_type="document",
                timeout=self._build_timeout,
            )
            index = GuardIndex(
                matrix=matrix,
                labels=labels,
                categories=categories,
                texts=texts,
                sentinel=sentinel[0],
            )
            elapsed = self._clock() - started
            self.builds += 1
            # The embedding spend is the CLIENT's money and repeats on every
            # policy edit, so make it visible rather than inferable.
            logger.info(
                "Built guard index %s: %d exemplars, dim %d, %.2fs, model %s.",
                resolved.index_key[:12], index.n, index.dim, elapsed,
                resolved.embed_model,
            )
            self._run_separability(resolved, index)
            await asyncio.to_thread(
                self._store.save_guard_index,
                resolved.index_key, resolved.embed_model, resolved.policy_hash,
                index.dim, index.n, index.to_blob(), index.sentinel_blob(),
                {"n": index.n, "dim": index.dim,
                 "embed_model": resolved.embed_model},
            )
        return index

    async def _load_from_store(
        self,
        resolved: ResolvedGuard,
        labels: Sequence[str],
        categories: Sequence[str],
        texts: Sequence[str],
        embedder: GuardEmbedder,
    ) -> Optional[GuardIndex]:
        row = await asyncio.to_thread(
            self._store.load_guard_index, resolved.index_key
        )
        if not row:
            return None
        if int(row.get("n", -1)) != len(texts):
            logger.warning(
                "Stored guard index %s has %s rows but the policy has %d; "
                "rebuilding.", resolved.index_key[:12], row.get("n"), len(texts),
            )
            await asyncio.to_thread(self._store.delete_guard_index,
                                    resolved.index_key)
            return None
        try:
            index = GuardIndex.from_blob(
                row["vectors"], row["sentinel"], int(row["dim"]),
                labels, categories, texts,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Stored guard index %s is unreadable (%s); rebuilding.",
                           resolved.index_key[:12], e)
            await asyncio.to_thread(self._store.delete_guard_index,
                                    resolved.index_key)
            return None

        # The identity check TEI's /info was really for: a model repointed
        # behind an unchanged alias at an unchanged URL keeps its name, so only
        # the vectors can reveal it.
        fresh = await embedder.embed(
            [SENTINEL_TEXT], input_type="document", timeout=self._build_timeout
        )
        if not index.verify_sentinel(fresh[0]):
            logger.warning(
                "Guard index %s failed its sentinel check — the embedding model "
                "behind %r appears to have changed. Rebuilding.",
                resolved.index_key[:12], resolved.embed_model,
            )
            await asyncio.to_thread(self._store.delete_guard_index,
                                    resolved.index_key)
            return None

        self.blob_hits += 1
        await asyncio.to_thread(self._store.touch_guard_index, resolved.index_key)
        return index

    def _run_separability(self, resolved: ResolvedGuard, index: GuardIndex) -> None:
        """Leave-one-out check that this policy can separate its own examples."""
        params = resolved.params
        scored = _leave_one_out(index, params.top_k, params.min_similarity)
        report = separability_report(
            scored, params.allow_threshold, params.block_threshold
        )
        logger.info(
            "Guard policy %s calibration: predicted judge-invocation rate %.0f%%.",
            resolved.policy_hash[:12], report.judge_rate * 100,
        )
        if report.judge_rate > 0.30:
            logger.warning(
                "Guard policy %s would send %.0f%% of traffic resembling its own "
                "exemplars to the judge. Consider widening the gap between "
                "allow_threshold and block_threshold.",
                resolved.policy_hash[:12], report.judge_rate * 100,
            )
        if not report.ok:
            first = report.failures[0]
            raise GuardConfigError(
                f"guard.policy cannot separate its own examples with these "
                f"thresholds: {first.problem} (exemplar: {first.text!r}). "
                f"{len(report.failures)} exemplar(s) affected."
            )

    def _evict_indexes(self) -> None:
        # Never evict down to nothing: one index over the byte budget still has
        # to be resident to be usable, and evicting the entry we just built —
        # and are about to search — would only guarantee rebuilding it.
        while len(self._indexes) > 1 and (
            len(self._indexes) > self._max_indexes
            or self._index_bytes > self._cache_max_bytes
        ):
            _, evicted = self._indexes.popitem(last=False)
            self._index_bytes -= evicted.nbytes
            # Dropping the entry drops its task reference too, so a policy edit
            # does not leak one build task per version.
            if evicted.task is not None and not evicted.task.done():
                evicted.task.cancel()

    def invalidate_index(self, index_key: str) -> None:
        """Forgets one index — used when a query hits a dimension mismatch."""
        entry = self._indexes.pop(index_key, None)
        if entry is not None:
            self._index_bytes -= entry.nbytes
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()

    # ------------------------------------------------------------------ #
    # 3. Segment memo
    # ------------------------------------------------------------------ #

    @staticmethod
    def segment_key(
        resolved: ResolvedGuard, text: str
    ) -> Tuple[str, str, str]:
        return (
            resolved.index_key,
            resolved.params_hash,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def memo_get(self, key: Tuple[str, str, str]) -> Optional[SegmentOutcome]:
        """A previous outcome for this exact segment, or None.

        A miss means "run the check", never "allow" — every caller treats it
        that way, and there is no code path where a memo failure produces a
        verdict.
        """
        entry = self._segments.get(key)
        if entry is None:
            return None
        expires_at, outcome = entry
        if expires_at <= self._clock():
            self._segments.pop(key, None)
            return None
        self._segments.move_to_end(key)
        return outcome

    def memo_put(self, key: Tuple[str, str, str], outcome: SegmentOutcome) -> None:
        self._segments[key] = (self._clock() + self._segment_memo_ttl, outcome)
        self._segments.move_to_end(key)
        while len(self._segments) > self._segment_memo_size:
            self._segments.popitem(last=False)

    # ------------------------------------------------------------------ #
    # 4. Circuit breaker
    # ------------------------------------------------------------------ #

    def breaker_is_open(self, index_key: str) -> bool:
        breaker = self._breakers.get(index_key)
        if breaker is None:
            return False
        if breaker.open_until > self._clock():
            return True
        if breaker.open_until:
            # Cooldown elapsed: half-open, one request gets to try again.
            breaker.open_until = 0.0
            breaker.failures = 0
        return False

    def record_failure(self, index_key: str) -> None:
        now = self._clock()
        breaker = self._breakers.setdefault(index_key, _Breaker())
        if now - breaker.window_started > self._breaker_window:
            breaker.window_started = now
            breaker.failures = 0
        breaker.failures += 1
        if breaker.failures >= self._breaker_threshold and not breaker.open_until:
            breaker.open_until = now + self._breaker_cooldown
            logger.critical(
                "Guard circuit OPEN for index %s after %d consecutive failures. "
                "Checks short-circuit for %.0fs. Clients that did not opt into "
                "degrade_to_unguarded are being REFUSED.",
                index_key[:12], breaker.failures, self._breaker_cooldown,
            )

    def record_success(self, index_key: str) -> None:
        self._breakers.pop(index_key, None)


# --------------------------------------------------------------------------- #
# Leave-one-out scoring
# --------------------------------------------------------------------------- #


def _leave_one_out(
    index: GuardIndex, top_k: int, min_similarity: float
) -> List[Tuple[str, str, float]]:
    """Scores every exemplar against all the OTHERS.

    Chunked: at the 5000-exemplar cap a full n x n similarity matrix would be
    100 MB of float32 held at once, during a build that already holds the
    matrix itself.
    """
    n = index.n
    if n < 2:
        return []
    out: List[Tuple[str, str, float]] = []
    k = min(max(1, top_k), n - 1)
    for start in range(0, n, _LOO_CHUNK):
        stop = min(start + _LOO_CHUNK, n)
        block = index.matrix[start:stop] @ index.matrix.T
        for offset in range(stop - start):
            row = block[offset]
            i = start + offset
            row[i] = -np.inf                     # leave itself out
            top = np.argpartition(-row, k - 1)[:k]
            neighbors = [
                Neighbor(
                    text=index.texts[j],
                    label=index.labels[j],       # type: ignore[arg-type]
                    category_id=index.categories[j],
                    similarity=float(row[j]),
                )
                for j in top
            ]
            result = classify(neighbors, min_similarity)
            out.append((index.texts[i], index.labels[i], result.score))
    return out


__all__ = ["GuardPool", "GuardBuildError", "SegmentOutcome"]
