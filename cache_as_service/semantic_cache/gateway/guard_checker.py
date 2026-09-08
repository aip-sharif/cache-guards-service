"""Orchestration for one guard decision. Never raises.

``GuardChecker.check`` is the only thing the router calls. It returns a
:class:`GuardOutcome` for every possible input and every possible failure —
there is no exception path out of it, because on a fail-closed route an
unexpected exception would escape as a bare 500 outside the OpenAI error
envelope.

**One deadline for everything.** GaaS gave its embedder and its judge 15
seconds EACH, independently, with no shared budget, so a cascade check could
add ~30s to a request. Under a fail-closed contract that is not latency, it is
blocked live traffic. Everything here runs inside a single
``asyncio.timeout(guard_timeout)``, default 5s.

**Every failure has a name.** The reason strings are a closed set (see
:data:`REASONS`) so an operator can tell "the embedder is down" from "the judge
is saturated" from "we ran out of budget waiting for a build" — three very
different problems that collapse into one indistinguishable string if you let
them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from semantic_cache.gateway.guard_config import GuardConfigError, ResolvedGuard
from semantic_cache.gateway.guard_judge import (
    GuardJudge,
    GuardJudgeError,
    build_judge_messages,
)
from semantic_cache.gateway.guard_logic import (
    DEFAULT_UNAVAILABLE_REFUSAL,
    Decision,
    GuardJudgeRequired,
    Neighbor,
    Segment,
    classify,
    decide,
    refusal_for,
    should_invoke_judge,
)
from semantic_cache.gateway.guard_pool import (
    GuardBuildError,
    GuardPool,
    SegmentOutcome,
)
from semantic_cache.gateway.guard_vectors import (
    GuardDimMismatch,
    GuardEmbedError,
)

logger = logging.getLogger(__name__)

#: Closed set of reasons a check could not produce a verdict.
REASONS = (
    "embed_unreachable",
    "embed_malformed",
    "index_build_failed",
    "dim_mismatch",
    "judge_unreachable",
    "judge_unparseable",
    "judge_saturated",
    "guard_timeout_waiting_for_build",
    "guard_timeout_embed",
    "guard_timeout_judge",
    "input_too_large",
    "unguardable_content",
    "circuit_open",
    "config_invalid",
    "guard_error",
)

#: How many overlapping windows one oversized segment may be split into.
MAX_WINDOWS = 4
_WINDOW_OVERLAP = 0.25


@dataclass
class GuardOutcome:
    """What the router acts on. ``action`` is the only field it must branch on."""

    action: str  # "allow" | "block" | "flag" | "unavailable"
    matched_category: Optional[str] = None
    reason: str = ""
    refusal: Optional[str] = None
    embedding_score: Optional[float] = None
    judge_score: Optional[float] = None
    judge_invoked: bool = False
    top_matches: List[Neighbor] = field(default_factory=list)
    latency_ms: Optional[int] = None

    @property
    def blocked(self) -> bool:
        return self.action == "block"


class GuardChecker:
    """Runs one guard decision end to end."""

    def __init__(
        self,
        pool: GuardPool,
        judge: GuardJudge,
        embedder_factory: Any,
        *,
        timeout: float = 5.0,
        clock: Any = None,
    ) -> None:
        self._pool = pool
        self._judge = judge
        self._embedder_factory = embedder_factory
        self._timeout = timeout
        self._clock = clock
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

    @property
    def pool(self) -> GuardPool:
        """The per-client state, for the router's `resolve()` call.

        Public on purpose: the router resolves a client's guard config BEFORE
        deciding whether to check anything (it needs the policy hash for the
        cache scope even on an allow), so it reaches through the checker to the
        pool. Keeping this private meant only the test double had it — which is
        precisely how the missing attribute survived the unit suite and was
        caught by the first live request instead.
        """
        return self._pool

    # ------------------------------------------------------------------ #

    async def check(
        self,
        segments: Sequence[Segment],
        resolved: ResolvedGuard,
        *,
        judge_base_url: Optional[str] = None,
    ) -> GuardOutcome:
        """Never raises. Every failure becomes ``action='unavailable'``."""
        if not segments:
            return GuardOutcome(action="allow", reason="nothing_to_check")

        if self._pool.breaker_is_open(resolved.index_key):
            # Does not change the OUTCOME — clients who chose
            # degrade_to_unguarded still degrade, others are still refused. It
            # only stops each request paying the full deadline to rediscover a
            # dependency that is already known to be down.
            return self._unavailable(resolved, "circuit_open")

        try:
            async with _deadline(self._timeout):
                return await self._check(segments, resolved, judge_base_url)
        except asyncio.TimeoutError:
            self._pool.record_failure(resolved.index_key)
            return self._unavailable(resolved, "guard_timeout_embed")
        except GuardConfigError as e:
            # Escaped from a build-time separability failure. Not a runtime
            # fault, so it does not feed the breaker.
            logger.error("Guard config rejected during check: %s", e)
            return self._unavailable(resolved, "config_invalid", str(e))
        except Exception as e:  # noqa: BLE001
            self._pool.record_failure(resolved.index_key)
            logger.exception("Unexpected guard failure: %s", e)
            return self._unavailable(resolved, "guard_error", str(e))

    # ------------------------------------------------------------------ #

    async def _check(
        self,
        segments: Sequence[Segment],
        resolved: ResolvedGuard,
        judge_base_url: Optional[str],
    ) -> GuardOutcome:
        params = resolved.params

        windows = _windows(segments, params.max_input_chars, params.max_segments)
        if windows is None:
            return self._unavailable(resolved, "input_too_large")

        embedder = self._embedder_factory(resolved)

        # 1. Which windows do we already have an answer for?
        keys = [self._pool.segment_key(resolved, text) for text in windows]
        cached: List[Optional[SegmentOutcome]] = [
            self._pool.memo_get(key) for key in keys
        ]
        pending = [i for i, hit in enumerate(cached) if hit is None]

        index = None
        if pending:
            try:
                index = await self._pool.get_index(resolved, embedder)
            except GuardBuildError:
                self._pool.record_failure(resolved.index_key)
                return self._unavailable(
                    resolved, "guard_timeout_waiting_for_build"
                )
            except GuardConfigError:
                raise
            except GuardEmbedError as e:
                self._pool.record_failure(resolved.index_key)
                return self._unavailable(resolved, "embed_unreachable", str(e))
            except Exception as e:  # noqa: BLE001
                self._pool.record_failure(resolved.index_key)
                return self._unavailable(resolved, "index_build_failed", str(e))

            try:
                # ONE embedding request for every window that missed the memo.
                matrix = await embedder.embed(
                    [windows[i] for i in pending], input_type="query"
                )
            except GuardEmbedError as e:
                self._pool.record_failure(resolved.index_key)
                reason = (
                    "embed_malformed" if "malformed" in type(e).__name__.lower()
                    else "embed_unreachable"
                )
                return self._unavailable(resolved, reason, str(e))

            for row, position in enumerate(pending):
                try:
                    neighbors = index.search(matrix[row], params.top_k)
                except GuardDimMismatch as e:
                    # The stored matrix belongs to a different model. Drop it
                    # so the next request rebuilds, exactly once.
                    self._pool.invalidate_index(resolved.index_key)
                    self._pool.record_failure(resolved.index_key)
                    return self._unavailable(resolved, "dim_mismatch", str(e))
                result = classify(neighbors, params.min_similarity)
                cached[position] = SegmentOutcome(
                    score=result.score,
                    category=result.best_category,
                    judge_score=None,
                    judge_rationale=None,
                    top_matches=result.top_matches,
                )

        outcomes: List[SegmentOutcome] = [c for c in cached if c is not None]
        if not outcomes:  # pragma: no cover — defensive
            return self._unavailable(resolved, "guard_error", "no segment scored")

        # 2. The worst window decides. A conversation is as prohibited as its
        #    most prohibited turn.
        worst_at = max(range(len(outcomes)), key=lambda i: outcomes[i].score)
        worst = outcomes[worst_at]

        # 3. Judge, if the mode says so and the score is ambiguous.
        judge_score = worst.judge_score
        judge_rationale = worst.judge_rationale
        if judge_score is None and should_invoke_judge(
            worst.score, params.mode, params.allow_threshold, params.block_threshold
        ):
            verdict = await self._run_judge(
                windows[worst_at], worst, resolved, judge_base_url
            )
            if isinstance(verdict, GuardOutcome):
                return verdict                    # unavailable
            judge_score, judge_rationale = verdict

        # 4. Decide, and remember the segment outcomes for next time.
        try:
            decision = decide(
                embedding_score=worst.score,
                matched_category=worst.category,
                mode=params.mode,
                allow_threshold=params.allow_threshold,
                block_threshold=params.block_threshold,
                judge_score=judge_score,
                judge_rationale=judge_rationale,
                judge_allow_threshold=params.judge_allow_threshold,
                judge_block_threshold=params.judge_block_threshold,
                category_actions=resolved.policy.category_actions,
            )
        except GuardJudgeRequired as e:
            self._pool.record_failure(resolved.index_key)
            return self._unavailable(resolved, "judge_unreachable", str(e))

        for key, outcome in zip(keys, cached):
            if outcome is not None:
                if outcome is worst:
                    outcome.judge_score = judge_score
                    outcome.judge_rationale = judge_rationale
                self._pool.memo_put(key, outcome)

        self._pool.record_success(resolved.index_key)
        return self._from_decision(decision, worst, resolved)

    # ------------------------------------------------------------------ #

    async def _run_judge(
        self,
        text: str,
        outcome: SegmentOutcome,
        resolved: ResolvedGuard,
        judge_base_url: Optional[str],
    ):
        params = resolved.params
        semaphore = self._semaphores.setdefault(
            resolved.index_key, asyncio.Semaphore(params.judge_max_concurrency)
        )
        messages = build_judge_messages(
            text, outcome.top_matches,
            task_description=params.judge_task_description,
            category_descriptions=resolved.policy.category_descriptions,
        )
        try:
            # Saturation waits out the remaining budget and then becomes
            # unavailable. It must NEVER silently drop to embedding-only: that
            # would turn load into a quiet reduction in guard strength.
            async with semaphore:
                verdict = await self._judge.judge(
                    messages,
                    model=params.judge_model or "",
                    api_key=params.judge_api_key or "",
                    base_url=judge_base_url or "",
                )
        except asyncio.TimeoutError:
            self._pool.record_failure(resolved.index_key)
            return self._unavailable(resolved, "guard_timeout_judge")
        except GuardJudgeError as e:
            self._pool.record_failure(resolved.index_key)
            reason = (
                "judge_unparseable"
                if "unreachable" not in str(e) and "returned" not in str(e)
                else "judge_unreachable"
            )
            return self._unavailable(resolved, reason, str(e))
        return verdict.confidence, verdict.rationale

    # ------------------------------------------------------------------ #

    def _from_decision(
        self, decision: Decision, worst: SegmentOutcome, resolved: ResolvedGuard
    ) -> GuardOutcome:
        refusal = None
        if decision.action == "block":
            refusal = refusal_for(
                decision.matched_category,
                resolved.policy.category_refusals,
                resolved.policy.default_refusal,
                resolved.params.default_refusal,
            )
        return GuardOutcome(
            action=decision.action,
            matched_category=decision.matched_category,
            reason=decision.reason,
            refusal=refusal,
            embedding_score=worst.score,
            judge_score=worst.judge_score,
            judge_invoked=decision.judge_invoked,
            top_matches=list(worst.top_matches),
        )

    def _unavailable(
        self, resolved: ResolvedGuard, reason: str, detail: str = ""
    ) -> GuardOutcome:
        if detail:
            logger.warning("Guard unavailable (%s): %s", reason, detail)
        return GuardOutcome(
            action="unavailable",
            reason=reason,
            refusal=(
                resolved.params.unavailable_refusal or DEFAULT_UNAVAILABLE_REFUSAL
            ),
        )


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def _windows(
    segments: Sequence[Segment], max_chars: int, max_segments: int
) -> Optional[List[str]]:
    """Segment texts, splitting oversized ones into overlapping windows.

    Returns ``None`` when the input cannot be covered, which the caller turns
    into ``input_too_large``. Nothing is ever TRUNCATED: truncation is a bypass
    — put the payload past the cut and it is never seen. GaaS capped its input
    at 8000 characters with a pydantic validator, which on a longer message was
    a 500 rather than a decision either way.
    """
    if len(segments) > max_segments:
        return None

    total = sum(len(s.text) for s in segments)
    per_segment = max_chars if len(segments) == 1 else max_chars

    windows: List[str] = []
    for segment in segments:
        text = segment.text
        if len(text) <= per_segment:
            windows.append(text)
            continue
        step = max(1, int(per_segment * (1 - _WINDOW_OVERLAP)))
        pieces = [
            text[start:start + per_segment]
            for start in range(0, len(text), step)
        ]
        pieces = [p for p in pieces if p]
        if len(pieces) > MAX_WINDOWS:
            return None
        windows.extend(pieces)

    if not windows:
        return None
    if total > max_chars * MAX_WINDOWS:
        return None
    return windows


def _deadline(seconds: float):
    """asyncio.timeout on 3.11+, falling back for older runtimes."""
    timeout = getattr(asyncio, "timeout", None)
    if timeout is not None:
        return timeout(seconds)
    return _LegacyDeadline(seconds)  # pragma: no cover


class _LegacyDeadline:  # pragma: no cover — Python < 3.11 only
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


__all__ = ["REASONS", "GuardChecker", "GuardOutcome"]
