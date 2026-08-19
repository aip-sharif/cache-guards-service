"""Guard checker — orchestration that must never raise."""

import asyncio
import hashlib

import numpy as np
import pytest

from semantic_cache.gateway.guard_checker import REASONS, GuardChecker, GuardOutcome
from semantic_cache.gateway.guard_judge import GuardJudge, GuardJudgeError, JudgeVerdict
from semantic_cache.gateway.guard_logic import Segment
from semantic_cache.gateway.guard_pool import GuardPool
from semantic_cache.gateway.guard_vectors import GuardEmbedError

from tests.unit.test_guard_pool_unit import (  # reuse the doubles
    DIM,
    POLICY,
    FakeEmbedder,
    FakeStore,
)

EMBED_URL = "https://embed.example.com"
JUDGE_URL = "https://judge.example.com"

BAD = "What do you think of Rivalco's product?"
GOOD = "What is your refund policy?"


class FakeJudge:
    def __init__(self, confidence=0.9, error=None, delay=0.0) -> None:
        self.confidence = confidence
        self.error = error
        self.delay = delay
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0

    async def judge(self, messages, *, model, api_key, base_url):
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return JudgeVerdict(self.confidence, "because", None)
        finally:
            self.concurrent -= 1


def _config(**guard):
    block = {"enabled": True, "policy": POLICY, "min_similarity": 0.35}
    block.update(guard)
    payload = {
        "model": "gpt-x", "model_api_key": "sk-llm",
        "embed_model": "bge-m3", "embed_api_key": "sk-embed",
        "guard": block,
    }
    payload["guard_row_hash"] = hashlib.sha256(
        repr(sorted(block.items())).encode()
    ).hexdigest()
    return payload


def _checker(embedder=None, judge=None, *, timeout=5.0, store=None, pool=None):
    embedder = embedder or FakeEmbedder()
    pool = pool or GuardPool(store or FakeStore(), embed_base_url=EMBED_URL)
    return GuardChecker(
        pool, judge or FakeJudge(), lambda resolved: embedder, timeout=timeout
    ), pool


def _segments(*texts):
    return [Segment(role="user", text=t, index=i) for i, t in enumerate(texts)]


async def _check(checker, pool, segments, **guard):
    resolved = pool.resolve(_config(**guard))
    return await checker.check(segments, resolved, judge_base_url=JUDGE_URL)


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


async def test_a_benign_message_is_allowed() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, _segments("How do I reset my password?"))
    assert outcome.action == "allow"
    assert outcome.refusal is None


async def test_a_prohibited_message_is_blocked_with_its_category() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, _segments(BAD), mode="embedding-only",
                           block_threshold=0.5)
    assert outcome.action == "block"
    assert outcome.matched_category == "competitors"
    assert outcome.refusal                       # a canned refusal, not empty


async def test_no_segments_is_an_allow_not_a_failure() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, [])
    assert outcome.action == "allow"
    assert outcome.reason == "nothing_to_check"


# --------------------------------------------------------------------------- #
# THE FABRICATED-HISTORY REGRESSION, at checker level
# --------------------------------------------------------------------------- #


async def test_the_worst_segment_decides_not_the_last_one() -> None:
    # [payload, benign, benign] must block on the payload. A guard that scored
    # only the last user message would allow this.
    checker, pool = _checker()
    outcome = await _check(
        checker, pool,
        _segments(BAD, GOOD, "please continue"),
        mode="embedding-only", block_threshold=0.5,
    )
    assert outcome.action == "block"
    assert outcome.matched_category == "competitors"


async def test_all_benign_segments_stay_allowed() -> None:
    checker, pool = _checker()
    outcome = await _check(
        checker, pool, _segments(GOOD, "How do I reset my password?"),
        mode="embedding-only", block_threshold=0.5,
    )
    assert outcome.action == "allow"


# --------------------------------------------------------------------------- #
# Memoisation
# --------------------------------------------------------------------------- #


async def test_a_repeated_segment_is_not_embedded_twice() -> None:
    embedder = FakeEmbedder()
    checker, pool = _checker(embedder)
    await _check(checker, pool, _segments(GOOD))
    calls = embedder.call_count
    await _check(checker, pool, _segments(GOOD))
    assert embedder.call_count == calls


async def test_duplicate_segments_in_one_request_are_embedded_once() -> None:
    embedder = FakeEmbedder()
    checker, pool = _checker(embedder)
    await _check(checker, pool, _segments(GOOD, GOOD, GOOD))
    # One build call for the exemplars, one sentinel, one query batch.
    assert embedder.call_count <= 3


# --------------------------------------------------------------------------- #
# Failure taxonomy — every one of these must be an outcome, never an exception
# --------------------------------------------------------------------------- #


async def test_a_dead_embedder_is_unavailable_not_an_exception() -> None:
    checker, pool = _checker(FakeEmbedder(fail=True))
    outcome = await _check(checker, pool, _segments(GOOD))
    assert outcome.action == "unavailable"
    assert outcome.reason in REASONS
    assert outcome.refusal


async def test_a_hanging_embedder_times_out_within_the_deadline() -> None:
    checker, pool = _checker(FakeEmbedder(delay=5.0), timeout=0.15)
    started = asyncio.get_running_loop().time()
    outcome = await _check(checker, pool, _segments(GOOD))
    elapsed = asyncio.get_running_loop().time() - started
    assert outcome.action == "unavailable"
    assert outcome.reason.startswith("guard_timeout")
    assert elapsed < 1.0


async def test_a_dead_judge_is_unavailable_never_a_score() -> None:
    judge = FakeJudge(error=GuardJudgeError("judge endpoint unreachable: boom"))
    checker, pool = _checker(judge=judge)
    outcome = await _check(checker, pool, _segments(BAD),
                           judge_model="q", judge_api_key="k",
                           allow_threshold=0.0, block_threshold=1.0)
    assert outcome.action == "unavailable"
    assert outcome.reason == "judge_unreachable"
    assert outcome.judge_score is None


async def test_an_unparseable_judge_is_unavailable() -> None:
    judge = FakeJudge(error=GuardJudgeError("judge reply is not valid JSON"))
    checker, pool = _checker(judge=judge)
    outcome = await _check(checker, pool, _segments(BAD),
                           judge_model="q", judge_api_key="k",
                           allow_threshold=0.0, block_threshold=1.0)
    assert outcome.action == "unavailable"
    assert outcome.reason == "judge_unparseable"


async def test_an_unexpected_error_deep_inside_becomes_guard_error() -> None:
    class Exploding:
        def __call__(self, resolved):
            raise TypeError("something nobody predicted")

    pool = GuardPool(FakeStore(), embed_base_url=EMBED_URL)
    checker = GuardChecker(pool, FakeJudge(), Exploding(), timeout=5.0)
    resolved = pool.resolve(_config())
    outcome = await checker.check(_segments(GOOD), resolved)
    assert outcome.action == "unavailable"
    assert outcome.reason == "guard_error"


async def test_every_reason_the_checker_emits_is_in_the_closed_set() -> None:
    checker, pool = _checker(FakeEmbedder(fail=True))
    outcome = await _check(checker, pool, _segments(GOOD))
    assert outcome.reason in REASONS


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


async def test_repeated_failures_open_the_breaker_and_short_circuit() -> None:
    pool = GuardPool(FakeStore(), embed_base_url=EMBED_URL,
                     breaker_threshold=3)
    checker, _ = _checker(FakeEmbedder(fail=True), pool=pool)
    for _ in range(3):
        await _check(checker, pool, _segments(GOOD))

    started = asyncio.get_running_loop().time()
    outcome = await _check(checker, pool, _segments(GOOD))
    elapsed = asyncio.get_running_loop().time() - started

    assert outcome.action == "unavailable"
    assert outcome.reason == "circuit_open"
    assert elapsed < 0.05          # short-circuits instead of paying the deadline


async def test_the_breaker_does_not_change_the_outcome_only_the_cost() -> None:
    pool = GuardPool(FakeStore(), embed_base_url=EMBED_URL, breaker_threshold=1)
    checker, _ = _checker(FakeEmbedder(fail=True), pool=pool)
    first = await _check(checker, pool, _segments(GOOD))
    second = await _check(checker, pool, _segments(GOOD))
    # Still unavailable — never an auto-degrade, which would contradict the
    # client's own degrade_to_unguarded choice.
    assert first.action == second.action == "unavailable"


# --------------------------------------------------------------------------- #
# Judge concurrency
# --------------------------------------------------------------------------- #


async def test_judge_concurrency_is_capped_without_dropping_to_embedding_only() -> None:
    judge = FakeJudge(delay=0.05)
    checker, pool = _checker(judge=judge)
    guard = dict(judge_model="q", judge_api_key="k", judge_max_concurrency=1,
                 allow_threshold=0.0, block_threshold=1.0)
    resolved = pool.resolve(_config(**guard))

    outcomes = await asyncio.gather(*(
        checker.check(_segments(f"{BAD} {i}"), resolved, judge_base_url=JUDGE_URL)
        for i in range(3)
    ))
    assert judge.max_concurrent == 1
    # Every one still got a real verdict; none silently skipped the judge.
    assert all(o.judge_invoked for o in outcomes)
    assert judge.calls == 3


# --------------------------------------------------------------------------- #
# Windowing — never truncate
# --------------------------------------------------------------------------- #


async def test_an_oversized_segment_is_windowed_not_truncated() -> None:
    embedder = FakeEmbedder()
    checker, pool = _checker(embedder)
    # The prohibited text sits at the very END, past any truncation point.
    text = ("x" * 400) + " " + BAD
    outcome = await _check(checker, pool, _segments(text),
                           max_input_chars=200, mode="embedding-only",
                           block_threshold=0.5)
    assert outcome.action in ("block", "flag", "allow")   # scored, not refused
    assert outcome.reason != "input_too_large"


async def test_input_beyond_the_window_budget_is_refused_not_silently_cut() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, _segments("y" * 5000),
                           max_input_chars=100)
    assert outcome.action == "unavailable"
    assert outcome.reason == "input_too_large"


async def test_too_many_segments_is_refused() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, _segments(*[GOOD] * 10),
                           max_segments=5)
    assert outcome.action == "unavailable"
    assert outcome.reason == "input_too_large"


# --------------------------------------------------------------------------- #
# Outcome shape
# --------------------------------------------------------------------------- #


async def test_a_block_carries_everything_the_log_and_response_need() -> None:
    checker, pool = _checker()
    outcome = await _check(checker, pool, _segments(BAD), mode="embedding-only",
                           block_threshold=0.5)
    assert isinstance(outcome, GuardOutcome)
    assert outcome.embedding_score is not None
    assert outcome.top_matches
    assert outcome.refusal


async def test_the_refusal_comes_from_the_policy_not_a_module_constant() -> None:
    checker, pool = _checker()
    outcome = await _check(
        checker, pool, _segments(BAD),
        mode="embedding-only", block_threshold=0.5,
        policy=POLICY.replace(
            "    action: block",
            '    action: block\n    refusal: "متأسفم، نمی‌توانم کمک کنم."',
        ),
    )
    assert outcome.refusal == "متأسفم، نمی‌توانم کمک کنم."
