"""Guard pool — resolve memo, pool-owned build, LRU, segment memo, breaker."""

import asyncio
import hashlib

import numpy as np
import pytest

from semantic_cache.gateway.guard_config import GuardConfigError
from semantic_cache.gateway.guard_pool import (
    GuardBuildError,
    GuardPool,
    SegmentOutcome,
)

EMBED_URL = "https://embed.example.com"
DIM = 64

POLICY = """
categories:
  - category_id: competitors
    action: block
    disallowed_exemplars:
      - "What do you think of Rivalco's product?"
      - "Is Rivalco better than you?"
    allowed_exemplars:
      - "What is your refund policy?"
      - "How do I reset my password?"
"""


class FakeEmbedder:
    """Hashed word-trigram double — texts sharing words get close vectors.

    A pure hash of the whole string would be uncorrelated noise, under which no
    policy could ever pass the separability self-test. This behaves enough like
    a real embedder for these tests: shared vocabulary means high cosine.
    """

    def __init__(self, *, delay: float = 0.0, fail: bool = False,
                 salt: str = "") -> None:
        self.call_count = 0
        self.text_count = 0
        self.delay = delay
        self.fail = fail
        self.salt = salt

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        words = "".join(
            c.lower() if c.isalnum() else " " for c in text
        ).split()
        for word in words:
            for size in (len(word), 3):
                for start in range(max(1, len(word) - size + 1)):
                    token = word[start:start + size]
                    bucket = int(
                        hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16
                    ) % DIM
                    vector[bucket] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        vector = vector / norm
        if self.salt:
            # A different MODEL, not a different quality: rolling the axes is a
            # permutation, so every pairwise cosine is preserved exactly while
            # every vector changes. That is what a swapped-but-comparable model
            # looks like, and it is precisely what only the sentinel can catch.
            shift = int(hashlib.sha256(self.salt.encode()).hexdigest()[:4], 16)
            vector = np.roll(vector, shift % DIM)
        return vector

    async def embed(self, texts, *, input_type="document", timeout=None):
        # `timeout` mirrors the real GuardEmbedder. A double that does not
        # accept what the pool actually passes is how a wiring gap survives a
        # green suite; TimeoutRecordingEmbedder below asserts on the value.
        self.call_count += 1
        self.text_count += len(texts)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("embedding endpoint is down")
        return np.ascontiguousarray(
            np.vstack([self._vector(t) for t in texts]), dtype=np.float32
        )


class FakeStore:
    def __init__(self) -> None:
        self.rows = {}
        self.saves = 0
        self.loads = 0
        self.deletes = 0
        self.touches = 0

    def save_guard_index(self, index_key, embed_model, policy_hash, dim, n,
                         vectors, sentinel, meta):
        self.saves += 1
        self.rows[index_key] = {
            "embed_model": embed_model, "policy_hash": policy_hash,
            "dim": dim, "n": n, "vectors": vectors, "sentinel": sentinel,
            "meta": meta,
        }
        return True

    def load_guard_index(self, index_key):
        self.loads += 1
        return self.rows.get(index_key)

    def delete_guard_index(self, index_key):
        self.deletes += 1
        return 1 if self.rows.pop(index_key, None) else 0

    def touch_guard_index(self, index_key):
        self.touches += 1
        return True


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(**guard):
    # min_similarity is tuned to FakeEmbedder, whose sparse trigram vectors top
    # out around 0.47 for related sentences where a real embedder reaches 0.8+.
    # These tests are about pool mechanics; scoring quality is guard_logic's.
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


def _pool(store=None, **kwargs):
    kwargs.setdefault("embed_base_url", EMBED_URL)
    return GuardPool(store or FakeStore(), **kwargs)


# --------------------------------------------------------------------------- #
# Resolve memo
# --------------------------------------------------------------------------- #


def test_resolve_returns_none_when_the_guard_is_off() -> None:
    pool = _pool()
    assert pool.resolve({"guard": None, "guard_row_hash": None}) is None


def test_resolve_is_memoised_on_the_precomputed_row_hash() -> None:
    pool = _pool()
    config = _config()
    first = pool.resolve(config)
    second = pool.resolve(config)
    assert first is second          # the identical object, not an equal one


def test_a_broken_policy_is_parsed_once_per_window_not_once_per_request() -> None:
    pool = _pool()
    config = _config(policy="categories: []")
    errors = []
    for _ in range(5):
        with pytest.raises(GuardConfigError) as excinfo:
            pool.resolve(config)
        errors.append(excinfo.value)
    # The same cached exception object comes back every time.
    assert all(e is errors[0] for e in errors)


def test_the_resolve_memo_is_bounded() -> None:
    pool = _pool(resolve_cache_size=3)
    for i in range(10):
        pool.resolve(_config(default_refusal=f"no {i}"))
    assert len(pool._resolved) == 3


def test_a_config_change_resolves_afresh() -> None:
    pool = _pool()
    a = pool.resolve(_config(block_threshold=0.85))
    b = pool.resolve(_config(block_threshold=0.90))
    assert a is not b
    assert a.params_hash != b.params_hash
    assert a.index_key == b.index_key       # thresholds never force a re-embed


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


async def test_a_first_request_builds_and_persists_the_index() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    pool = _pool(store)
    resolved = pool.resolve(_config())
    index = await pool.get_index(resolved, embedder)

    assert index.n == 4                       # 2 disallowed + 2 allowed
    assert index.dim == DIM
    assert store.saves == 1
    assert pool.builds == 1


async def test_a_second_request_reuses_the_in_process_index() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    pool = _pool(store)
    resolved = pool.resolve(_config())
    first = await pool.get_index(resolved, embedder)
    calls = embedder.call_count
    second = await pool.get_index(resolved, embedder)
    assert second is first
    assert embedder.call_count == calls


async def test_fifty_concurrent_first_requests_build_exactly_once() -> None:
    store, embedder = FakeStore(), FakeEmbedder(delay=0.02)
    pool = _pool(store)
    resolved = pool.resolve(_config())
    results = await asyncio.gather(
        *(pool.get_index(resolved, embedder) for _ in range(50))
    )
    assert all(r is results[0] for r in results)
    assert pool.builds == 1


async def test_a_stored_blob_is_reused_with_no_exemplar_embedding() -> None:
    store = FakeStore()
    resolved = _pool(store).resolve(_config())
    await _pool(store).get_index(resolved, FakeEmbedder())
    assert store.saves == 1

    # A fresh process, same store: only the sentinel is embedded.
    fresh_pool, fresh_embedder = _pool(store), FakeEmbedder()
    index = await fresh_pool.get_index(resolved, fresh_embedder)
    assert index.n == 4
    assert fresh_pool.builds == 0
    assert fresh_pool.blob_hits == 1
    assert fresh_embedder.text_count == 1          # the sentinel, nothing else


async def test_a_model_swapped_behind_the_same_alias_forces_a_rebuild() -> None:
    store = FakeStore()
    resolved = _pool(store).resolve(_config())
    await _pool(store).get_index(resolved, FakeEmbedder())

    # Same model NAME and URL, different vectors — only the sentinel reveals it.
    swapped = FakeEmbedder(salt="different-model")
    pool = _pool(store)
    index = await pool.get_index(resolved, swapped)
    assert store.deletes == 1
    assert pool.builds == 1
    assert index.n == 4


async def test_a_stored_row_whose_size_no_longer_matches_is_discarded() -> None:
    store = FakeStore()
    resolved = _pool(store).resolve(_config())
    await _pool(store).get_index(resolved, FakeEmbedder())
    store.rows[resolved.index_key]["n"] = 99

    pool = _pool(store)
    await pool.get_index(resolved, FakeEmbedder())
    assert store.deletes == 1
    assert pool.builds == 1


async def test_a_persist_failure_does_not_fail_the_build() -> None:
    store = FakeStore()
    store.save_guard_index = lambda *a, **k: False
    pool = _pool(store)
    index = await pool.get_index(pool.resolve(_config()), FakeEmbedder())
    assert index.n == 4


# --------------------------------------------------------------------------- #
# THE LIVELOCK REGRESSION — why the build is shielded
# --------------------------------------------------------------------------- #


async def test_a_caller_that_times_out_does_not_kill_the_build() -> None:
    """Without asyncio.shield this test fails, which is the point.

    Each request would cancel the build it is waiting on, the next would start
    from zero, and a policy slower to build than one request's budget could
    never finish at all.
    """
    store, embedder = FakeStore(), FakeEmbedder(delay=0.20)
    pool = _pool(store)
    resolved = pool.resolve(_config())

    with pytest.raises(GuardBuildError):
        await pool.get_index(resolved, embedder, budget=0.01)

    entry = pool._indexes[resolved.index_key]
    assert entry.task is not None
    assert not entry.task.done()          # still running, not cancelled

    # A later request finds the SAME build, finished.
    index = await pool.get_index(resolved, embedder, budget=2.0)
    assert index.n == 4
    assert pool.builds == 1               # one build total, not two


async def test_a_failed_build_is_retried_rather_than_cached_forever() -> None:
    store = FakeStore()
    pool = _pool(store)
    resolved = pool.resolve(_config())

    broken = FakeEmbedder(fail=True)
    with pytest.raises(RuntimeError):
        await pool.get_index(resolved, broken)

    index = await pool.get_index(resolved, FakeEmbedder())
    assert index.n == 4


# --------------------------------------------------------------------------- #
# Separability self-test
# --------------------------------------------------------------------------- #


async def test_a_policy_that_cannot_separate_its_own_examples_is_rejected() -> None:
    # allow_threshold == block_threshold == 0 makes every allowed exemplar
    # score at or above the block threshold.
    pool = _pool()
    resolved = pool.resolve(_config(allow_threshold=0.0, block_threshold=0.0))
    with pytest.raises(GuardConfigError, match="cannot separate its own examples"):
        await pool.get_index(resolved, FakeEmbedder())


# --------------------------------------------------------------------------- #
# LRU
# --------------------------------------------------------------------------- #


async def test_the_index_lru_is_bounded_by_entry_count() -> None:
    pool = _pool(max_indexes=2)
    for i in range(4):
        resolved = pool.resolve(_config(
            policy=POLICY.replace("Rivalco's", f"Rivalco{i}'s")
        ))
        await pool.get_index(resolved, FakeEmbedder())
    assert len(pool._indexes) == 2


async def test_the_index_lru_is_also_bounded_by_bytes() -> None:
    # One policy can be hundreds of times bigger than another, so entry count
    # alone is not a memory bound.
    pool = _pool(max_indexes=100, cache_max_bytes=1)
    for i in range(3):
        resolved = pool.resolve(_config(
            policy=POLICY.replace("Rivalco's", f"Rivalco{i}'s")
        ))
        await pool.get_index(resolved, FakeEmbedder())
    assert len(pool._indexes) == 1


async def test_eviction_drops_the_entry_and_its_task() -> None:
    pool = _pool(max_indexes=1)
    keys = []
    for i in range(3):
        resolved = pool.resolve(_config(
            policy=POLICY.replace("Rivalco's", f"Rivalco{i}'s")
        ))
        keys.append(resolved.index_key)
        await pool.get_index(resolved, FakeEmbedder())
    # No task or lock survives for an evicted policy — otherwise every policy
    # edit would leak one, uncounted against the byte budget.
    assert set(pool._indexes) == {keys[-1]}


async def test_invalidate_index_forgets_it() -> None:
    pool = _pool()
    resolved = pool.resolve(_config())
    await pool.get_index(resolved, FakeEmbedder())
    pool.invalidate_index(resolved.index_key)
    assert resolved.index_key not in pool._indexes


# --------------------------------------------------------------------------- #
# Segment memo
# --------------------------------------------------------------------------- #


def _outcome(score=0.5):
    return SegmentOutcome(score=score, category=None, judge_score=None,
                          judge_rationale=None)


def test_the_memo_round_trips_a_segment_outcome() -> None:
    pool = _pool()
    resolved = pool.resolve(_config())
    key = pool.segment_key(resolved, "hello there")
    assert pool.memo_get(key) is None            # a miss means "run the check"
    pool.memo_put(key, _outcome(0.7))
    assert pool.memo_get(key).score == 0.7


def test_changing_any_decision_parameter_invalidates_the_memo() -> None:
    pool = _pool()
    base = pool.resolve(_config())
    pool.memo_put(pool.segment_key(base, "hello"), _outcome(0.7))
    for change in ({"block_threshold": 0.9}, {"top_k": 3},
                   {"min_similarity": 0.1}, {"check_roles": ["user"]}):
        variant = pool.resolve(_config(**change))
        assert pool.memo_get(pool.segment_key(variant, "hello")) is None, change


def test_editing_the_policy_invalidates_the_memo() -> None:
    pool = _pool()
    base = pool.resolve(_config())
    pool.memo_put(pool.segment_key(base, "hello"), _outcome(0.7))
    edited = pool.resolve(_config(policy=POLICY + '      - "one more"\n'))
    assert pool.memo_get(pool.segment_key(edited, "hello")) is None


def test_memo_entries_expire() -> None:
    clock = FakeClock()
    pool = _pool(segment_memo_ttl=600.0, clock=clock)
    resolved = pool.resolve(_config())
    key = pool.segment_key(resolved, "hello")
    pool.memo_put(key, _outcome())
    clock.advance(599)
    assert pool.memo_get(key) is not None
    clock.advance(2)
    assert pool.memo_get(key) is None


def test_the_memo_is_bounded() -> None:
    pool = _pool(segment_memo_size=5)
    resolved = pool.resolve(_config())
    for i in range(20):
        pool.memo_put(pool.segment_key(resolved, f"text {i}"), _outcome())
    assert len(pool._segments) == 5


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


def test_the_breaker_opens_after_repeated_failures() -> None:
    clock = FakeClock()
    pool = _pool(breaker_threshold=10, clock=clock)
    assert pool.breaker_is_open("idx") is False
    for _ in range(9):
        pool.record_failure("idx")
    assert pool.breaker_is_open("idx") is False
    pool.record_failure("idx")
    assert pool.breaker_is_open("idx") is True


def test_the_breaker_half_opens_after_the_cooldown() -> None:
    clock = FakeClock()
    pool = _pool(breaker_threshold=2, breaker_cooldown=30.0, clock=clock)
    pool.record_failure("idx")
    pool.record_failure("idx")
    assert pool.breaker_is_open("idx") is True
    clock.advance(31)
    assert pool.breaker_is_open("idx") is False


def test_a_success_resets_the_breaker() -> None:
    pool = _pool(breaker_threshold=3)
    pool.record_failure("idx")
    pool.record_failure("idx")
    pool.record_success("idx")
    pool.record_failure("idx")
    assert pool.breaker_is_open("idx") is False


def test_failures_spread_beyond_the_window_do_not_accumulate() -> None:
    clock = FakeClock()
    pool = _pool(breaker_threshold=3, breaker_window=60.0, clock=clock)
    pool.record_failure("idx")
    pool.record_failure("idx")
    clock.advance(61)
    pool.record_failure("idx")
    assert pool.breaker_is_open("idx") is False


def test_breakers_are_per_index_not_global() -> None:
    pool = _pool(breaker_threshold=2)
    pool.record_failure("a")
    pool.record_failure("a")
    assert pool.breaker_is_open("a") is True
    assert pool.breaker_is_open("b") is False


# --------------------------------------------------------------------------- #
# Build budget
# --------------------------------------------------------------------------- #


class TimeoutRecordingEmbedder(FakeEmbedder):
    """FakeEmbedder that records the per-call timeout the pool asked for."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.timeouts = []

    async def embed(self, texts, *, input_type="document", timeout=None):
        self.timeouts.append(timeout)
        return await super().embed(texts, input_type=input_type)


async def test_the_build_runs_on_the_build_budget_not_the_request_budget() -> None:
    """SC_GUARD_BUILD_TIMEOUT is the documented budget for embedding a policy,
    and the build is shielded and server-owned precisely so it may take that
    long. It shared the REQUEST embedder's much shorter timeout, so the build
    budget was unreachable and a policy could never finish building against an
    endpoint slower than one request."""
    pool = _pool(build_timeout=45.0)
    embedder = TimeoutRecordingEmbedder()
    await pool.get_index(pool.resolve(_config()), embedder)

    assert embedder.timeouts, "the build made no embedding call"
    assert all(t == 45.0 for t in embedder.timeouts), embedder.timeouts


async def test_the_stored_index_sentinel_check_also_gets_the_build_budget() -> None:
    """The warm path still makes one live embedding call to verify the
    sentinel. On the request timeout that call fails on exactly the endpoints
    the cold build already failed on — a warm start was no safer."""
    store = FakeStore()
    pool = _pool(store, build_timeout=45.0)
    resolved = pool.resolve(_config())
    await pool.get_index(resolved, TimeoutRecordingEmbedder())

    reloaded = _pool(store, build_timeout=45.0)
    embedder = TimeoutRecordingEmbedder()
    await reloaded.get_index(reloaded.resolve(_config()), embedder)

    assert embedder.timeouts == [45.0]
