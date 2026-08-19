"""Bounded guard decision log — batching, overflow, resilience, drain."""

import asyncio

import pytest

from semantic_cache.gateway.guard_log import GuardDecisionLog


class FakeStore:
    """Records every batch handed to log_guard_decisions."""

    def __init__(self) -> None:
        self.batches = []
        self.fail_times = 0

    def log_guard_decisions(self, rows):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("database is down")
        self.batches.append(list(rows))
        return len(rows)

    @property
    def rows(self):
        return [row for batch in self.batches for row in batch]


def _row(i: int):
    return {"scope": "s", "policy_hash": "p", "action": "allow", "n": i}


async def _settle(log: GuardDecisionLog, expected: int, timeout: float = 2.0):
    """Waits until the store has seen `expected` rows, or gives up."""
    deadline = asyncio.get_running_loop().time() + timeout
    while log.written < expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


async def test_rows_are_written_in_batches_not_one_round_trip_each() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store, batch=50)
    log.start()
    for i in range(120):
        log.submit(_row(i))
    await _settle(log, 120)
    await log.drain()

    assert len(store.rows) == 120
    # 120 rows must not cost 120 round trips on an 8-connection pool.
    assert len(store.batches) <= 10
    assert max(len(b) for b in store.batches) > 1


async def test_every_submitted_row_survives_in_order() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store, batch=10)
    log.start()
    for i in range(25):
        log.submit(_row(i))
    await _settle(log, 25)
    await log.drain()
    assert [r["n"] for r in store.rows] == list(range(25))


async def test_submit_is_non_blocking_and_returns_immediately() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store)
    # Not started: nothing consumes, yet submit still returns synchronously.
    assert log.submit(_row(0)) is True
    assert log.backlog == 1
    assert store.batches == []


# --------------------------------------------------------------------------- #
# Overflow
# --------------------------------------------------------------------------- #


async def test_a_full_queue_drops_and_counts_rather_than_blocking() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store, maxsize=10)   # drainer deliberately not started
    accepted = sum(1 for i in range(20) if log.submit(_row(i)))
    assert accepted == 10
    assert log.dropped == 10
    assert log.backlog == 10


async def test_dropping_never_raises() -> None:
    log = GuardDecisionLog(FakeStore(), maxsize=1)
    for i in range(50):
        log.submit(_row(i))          # must not raise
    assert log.dropped == 49


# --------------------------------------------------------------------------- #
# Resilience
# --------------------------------------------------------------------------- #


async def test_a_failing_write_does_not_kill_the_drainer() -> None:
    store = FakeStore()
    store.fail_times = 1
    log = GuardDecisionLog(store, batch=5, flush_interval=0.01)
    log.start()
    log.submit(_row(0))
    await asyncio.sleep(0.15)        # let the failure happen and be logged

    for i in range(1, 4):
        log.submit(_row(i))
    await _settle(log, 3)
    await log.drain()

    # The first batch was lost, but logging continued afterwards.
    assert len(store.rows) >= 3


async def test_start_is_idempotent() -> None:
    log = GuardDecisionLog(FakeStore())
    log.start()
    first = log._task
    log.start()
    assert log._task is first
    await log.drain()


# --------------------------------------------------------------------------- #
# Drain
# --------------------------------------------------------------------------- #


async def test_drain_flushes_what_is_still_queued() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store, batch=1000)   # never fills, so nothing auto-flushes
    log.start()
    await asyncio.sleep(0)
    for i in range(7):
        log.submit(_row(i))
    await log.drain()
    assert len(store.rows) == 7


async def test_drain_on_an_empty_queue_is_a_no_op() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store)
    log.start()
    await log.drain()
    assert store.batches == []


async def test_drain_is_idempotent() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store)
    log.start()
    log.submit(_row(0))
    await log.drain()
    await log.drain()
    assert len(store.rows) == 1


async def test_submitting_after_drain_is_refused_not_silently_queued() -> None:
    log = GuardDecisionLog(FakeStore())
    log.start()
    await log.drain()
    assert log.submit(_row(0)) is False


async def test_drain_without_start_still_flushes() -> None:
    store = FakeStore()
    log = GuardDecisionLog(store)
    log.submit(_row(0))
    await log.drain()
    assert len(store.rows) == 1
