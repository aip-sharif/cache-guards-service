"""Bounded, batched writer for the guard's decision log.

Every guarded request produces a decision row — allows included, since "the
guard saw this and let it through" is the half of the audit trail people
actually need when something gets past.

**Why not one await per request.** ``PostgresGatewayStore`` opens a pool of at
most ``max_size`` connections (8 by default), and the router already awaits one
``asyncio.to_thread`` for the message log on every completion, on top of the
cache's write-through backup hooks. Adding a second synchronous round trip per
request — for allows too — would multiply contention on that handful of
connections for data nobody reads synchronously. So ``submit()`` is a
non-blocking enqueue and one background task writes in batches.

**Why not bare create_task per row.** A task per row is unbounded: a Postgres
stall would accumulate rows and tasks without limit, and the server's shutdown
hook closes the store without awaiting any of them. A bounded queue plus one
drainer makes the backlog explicit (and droppable, loudly) and gives shutdown
something to await.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DROP_LOG_EVERY = 100


class GuardDecisionLog:
    """Queue in front of ``PostgresGatewayStore.log_guard_decisions``."""

    def __init__(
        self,
        store: Any,
        *,
        maxsize: int = 10000,
        batch: int = 50,
        flush_interval: float = 1.0,
    ) -> None:
        self._store = store
        self._batch = max(1, batch)
        self._flush_interval = flush_interval
        self._queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=maxsize)
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        #: Rows lost to a full queue. A guard that quietly stops recording its
        #: decisions is a guard nobody can audit, so this is counted and logged.
        self.dropped = 0
        self.written = 0

    # -- producer side ------------------------------------------------------ #

    def submit(self, row: Dict[str, Any]) -> bool:
        """Enqueues one decision. Synchronous, non-blocking, never raises.

        Starts the drainer on first use if nothing else has. The server calls
        ``start()`` from its startup hook, but relying on that alone is a quiet
        failure mode: if the hook is ever dropped or the app is embedded
        somewhere that skips lifespan events, rows queue into a bounded buffer
        and are silently discarded — the guard still ENFORCES, but its audit
        trail is gone with no error anywhere. Self-starting removes that
        coupling.
        """
        if self._closed:
            return False
        self._ensure_started()
        try:
            self._queue.put_nowait(row)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % _DROP_LOG_EVERY == 1:
                logger.warning(
                    "Guard decision log is full; dropped %d row(s) so far. "
                    "Decisions are still being ENFORCED — only the audit trail "
                    "is losing rows.",
                    self.dropped,
                )
            return False

    # -- consumer side ------------------------------------------------------ #

    def start(self) -> None:
        """Starts the drainer. Idempotent."""
        if self._task is None or self._task.done():
            self._closed = False
            self._task = asyncio.create_task(self._run(), name="guard-decision-log")

    def _ensure_started(self) -> None:
        """Starts the drainer if a loop is running and it is not up yet."""
        if self._task is not None and not self._task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (a sync test, or process teardown) — drain() covers it
        self.start()

    async def _run(self) -> None:
        while True:
            try:
                rows = await self._collect()
                if rows:
                    await self._write(rows)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # The drainer must outlive any single failure — otherwise one
                # bad batch silently ends all decision logging for the process.
                logger.error("Guard decision log drainer error: %s", e)
                await asyncio.sleep(self._flush_interval)

    async def _collect(self) -> List[Dict[str, Any]]:
        """Waits for one row, then takes up to ``batch`` more without waiting."""
        first = await self._queue.get()
        rows = [first]
        while len(rows) < self._batch:
            try:
                rows.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return rows

    async def _write(self, rows: List[Dict[str, Any]]) -> None:
        written = await asyncio.to_thread(self._store.log_guard_decisions, rows)
        self.written += int(written or 0)

    # -- shutdown ------------------------------------------------------------ #

    async def drain(self) -> None:
        """Flushes what is queued, then stops the drainer. Idempotent.

        Called from the server's shutdown hook BEFORE the store is closed, so
        in-flight decisions land rather than vanishing with the process.
        """
        self._closed = True
        pending: List[Dict[str, Any]] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass

        if pending:
            try:
                await self._write(pending)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Guard decision log drain lost %d row(s): %s", len(pending), e
                )

    @property
    def backlog(self) -> int:
        return self._queue.qsize()


__all__ = ["GuardDecisionLog"]
