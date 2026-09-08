"""Bounded readiness probing for the serving path.

WHY THIS EXISTS
---------------
`/health` used to ping Redis and call that both liveness and readiness. Two
things were wrong with conflating them:

* **Readiness was a lie.** Gateway serving also needs Postgres, the APP's
  config endpoint, and the mlops endpoints. A pod with Redis up and Postgres
  down reported healthy and took traffic it could not serve.
* **Liveness was a restart loop.** The compose healthcheck drives
  `restart: unless-stopped`, so a Redis blip restarted a perfectly alive
  process — which does not bring Redis back.

So: `/health` answers "this process is running" and `/ready` answers "this
process can serve", with one line per dependency.

EVERY PROBE IS BOUNDED. A readiness endpoint that hangs because its dependency
hangs is the same outage as the dependency, plus a stuck load balancer — so
each check runs in a worker thread and is abandoned at the deadline. A probe
that overruns is reported down, never awaited.

Probes are also allowed to be *optional*: a dependency that degrades a feature
rather than stopping serving is reported honestly and does not fail the pod.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Check:
    """One dependency probe.

    `probe` returns truthy for up. It may raise — an exception is "down", never
    a 500 on the readiness endpoint itself.

    `required=False` means the check is reported but does not gate traffic."""

    name: str
    probe: Callable[[], Any]
    required: bool = True


class ReadinessProbe:
    """Runs a fixed set of checks under one wall-clock deadline."""

    def __init__(self, checks: Sequence[Check], timeout: float = 2.0) -> None:
        self._checks: List[Check] = list(checks)
        self._timeout = float(timeout)

    @property
    def checks(self) -> List[Check]:
        return list(self._checks)

    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """``(ready, detail)``. `detail` names every check and its state, so an
        operator reading a 503 knows WHICH dependency to go look at rather than
        being told only that something is wrong."""
        if not self._checks:
            return True, {"status": "ok", "checks": {}}

        results: Dict[str, Any] = {}
        # One executor per call: the checks are few and infrequent, and a
        # shared pool would let a hung probe from a previous scrape starve
        # this one.
        #
        # Deliberately NOT a `with` block. ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which joins the very worker the deadline just
        # gave up on — so a probe that hangs for 30s produced a /ready that
        # hung for 30s, timeout and all. Abandoning the thread is the point.
        pool = ThreadPoolExecutor(max_workers=len(self._checks))
        try:
            deadline = time.monotonic() + self._timeout
            futures = {c.name: pool.submit(self._safe, c) for c in self._checks}
            for check in self._checks:
                try:
                    # One shared deadline, not one per check: N checks must not
                    # cost N timeouts, or adding a dependency silently makes
                    # the endpoint slower than the scrape interval.
                    results[check.name] = futures[check.name].result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except FutureTimeout:
                    results[check.name] = "timeout"
        finally:
            # The stuck worker keeps running and is never joined here. Its own
            # socket timeouts end it; the interpreter joins any stragglers at
            # exit.
            pool.shutdown(wait=False, cancel_futures=True)

        required_down = [
            c.name
            for c in self._checks
            if c.required and results.get(c.name) != "ok"
        ]
        ready = not required_down
        detail: Dict[str, Any] = {
            "status": "ok" if ready else "unready",
            "checks": results,
        }
        if required_down:
            detail["failed"] = required_down
        return ready, detail

    @staticmethod
    def _safe(check: Check) -> str:
        try:
            return "ok" if check.probe() else "down"
        except Exception as e:  # noqa: BLE001 — a failed probe is "down"
            logger.debug("Readiness check %s failed: %s: %s", check.name,
                         type(e).__name__, e)
            return "down"


__all__ = ["Check", "ReadinessProbe"]
