"""Conftest for the unit lane.

Tests here touch NO backend — no Redis, no embedding model, no network — so
they run fast in CI on a bare runner. There are intentionally no connection-
opening fixtures; anything a unit test needs is constructed inline with mocks.
(The Redis-backed fixtures live in `tests/integration/conftest.py`.)
"""
