"""Client for the APP's config endpoint.

The APP (a separate project) owns which models each client uses, and it also
MINTS the client keys. This service never sees a project id and never creates
a key: a client calls us with the key the APP gave it, and we turn around and
ask the APP — presenting that same key — for six things:

    model            + model_api_key        (chat LLM on the mlops endpoint)
    embed_model      + embed_api_key        (embedding model for the cache)
    extractor_model  + extractor_api_key    (entity extractor; null → unused)

The mlops serving BASE URLS come from our own env (SC_LLM_BASE_URL,
SC_EMBED_BASE_URL, SC_EXTRACTOR_BASE_URL) — the APP only supplies model names
and keys.

The APP contract:

    GET {SC_APP_CONFIG_URL}                    (a single fixed URL)
    Authorization: Bearer <the caller's own key>
    Accept: application/json

    → 200 {
        "model": "...",           "model_api_key": "...",
        "embed_model": "...",     "embed_api_key": "...",
        "extractor_model": null,  "extractor_api_key": null,
        "extractor_domain": null,          # "medical" | "legal" when set
        "project_id": "...",               # optional — used as cache scope
        "cache_config": {}                 # optional per-project overrides
      }
    → 401/403/404 when the key is unknown / has no config.

The APP identifies the client from the bearer, so the URL is used verbatim —
there is no id to substitute. Responses are cached in memory (keyed by the
key) for a short TTL so the APP is not called on every completion.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx

_REQUIRED = ("model", "model_api_key", "embed_model", "embed_api_key")


class AppConfigError(Exception):
    """The APP could not supply a config for this key."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class AppConfigClient:
    """Fetches (and briefly caches) per-key model configs from the APP."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        ttl: float = 60.0,
        timeout: float = 10.0,
    ) -> None:
        self._client = client
        self._url = url
        self._ttl = ttl
        self._timeout = timeout
        self._lock = threading.Lock()
        # api_key -> (expires_at_monotonic, config)
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    async def resolve(self, api_key: str) -> Dict[str, Any]:
        """The config for a key, from cache or fresh from the APP.

        Raises AppConfigError: 403 when the APP rejects the key, 502 when the
        APP itself is unreachable or answers garbage."""
        with self._lock:
            entry = self._cache.get(api_key)
            if entry and entry[0] > time.monotonic():
                return entry[1]

        try:
            response = await self._client.get(
                self._url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
        except httpx.HTTPError as e:
            raise AppConfigError(502, f"APP config endpoint unreachable: {e}")

        if response.status_code in (401, 403, 404):
            raise AppConfigError(
                403, "The APP has no model config for this key."
            )
        if response.status_code != 200:
            raise AppConfigError(
                502, f"APP config endpoint returned {response.status_code}."
            )

        try:
            raw = response.json()
        except ValueError:
            raise AppConfigError(502, "APP config endpoint returned invalid JSON.")
        config = self._normalize(raw)

        with self._lock:
            self._cache[api_key] = (time.monotonic() + self._ttl, config)
        return config

    def invalidate(self, api_key: Optional[str] = None) -> None:
        with self._lock:
            if api_key is None:
                self._cache.clear()
            else:
                self._cache.pop(api_key, None)

    @staticmethod
    def _pick(raw: Dict[str, Any], *names: str) -> Any:
        """First non-empty value among ``names`` — the APP's field names are
        matched by alias so both spellings work."""
        for name in names:
            value = raw.get(name)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _normalize(cls, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise AppConfigError(502, "APP config must be a JSON object.")

        # The APP's own names come first; the generic ones are accepted too.
        # 'extaractor' is the APP's spelling — kept verbatim on purpose.
        model = cls._pick(raw, "llm_model", "model")
        model_key = cls._pick(raw, "llm_key", "model_api_key", "llm_api_key")
        embed_model = cls._pick(raw, "embedd_model", "embed_model", "embedding_model")
        embed_key = cls._pick(
            raw, "embedd_key", "embed_api_key", "embedding_key", "embedding_api_key"
        )

        missing = [
            name for name, value in (
                ("llm_model", model), ("llm_key", model_key),
                ("embedd_model", embed_model), ("embedd_key", embed_key),
            ) if not value
        ]
        if missing:
            raise AppConfigError(
                502, f"APP config is missing required fields: {missing}."
            )

        return {
            "model": str(model),
            "model_api_key": str(model_key),
            "embed_model": str(embed_model),
            "embed_api_key": str(embed_key),
            "extractor_model": cls._pick(
                raw, "extaractor", "extractor", "extractor_model"
            ),
            "extractor_api_key": cls._pick(
                raw, "extaractor_key", "extractor_key", "extractor_api_key"
            ),
            "extractor_domain": cls._pick(raw, "extractor_domain", "domain"),
            "cache_config": raw.get("cache_config") or {},
            "project_id": raw.get("project_id"),
            **cls._guard_fields(raw),
        }

    @classmethod
    def _guard_fields(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Passes the optional guard block through, plus two hashes.

        UNVALIDATED, exactly like ``cache_config``: a bad guard block should
        surface as a router-level 502 with a client-safe message, not as a
        config-client failure that also takes down the cache path.

        The hashes are computed HERE, not per request, because ``resolve``
        TTL-caches this dict and hands back the identical object for the whole
        window — so they cost once per TTL per key. At the 5000-exemplar cap a
        policy is a few hundred KB, and json.dumps + sha256 on it per request
        would be real event-loop-blocking CPU in a single-worker process.
        """
        guard = cls._pick(raw, "guard", "guardrail")
        if not guard:
            return {"guard": {}, "guard_row_hash": None, "guard_policy_hash": None}

        try:
            row_hash = hashlib.sha256(
                json.dumps(guard, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception:  # noqa: BLE001 — a guard block we cannot even hash
            row_hash = None

        policy_hash = None
        if isinstance(guard, dict) and isinstance(guard.get("policy"), str):
            policy_hash = hashlib.sha256(
                guard["policy"].encode("utf-8")
            ).hexdigest()

        return {
            "guard": guard,
            "guard_row_hash": row_hash,
            "guard_policy_hash": policy_hash,
        }
