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
    X-Service-Key: <SC_APP_SERVICE_KEY>        (optional; see below)
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

TWO CREDENTIALS, TWO DIFFERENT QUESTIONS
----------------------------------------
The bearer answers "WHICH CLIENT is this for?" and is the caller's own key,
forwarded. The service key answers "IS THIS THE CACHE SERVICE ASKING?" and is a
long-lived secret shared between this service and the APP. They are not
interchangeable and they fail differently.

The service key matters because without it the config endpoint is guarded by
the client key alone — so anyone who obtains a client key can query the APP
directly and receive that client's MODEL PROVIDER CREDENTIALS. A client key is
handed to end users and travels through their infrastructure; a service key
lives only in this service's environment. Requiring both means a leaked client
key no longer yields provider credentials on its own.

It is sent in a SEPARATE HEADER because Authorization is already carrying the
client's key, and the name is configurable (SC_APP_SERVICE_KEY_HEADER, default
X-Service-Key) so the APP team can pick it without a code change here.

It is OPTIONAL: unset, nothing is sent and the contract is exactly as it was.
That keeps the rollout ordered — deploy us with the key, then have the APP
start requiring it — rather than requiring both sides to change at once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_REQUIRED = ("model", "model_api_key", "embed_model", "embed_api_key")

#: Default ceiling on cached configs. The cache is keyed by CLIENT KEY, so its
#: size is chosen by our callers, not by us: an unbounded dict grows with every
#: distinct bearer ever presented — including every wrong one, which makes it a
#: memory-exhaustion primitive anyone can pull. Each entry also holds live
#: provider credentials, so "how many secrets does this process hold" was
#: likewise unbounded.
DEFAULT_MAX_ENTRIES = 10_000

#: Header the service credential travels in. NOT Authorization — that one is
#: already carrying the client's key, and collapsing two different claims into
#: one header is how you end up unable to tell them apart at the other end.
DEFAULT_SERVICE_KEY_HEADER = "X-Service-Key"

#: Per-PROCESS pepper. Cache keys are HMACs under it, so the dict holds no
#: value derived from a client key that survives the process or means anything
#: outside it — a heap dump cannot be replayed against a captured key to
#: confirm it was used here. It is deliberately not configurable: nothing needs
#: these fingerprints to be stable across restarts.
_PEPPER = os.urandom(32)


def _fingerprint(api_key: str) -> str:
    return hmac.new(_PEPPER, api_key.encode("utf-8"), hashlib.sha256).hexdigest()


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
        max_entries: int = DEFAULT_MAX_ENTRIES,
        service_key: Optional[str] = None,
        service_key_header: str = DEFAULT_SERVICE_KEY_HEADER,
    ) -> None:
        self._client = client
        self._url = url
        self._ttl = ttl
        self._timeout = timeout
        self._service_key = service_key or None
        self._service_key_header = service_key_header or DEFAULT_SERVICE_KEY_HEADER
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        # fingerprint(api_key) -> (expires_at_monotonic, config)
        # An OrderedDict used as an LRU: bounded in size, and eviction takes
        # the least recently USED entry rather than the oldest inserted, so a
        # burst of unknown keys cannot flush the working set of live ones.
        self._cache: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()

    async def resolve(self, api_key: str) -> Dict[str, Any]:
        """The config for a key, from cache or fresh from the APP.

        Raises AppConfigError: 403 when the APP rejects the key, 502 when the
        APP itself is unreachable or answers garbage."""
        cache_key = _fingerprint(api_key)
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if entry[0] > time.monotonic():
                    self._cache.move_to_end(cache_key)
                    return entry[1]
                # Expired: drop it now rather than leaving it to eviction, so a
                # revoked key's credentials do not linger in memory until
                # something else needs the space.
                del self._cache[cache_key]

        try:
            response = await self._client.get(
                self._url,
                headers=self._headers(api_key),
                timeout=self._timeout,
            )
        except httpx.HTTPError as e:
            raise AppConfigError(502, f"APP config endpoint unreachable: {e}")

        if response.status_code in (401, 403, 404):
            # The CLIENT-FACING message stays identical for all three, and says
            # nothing about which credential failed — otherwise this endpoint
            # becomes an oracle for probing which client keys exist.
            #
            # The LOG is where the two causes get separated. A rejected service
            # key and an unknown client key are different incidents with
            # different fixes, and "no model config for this key" sends an
            # operator to look at the client's key when the real answer may be
            # that SC_APP_SERVICE_KEY is wrong or missing. 401 in particular
            # means "caller not authenticated", which is our credential, not
            # theirs.
            logger.warning(
                "APP config refused: HTTP %s from %s (we %s a service key). "
                "401 usually means OUR SC_APP_SERVICE_KEY is wrong or absent; "
                "403/404 usually means the APP does not know this client key.",
                response.status_code,
                self._url,
                "presented" if self._service_key else "did NOT present",
            )
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
            self._cache[cache_key] = (time.monotonic() + self._ttl, config)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return config

    def _headers(self, api_key: str) -> Dict[str, str]:
        """The caller's bearer, plus our own service credential when set.

        Two credentials answering two questions: the bearer says which client,
        the service key says that it is us asking."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if self._service_key:
            headers[self._service_key_header] = self._service_key
        return headers

    @property
    def authenticates_as_a_service(self) -> bool:
        """False → we present only the caller's key, and a leaked client key is
        enough to pull that client's provider credentials from the APP."""
        return bool(self._service_key)

    def invalidate(self, api_key: Optional[str] = None) -> None:
        """Drops one key's config, or all of them. Called on rotation and
        revocation — a revoked key must stop working now, not in up to
        SC_APP_CONFIG_TTL seconds."""
        with self._lock:
            if api_key is None:
                self._cache.clear()
            else:
                self._cache.pop(_fingerprint(api_key), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

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
