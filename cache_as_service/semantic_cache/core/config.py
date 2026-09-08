"""Configuration management for Semantic Cache using Pydantic."""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    CUSTOM_API = "custom_api"


class CacheDomain(str, Enum):
    """Domains supported by the entity-aware extraction layer."""

    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"


class CacheMode(str, Enum):
    """The retrieval method a lookup uses — chosen per client (the APP returns it).

    - ``semantic``: vector KNN + cosine-similarity floor (the default). When an
      entity extractor is also configured this becomes the entity-aware variant
      automatically (domain prefilter + entity overlap + stricter threshold).
      This is the only method that computes embeddings.
    - ``bm25``: lexical full-text search over the stored question text using a
      RediSearch scorer (``lexical_scorer``, default BM25). No embeddings — the
      query's tokens are matched (intersection) against stored questions and the
      best-scoring hit above ``lexical_min_score`` wins. Cheap and good for
      keyword/repeat traffic where paraphrase generalization isn't needed.
    - ``fuzzy``: like ``bm25`` but each token is matched within a Levenshtein
      distance (``fuzzy_distance``, 1–3), so typos and small spelling variants
      still hit. No embeddings. Language-agnostic.
    - ``exact``: L0 exact-match ONLY — a single normalized-text (+scope) HGET.
      No embedding, no KNN, no extraction on read; no embedding on write. Zero
      false-hit risk and the cheapest tier.
    - ``off``: caching disabled — every request is a pure passthrough to the
      upstream LLM (the message log still records it).
    """

    SEMANTIC = "semantic"
    BM25 = "bm25"
    FUZZY = "fuzzy"
    EXACT = "exact"
    OFF = "off"


# RediSearch full-text scorers usable by the ``bm25`` (lexical) method. BM25 is
# the default and is available on every redis-stack build; the others let the
# APP pick a different ranking function without new code.
LEXICAL_SCORERS = frozenset(
    {"BM25", "BM25STD", "TFIDF", "TFIDF.DOCNORM", "DISMAX", "DOCSCORE"}
)


def _check_threshold(v: Optional[float]) -> Optional[float]:
    if v is not None and not 0.0 <= v <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")
    return v


def _check_scorer(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    canonical = v.strip().upper()
    if canonical not in LEXICAL_SCORERS:
        raise ValueError(f"scorer must be one of {sorted(LEXICAL_SCORERS)}, got {v!r}.")
    return canonical


class SemanticParams(BaseModel):
    """Per-client hyperparameters for the ``semantic`` (vector) method. Any field
    left None falls back to the flat cache config default."""

    model_config = ConfigDict(extra="forbid")
    similarity_threshold: Optional[float] = None
    entity_threshold: Optional[float] = None

    _v = field_validator("similarity_threshold", "entity_threshold")(
        lambda cls, v: _check_threshold(v)
    )


class Bm25Params(BaseModel):
    """Per-client hyperparameters for the ``bm25`` (lexical) method."""

    model_config = ConfigDict(extra="forbid")
    scorer: Optional[str] = None
    min_score: Optional[float] = None

    _v = field_validator("scorer")(lambda cls, v: _check_scorer(v))


class FuzzyParams(BaseModel):
    """Per-client hyperparameters for the ``fuzzy`` (Levenshtein) method."""

    model_config = ConfigDict(extra="forbid")
    scorer: Optional[str] = None
    min_score: Optional[float] = None
    distance: Optional[int] = None

    _vs = field_validator("scorer")(lambda cls, v: _check_scorer(v))

    @field_validator("distance")
    @classmethod
    def _vd(cls, v):
        if v is not None and not 1 <= v <= 3:
            raise ValueError("distance must be between 1 and 3.")
        return v


class SemanticCacheConfig(BaseSettings):
    """Configuration for the Semantic Caching system.

    Loads from environment variables by default. All environment variables
    can be prefixed with `SC_` (e.g., SC_REDIS_HOST).
    """

    model_config = SettingsConfigDict(env_prefix="SC_", case_sensitive=False)

    # Redis Settings
    redis_host: str = Field(default="localhost", description="Redis server host.")
    redis_port: int = Field(default=6379, description="Redis server port.")
    redis_password: Optional[SecretStr] = Field(
        default=None, description="Redis password."
    )
    redis_is_cluster: bool = Field(
        default=False, description="Set to True if connecting to a Redis Cluster."
    )
    redis_sentinels: Optional[List[str]] = Field(
        default=None, description="List of sentinel nodes formatted as 'host:port'."
    )
    redis_sentinel_master: Optional[str] = Field(
        default=None, description="Sentinel master name if using sentinels."
    )

    # Cache & Similarity Settings
    similarity_threshold: float = Field(
        default=0.85, description="Cosine similarity threshold for a cache hit."
    )
    default_ttl: int = Field(
        default=86400, description="Default Time-to-Live for cache entries in seconds."
    )
    permanent_hit_threshold: int = Field(
        default=5, description="Number of hits required to promote an entry to be kept forever (0 to disable)."
    )
    cache_index_name: str = Field(
        default="semantic_cache_idx", description="RedisSearch index name."
    )
    key_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Redis key prefix this cache's index is defined over. Default None "
            "→ the historical 'scache:' prefix. Set a UNIQUE prefix when running "
            "multiple indexes with different embedding models on one Redis — "
            "indexes sharing a prefix would index each other's vectors."
        ),
    )

    # SaaS / multi-tenant server
    admin_api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Bearer key for the SaaS admin endpoints (tenant creation) when "
            "serving via semantic_cache.server. Set SC_ADMIN_API_KEY. Unset → "
            "admin endpoints return 503 (self-serve signup disabled)."
        ),
    )

    api_key_pepper: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Server-side pepper for API-key lookup fingerprints "
            "(SC_API_KEY_PEPPER). Without it keys are stored as a bare sha256, "
            "which anyone who can read the Redis keyspace — a backup, a "
            "snapshot, a support dump — can check a guess against offline. "
            "Set it to a long random string and keep it OUT of the data store. "
            "Existing keys keep working: they migrate to the peppered "
            "fingerprint the next time they are used."
        ),
    )

    max_request_bytes: int = Field(
        default=1_048_576,
        description=(
            "Ceiling on a request body in bytes (SC_MAX_REQUEST_BYTES, default "
            "1 MiB); 0 disables. Enforced on the STREAM, not just on "
            "Content-Length. Raise it if your clients legitimately send long "
            "conversations - a 1 MiB chat body is roughly 250k tokens."
        ),
    )

    readiness_timeout: float = Field(
        default=2.0,
        description=(
            "Per-dependency deadline for GET /ready, seconds "
            "(SC_READINESS_TIMEOUT). Keep it well under the probe interval: a "
            "readiness check that outlives its own scrape stops being a signal."
        ),
    )

    # SSO JWT verification. A verified JWT AUTO-PROVISIONS a tenant, so these
    # settings are the difference between "our SSO can sign people in" and
    # "anyone can mint a tenant". All default to unset, which DISABLES the JWT
    # path entirely and leaves only minted sc-... keys — fail closed.
    sso_jwks_url: Optional[str] = Field(
        default=None,
        description=(
            "JWKS endpoint of the SSO that signs user tokens (SC_SSO_JWKS_URL, "
            "e.g. https://sso.example.com/.well-known/jwks.json). Required for "
            "RS*/ES*/PS* tokens; needs the [sso] extra (PyJWT)."
        ),
    )
    sso_shared_secret: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Shared secret for HS256/384/512 SSO tokens (SC_SSO_SHARED_SECRET). "
            "Verified in-process with stdlib hmac — no extra needed."
        ),
    )
    sso_issuer: Optional[str] = Field(
        default=None,
        description=(
            "Expected 'iss' claim (SC_SSO_ISSUER). Unset → issuer is not "
            "checked, which is safe only when the signing key is single-purpose."
        ),
    )
    sso_audience: Optional[str] = Field(
        default=None,
        description=(
            "Expected 'aud' claim (SC_SSO_AUDIENCE). Unset → audience is not "
            "checked, so a token minted for a DIFFERENT service by the same SSO "
            "is accepted here. Set it whenever the SSO serves more than us."
        ),
    )
    sso_algorithms: List[str] = Field(
        default_factory=lambda: ["RS256"],
        description=(
            "Allowlist of accepted 'alg' values (SC_SSO_ALGORITHMS, "
            'e.g. \'["RS256"]\'). Enforced before a key is chosen, so '
            "alg-confusion and alg=none are unreachable."
        ),
    )
    sso_leeway: float = Field(
        default=60.0,
        description="Clock-skew tolerance in seconds for exp/nbf (SC_SSO_LEEWAY).",
    )
    sso_jwks_ttl: float = Field(
        default=300.0,
        description="Seconds to cache the SSO's JWKS before refetching (SC_SSO_JWKS_TTL).",
    )

    # OpenAI-compatible gateway (semantic_cache.gateway)
    pg_dsn: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Postgres DSN for the gateway's projects/keys and message log "
            "(SC_PG_DSN, e.g. postgresql://user:pass@host:5432/db). "
            "Points at an EXISTING Postgres — only the 'gw' schema is created. "
            "Unset → the gateway is not mounted."
        ),
    )
    gateway_upstream_timeout: float = Field(
        default=120.0,
        description="Timeout (seconds) for upstream LLM calls from the gateway.",
    )
    app_config_url: Optional[str] = Field(
        default=None,
        description=(
            "The APP's config endpoint (SC_APP_CONFIG_URL), a single fixed URL "
            "e.g. http://app-host:8000/cache. The gateway GETs it presenting "
            "the caller's own key as the bearer; the APP identifies the client "
            "from that key and returns the model / embedding model / "
            "entity-extractor model and their three API keys. Unset → gateway "
            "serving is not mounted."
        ),
    )
    app_config_ttl: float = Field(
        default=60.0,
        description=(
            "Seconds a key's config fetched from the APP is cached in memory "
            "before being re-fetched (SC_APP_CONFIG_TTL)."
        ),
    )
    app_service_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Shared secret proving to the APP that the caller is THIS SERVICE "
            "(SC_APP_SERVICE_KEY). Distinct from the client key: the bearer we "
            "forward says which client, this says it is us asking. Without it "
            "the APP's config endpoint is guarded by the client key alone, so "
            "anyone holding a client key can pull that client's model-provider "
            "credentials straight from the APP. Unset -> not sent, and the "
            "contract is unchanged."
        ),
    )
    app_service_key_header: str = Field(
        default="X-Service-Key",
        description=(
            "Header the service key travels in (SC_APP_SERVICE_KEY_HEADER). "
            "Configurable so the APP team can choose the name without a code "
            "change here. Never Authorization - that carries the client key."
        ),
    )
    app_config_max_entries: int = Field(
        default=10_000,
        description=(
            "Ceiling on cached per-key configs (SC_APP_CONFIG_MAX_ENTRIES). The "
            "cache is keyed by the CALLER's key, so without a bound its size is "
            "chosen by our callers — every wrong bearer included — and each "
            "entry holds live provider credentials. LRU eviction beyond this."
        ),
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description=(
            "MLOps serving endpoint for chat completions (SC_LLM_BASE_URL). "
            "The gateway calls {llm_base_url}/v1/chat/completions with the "
            "model + key returned by the APP."
        ),
    )
    embed_base_url: Optional[str] = Field(
        default=None,
        description=(
            "MLOps serving endpoint for embeddings (SC_EMBED_BASE_URL). Used "
            "with the embedding model + key returned by the APP."
        ),
    )
    extractor_base_url: Optional[str] = Field(
        default=None,
        description=(
            "MLOps serving endpoint for the entity extractor "
            "(SC_EXTRACTOR_BASE_URL). Only used when the APP returns an "
            "extractor model for the project."
        ),
    )

    # Input guard (optional gateway module). Whether a client is guarded, and
    # with which models, comes ENTIRELY from the APP's config response — these
    # are operator limits and fallbacks only. NONE of them is required and none
    # gates the /v1 mount (SC_EXTRACTOR_BASE_URL is the precedent).
    guard_enabled: bool = Field(
        default=True,
        description=(
            "Operator break-glass for the input guard (SC_GUARD_ENABLED). "
            "False disables it for EVERY client without touching the APP. "
            "Also flippable at runtime via POST /admin/guard."
        ),
    )
    guard_timeout: float = Field(
        default=5.0,
        description=(
            "Whole-check deadline in seconds for one guard decision, covering "
            "embedding + index + judge (SC_GUARD_TIMEOUT). Deliberately NOT "
            "gateway_upstream_timeout: the guard is inline and fail-closed, so "
            "a check that stalls two minutes is an outage, not latency."
        ),
    )
    guard_build_timeout: float = Field(
        default=60.0,
        description=(
            "Deadline in seconds for embedding a policy's exemplars into an "
            "index (SC_GUARD_BUILD_TIMEOUT). The build is server-owned, so a "
            "request that times out waiting does not cancel it."
        ),
    )
    guard_max_exemplars: int = Field(
        default=5000,
        description=(
            "Maximum exemplars in one policy (SC_GUARD_MAX_EXEMPLARS). Beyond "
            "this the APP's config is rejected rather than silently truncated."
        ),
    )
    guard_max_policy_bytes: int = Field(
        default=262144,
        description=(
            "Maximum size of the inline policy YAML (SC_GUARD_MAX_POLICY_BYTES). "
            "Checked BEFORE parsing, so a hostile document cannot exhaust the "
            "YAML parser."
        ),
    )
    guard_cache_max_bytes: int = Field(
        default=268435456,
        description=(
            "Byte budget for in-process guard exemplar matrices "
            "(SC_GUARD_CACHE_MAX_BYTES, default 256 MiB). One policy can be "
            "hundreds of times larger than another, so the LRU is bounded by "
            "bytes as well as by entry count."
        ),
    )
    guard_segment_memo_size: int = Field(
        default=50000,
        description=(
            "Entries in the per-segment decision memo (SC_GUARD_SEGMENT_MEMO_SIZE). "
            "Keyed by policy+embedder AND every decision parameter, with a short "
            "TTL, so any config change invalidates it by construction."
        ),
    )
    guard_embed_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Serving endpoint for the guard's embeddings (SC_GUARD_EMBED_BASE_URL). "
            "Falls back to SC_EMBED_BASE_URL when unset, so the guard adds no "
            "required env var."
        ),
    )
    guard_judge_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Serving endpoint for the guard's judge LLM (SC_GUARD_JUDGE_BASE_URL). "
            "Falls back to SC_LLM_BASE_URL when unset."
        ),
    )

    # Observability
    log_level: str = Field(
        default="INFO",
        description="Root log level for the structured JSON logger (SC_LOG_LEVEL).",
    )
    environment: str = Field(
        default="production",
        description="Deployment environment tag, surfaced to logs + Sentry (SC_ENVIRONMENT).",
    )
    sentry_dsn: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Sentry DSN for error reporting (SC_SENTRY_DSN). Unset → disabled. "
            "Requires the 'sentry' extra: pip install 'semantic-cache[sentry]'."
        ),
    )
    sentry_traces_sample_rate: float = Field(
        default=0.0,
        description="Sentry performance-trace sample rate 0.0–1.0 (SC_SENTRY_TRACES_SAMPLE_RATE).",
    )

    # Resilience
    fail_open: bool = Field(
        default=True,
        description=(
            "When True (default), a backend failure on search/set (Redis outage, "
            "embedding/extraction error) is swallowed — search returns None (a "
            "miss) and set becomes a no-op — so a cache problem NEVER breaks the "
            "caller. Set False to re-raise CacheOperationError for callers that "
            "want to handle it themselves."
        ),
    )
    cache_mode: Union[CacheMode, List[CacheMode]] = Field(
        default=CacheMode.SEMANTIC,
        description=(
            "Which retrieval method(s) a lookup uses (the APP chooses this per "
            "client). SAME field for one method or several — pass EITHER:\n"
            "  • a single method: 'semantic' (default, vector KNN — entity-aware "
            "when an extractor is configured), 'bm25' (lexical full-text), "
            "'fuzzy' (Levenshtein), 'exact' (L0 exact-match), or 'off' (no "
            "caching, pure passthrough); OR\n"
            "  • an ordered list = a CASCADE, tried in order, first hit wins, e.g. "
            "['exact','bm25','semantic'] — embedding is paid only if the earlier "
            "tiers miss. A list must be non-empty and cannot contain 'off'.\n"
            "A single value applies the exact pre-tier in front when exact_tier "
            "is set; an explicit list is taken verbatim (it supersedes exact_tier)."
        ),
    )
    lexical_scorer: str = Field(
        default="BM25",
        description=(
            "RediSearch full-text scorer used when cache_mode='bm25'. One of "
            "BM25, BM25STD, TFIDF, TFIDF.DOCNORM, DISMAX, DOCSCORE. Ignored by "
            "the other modes."
        ),
    )
    lexical_min_score: float = Field(
        default=0.0,
        description=(
            "Minimum full-text score for a HIT in the bm25/fuzzy methods. Scores "
            "are corpus-relative and unbounded, so the default 0.0 accepts any "
            "all-token match; raise it to demand a stronger lexical overlap."
        ),
    )
    fuzzy_distance: int = Field(
        default=1,
        description=(
            "Max Levenshtein distance per token for cache_mode='fuzzy' (RediSearch "
            "supports 1–3). 1 catches single-character typos; higher is more "
            "forgiving but looser. Ignored by the other methods."
        ),
    )
    # Per-method hyperparameter blocks. Each lets a client tune ONE method
    # independently (important in a cascade, where tiers may want different
    # values). Any field left unset falls back to the flat defaults above.
    semantic: Optional[SemanticParams] = Field(
        default=None,
        description="Per-client overrides for the semantic method (similarity_threshold, entity_threshold).",
    )
    bm25: Optional[Bm25Params] = Field(
        default=None,
        description="Per-client overrides for the bm25 method (scorer, min_score).",
    )
    fuzzy: Optional[FuzzyParams] = Field(
        default=None,
        description="Per-client overrides for the fuzzy method (scorer, min_score, distance).",
    )
    exact_tier: bool = Field(
        default=False,
        description=(
            "L0 exact-match tier: on search, check for an EXACT normalized-text "
            "(+scope) repeat with one Redis HGET BEFORE embedding/KNN/extraction. "
            "An exact repeat returns instantly and skips the embedding + entity "
            "calls; a non-exact query falls through to the semantic search "
            "unchanged. Off by default (adds one HGET on every miss); enable for "
            "workloads with many exact repeats to cut latency + embedding cost."
        ),
    )

    # Normalization Settings
    enable_translation: bool = Field(
        default=False,
        description="Translate inputs to a canonical language (e.g., English) before embedding."
    )
    normalize_aggressive: bool = Field(
        default=False,
        description=(
            "Aggressive query canonicalization before embedding + keying: Unicode "
            "NFKC, Persian/Arabic-Indic digit folding, lowercasing, apostrophe "
            "removal, punctuation stripping, and English stop-word dropping "
            "(order-preserving). Collapses phrasing-only variants to one cache "
            "entry — higher hit rate. Off by default because it changes the "
            "storage key (existing entries are keyed under the old normalization). "
            "Enable on a fresh keyspace / new deployment."
        ),
    )
    translation_target_language: str = Field(
        default="en",
        description="The canonical language code to translate into if enable_translation is True."
    )

    # Embedding Settings
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.HUGGINGFACE,
        description="Provider for embedding generation.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Name of the model to use. Defaults to a multilingual model to map Persian and English to the same semantic space.",
    )
    embedding_base_url: Optional[str] = Field(
        default=None, description="Base URL for third-party API providers."
    )
    embedding_api_key: Optional[SecretStr] = Field(
        default=None, description="API key for third-party providers."
    )
    embedding_dim: Optional[int] = Field(
        default=None,
        description=(
            "Vector dimension for API embeddings. Leave None to auto-detect from "
            "the provider on first use (one probe call); set explicitly to skip "
            "the probe. Ignored for the HuggingFace backend (it self-reports)."
        ),
    )
    embedding_timeout: int = Field(
        default=10, description="Timeout (seconds) for API embedding HTTP calls."
    )

    # HTTP resilience (applies to embedding + entity-extraction API calls)
    http_max_retries: int = Field(
        default=2,
        description=(
            "Retries on transient HTTP failures (timeouts, connection errors, "
            "429, 5xx). 0 disables retrying."
        ),
    )
    http_backoff_base: float = Field(
        default=0.5,
        description="Base seconds for exponential backoff (delay = base * 2**attempt).",
    )

    # Entity-Aware Caching Settings
    # Maps to user-facing names SEMANTIC_CACHE_ENTITY_AWARE / _DOMAIN / _ENTITY_MODEL / _ENTITY_THRESHOLD
    # via the SC_ prefix (e.g. SC_ENTITY_AWARE=true).
    entity_aware: bool = Field(
        default=False,
        description="Enable entity-aware caching: require exact entity set match before similarity check."
    )
    domain: CacheDomain = Field(
        default=CacheDomain.GENERAL,
        description="Domain used to select entity extraction prompt and tag stored entries."
    )
    entity_model: str = Field(
        default="gpt-4o-mini",
        description="LLM used for entity extraction when entity_aware is True."
    )
    entity_threshold: float = Field(
        default=0.95,
        description="Stricter similarity threshold applied when entity_aware is True."
    )
    entity_llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the entity-extraction LLM (OpenAI-compatible). Falls back to embedding_base_url."
    )
    entity_llm_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the entity-extraction LLM. Falls back to embedding_api_key."
    )
    entity_llm_timeout: int = Field(
        default=10,
        description="Timeout in seconds for entity-extraction LLM calls."
    )
    entity_use_json_mode: bool = Field(
        default=True,
        description=(
            "Send OpenAI-style response_format={'type': 'json_object'}. "
            "Disable for OpenAI-compatible gateways (vLLM, some Anthropic shims) "
            "that 400 on the field. The tolerant parser still extracts the JSON object."
        ),
    )
    entity_extraction_cache_ttl: int = Field(
        default=3600,
        description=(
            "TTL in seconds for the per-query entity-extraction cache "
            "(stored in Redis when CachedEntityExtractor is used). 0 disables."
        ),
    )

    # --- effective per-method hyperparameters (block → flat → default) --------
    def eff_similarity_threshold(self) -> float:
        if self.semantic and self.semantic.similarity_threshold is not None:
            return self.semantic.similarity_threshold
        return self.similarity_threshold

    def eff_entity_threshold(self) -> float:
        if self.semantic and self.semantic.entity_threshold is not None:
            return self.semantic.entity_threshold
        return self.entity_threshold

    def eff_lexical_scorer(self, method: "CacheMode") -> str:
        block = self.bm25 if method == CacheMode.BM25 else self.fuzzy
        if block and block.scorer:
            return block.scorer
        return self.lexical_scorer

    def eff_lexical_min_score(self, method: "CacheMode") -> float:
        block = self.bm25 if method == CacheMode.BM25 else self.fuzzy
        if block and block.min_score is not None:
            return block.min_score
        return self.lexical_min_score

    def eff_fuzzy_distance(self) -> int:
        if self.fuzzy and self.fuzzy.distance is not None:
            return self.fuzzy.distance
        return self.fuzzy_distance

    @field_validator("cache_mode")
    @classmethod
    def validate_cache_mode(cls, v):
        """A cache_mode LIST (cascade) must be non-empty and cannot contain 'off'
        ('off' only makes sense as the single scalar value)."""
        if isinstance(v, list):
            if not v:
                raise ValueError("cache_mode list must be non-empty.")
            if any(m == CacheMode.OFF for m in v):
                raise ValueError(
                    "cache_mode list cannot contain 'off' — use cache_mode='off' "
                    "(the scalar) to disable caching."
                )
        return v

    def retrieval_cascade(self) -> "List[CacheMode]":
        """The canonical ordered list of retrieval methods for a lookup — the
        single source of truth used by the manager.

        ``cache_mode`` may be a LIST (an explicit cascade, taken verbatim) or a
        SINGLE method. A single method derives its cascade from ``exact_tier``
        (an exact pre-tier in front of the bm25/fuzzy/semantic method). The
        scalar ``off`` yields an empty cascade (caching disabled)."""
        mode = self.cache_mode
        if isinstance(mode, list):
            return list(mode)
        if mode == CacheMode.OFF:
            return []
        if mode == CacheMode.EXACT:
            return [CacheMode.EXACT]
        tiers: "List[CacheMode]" = []
        if self.exact_tier:
            tiers.append(CacheMode.EXACT)
        tiers.append(mode)  # BM25 / FUZZY / SEMANTIC
        return tiers

    @field_validator("fuzzy_distance")
    @classmethod
    def validate_fuzzy_distance(cls, v: int) -> int:
        """RediSearch caps Levenshtein fuzzy matching at distance 3."""
        if not 1 <= v <= 3:
            raise ValueError("fuzzy_distance must be between 1 and 3.")
        return v

    @field_validator("lexical_scorer")
    @classmethod
    def validate_lexical_scorer(cls, v: str) -> str:
        """Restrict to RediSearch scorers we support (case-insensitive)."""
        canonical = v.strip().upper()
        if canonical not in LEXICAL_SCORERS:
            raise ValueError(
                f"lexical_scorer must be one of {sorted(LEXICAL_SCORERS)}, got {v!r}."
            )
        return canonical

    @field_validator("similarity_threshold", "entity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure similarity thresholds are between 0 and 1.

        Args:
            v: The similarity threshold value.

        Returns:
            The validated threshold.

        Raises:
            ValueError: If the threshold is outside the [0.0, 1.0] range.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0.")
        return v
