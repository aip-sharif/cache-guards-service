# Semantic Cache SaaS + OpenAI gateway service.
#
# Build:  docker build -t semantic-cache-saas .
# Run:    docker run -p 8080:8080 \
#           -e SC_REDIS_HOST=redis -e SC_REDIS_PORT=6379 \
#           -e SC_EMBEDDING_PROVIDER=custom_api \
#           -e SC_EMBEDDING_BASE_URL=https://... \
#           semantic-cache-saas
#
# Embeddings are API-BASED (OpenRouter / mlops / any OpenAI-compatible
# endpoint) — no local model is baked into the image. If you want local
# HuggingFace embeddings instead, add "embeddings" to the extras below and
# set SC_EMBEDDING_PROVIDER=huggingface.
#
# Pinned by DIGEST, not by tag. `python:3.12-slim` is a moving target: the same
# Dockerfile silently produces a different image next month, so a build that
# reproduces a reported bug may not reproduce the bug at all. Bump this
# deliberately.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (better layer caching). Copy the package metadata and
# the package source (flat layout: the package lives in ./semantic_cache/).
COPY pyproject.toml README.md ./
COPY semantic_cache ./semantic_cache

# `.[fastapi]` the web server; `.[async]` the async manager path used by the
# SaaS data routes + gateway; `.[postgres]` the gateway's key/message store;
# `.[metrics]` prometheus_client, WITHOUT WHICH /metrics answers 503 — the
# counters exist in the code either way, which is exactly how their absence
# from this line went unnoticed; `.[sso]` PyJWT, for JWKS-verified SSO tokens.
RUN pip install --upgrade pip && \
    pip install --prefix=/install ".[fastapi,async,sentry,postgres,metrics,sso]"


FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root. The process needs no write access to anything it ships: an RCE in
# a dependency should land on a uid that cannot rewrite the application it just
# got into. Created before the COPY so the layer is cached.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin appuser

# Only the install prefix crosses the stage boundary — the source tree, pip,
# and the build cache stay behind. `semantic_cache` is installed into
# site-packages by the build stage, so there is deliberately no copy of the
# source here for the running process to import (or for anything else to edit).
COPY --from=build /install /usr/local

WORKDIR /home/appuser
USER 10001

EXPOSE 8080

# Liveness only — /health checks nothing external on purpose, so a Redis blip
# cannot turn `restart: unless-stopped` into a restart loop. Dependency state
# is at /ready, which belongs to the orchestrator's readiness probe, not here.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=4).status==200 else 1)"

# Uvicorn factory entrypoint. Override SC_* via -e / --env-file at runtime.
#
# ONE worker on purpose. The guard's exemplar matrices, its segment memo, and
# the APP-config cache are all per-process, so N workers means N copies of
# every matrix and N cold builds. Scale with replicas, not workers, and size
# the container for one process.
#
# --timeout-graceful-shutdown gives in-flight completions (which can stream for
# minutes) a bounded chance to finish, and lets the guard decision log drain
# on shutdown instead of vanishing with the process.
CMD ["uvicorn", "semantic_cache.server:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "30", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
