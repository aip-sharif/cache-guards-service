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
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (better layer caching). Copy the package metadata and
# the package source (flat layout: the package lives in ./semantic_cache/).
COPY pyproject.toml README.md ./
COPY semantic_cache ./semantic_cache

# `.[fastapi]` the web server; `.[async]` the async manager path used by the
# SaaS data routes + gateway; `.[postgres]` the gateway's key/message store.
RUN pip install --upgrade pip && \
    pip install ".[fastapi,async,sentry,postgres]"

EXPOSE 8080

# Uvicorn factory entrypoint. Override SC_* via -e / --env-file at runtime.
CMD ["uvicorn", "semantic_cache.server:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080"]
