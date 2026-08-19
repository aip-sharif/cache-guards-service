"""OpenAI-compatible gateway: cache-first chat completions over any upstream.

Requests hit the semantic cache first (Redis hot tier, Postgres durable tier);
misses are proxied to the upstream LLM configured per model in Postgres, and
the answer is written through to both tiers on the way back.
"""
