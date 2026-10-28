"""Guard embedder + in-process index — pure unit via httpx.MockTransport."""

import hashlib

import httpx
import numpy as np
import pytest

from semantic_cache.gateway.guard_logic import Neighbor
from semantic_cache.gateway.guard_vectors import (
    SENTINEL_TEXT,
    GuardDimMismatch,
    GuardEmbedder,
    GuardEmbedError,
    GuardEmbedMalformed,
    GuardIndex,
    cosine,
    request_embed_timeout,
)

BASE = "https://embed.example.com"
DIM = 8


def _vector_for(text: str, dim: int = DIM):
    """Deterministic pseudo-embedding — same text, same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8)
    return [float(x) + 1.0 for x in raw]


class EmbedSpy:
    """MockTransport handler standing in for an OpenAI-compatible embedder."""

    def __init__(self) -> None:
        self.requests = []
        self.batch_sizes = []
        self.status = 200
        self.shuffle = False
        self.zero_vector = False
        self.payload_override = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.content).json()
        self.requests.append(body)
        inputs = body["input"]
        self.batch_sizes.append(len(inputs))

        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "no"}})
        if self.payload_override is not None:
            return httpx.Response(200, json=self.payload_override)

        items = []
        for position, text in enumerate(inputs):
            vector = [0.0] * DIM if self.zero_vector else _vector_for(text)
            items.append({"object": "embedding", "index": position,
                          "embedding": vector})
        if self.shuffle:
            items = list(reversed(items))
        return httpx.Response(200, json={"object": "list", "data": items})


def _embedder(spy: EmbedSpy, **kwargs) -> GuardEmbedder:
    client = httpx.AsyncClient(transport=httpx.MockTransport(spy))
    kwargs.setdefault("model", "test-embedder")
    return GuardEmbedder(client, base_url=BASE, api_key="sk-embed", **kwargs)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


async def test_a_large_policy_is_embedded_in_batches() -> None:
    spy = EmbedSpy()
    matrix = await _embedder(spy).embed([f"text {i}" for i in range(200)])
    assert matrix.shape == (200, DIM)
    assert len(spy.requests) == 4            # ceil(200 / 64)
    assert spy.batch_sizes == [64, 64, 64, 8]


async def test_empty_input_makes_no_request() -> None:
    spy = EmbedSpy()
    matrix = await _embedder(spy).embed([])
    assert matrix.shape[0] == 0
    assert spy.requests == []


async def test_the_endpoint_and_auth_header_are_openai_shaped() -> None:
    spy = EmbedSpy()
    embedder = _embedder(spy)
    assert embedder._url == f"{BASE}/v1/embeddings"
    await embedder.embed(["hello"])
    assert spy.requests[0]["model"] == "test-embedder"


async def test_a_full_endpoint_url_is_not_double_suffixed() -> None:
    spy = EmbedSpy()
    embedder = _embedder(spy)
    embedder._url = ""  # sanity: the helper is what we rely on
    from semantic_cache.core.embedding_manager import _embeddings_endpoint
    assert _embeddings_endpoint(f"{BASE}/v1/embeddings") == f"{BASE}/v1/embeddings"


# --------------------------------------------------------------------------- #
# Ordering — the defect that yields a plausible, wrong classifier
# --------------------------------------------------------------------------- #


async def test_out_of_order_response_is_reordered_by_index() -> None:
    ordered = EmbedSpy()
    shuffled = EmbedSpy()
    shuffled.shuffle = True
    texts = ["alpha", "beta", "gamma", "delta"]

    a = await _embedder(ordered).embed(texts)
    b = await _embedder(shuffled).embed(texts)

    # Same texts must produce the same rows regardless of response order.
    assert np.allclose(a, b)
    # And row i really is the vector of texts[i].
    expected = np.asarray(_vector_for("alpha"), dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(a[0], expected, atol=1e-5)


async def test_a_repeated_index_is_rejected() -> None:
    spy = EmbedSpy()
    spy.payload_override = {"data": [
        {"index": 0, "embedding": _vector_for("a")},
        {"index": 0, "embedding": _vector_for("b")},
    ]}
    with pytest.raises(GuardEmbedMalformed, match="repeats index"):
        await _embedder(spy).embed(["a", "b"])


async def test_an_out_of_range_index_is_rejected() -> None:
    spy = EmbedSpy()
    spy.payload_override = {"data": [{"index": 7, "embedding": _vector_for("a")}]}
    with pytest.raises(GuardEmbedMalformed, match="out-of-range index"):
        await _embedder(spy).embed(["a"])


async def test_a_short_response_is_rejected() -> None:
    spy = EmbedSpy()
    spy.payload_override = {"data": [{"index": 0, "embedding": _vector_for("a")}]}
    with pytest.raises(GuardEmbedMalformed, match="expected 2"):
        await _embedder(spy).embed(["a", "b"])


async def test_mixed_dimensions_are_rejected() -> None:
    spy = EmbedSpy()
    spy.payload_override = {"data": [
        {"index": 0, "embedding": [1.0] * 8},
        {"index": 1, "embedding": [1.0] * 4},
    ]}
    with pytest.raises(GuardEmbedMalformed, match="mixes dimensions"):
        await _embedder(spy).embed(["a", "b"])


async def test_a_200_carrying_an_error_object_is_rejected() -> None:
    spy = EmbedSpy()
    spy.payload_override = {"error": {"message": "quota exceeded"}}
    with pytest.raises(GuardEmbedError, match="quota exceeded"):
        await _embedder(spy).embed(["a"])


# --------------------------------------------------------------------------- #
# Normalisation — load-bearing for the classifier, unlike everywhere else
# --------------------------------------------------------------------------- #


async def test_every_returned_vector_is_unit_length() -> None:
    matrix = await _embedder(EmbedSpy()).embed(["one", "two", "three"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    assert matrix.dtype == np.float32


async def test_a_zero_vector_is_refused_rather_than_scored() -> None:
    # Unnoticed, it makes the classifier's denominator zero -> score 0.0 -> ALLOW.
    spy = EmbedSpy()
    spy.zero_vector = True
    with pytest.raises(GuardEmbedMalformed, match="zero-length"):
        await _embedder(spy).embed(["   "])


async def test_nan_vectors_are_refused() -> None:
    # Sent as raw bytes: httpx refuses to SERIALISE NaN, but Python's json
    # parser accepts it, so a real server can put one on the wire.
    body = b'{"data": [{"index": 0, "embedding": [NaN, 1, 1, 1, 1, 1, 1, 1]}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )

    with pytest.raises(GuardEmbedMalformed, match="NaN or inf"):
        await _embedder(handler).embed(["a"])


# --------------------------------------------------------------------------- #
# Prefixes — one enum, both sides
# --------------------------------------------------------------------------- #


async def test_e5_prefixes_differ_between_query_and_document() -> None:
    spy = EmbedSpy()
    embedder = _embedder(spy, prefix_style="e5")
    await embedder.embed(["capital of france"], input_type="query")
    await embedder.embed(["capital of france"], input_type="document")
    assert spy.requests[0]["input"] == ["query: capital of france"]
    assert spy.requests[1]["input"] == ["passage: capital of france"]


async def test_prefix_style_none_sends_the_text_verbatim() -> None:
    spy = EmbedSpy()
    embedder = _embedder(spy, prefix_style="none")
    await embedder.embed(["hello"], input_type="query")
    await embedder.embed(["hello"], input_type="document")
    assert spy.requests[0]["input"] == ["hello"]
    assert spy.requests[1]["input"] == ["hello"]


def test_an_unknown_prefix_style_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="prefix_style"):
        _embedder(EmbedSpy(), prefix_style="bge")


# --------------------------------------------------------------------------- #
# Single-text fallback
# --------------------------------------------------------------------------- #


class RejectsListInput:
    """A server that 400s on list input but accepts a single string."""

    def __init__(self) -> None:
        self.batch_sizes = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.content).json()
        inputs = body["input"]
        self.batch_sizes.append(len(inputs))
        if len(inputs) > 1:
            return httpx.Response(400, json={"error": {"message": "one at a time"}})
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": _vector_for(inputs[0])}
        ]})


async def test_a_server_that_rejects_lists_falls_back_and_remembers() -> None:
    spy = RejectsListInput()
    embedder = _embedder(spy)
    first = await embedder.embed(["a", "b", "c"])
    assert first.shape == (3, DIM)
    assert spy.batch_sizes == [3, 1, 1, 1]      # one rejected probe, then singles

    spy.batch_sizes.clear()
    await embedder.embed(["d", "e"])
    assert spy.batch_sizes == [1, 1]            # the mode stuck; no second probe


async def test_a_non_400_error_is_not_retried_into_the_fallback() -> None:
    spy = EmbedSpy()
    spy.status = 503
    with pytest.raises(GuardEmbedError):
        await _embedder(spy, retries=0).embed(["a", "b"])


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def _index(dim: int = DIM, n: int = 5) -> GuardIndex:
    rows = []
    for i in range(n):
        v = np.asarray(_vector_for(f"exemplar {i}", dim), dtype=np.float32)
        rows.append(v / np.linalg.norm(v))
    sentinel = np.asarray(_vector_for(SENTINEL_TEXT, dim), dtype=np.float32)
    sentinel /= np.linalg.norm(sentinel)
    return GuardIndex(
        matrix=np.ascontiguousarray(np.vstack(rows), dtype=np.float32),
        labels=tuple("disallowed" if i % 2 == 0 else "allowed" for i in range(n)),
        categories=tuple(f"cat-{i % 2}" for i in range(n)),
        texts=tuple(f"exemplar {i}" for i in range(n)),
        sentinel=sentinel,
    )


def test_search_returns_neighbors_sorted_by_similarity() -> None:
    index = _index()
    query = index.matrix[2].copy()
    hits = index.search(query, top_k=3)
    assert len(hits) == 3
    assert all(isinstance(h, Neighbor) for h in hits)
    assert hits[0].text == "exemplar 2"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert [h.similarity for h in hits] == sorted(
        (h.similarity for h in hits), reverse=True
    )


def test_search_carries_the_row_aligned_metadata() -> None:
    index = _index()
    hit = index.search(index.matrix[0].copy(), top_k=1)[0]
    assert hit.label == "disallowed"
    assert hit.category_id == "cat-0"


def test_top_k_larger_than_the_policy_is_clamped() -> None:
    index = _index(n=3)
    assert len(index.search(index.matrix[0].copy(), top_k=99)) == 3


def test_a_query_of_the_wrong_dimension_is_a_named_error() -> None:
    index = _index(dim=8)
    with pytest.raises(GuardDimMismatch, match="embedding model changed"):
        index.search(np.ones(16, dtype=np.float32), top_k=3)


def test_metadata_that_is_not_row_aligned_is_refused() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        GuardIndex(
            matrix=np.zeros((3, DIM), dtype=np.float32),
            labels=("allowed",),
            categories=("a", "b", "c"),
            texts=("x", "y", "z"),
            sentinel=np.zeros(DIM, dtype=np.float32),
        )


@pytest.mark.parametrize("dim", [384, 1024])
def test_the_matrix_round_trips_through_a_blob_bit_identically(dim) -> None:
    index = _index(dim=dim, n=7)
    restored = GuardIndex.from_blob(
        index.to_blob(), index.sentinel_blob(), dim,
        index.labels, index.categories, index.texts,
    )
    assert restored.dim == dim
    assert restored.n == 7
    assert np.array_equal(restored.matrix, index.matrix)
    assert np.array_equal(restored.sentinel, index.sentinel)


def test_a_blob_that_does_not_divide_by_dim_is_refused() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        GuardIndex.matrix_from_blob(np.zeros(10, dtype="<f4").tobytes(), 4)


def test_the_sentinel_detects_a_model_swapped_behind_the_same_alias() -> None:
    index = _index()
    assert index.verify_sentinel(index.sentinel.copy()) is True

    other = np.asarray(_vector_for("a completely different model"), dtype=np.float32)
    other /= np.linalg.norm(other)
    assert index.verify_sentinel(other) is False
    # A different dimension is a mismatch, not a crash.
    assert index.verify_sentinel(np.ones(16, dtype=np.float32)) is False


def test_nbytes_tracks_the_matrix_for_the_pool_budget() -> None:
    index = _index(dim=1024, n=100)
    assert index.nbytes >= 100 * 1024 * 4
    assert index.nbytes < 100 * 1024 * 4 + 100_000


def test_cosine_is_zero_for_a_zero_vector_rather_than_nan() -> None:
    assert cosine(np.zeros(4), np.ones(4)) == 0.0
    assert cosine(np.ones(4), np.ones(4)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Timeouts
#
# The guard has TWO embedding paths with deliberately different budgets: the
# per-request query embed, which must fit inside SC_GUARD_TIMEOUT, and the
# policy index build, which is shielded, server-owned and budgeted separately
# by SC_GUARD_BUILD_TIMEOUT. They ran on one shared value, so the build could
# never use its own budget and any endpoint slower than the request budget was
# permanently unreachable.
# --------------------------------------------------------------------------- #


class TimeoutSpy:
    """An http double that records the timeout each call was given."""

    def __init__(self, raises: Exception = None) -> None:
        self.timeouts = []
        self.raises = raises

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.timeouts.append(timeout)
        if self.raises is not None:
            raise self.raises
        items = [{"object": "embedding", "index": i, "embedding": _vector_for(t)}
                 for i, t in enumerate(json["input"])]
        return httpx.Response(200, json={"object": "list", "data": items},
                              request=httpx.Request("POST", url))


async def test_embed_uses_the_instance_timeout_by_default() -> None:
    spy = TimeoutSpy()
    await GuardEmbedder(spy, base_url=BASE, api_key="k", model="m",
                        timeout=2.0).embed(["a"])
    assert spy.timeouts == [2.0]


async def test_embed_accepts_a_per_call_timeout_for_the_build_path() -> None:
    """The build's budget is 60s while a request's is ~1.5s; one embedder
    object serves both, so the budget must travel with the CALL."""
    spy = TimeoutSpy()
    embedder = GuardEmbedder(spy, base_url=BASE, api_key="k", model="m",
                             timeout=1.5)
    await embedder.embed(["a"], input_type="query")          # request path
    await embedder.embed(["b"], timeout=60.0)                # build path
    assert spy.timeouts == [1.5, 60.0]


async def test_a_timeout_names_its_cause() -> None:
    """httpx timeout exceptions stringify to '', so the operator got
    'embedding request failed: ' with no cause at all."""
    spy = TimeoutSpy(raises=httpx.ReadTimeout(""))
    embedder = GuardEmbedder(spy, base_url=BASE, api_key="k", model="m",
                             retries=0)
    with pytest.raises(GuardEmbedError) as excinfo:
        await embedder.embed(["a"])
    assert "ReadTimeout" in str(excinfo.value)


def test_the_request_embed_timeout_scales_with_the_guard_deadline() -> None:
    """It used to be min(1.5, guard_timeout / 3) — an absolute ceiling that
    made SC_GUARD_TIMEOUT inert, so an operator whose endpoint answered in
    2s had no setting anywhere that could fix it."""
    assert request_embed_timeout(30.0, "cascade") > request_embed_timeout(5.0, "cascade")
    assert request_embed_timeout(60.0, "cascade") >= 15.0


def test_embedding_only_reserves_nothing_for_a_judge_that_cannot_run() -> None:
    assert (request_embed_timeout(5.0, "embedding-only")
            > request_embed_timeout(5.0, "cascade"))
    assert request_embed_timeout(5.0, "embedding-only") <= 5.0


def test_the_request_embed_timeout_has_a_floor_for_tiny_deadlines() -> None:
    assert request_embed_timeout(0.1, "cascade") >= 1.5
