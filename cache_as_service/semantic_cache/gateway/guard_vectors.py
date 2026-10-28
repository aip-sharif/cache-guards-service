"""Embedding + nearest-neighbour search for the guard's policy exemplars.

Two objects: :class:`GuardEmbedder` turns text into unit vectors against the
APP-supplied OpenAI-compatible endpoint, and :class:`GuardIndex` holds one
policy's exemplar matrix in process and answers top-k queries.

**Why not pgvector.** The gateway's Postgres is a plain ``postgres:16-alpine``
in compose and, in production, the customer's own database — pgvector cannot be
assumed. And because every client may use a different embedding model, the
vector DIMENSION varies per client, which a fixed ``vector(N)`` column cannot
express. Storing the matrix as ``bytea`` makes ``dim`` an ordinary integer
column and the problem disappears; ``gw.cache_entries.vector bytea`` is already
this codebase's precedent. Exemplar sets are tens to hundreds per policy, so
one matmul beats an ANN index anyway.

**Why normalisation is load-bearing here.** Nothing else in the gateway
normalises — RediSearch computes cosine internally. The guard's classifier SUMS
raw similarity values, so unnormalised vectors would silently skew the vote. A
zero vector (some servers return one for whitespace-only input) would make the
denominator zero, score 0.0, and the input would be ALLOWED. Hence the explicit
norm check.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from semantic_cache.core.embedding_manager import _embeddings_endpoint
from semantic_cache.core.retry import acall_with_retries
from semantic_cache.gateway.guard_logic import Neighbor

logger = logging.getLogger(__name__)

#: query prefix, document prefix. ONE enum drives both sides on purpose: two
#: independent prefix strings would make one-sided prefixing — a silent,
#: unmeasurable retrieval-quality loss — a single omitted line away. GaaS
#: sniffed the model NAME for "e5", which guesses wrong on any private alias.
PREFIX_STYLES: Dict[str, Tuple[str, str]] = {
    "none": ("", ""),
    "e5": ("query: ", "passage: "),
}

#: Text embedded once per index and stored beside the matrix, to detect a model
#: swapped behind an unchanged alias at an unchanged URL.
SENTINEL_TEXT = "semantic-cache guard index sentinel v1"

_MAX_BATCH = 64
_NORM_TOLERANCE = 1e-3

#: Floor for the request-path embed timeout, so a very small SC_GUARD_TIMEOUT
#: does not produce a budget no endpoint could ever meet. It is a FLOOR, never
#: a ceiling — see request_embed_timeout.
_MIN_REQUEST_TIMEOUT = 1.5

#: Fraction of the whole-check deadline the query embed may spend when a judge
#: might still have to run after it.
_EMBED_SHARE_WITH_JUDGE = 1.0 / 3.0

#: Never spend the entire deadline in one stage: leaving a sliver means the
#: failure surfaces as a named "embed_unreachable" rather than as the outer
#: deadline firing, which is the difference between a diagnosis and a shrug.
_DEADLINE_MARGIN = 0.9


def request_embed_timeout(guard_timeout: float, mode: str) -> float:
    """HTTP timeout for the per-request query embed, from the check deadline.

    ``GuardChecker`` already wraps every stage in one ``asyncio.timeout``, so
    this per-call value is not the real bound. Its only job is to stop ONE
    stage from consuming the budget the next stage needs — which means a mode
    that can never invoke a judge has nothing to reserve for one and may use
    (almost) the whole deadline.

    This MUST scale with the deadline. It used to be
    ``min(1.5, guard_timeout / 3)``, whose ``min`` made 1.5s an absolute
    ceiling: raising SC_GUARD_TIMEOUT changed nothing, no other setting
    touched this path, and so an endpoint that answered in, say, 2s was
    permanently unreachable — reported as a connection timeout while curl
    against the same URL succeeded, because curl has no such deadline.
    """
    share = (
        guard_timeout
        if mode == "embedding-only"
        else guard_timeout * _EMBED_SHARE_WITH_JUDGE
    )
    return max(_MIN_REQUEST_TIMEOUT, share * _DEADLINE_MARGIN)


def judge_timeout(guard_timeout: float) -> float:
    """HTTP timeout for one judge call. Same reasoning, same former defect.

    The judge was wired as ``min(3.0, guard_timeout)``, equally inert above a
    deadline of 3s. It is the complement of the embed share, so the two stages
    together fit the deadline they were derived from.
    """
    share = guard_timeout * (1.0 - _EMBED_SHARE_WITH_JUDGE)
    return max(3.0, share * _DEADLINE_MARGIN)


class GuardEmbedError(Exception):
    """The embedding endpoint could not be used."""


class GuardEmbedMalformed(GuardEmbedError):
    """The endpoint answered, but the vectors are unusable."""


class GuardDimMismatch(Exception):
    """A query vector does not match the dimension of the stored matrix."""


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #


class GuardEmbedder:
    """Batching client for an OpenAI-compatible ``/v1/embeddings`` endpoint.

    The model, key and base URL all come from the APP's config response — there
    is no local or self-hosted fallback anywhere in the guard.
    """

    def __init__(
        self,
        http: Any,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prefix_style: str = "none",
        batch_size: int = _MAX_BATCH,
        timeout: float = 10.0,
        retries: int = 1,
        backoff_base: float = 0.05,
    ) -> None:
        if prefix_style not in PREFIX_STYLES:
            raise ValueError(f"unknown prefix_style {prefix_style!r}")
        self._http = http
        self._url = _embeddings_endpoint(base_url)
        self._api_key = api_key
        self._model = model
        self._query_prefix, self._doc_prefix = PREFIX_STYLES[prefix_style]
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._retries = retries
        self._backoff_base = backoff_base
        #: Set once a server rejects list input, so we stop paying for the
        #: round trip that discovers it again.
        self._one_at_a_time = False
        self.call_count = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str = "document",
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        """Embeds ``texts`` and returns an ``(n, dim)`` float32 unit matrix.

        ``timeout`` overrides the instance default for THIS call. The guard has
        two embedding paths with deliberately different budgets — a request's
        query embed, which must fit inside SC_GUARD_TIMEOUT, and a policy index
        build, which is shielded, server-owned and budgeted by
        SC_GUARD_BUILD_TIMEOUT because embedding a whole policy is slow. One
        embedder object serves both, so the budget has to travel with the CALL.
        Sharing the request's timeout made SC_GUARD_BUILD_TIMEOUT unreachable:
        against any endpoint slower than one request's share, the build failed,
        was retried by the next request, and failed again forever.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefix = self._query_prefix if input_type == "query" else self._doc_prefix
        prepared = [prefix + t for t in texts]
        budget = self._timeout if timeout is None else timeout

        vectors: List[List[float]] = []
        size = 1 if self._one_at_a_time else self._batch_size
        index = 0
        while index < len(prepared):
            batch = prepared[index:index + size]
            vectors.extend(await self._embed_batch(batch, budget))
            index += size
            if self._one_at_a_time:
                size = 1

        matrix = np.asarray(vectors, dtype=np.float32)
        return self._normalize(matrix)

    async def _embed_batch(
        self, batch: Sequence[str], budget: float
    ) -> List[List[float]]:
        try:
            return await self._post(batch, budget)
        except GuardEmbedError:
            raise
        except Exception as e:  # noqa: BLE001
            if len(batch) > 1 and _is_bad_request(e):
                # Some servers only accept a single string. Remember it, so the
                # rest of this build does not rediscover it batch by batch.
                logger.warning(
                    "Embedding endpoint rejected list input (%s); falling back "
                    "to one text per request for this embedder.", e,
                )
                self._one_at_a_time = True
                out: List[List[float]] = []
                for text in batch:
                    out.extend(await self._post([text], budget))
                return out
            # httpx's timeout exceptions stringify to the EMPTY STRING, so the
            # bare {e} produced "embedding request failed: " — an operator
            # staring at that cannot tell a timeout from a DNS failure from a
            # refused connection, which is the whole diagnosis. Name the type,
            # and name the budget that was exceeded.
            raise GuardEmbedError(
                f"embedding request to {self._url} failed after {budget:.2f}s "
                f"({type(e).__name__}): {e}"
            ) from e

    async def _post(
        self, batch: Sequence[str], budget: float
    ) -> List[List[float]]:
        async def _once() -> List[List[float]]:
            self.call_count += 1
            response = await self._http.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": list(batch), "model": self._model},
                timeout=budget,
            )
            response.raise_for_status()
            return _extract_vectors(response.json(), len(batch))

        return await acall_with_retries(
            _once, retries=self._retries, backoff_base=self._backoff_base
        )

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise GuardEmbedMalformed(
                f"embedding endpoint returned an unusable shape {matrix.shape}"
            )
        norms = np.linalg.norm(matrix, axis=1)
        if not np.all(np.isfinite(matrix)):
            raise GuardEmbedMalformed("embedding endpoint returned NaN or inf")
        if np.any(norms <= _NORM_TOLERANCE):
            # A zero vector would make the classifier's denominator zero, score
            # 0.0, and ALLOW the input. Refuse to build on one.
            raise GuardEmbedMalformed(
                "embedding endpoint returned a zero-length vector"
            )
        normalized = matrix / norms[:, None]
        drift = np.abs(np.linalg.norm(normalized, axis=1) - 1.0).max()
        if drift > _NORM_TOLERANCE:  # pragma: no cover — arithmetic guard
            raise GuardEmbedMalformed(
                f"vectors could not be normalised (max drift {drift})"
            )
        return np.ascontiguousarray(normalized, dtype=np.float32)


def _is_bad_request(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 400


def _extract_vectors(payload: Any, expected: int) -> List[List[float]]:
    """Pulls vectors out of an OpenAI embeddings response, IN INPUT ORDER.

    The sort is the whole point. TEI answered positionally and GaaS relied on
    that via ``zip()``, but the OpenAI schema carries an explicit ``index`` and
    does not promise order. Getting this wrong pairs each exemplar with another
    exemplar's vector and yields a classifier that looks plausible and is wrong.
    """
    if not isinstance(payload, dict):
        raise GuardEmbedMalformed("embedding response was not a JSON object")
    if isinstance(payload.get("error"), dict):
        message = payload["error"].get("message", "unknown error")
        raise GuardEmbedError(f"embedding endpoint returned an error: {message}")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise GuardEmbedMalformed(
            f"embedding response has {len(data) if isinstance(data, list) else '?'} "
            f"items, expected {expected}"
        )

    ordered: List[Optional[List[float]]] = [None] * expected
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise GuardEmbedMalformed("embedding response item was not an object")
        slot = item.get("index", position)
        if not isinstance(slot, int) or not 0 <= slot < expected:
            raise GuardEmbedMalformed(
                f"embedding response carries an out-of-range index {slot!r}"
            )
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise GuardEmbedMalformed("embedding response item had no embedding")
        if ordered[slot] is not None:
            raise GuardEmbedMalformed(
                f"embedding response repeats index {slot}"
            )
        ordered[slot] = [float(x) for x in vector]

    if any(v is None for v in ordered):
        raise GuardEmbedMalformed("embedding response skipped an index")
    widths = {len(v) for v in ordered}  # type: ignore[arg-type]
    if len(widths) != 1:
        raise GuardEmbedMalformed(
            f"embedding response mixes dimensions {sorted(widths)}"
        )
    return ordered  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


@dataclass
class GuardIndex:
    """One policy's exemplar vectors, searchable in process."""

    matrix: np.ndarray            # (n, dim) float32, unit rows, C-contiguous
    labels: Tuple[str, ...]       # "disallowed" | "allowed", row-aligned
    categories: Tuple[str, ...]   # category_id, row-aligned
    texts: Tuple[str, ...]        # exemplar text, row-aligned
    sentinel: np.ndarray          # (dim,) float32

    def __post_init__(self) -> None:
        n = self.matrix.shape[0]
        if not (len(self.labels) == len(self.categories) == len(self.texts) == n):
            raise ValueError(
                "GuardIndex metadata is not row-aligned with the matrix"
            )

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def nbytes(self) -> int:
        """Approximate resident size, for the pool's byte budget."""
        text_bytes = sum(len(t.encode("utf-8")) for t in self.texts)
        return int(self.matrix.nbytes + self.sentinel.nbytes + text_bytes)

    def search(self, query: np.ndarray, top_k: int) -> List[Neighbor]:
        """Top-``k`` most similar exemplars. Rows are unit, so dot == cosine."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise GuardDimMismatch(
                f"query vector has dimension {query.shape[0]}, index has "
                f"{self.dim} — the embedding model changed under this index"
            )
        if self.n == 0:  # pragma: no cover — an empty policy cannot be built
            return []
        scores = self.matrix @ query
        k = min(max(1, top_k), self.n)
        # argpartition finds the top k without sorting all n.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Neighbor(
                text=self.texts[i],
                label=self.labels[i],       # type: ignore[arg-type]
                category_id=self.categories[i],
                similarity=float(scores[i]),
            )
            for i in top
        ]

    def verify_sentinel(self, fresh: np.ndarray) -> bool:
        """Whether a freshly embedded sentinel still matches the stored one.

        This is what TEI's ``/info`` check was really for, rebuilt as an
        identity test: an alias repointed at a different model keeps the same
        name and URL, so only the vectors can reveal it.
        """
        fresh = np.asarray(fresh, dtype=np.float32).reshape(-1)
        if fresh.shape[0] != self.dim:
            return False
        return float(fresh @ self.sentinel) > 0.999

    def to_blob(self) -> bytes:
        return self.matrix.astype("<f4", copy=False).tobytes(order="C")

    def sentinel_blob(self) -> bytes:
        return self.sentinel.astype("<f4", copy=False).tobytes(order="C")

    @staticmethod
    def matrix_from_blob(blob: bytes, dim: int) -> np.ndarray:
        if dim <= 0:
            raise ValueError("dim must be positive")
        flat = np.frombuffer(blob, dtype="<f4")
        if flat.size % dim:
            raise ValueError(
                f"stored matrix of {flat.size} floats is not divisible by "
                f"dim {dim}"
            )
        return np.ascontiguousarray(flat.reshape(-1, dim), dtype=np.float32)

    @classmethod
    def from_blob(
        cls,
        blob: bytes,
        sentinel_blob: bytes,
        dim: int,
        labels: Sequence[str],
        categories: Sequence[str],
        texts: Sequence[str],
    ) -> "GuardIndex":
        return cls(
            matrix=cls.matrix_from_blob(blob, dim),
            labels=tuple(labels),
            categories=tuple(categories),
            texts=tuple(texts),
            sentinel=cls.matrix_from_blob(sentinel_blob, dim).reshape(-1),
        )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, tolerant of unnormalised input."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0 or math.isnan(denominator):
        return 0.0
    return float(a @ b) / denominator


__all__ = [
    "PREFIX_STYLES",
    "SENTINEL_TEXT",
    "GuardEmbedError",
    "GuardEmbedMalformed",
    "GuardDimMismatch",
    "GuardEmbedder",
    "GuardIndex",
    "cosine",
    "judge_timeout",
    "request_embed_timeout",
]
