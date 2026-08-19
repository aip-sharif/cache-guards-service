"""Cache-mode behavior at the manager level — semantic / exact / off.

Backend is a tiny in-memory Redis double (just the calls the exact + off paths
touch) and a fake embedder that counts its calls, so the test stays in the unit
lane (no Redis, no model, no network) while still exercising the real
SemanticCacheManager control flow.
"""

import pytest
from redis.exceptions import ResponseError

from semantic_cache.core.cache_manager import SemanticCacheManager
from semantic_cache.core.config import CacheMode, SemanticCacheConfig


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _Doc:
    def __init__(self, doc_id, **fields) -> None:
        self.id = doc_id
        for k, v in fields.items():
            setattr(self, k, v)


class _Result:
    def __init__(self, docs) -> None:
        self.docs = docs


class _FakeFt:
    def __init__(self) -> None:
        self.created = False
        self.searches = 0
        self.result_docs = []   # what the next .search() returns
        self.last_query = None  # the Query object last passed to .search()

    def info(self):
        if not self.created:
            raise ResponseError("Unknown Index name")
        return {"num_docs": 0}

    def create_index(self, fields, definition):
        self.created = True

    def search(self, query, query_params=None):
        self.searches += 1
        self.last_query = query
        return _Result(list(self.result_docs))


class _FakePipe:
    def __init__(self, redis) -> None:
        self.redis = redis
        self.q = []

    def hincrby(self, key, field, n):
        self.q.append(("hincrby", key, field, n))
        return self

    def ttl(self, key):
        self.q.append(("ttl", key))
        return self

    def execute(self):
        out = []
        for op in self.q:
            if op[0] == "hincrby":
                _, key, field, n = op
                d = self.redis.h.setdefault(key, {})
                d[field] = int(d.get(field, 0)) + n
                out.append(d[field])
            else:
                out.append(self.redis.ttl(op[1]))
        return out


class FakeRedis:
    def __init__(self) -> None:
        self.h = {}
        self.ttls = {}
        self._ft = _FakeFt()

    def ft(self, name):
        return self._ft

    def hset(self, key, mapping=None):
        self.h.setdefault(key, {}).update(mapping or {})

    def hmget(self, key, *fields):
        d = self.h.get(key, {})
        return [d.get(f) for f in fields]

    def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    def ttl(self, key):
        return self.ttls.get(key, -1)

    def expire(self, key, seconds):
        self.ttls[key] = seconds

    def persist(self, key):
        self.ttls[key] = -1

    def pipeline(self, transaction=False):
        return _FakePipe(self)

    def scan_iter(self, match=None, count=None):
        return iter(list(self.h.keys()))


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def get_dimension(self):
        return 3

    def get_embedding(self, text):
        self.calls += 1
        return [0.1, 0.2, 0.3]


def _manager(mode: CacheMode, **cfg_kwargs):
    cfg = SemanticCacheConfig(cache_mode=mode, default_ttl=100, **cfg_kwargs)
    emb = FakeEmbedding()
    redis = FakeRedis()
    mgr = SemanticCacheManager(
        config=cfg, redis_client=redis, embedding_manager=emb
    )
    return mgr, redis, emb


def _manager_cascade(methods):
    cfg = SemanticCacheConfig(cache_mode=methods, default_ttl=100)
    emb = FakeEmbedding()
    redis = FakeRedis()
    mgr = SemanticCacheManager(
        config=cfg, redis_client=redis, embedding_manager=emb
    )
    return mgr, redis, emb


def _only_doc(redis: FakeRedis):
    return next(v for k, v in redis.h.items() if k.startswith("scache:"))


def _only_key(redis: FakeRedis):
    return next(k for k in redis.h if k.startswith("scache:"))


# --------------------------------------------------------------------------- #
# Regression: index is set up in __init__ (was misindented into _run_hook)
# --------------------------------------------------------------------------- #


def test_init_creates_the_index_without_any_hook() -> None:
    mgr, redis, _ = _manager(CacheMode.SEMANTIC)
    # A freshly built manager — no persist/touch/delete hook wired — must
    # already have its versioned index ready.
    assert mgr.active_index_name.endswith(":v3")
    assert redis._ft.created is True


# --------------------------------------------------------------------------- #
# exact mode
# --------------------------------------------------------------------------- #


def test_exact_mode_stores_without_a_vector_and_never_embeds() -> None:
    mgr, redis, emb = _manager(CacheMode.EXACT)
    mgr.set("Capital of France?", "Paris")
    doc = _only_doc(redis)
    assert doc["response"] == "Paris"
    assert "vector" not in doc          # no embedding written
    assert emb.calls == 0               # …and none computed


def test_exact_mode_hits_on_repeat_and_misses_on_variation() -> None:
    mgr, redis, emb = _manager(CacheMode.EXACT)
    mgr.set("Capital of France?", "Paris")

    hit = mgr.search("Capital of France?")
    assert hit is not None
    assert hit["response"] == "Paris"
    assert hit["similarity"] == 1.0
    assert hit["exact"] is True

    # A non-identical phrasing is a MISS — exact mode never falls through to KNN.
    assert mgr.search("what is the capital of france") is None
    assert emb.calls == 0               # no embedding on read at all
    assert redis._ft.searches == 0      # KNN never invoked


# --------------------------------------------------------------------------- #
# off mode
# --------------------------------------------------------------------------- #


def test_off_mode_is_a_no_op_for_both_set_and_search() -> None:
    mgr, redis, emb = _manager(CacheMode.OFF)
    mgr.set("anything", "value")
    assert redis.h == {}                # nothing stored
    assert mgr.search("anything") is None
    assert emb.calls == 0


# --------------------------------------------------------------------------- #
# bm25 (lexical) mode — full-text, no embeddings
# --------------------------------------------------------------------------- #


def test_bm25_mode_stores_without_a_vector_and_never_embeds() -> None:
    mgr, redis, emb = _manager(CacheMode.BM25)
    mgr.set("Capital of France?", "Paris")
    doc = _only_doc(redis)
    assert doc["response"] == "Paris"
    assert "vector" not in doc          # lexical writes carry no vector
    assert emb.calls == 0               # …and compute no embedding


def test_bm25_mode_hits_via_fulltext_with_the_configured_scorer() -> None:
    mgr, redis, emb = _manager(CacheMode.BM25)
    mgr.set("Capital of France?", "Paris")
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="4.2", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    hit = mgr.search("what is the capital of france")
    assert hit is not None
    assert hit["response"] == "Paris"
    assert hit["method"] == "bm25"
    assert emb.calls == 0                       # no embedding on read
    # the lookup ran a full-text query with the configured scorer, not a KNN
    assert redis._ft.last_query._scorer == "BM25"


def test_bm25_min_score_gate_turns_weak_matches_into_misses() -> None:
    mgr, redis, _ = _manager(CacheMode.BM25, lexical_min_score=5.0)
    mgr.set("Capital of France?", "Paris")
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="1.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    assert mgr.search("capital of france") is None   # 1.0 < min 5.0


def test_bm25_honors_a_non_default_scorer() -> None:
    mgr, redis, _ = _manager(CacheMode.BM25, lexical_scorer="tfidf")
    mgr.set("Capital of France?", "Paris")
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="2.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    mgr.search("capital of france")
    assert redis._ft.last_query._scorer == "TFIDF"   # normalized upper-case


# --------------------------------------------------------------------------- #
# per-method hyperparameters (block → flat → default)
# --------------------------------------------------------------------------- #


def test_effective_params_prefer_block_then_flat() -> None:
    # blocks present → block wins
    c = SemanticCacheConfig(
        similarity_threshold=0.8, lexical_scorer="TFIDF", lexical_min_score=0.1,
        fuzzy_distance=1,
        semantic={"similarity_threshold": 0.95},
        bm25={"scorer": "BM25", "min_score": 2.0},
        fuzzy={"min_score": 3.0, "distance": 3},
    )
    assert c.eff_similarity_threshold() == 0.95
    assert c.eff_lexical_scorer(CacheMode.BM25) == "BM25"
    assert c.eff_lexical_min_score(CacheMode.BM25) == 2.0
    assert c.eff_lexical_min_score(CacheMode.FUZZY) == 3.0
    assert c.eff_fuzzy_distance() == 3
    # fuzzy block has no scorer → falls back to the flat value
    assert c.eff_lexical_scorer(CacheMode.FUZZY) == "TFIDF"


def test_effective_params_fall_back_to_flat_when_no_block() -> None:
    c = SemanticCacheConfig(similarity_threshold=0.7, lexical_min_score=0.5, fuzzy_distance=2)
    assert c.eff_similarity_threshold() == 0.7
    assert c.eff_lexical_min_score(CacheMode.BM25) == 0.5
    assert c.eff_fuzzy_distance() == 2


def test_per_method_blocks_are_validated() -> None:
    with pytest.raises(Exception):
        SemanticCacheConfig(semantic={"similarity_threshold": 5.0})
    with pytest.raises(Exception):
        SemanticCacheConfig(bm25={"scorer": "nope"})
    with pytest.raises(Exception):
        SemanticCacheConfig(fuzzy={"distance": 9})
    with pytest.raises(Exception):
        SemanticCacheConfig(semantic={"typo_field": 1})   # extra=forbid


def test_per_method_min_score_gates_bm25_tier() -> None:
    # An impossibly-high per-method min_score turns a scored candidate into a miss.
    mgr, redis, _ = _manager(["bm25"], bm25={"min_score": 100000.0})
    mgr.set("Capital of France?", "Paris")
    redis._ft.result_docs = [
        _Doc(_only_key(redis), score="2.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    assert mgr.search("capital france") is None

    mgr2, redis2, _ = _manager(["bm25"], bm25={"min_score": 0.0})
    mgr2.set("Capital of France?", "Paris")
    redis2._ft.result_docs = [
        _Doc(_only_key(redis2), score="2.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    assert (mgr2.search("capital france") or {}).get("response") == "Paris"


def test_per_method_similarity_threshold_flows_to_vector_search() -> None:
    # A candidate at cosine distance 0.2 (similarity 0.8): a 0.9 floor misses,
    # a 0.5 floor hits — driven purely by the per-method semantic block.
    for thr, expect_hit in [(0.9, False), (0.5, True)]:
        mgr, redis, _ = _manager(["semantic"],
                                 semantic={"similarity_threshold": thr})
        mgr.set("Capital of France?", "Paris")
        redis._ft.result_docs = [
            _Doc(_only_key(redis), vector_score="0.2", response="Paris",
                 metadata="{}", entities="[]", domain="general")
        ]
        hit = mgr.search("capital of the france nation")
        assert (hit is not None) is expect_hit


# --------------------------------------------------------------------------- #
# semantic mode (default) still writes a vector
# --------------------------------------------------------------------------- #


def test_semantic_mode_writes_a_vector() -> None:
    mgr, redis, emb = _manager(CacheMode.SEMANTIC)
    mgr.set("Capital of France?", "Paris")
    doc = _only_doc(redis)
    assert "vector" in doc
    assert emb.calls == 1


# --------------------------------------------------------------------------- #
# fuzzy (lexical Levenshtein) mode
# --------------------------------------------------------------------------- #


def test_fuzzy_mode_wraps_tokens_in_levenshtein_markers() -> None:
    mgr, redis, emb = _manager(CacheMode.FUZZY, fuzzy_distance=2)
    mgr.set("Capital of France?", "Paris")
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="2.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    hit = mgr.search("capitol france")
    assert hit is not None and hit["method"] == "fuzzy"
    assert emb.calls == 0                        # lexical: no embedding
    # distance 2 → each token wrapped in %% … %%
    qs = redis._ft.last_query.query_string()
    assert "%%capitol%%" in qs and "%%france%%" in qs


def test_fuzzy_distance_must_be_1_to_3() -> None:
    for bad in (0, 4):
        with pytest.raises(ValueError):
            SemanticCacheConfig(fuzzy_distance=bad)
    for good in (1, 2, 3):
        assert SemanticCacheConfig(fuzzy_distance=good).fuzzy_distance == good


def test_fuzzy_mode_writes_no_vector() -> None:
    mgr, redis, emb = _manager(CacheMode.FUZZY)
    mgr.set("Capital of France?", "Paris")
    assert "vector" not in _only_doc(redis)
    assert emb.calls == 0


# --------------------------------------------------------------------------- #
# full-text value alignment (regression: Persian punctuation glued to tokens)
# --------------------------------------------------------------------------- #


def test_fulltext_value_is_tokenizer_aligned() -> None:
    mgr, _, _ = _manager(CacheMode.BM25)
    # Persian question mark (؟) must not stick to the last word — the stored
    # full-text value is the space-joined \w+ tokens.
    assert mgr._fulltext_value("پایتخت فرانسه کجاست؟") == "پایتخت فرانسه کجاست"
    assert mgr._fulltext_value("What is the capital of France?") == "What is the capital of France"


def test_bm25_stores_the_tokenized_text_not_raw() -> None:
    mgr, redis, _ = _manager(CacheMode.BM25)
    mgr.set("پایتخت فرانسه کجاست؟", "پاریس")
    doc = _only_doc(redis)
    assert doc["text"] == "پایتخت فرانسه کجاست"   # ؟ stripped for indexing


# --------------------------------------------------------------------------- #
# ordered cascade — clients can combine methods
# --------------------------------------------------------------------------- #


def test_cascade_resolution_from_config() -> None:
    E, B, S = CacheMode.EXACT, CacheMode.BM25, CacheMode.SEMANTIC
    assert SemanticCacheConfig(cache_mode="semantic").retrieval_cascade() == [S]
    assert SemanticCacheConfig(cache_mode="semantic", exact_tier=True).retrieval_cascade() == [E, S]
    assert SemanticCacheConfig(cache_mode="bm25", exact_tier=True).retrieval_cascade() == [E, B]
    assert SemanticCacheConfig(cache_mode="fuzzy").retrieval_cascade() == [CacheMode.FUZZY]
    assert SemanticCacheConfig(cache_mode="exact").retrieval_cascade() == [E]
    assert SemanticCacheConfig(cache_mode="off").retrieval_cascade() == []
    # a cache_mode LIST is an explicit cascade, taken verbatim (supersedes exact_tier)
    assert SemanticCacheConfig(
        cache_mode=["bm25", "semantic"], exact_tier=True
    ).retrieval_cascade() == [B, S]


def test_cascade_list_cannot_contain_off() -> None:
    with pytest.raises(ValueError):
        SemanticCacheConfig(cache_mode=["exact", "off"])
    with pytest.raises(ValueError):
        SemanticCacheConfig(cache_mode=[])


def test_cascade_without_semantic_writes_no_vector() -> None:
    mgr, redis, emb = _manager(["exact", "bm25"])
    mgr.set("Capital of France?", "Paris")
    doc = _only_doc(redis)
    assert "vector" not in doc
    assert emb.calls == 0


def test_cascade_with_semantic_writes_a_vector() -> None:
    mgr, redis, emb = _manager(["exact", "bm25", "semantic"])
    mgr.set("Capital of France?", "Paris")
    doc = _only_doc(redis)
    assert "vector" in doc
    assert emb.calls == 1


def test_cascade_first_hit_wins_exact_before_bm25() -> None:
    mgr, redis, _ = _manager(["exact", "bm25"])
    mgr.set("Capital of France?", "Paris")

    # Identical repeat → the exact tier hits, the bm25/full-text tier is never
    # consulted.
    hit = mgr.search("Capital of France?")
    assert hit is not None and hit.get("exact") is True
    assert redis._ft.searches == 0

    # A reworded query misses exact and falls through to the bm25 tier.
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="3.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    hit2 = mgr.search("the capital of France")
    assert hit2 is not None and hit2.get("method") == "bm25"
    assert redis._ft.searches == 1


# --------------------------------------------------------------------------- #
# retrieval_cascade() — how cache_mode (scalar/list) + exact_tier resolve to an order
# --------------------------------------------------------------------------- #


def test_cascade_from_cache_mode_list() -> None:
    cfg = SemanticCacheConfig(cache_mode=["exact", "bm25", "semantic"])
    assert cfg.retrieval_cascade() == [
        CacheMode.EXACT, CacheMode.BM25, CacheMode.SEMANTIC
    ]


def test_cascade_derived_from_scalar_mode_and_exact_tier() -> None:
    assert SemanticCacheConfig(
        cache_mode="semantic", exact_tier=True
    ).retrieval_cascade() == [CacheMode.EXACT, CacheMode.SEMANTIC]
    assert SemanticCacheConfig(
        cache_mode="semantic", exact_tier=False
    ).retrieval_cascade() == [CacheMode.SEMANTIC]
    assert SemanticCacheConfig(
        cache_mode="bm25", exact_tier=True
    ).retrieval_cascade() == [CacheMode.EXACT, CacheMode.BM25]
    assert SemanticCacheConfig(cache_mode="exact").retrieval_cascade() == [
        CacheMode.EXACT
    ]
    assert SemanticCacheConfig(cache_mode="off").retrieval_cascade() == []


def test_cache_mode_list_supersedes_exact_tier() -> None:
    cfg = SemanticCacheConfig(cache_mode=["bm25"], exact_tier=True)
    assert cfg.retrieval_cascade() == [CacheMode.BM25]


def test_cache_mode_list_rejects_off_and_empty() -> None:
    with pytest.raises(Exception):
        SemanticCacheConfig(cache_mode=["off"])
    with pytest.raises(Exception):
        SemanticCacheConfig(cache_mode=[])


# --------------------------------------------------------------------------- #
# cascade behavior — first hit wins, lazy work, one write serves all tiers
# --------------------------------------------------------------------------- #


def test_cascade_exact_then_bm25_short_circuits_on_exact_hit() -> None:
    mgr, redis, emb = _manager_cascade(["exact", "bm25"])
    mgr.set("Capital of France?", "Paris")
    hit = mgr.search("Capital of France?")     # exact hits first
    assert hit["response"] == "Paris"
    assert hit.get("exact") is True
    assert redis._ft.searches == 0             # bm25 tier never consulted
    assert emb.calls == 0


def test_cascade_falls_through_exact_miss_to_bm25() -> None:
    mgr, redis, emb = _manager_cascade(["exact", "bm25"])
    mgr.set("Capital of France?", "Paris")
    key = _only_key(redis)
    redis._ft.result_docs = [
        _Doc(key, score="3.0", response="Paris",
             metadata="{}", entities="[]", domain="general")
    ]
    hit = mgr.search("whats the capital of france")   # exact miss → bm25 hit
    assert hit["response"] == "Paris"
    assert hit["method"] == "bm25"
    assert redis._ft.searches == 1
    assert emb.calls == 0                      # still no embedding


def test_cascade_with_semantic_writes_a_vector_others_dont() -> None:
    with_sem, redis_a, emb_a = _manager_cascade(["exact", "bm25", "semantic"])
    with_sem.set("Capital of France?", "Paris")
    assert "vector" in _only_doc(redis_a)      # semantic in cascade → full write
    assert emb_a.calls == 1

    no_sem, redis_b, emb_b = _manager_cascade(["exact", "bm25"])
    no_sem.set("Capital of France?", "Paris")
    assert "vector" not in _only_doc(redis_b)  # text-only write
    assert emb_b.calls == 0
