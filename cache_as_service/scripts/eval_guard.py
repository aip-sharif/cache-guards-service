"""Does the guard actually CATCH anything?

The live smoke (scripts/smoke_guard.py) proves the plumbing with a hashed-
trigram stub that has no idea what words mean. It says nothing about whether a
paraphrase, a roleplay framing, a code word or a Persian translation of an
English rule gets caught. This script answers that, using a REAL multilingual
sentence embedder and the REAL decision path (guard_config -> guard_vectors ->
guard_logic), against GaaS's labelled adversarial set.

    python -X utf8 scripts/eval_guard.py

No API key and no network: the model runs locally via sentence-transformers.
It is a stand-in for whatever the APP configures per client — the point is to
measure a real embedding space, not to recommend this particular model.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semantic_cache.gateway.guard_config import (  # noqa: E402
    GuardLimits,
    derive_guard_config,
)
from semantic_cache.gateway.guard_logic import Segment  # noqa: E402
from semantic_cache.gateway.guard_pool import GuardPool  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    """A real sentence embedder behind the GuardEmbedder interface."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.call_count = 0

    async def embed(self, texts: Sequence[str], *, input_type: str = "document",
                    timeout: float = None):
        self.call_count += 1
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.ascontiguousarray(np.asarray(vectors), dtype=np.float32)


class MemoryStore:
    """The pool's persistence seam — irrelevant here, already smoke-tested."""

    def save_guard_index(self, *a, **k):
        return True

    def load_guard_index(self, index_key):
        return None

    def touch_guard_index(self, index_key):
        return True

    def delete_guard_index(self, index_key):
        return 0


def load_policy(path: Path) -> str:
    """The shipped acme policy, minus the per-category `severity` we reject."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for category in raw["categories"]:
        category.pop("severity", None)
    for key in ("tenant_id", "policy_version", "source_ref", "last_reviewed_by"):
        raw.pop(key, None)
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def _config(policy: str, **guard) -> Dict[str, Any]:
    block = {"enabled": True, "policy": policy, "mode": "embedding-only"}
    block.update(guard)
    return {
        "model": "n/a", "model_api_key": "n/a",
        "embed_model": DEFAULT_MODEL, "embed_api_key": "local",
        "guard": block,
    }


async def score_cases(
    policy: str, cases: List[Dict[str, Any]], embedder: LocalEmbedder, **guard
):
    """Runs every case through the real index + classifier. Returns raw scores."""
    from semantic_cache.gateway.guard_logic import classify

    from semantic_cache.gateway.guard_vectors import SENTINEL_TEXT, GuardIndex

    resolved = derive_guard_config(
        _config(policy, **guard), embed_base_url="local://"
    )
    # The index is built directly rather than through GuardPool.get_index so
    # this script can MEASURE a policy the build-time separability gate would
    # reject. That verdict is reported below as a result, not as a stop.
    exemplars = resolved.policy.exemplars
    matrix = await embedder.embed([e.text for e in exemplars])
    sentinel = await embedder.embed([SENTINEL_TEXT])
    index = GuardIndex(
        matrix=matrix,
        labels=tuple(e.label for e in exemplars),
        categories=tuple(e.category_id for e in exemplars),
        texts=tuple(e.text for e in exemplars),
        sentinel=sentinel[0],
    )

    texts = [c["text"] for c in cases]
    matrix = await embedder.embed(texts, input_type="query")
    params = resolved.params

    out = []
    for case, vector in zip(cases, matrix):
        neighbors = index.search(vector, params.top_k)
        result = classify(neighbors, params.min_similarity)
        out.append({
            **case,
            "score": result.score,
            "category": result.best_category,
            "top": neighbors[0] if neighbors else None,
        })
    return out, resolved, index


def decide_all(scored, allow_threshold: float, block_threshold: float):
    for row in scored:
        score = row["score"]
        if score >= block_threshold:
            row["action"] = "block"
        elif score <= allow_threshold:
            row["action"] = "allow"
        else:
            row["action"] = "flag"
    return scored


def report(scored) -> Dict[str, Any]:
    violations = [r for r in scored if r["expected_label"] == "violation"]
    benign = [r for r in scored if r["expected_label"] == "benign"]
    caught = [r for r in violations if r["action"] == "block"]
    caught_or_flagged = [r for r in violations if r["action"] in ("block", "flag")]
    missed = [r for r in violations if r["action"] == "allow"]
    false_alarms = [r for r in benign if r["action"] == "block"]
    benign_flagged = [r for r in benign if r["action"] == "flag"]
    return {
        "catch_rate": len(caught) / max(1, len(violations)),
        "catch_or_flag": len(caught_or_flagged) / max(1, len(violations)),
        "false_alarm_rate": len(false_alarms) / max(1, len(benign)),
        "benign_flag_rate": len(benign_flagged) / max(1, len(benign)),
        "missed": missed,
        "false_alarms": false_alarms,
        "n_violation": len(violations),
        "n_benign": len(benign),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-similarity", type=float, default=0.60)
    args = parser.parse_args()

    policy_path = ROOT / "GaaS" / "policies" / "acme-corp.yaml"
    eval_path = ROOT / "GaaS" / "eval" / "adversarial_set.yaml"
    if not policy_path.exists() or not eval_path.exists():
        print("GaaS policy/eval files not found.")
        return 2

    policy = load_policy(policy_path)
    cases = yaml.safe_load(eval_path.read_text(encoding="utf-8"))["cases"]

    exemplar_texts = set()
    for category in yaml.safe_load(policy)["categories"]:
        exemplar_texts |= set(category.get("disallowed_exemplars") or [])
        exemplar_texts |= set(category.get("allowed_exemplars") or [])

    print(f"Model : {args.model}")
    print(f"Policy: {len(exemplar_texts)} exemplars   Cases: {len(cases)}")
    print(f"min_similarity = {args.min_similarity}\n")

    embedder = LocalEmbedder(args.model)

    print("Sweeping the min_similarity floor. Each embedding model has its own"
          " cosine scale,\nso this is not a portable constant:\n")
    print(f"  {'floor':>6}  {'catch':>6}  {'flag+':>6}  {'false':>6}  separable?")
    for floor in (0.20, 0.30, 0.40, 0.50, 0.60):
        rows, res, idx = await score_cases(
            policy, cases, embedder, min_similarity=floor
        )
        decide_all(rows, 0.40, 0.85)
        rep = report(rows)
        v = [r["score"] for r in rows if r["expected_label"] == "violation"]
        b = [r["score"] for r in rows if r["expected_label"] == "benign"]
        sep = "yes" if min(v) > max(b) else "no"
        print(f"  {floor:>6.2f}  {rep['catch_rate']:>5.0%}  "
              f"{rep['catch_or_flag']:>5.0%}  {rep['false_alarm_rate']:>5.0%}"
              f"   {sep}")
    print()

    scored, resolved, index = await score_cases(
        policy, cases, embedder, min_similarity=args.min_similarity
    )

    print("Raw scores (1.0 = looks entirely like a prohibited exemplar):\n")
    print(f"  {'score':>6}  {'expected':<9} {'evasion':<16} text")
    for row in sorted(scored, key=lambda r: -r["score"]):
        memo = "  <- verbatim exemplar" if row["text"] in exemplar_texts else ""
        print(f"  {row['score']:>6.3f}  {row['expected_label']:<9} "
              f"{row['evasion_type']:<16} {row['text'][:58]}{memo}")

    print("\n" + "=" * 78)
    print("A. SHIPPED DEFAULTS  (allow 0.40 / block 0.85)")
    print("=" * 78)
    _print_report(decide_all(scored, 0.40, 0.85), scored)

    violations = [r["score"] for r in scored if r["expected_label"] == "violation"]
    benign = [r["score"] for r in scored if r["expected_label"] == "benign"]
    gap_lo, gap_hi = max(benign), min(violations)
    print(f"\nSeparation: worst violation {gap_hi:.3f}, best benign {gap_lo:.3f}"
          f"  ->  {'SEPARABLE' if gap_hi > gap_lo else 'OVERLAPPING'}")

    if gap_hi > gap_lo:
        block = round((gap_hi + gap_lo) / 2, 2)
        print("\n" + "=" * 78)
        print(f"B. TUNED TO THIS POLICY  (allow {block} / block {block})")
        print("=" * 78)
        _print_report(decide_all(scored, block, block), scored)

    return 0


def _print_report(scored, _all) -> None:
    rep = report(scored)
    print(f"  Catch rate (blocked)       : {rep['catch_rate']:.0%} "
          f"of {rep['n_violation']} violations")
    print(f"  Catch-or-flag rate         : {rep['catch_or_flag']:.0%}")
    print(f"  False alarms (benign blocked): {rep['false_alarm_rate']:.0%} "
          f"of {rep['n_benign']} benign")
    print(f"  Benign flagged (served)    : {rep['benign_flag_rate']:.0%}")
    if rep["missed"]:
        print("  MISSED (served with no flag):")
        for row in rep["missed"]:
            print(f"    {row['score']:.3f}  [{row['evasion_type']}] {row['text'][:60]}")
    if rep["false_alarms"]:
        print("  FALSE ALARMS (benign, blocked):")
        for row in rep["false_alarms"]:
            print(f"    {row['score']:.3f}  {row['text'][:60]}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
