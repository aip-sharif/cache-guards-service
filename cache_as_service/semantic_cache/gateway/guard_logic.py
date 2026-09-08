"""Pure decision logic for the input guard. No I/O, no config singletons.

Everything here is a function of its arguments, so the whole scoring and
threshold behaviour is unit-testable without Redis, Postgres, an embedding
endpoint or a judge.

This is a port of GaaS's ``decision.py`` + ``classifiers/embedding_classifier.py``
+ ``response_shaping.py``, with four behavioural defects fixed. Each fix is
marked FIX-n below and has a dedicated test:

FIX-1  GaaS tested the judge's low-confidence BAND before its decision
       THRESHOLD, so a documented ``decision_threshold: 0.50`` behaved as 0.60.
       Here there are two ordered thresholds and no band, so the boundaries are
       explicit and cannot overlap.
FIX-2  GaaS carried ``ambiguous_band`` as a third copy of the allow/block
       thresholds, read by ``should_invoke_judge`` but NOT by ``decide`` — so
       ``[0.0, 1.0]`` sent 100% of traffic to the judge and still validated.
       The band is derived from the two thresholds and cannot drift.
FIX-3  GaaS's ``decide`` had ``or not judge_invoked`` in the embedding branch,
       so a mode that REQUIRES a judge, whose judge did not run, fell through
       to an in-band ``flag`` — and flag SERVES. Here that raises.
FIX-4  GaaS summed raw similarities with no floor, so on a small policy the
       top-k neighbours were most of the corpus and the score measured class
       balance rather than resemblance — the judge fired on "hi". A
       ``min_similarity`` floor drops neighbours that resemble nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Mapping, Optional, Sequence

Action = Literal["block", "allow", "flag"]
Mode = Literal["cascade", "embedding-only", "judge-only", "max"]

#: Last-resort refusal. Only reached when neither the matched category, nor the
#: policy, nor the client's guard config supplies one. GaaS hardcoded English
#: module constants with a tenant id baked in; here every layer above this is
#: client-authored, so a Persian client is not refused in English.
DEFAULT_REFUSAL = "I'm not able to help with that request."

DEFAULT_UNAVAILABLE_REFUSAL = (
    "I can't process that request right now. Please try again."
)


class GuardJudgeRequired(Exception):
    """The configured mode requires a judge verdict that was not supplied.

    Raised instead of quietly degrading. See FIX-3: the degraded outcome in
    GaaS was ``flag``, which serves the request — i.e. the failure mode of a
    missing judge was to let traffic through.
    """


@dataclass(frozen=True)
class Segment:
    """One checkable piece of the request: a message the guard must score."""

    role: str
    text: str
    index: int


@dataclass(frozen=True)
class Neighbor:
    """One retrieved policy exemplar and how close the input was to it."""

    text: str
    label: Literal["disallowed", "allowed"]
    category_id: str
    similarity: float


@dataclass(frozen=True)
class Classification:
    #: 0.0 = resembles only allowed exemplars, 1.0 = resembles only disallowed.
    score: float
    best_category: Optional[str]
    #: How many neighbours actually cleared the floor and cast a vote. Zero
    #: means "this input resembles nothing in the policy".
    n_scored: int
    top_matches: List[Neighbor] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    action: Action
    matched_category: Optional[str]
    reason: str
    judge_invoked: bool
    score: float


@dataclass(frozen=True)
class SeparabilityFailure:
    text: str
    label: str
    score: float
    problem: str


@dataclass(frozen=True)
class SeparabilityReport:
    failures: List[SeparabilityFailure]
    #: Fraction of exemplars whose own score lands in the judge band. A high
    #: number predicts a high judge bill on real traffic.
    judge_rate: float

    @property
    def ok(self) -> bool:
        return not self.failures


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def classify(
    neighbors: Sequence[Neighbor], min_similarity: float
) -> Classification:
    """Similarity-weighted vote over the retrieved exemplars.

    Weighted rather than a plain majority so one near-exact match to a
    disallowed exemplar outweighs several loosely similar allowed ones —
    deliberately biased towards catching disguised near-duplicates, since
    recall is the primary metric for a guard.

    ``min_similarity`` (FIX-4) is the floor below which a neighbour casts no
    vote at all. Without it, ``top_k=8`` over a 13-exemplar policy retrieves
    most of the corpus for EVERY input, so the score reports the policy's
    class balance rather than any resemblance to the input.
    """
    scored = [n for n in neighbors if n.similarity >= min_similarity]
    if not scored:
        # Nothing in this policy resembles the input. That is a genuine allow,
        # NOT GaaS's empty-policy branch: a policy with no exemplars is
        # rejected at parse time (guard_config) and cannot reach here.
        return Classification(score=0.0, best_category=None, n_scored=0,
                              top_matches=list(neighbors[:3]))

    # Cosine is in [-1, 1] and we SUM the values, so a negative similarity
    # could push the ratio outside [0, 1]. Clamp per-neighbour.
    def weight(n: Neighbor) -> float:
        return max(0.0, n.similarity)

    disallowed = [n for n in scored if n.label == "disallowed"]
    disallowed_weight = sum(weight(n) for n in disallowed)
    total = sum(weight(n) for n in scored)
    score = (disallowed_weight / total) if total > 0 else 0.0

    # Only name a category when a disallowed exemplar actually cleared the
    # floor — otherwise a user could be shown a refusal for the wrong reason.
    best_category = (
        max(disallowed, key=lambda n: n.similarity).category_id
        if disallowed
        else None
    )
    return Classification(
        score=score,
        best_category=best_category,
        n_scored=len(scored),
        top_matches=sorted(scored, key=lambda n: -n.similarity)[:3],
    )


def should_invoke_judge(
    score: float, mode: Mode, allow_threshold: float, block_threshold: float
) -> bool:
    """Whether the embedding score is ambiguous enough to pay for a judge.

    The ambiguous band is DERIVED (FIX-2): it is exactly the open interval
    between the two thresholds a human already had to set. There is no
    separate field that can silently disagree with them.
    """
    if mode == "embedding-only":
        return False
    if mode in ("judge-only", "max"):
        return True
    return allow_threshold < score < block_threshold


def decide(
    *,
    embedding_score: float,
    matched_category: Optional[str],
    mode: Mode,
    allow_threshold: float,
    block_threshold: float,
    judge_score: Optional[float],
    judge_rationale: Optional[str],
    judge_allow_threshold: float,
    judge_block_threshold: float,
    category_actions: Optional[Mapping[str, str]] = None,
) -> Decision:
    """Scores + thresholds -> exactly one of block / allow / flag.

    Precedence, stated explicitly because GaaS's was implicit and wrong:

    * ``embedding-only`` never consults a judge.
    * ``cascade`` short-circuits on the embedding score OUTSIDE the band, and
      only consults the judge inside it.
    * ``judge-only`` and ``max`` ALWAYS require a judge verdict; the embedding
      short-circuits do not apply to them.

    Raises :class:`GuardJudgeRequired` when a mode requires a judge and none
    was supplied (FIX-3).
    """
    judge_invoked = judge_score is not None

    if mode == "embedding-only":
        return _apply_category_action(
            _decide_on_embedding(
                embedding_score, matched_category,
                allow_threshold, block_threshold, judge_invoked=False,
            ),
            category_actions,
        )

    if mode == "cascade" and not should_invoke_judge(
        embedding_score, mode, allow_threshold, block_threshold
    ):
        # Outside the band: the embedding score is decisive on its own.
        return _apply_category_action(
            _decide_on_embedding(
                embedding_score, matched_category,
                allow_threshold, block_threshold, judge_invoked=judge_invoked,
            ),
            category_actions,
        )

    if not judge_invoked:
        raise GuardJudgeRequired(
            f"mode={mode!r} requires a judge verdict for embedding score "
            f"{embedding_score:.3f}, but none was supplied"
        )

    final = (
        max(embedding_score, judge_score)  # type: ignore[arg-type]
        if mode == "max"
        else judge_score
    )
    assert final is not None  # narrowed by judge_invoked

    # Three ORDERED zones with no overlap — see FIX-1. A score is compared to
    # the block threshold first, then the allow threshold, and whatever is
    # left between them is the flag zone by construction.
    if final >= judge_block_threshold:
        decision = Decision(
            "block", matched_category,
            judge_rationale or "judge confidence at or above the block threshold",
            True, final,
        )
    elif final <= judge_allow_threshold:
        decision = Decision(
            "allow", None,
            judge_rationale or "judge confidence at or below the allow threshold",
            True, final,
        )
    else:
        decision = Decision(
            "flag", matched_category,
            judge_rationale or "judge confidence between the allow and block thresholds",
            True, final,
        )
    return _apply_category_action(decision, category_actions)


def _decide_on_embedding(
    score: float,
    matched_category: Optional[str],
    allow_threshold: float,
    block_threshold: float,
    judge_invoked: bool,
) -> Decision:
    if score >= block_threshold:
        return Decision("block", matched_category,
                        "embedding score at or above the block threshold",
                        judge_invoked, score)
    if score <= allow_threshold:
        return Decision("allow", None,
                        "embedding score at or below the allow threshold",
                        judge_invoked, score)
    return Decision("flag", matched_category,
                    "embedding score between the allow and block thresholds",
                    judge_invoked, score)


def _apply_category_action(
    decision: Decision, category_actions: Optional[Mapping[str, str]]
) -> Decision:
    """Honours a policy category's ``action:`` — DOWNGRADE ONLY.

    A category may soften a block into a flag. It may never turn a flag or an
    allow into a block: escalation would let one category's setting override
    the thresholds the client tuned, in the direction that refuses traffic.
    """
    if not category_actions or decision.action != "block":
        return decision
    if not decision.matched_category:
        return decision
    if category_actions.get(decision.matched_category) == "flag":
        return Decision(
            "flag", decision.matched_category,
            decision.reason + " (category action: flag)",
            decision.judge_invoked, decision.score,
        )
    return decision


# --------------------------------------------------------------------------- #
# Refusal text
# --------------------------------------------------------------------------- #


def refusal_for(
    matched_category: Optional[str],
    category_refusals: Mapping[str, str],
    policy_default: Optional[str],
    params_default: Optional[str],
) -> str:
    """Template lookup ONLY — a refusal is never generated at request time.

    Precedence: the matched category's own text, then the policy's
    ``default_refusal``, then the client's ``guard.default_refusal``, then the
    module constant. Every layer above the constant is client-authored, which
    is what lets a Persian client refuse in Persian.
    """
    if matched_category:
        text = category_refusals.get(matched_category)
        if text:
            return text
    return policy_default or params_default or DEFAULT_REFUSAL


# --------------------------------------------------------------------------- #
# Build-time calibration
# --------------------------------------------------------------------------- #


def separability_report(
    scores_by_exemplar: Sequence[tuple],
    allow_threshold: float,
    block_threshold: float,
) -> SeparabilityReport:
    """Leave-one-out sanity check on a policy, run once at index build time.

    ``scores_by_exemplar`` is a sequence of ``(text, label, score)`` where the
    score is what :func:`classify` returns for that exemplar when scored
    against every OTHER exemplar in the policy.

    An ALLOWED exemplar that scores at or above the block threshold means the
    policy will refuse its own example of acceptable traffic. A DISALLOWED
    exemplar that scores at or below the allow threshold means the policy will
    wave through its own example of an attack. Either is a config error the
    APP should see immediately, not a surprise in production.
    """
    failures: List[SeparabilityFailure] = []
    in_band = 0
    for text, label, score in scores_by_exemplar:
        if allow_threshold < score < block_threshold:
            in_band += 1
        if label == "disallowed" and score == 0.0:
            # Score 0.0 means NOTHING cleared min_similarity — not that the
            # input looked benign. Cosine scales differ per embedding model
            # (a paraphrase model peaks far lower than e5/bge), so a floor
            # carried over from another model silently discards all the
            # evidence and the answer becomes "allow". Name that cause
            # directly; the generic message below sends people to the wrong
            # knob.
            failures.append(SeparabilityFailure(
                text, label, score,
                "scores 0.000 because NO exemplar cleared min_similarity — "
                "the floor is too high for this embedding model, so every "
                "neighbour was discarded and this violation would be ALLOWED. "
                "Lower guard.min_similarity (it is model-dependent, not a "
                "portable constant)",
            ))
            continue
        if label == "allowed" and score >= block_threshold:
            failures.append(SeparabilityFailure(
                text, label, score,
                f"allowed exemplar scores {score:.3f}, at or above "
                f"block_threshold {block_threshold} — this policy would refuse "
                f"its own example of acceptable traffic",
            ))
        elif label == "disallowed" and score <= allow_threshold:
            failures.append(SeparabilityFailure(
                text, label, score,
                f"disallowed exemplar scores {score:.3f}, at or below "
                f"allow_threshold {allow_threshold} — this policy would allow "
                f"its own example of a violation",
            ))
    total = len(scores_by_exemplar) or 1
    return SeparabilityReport(failures=failures, judge_rate=in_band / total)


__all__ = [
    "Action",
    "Mode",
    "DEFAULT_REFUSAL",
    "DEFAULT_UNAVAILABLE_REFUSAL",
    "GuardJudgeRequired",
    "Segment",
    "Neighbor",
    "Classification",
    "Decision",
    "SeparabilityFailure",
    "SeparabilityReport",
    "classify",
    "should_invoke_judge",
    "decide",
    "refusal_for",
    "separability_report",
]
