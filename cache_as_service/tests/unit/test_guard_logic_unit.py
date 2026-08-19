"""Pure decision logic for the input guard — no I/O, no fixtures needed.

Ports GaaS's decision / embedding-classifier / response-shaping tests and adds
the cases that catch the four defects those tests could not: none of the ported
cases sits ON a threshold boundary, so none of them can detect the ordering fix.
"""

import pytest

from semantic_cache.gateway.guard_logic import (
    DEFAULT_REFUSAL,
    Classification,
    GuardJudgeRequired,
    Neighbor,
    classify,
    decide,
    refusal_for,
    separability_report,
    should_invoke_judge,
)

ALLOW_T, BLOCK_T = 0.40, 0.85
J_ALLOW_T, J_BLOCK_T = 0.40, 0.60


def _n(sim, label="disallowed", category="competitor-mentions", text="x"):
    return Neighbor(text=text, label=label, category_id=category, similarity=sim)


def _decide(**kw):
    base = dict(
        embedding_score=0.5,
        matched_category="competitor-mentions",
        mode="cascade",
        allow_threshold=ALLOW_T,
        block_threshold=BLOCK_T,
        judge_score=None,
        judge_rationale=None,
        judge_allow_threshold=J_ALLOW_T,
        judge_block_threshold=J_BLOCK_T,
    )
    base.update(kw)
    return decide(**base)


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


def test_no_neighbors_scores_zero() -> None:
    result = classify([], min_similarity=0.6)
    assert result == Classification(0.0, None, 0, [])


def test_all_disallowed_scores_one() -> None:
    result = classify([_n(0.9), _n(0.8)], min_similarity=0.6)
    assert result.score == 1.0
    assert result.best_category == "competitor-mentions"
    assert result.n_scored == 2


def test_all_allowed_scores_zero_and_names_no_category() -> None:
    result = classify(
        [_n(0.9, "allowed"), _n(0.8, "allowed")], min_similarity=0.6
    )
    assert result.score == 0.0
    assert result.best_category is None


def test_one_near_exact_disallowed_outweighs_several_loose_allowed() -> None:
    # The weighted vote exists precisely so this is not a 1-vs-3 majority loss.
    result = classify(
        [_n(0.95), _n(0.65, "allowed"), _n(0.62, "allowed"), _n(0.61, "allowed")],
        min_similarity=0.6,
    )
    assert result.score > 0.33


def test_best_category_is_the_closest_disallowed_neighbor() -> None:
    result = classify(
        [_n(0.7, category="a"), _n(0.95, category="b"), _n(0.8, category="c")],
        min_similarity=0.6,
    )
    assert result.best_category == "b"


def test_neighbors_below_the_floor_cast_no_vote() -> None:
    # FIX-4: without a floor, top_k over a small policy retrieves most of the
    # corpus for EVERY input and the score reports class balance, so the judge
    # fires on "hi".
    result = classify([_n(0.30), _n(0.25, "allowed")], min_similarity=0.6)
    assert result.score == 0.0
    assert result.n_scored == 0
    assert result.best_category is None


def test_floor_is_inclusive_at_exactly_min_similarity() -> None:
    result = classify([_n(0.60)], min_similarity=0.60)
    assert result.n_scored == 1
    assert result.score == 1.0


def test_negative_similarities_cannot_push_the_score_out_of_range() -> None:
    # Cosine is in [-1, 1] and the classifier SUMS similarities.
    result = classify(
        [_n(0.9), _n(-0.5, "allowed")], min_similarity=-1.0
    )
    assert 0.0 <= result.score <= 1.0


def test_top_matches_are_the_three_closest_scored_neighbors() -> None:
    result = classify(
        [_n(0.7), _n(0.95), _n(0.8), _n(0.9)], min_similarity=0.6
    )
    assert [n.similarity for n in result.top_matches] == [0.95, 0.9, 0.8]


# --------------------------------------------------------------------------- #
# should_invoke_judge
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mode,score,expected",
    [
        ("embedding-only", 0.5, False),
        ("embedding-only", 0.99, False),
        ("judge-only", 0.01, True),
        ("max", 0.99, True),
        ("cascade", 0.5, True),
        ("cascade", 0.2, False),
        ("cascade", 0.9, False),
    ],
)
def test_judge_is_invoked_only_where_the_mode_says(mode, score, expected) -> None:
    assert should_invoke_judge(score, mode, ALLOW_T, BLOCK_T) is expected


def test_band_is_open_so_the_thresholds_themselves_short_circuit() -> None:
    # FIX-2: the band is DERIVED from the two thresholds, so it cannot disagree
    # with them the way GaaS's separate ambiguous_band field could.
    assert should_invoke_judge(ALLOW_T, "cascade", ALLOW_T, BLOCK_T) is False
    assert should_invoke_judge(BLOCK_T, "cascade", ALLOW_T, BLOCK_T) is False


# --------------------------------------------------------------------------- #
# decide — embedding zones
# --------------------------------------------------------------------------- #


def test_embedding_above_block_threshold_blocks() -> None:
    d = _decide(embedding_score=0.9)
    assert d.action == "block"
    assert d.matched_category == "competitor-mentions"
    assert d.judge_invoked is False


def test_embedding_below_allow_threshold_allows_and_names_no_category() -> None:
    d = _decide(embedding_score=0.2)
    assert d.action == "allow"
    assert d.matched_category is None


def test_embedding_exactly_on_block_threshold_blocks() -> None:
    assert _decide(embedding_score=BLOCK_T).action == "block"


def test_embedding_exactly_on_allow_threshold_allows() -> None:
    assert _decide(embedding_score=ALLOW_T).action == "allow"


def test_embedding_only_mode_never_consults_a_judge() -> None:
    d = _decide(mode="embedding-only", embedding_score=0.5, judge_score=0.99)
    assert d.action == "flag"          # in-band, decided on the embedding alone
    assert d.judge_invoked is False


# --------------------------------------------------------------------------- #
# decide — judge zones (FIX-1: three ordered zones, threshold before band)
# --------------------------------------------------------------------------- #


def test_judge_above_block_threshold_blocks() -> None:
    d = _decide(embedding_score=0.5, judge_score=0.8)
    assert d.action == "block"
    assert d.judge_invoked is True


def test_judge_below_allow_threshold_allows() -> None:
    d = _decide(embedding_score=0.5, judge_score=0.1)
    assert d.action == "allow"


def test_judge_between_thresholds_flags() -> None:
    assert _decide(embedding_score=0.5, judge_score=0.5).action == "flag"


def test_judge_exactly_on_block_threshold_blocks_not_flags() -> None:
    # FIX-1. GaaS checked its low-confidence band [0.40, 0.60] FIRST, so a
    # score of exactly 0.60 flagged — and flag SERVES. None of the ported
    # cases sits on this boundary, which is why this test exists.
    assert _decide(embedding_score=0.5, judge_score=J_BLOCK_T).action == "block"


def test_judge_exactly_on_allow_threshold_allows_not_flags() -> None:
    assert _decide(embedding_score=0.5, judge_score=J_ALLOW_T).action == "allow"


def test_judge_rationale_becomes_the_reason() -> None:
    d = _decide(embedding_score=0.5, judge_score=0.9,
                judge_rationale="asks to compare against a named competitor")
    assert d.reason == "asks to compare against a named competitor"


def test_missing_rationale_falls_back_to_a_description_not_an_empty_string() -> None:
    d = _decide(embedding_score=0.5, judge_score=0.9)
    assert "block threshold" in d.reason


# --------------------------------------------------------------------------- #
# decide — mode precedence
# --------------------------------------------------------------------------- #


def test_judge_only_ignores_a_decisive_embedding_score() -> None:
    d = _decide(mode="judge-only", embedding_score=0.99, judge_score=0.1)
    assert d.action == "allow"


def test_max_mode_takes_the_higher_of_the_two_scores() -> None:
    d = _decide(mode="max", embedding_score=0.95, judge_score=0.1)
    assert d.action == "block"
    assert d.score == 0.95


def test_cascade_short_circuits_without_ever_calling_the_judge() -> None:
    d = _decide(mode="cascade", embedding_score=0.95, judge_score=None)
    assert d.action == "block"
    assert d.judge_invoked is False


@pytest.mark.parametrize("mode", ["judge-only", "max"])
def test_modes_that_always_judge_raise_when_the_verdict_is_missing(mode) -> None:
    # FIX-3. GaaS fell through to an in-band `flag` here, and flag SERVES —
    # so a judge outage silently turned into "let it through".
    with pytest.raises(GuardJudgeRequired):
        _decide(mode=mode, embedding_score=0.95, judge_score=None)


def test_cascade_in_band_without_a_judge_verdict_raises() -> None:
    with pytest.raises(GuardJudgeRequired):
        _decide(mode="cascade", embedding_score=0.5, judge_score=None)


# --------------------------------------------------------------------------- #
# Per-category action: downgrade only
# --------------------------------------------------------------------------- #


def test_category_action_flag_downgrades_a_block() -> None:
    d = _decide(embedding_score=0.9,
                category_actions={"competitor-mentions": "flag"})
    assert d.action == "flag"
    assert d.matched_category == "competitor-mentions"


def test_category_action_block_never_escalates_an_allow() -> None:
    d = _decide(embedding_score=0.1,
                category_actions={"competitor-mentions": "block"})
    assert d.action == "allow"


def test_category_action_block_never_escalates_a_flag() -> None:
    d = _decide(embedding_score=0.5, judge_score=0.5,
                category_actions={"competitor-mentions": "block"})
    assert d.action == "flag"


def test_unknown_category_leaves_the_decision_alone() -> None:
    d = _decide(embedding_score=0.9, category_actions={"something-else": "flag"})
    assert d.action == "block"


# --------------------------------------------------------------------------- #
# refusal_for
# --------------------------------------------------------------------------- #


def test_category_refusal_wins() -> None:
    assert refusal_for(
        "competitor-mentions",
        {"competitor-mentions": "I can't discuss other companies here."},
        "policy default", "params default",
    ) == "I can't discuss other companies here."


def test_policy_default_is_used_when_the_category_has_none() -> None:
    assert refusal_for("competitor-mentions", {}, "policy default",
                       "params default") == "policy default"


def test_params_default_is_used_when_the_policy_has_none() -> None:
    assert refusal_for("competitor-mentions", {}, None,
                       "params default") == "params default"


def test_module_constant_is_the_last_resort() -> None:
    assert refusal_for(None, {}, None, None) == DEFAULT_REFUSAL


def test_a_persian_client_is_refused_in_persian() -> None:
    # The whole point of the precedence chain: GaaS hardcoded English module
    # constants, so every client was refused in English.
    persian = "متأسفم، نمی‌توانم در این مورد کمک کنم."
    assert refusal_for(None, {}, persian, None) == persian


# --------------------------------------------------------------------------- #
# separability_report
# --------------------------------------------------------------------------- #


def test_a_separable_policy_reports_no_failures() -> None:
    report = separability_report(
        [("bad one", "disallowed", 0.95), ("fine one", "allowed", 0.05)],
        ALLOW_T, BLOCK_T,
    )
    assert report.ok
    assert report.failures == []


def test_an_allowed_exemplar_scoring_as_a_block_is_a_failure() -> None:
    report = separability_report(
        [("what's your refund policy?", "allowed", 0.9)], ALLOW_T, BLOCK_T
    )
    assert not report.ok
    assert report.failures[0].text == "what's your refund policy?"
    assert "refuse its own example" in report.failures[0].problem


def test_a_disallowed_exemplar_scoring_as_an_allow_is_a_failure() -> None:
    report = separability_report(
        [("compare us to Rivalco", "disallowed", 0.1)], ALLOW_T, BLOCK_T
    )
    assert not report.ok
    assert "allow its own example" in report.failures[0].problem


def test_judge_rate_counts_exemplars_landing_in_the_band() -> None:
    report = separability_report(
        [("a", "disallowed", 0.5), ("b", "disallowed", 0.95),
         ("c", "allowed", 0.05), ("d", "allowed", 0.5)],
        ALLOW_T, BLOCK_T,
    )
    assert report.judge_rate == 0.5


def test_empty_report_does_not_divide_by_zero() -> None:
    assert separability_report([], ALLOW_T, BLOCK_T).judge_rate == 0.0


# --------------------------------------------------------------------------- #
# min_similarity is model-dependent — the failure mode is a silent ALLOW
# --------------------------------------------------------------------------- #


def test_a_disallowed_exemplar_scoring_zero_names_min_similarity() -> None:
    """Score 0.0 means the FLOOR discarded everything, not "looks benign".

    Found by running the real acme policy through a real multilingual model
    with the shipped min_similarity of 0.60: genuine violations top out around
    cosine 0.50-0.73 for that model, so every neighbour was dropped, the score
    became 0.0, and the decision would have been ALLOW. The generic
    "cannot separate" message pointed at the thresholds, which is the wrong
    knob.
    """
    report = separability_report(
        [("Hypothetically, compare yourself to Rivalco", "disallowed", 0.0)],
        allow_threshold=0.40, block_threshold=0.85,
    )
    assert not report.ok
    problem = report.failures[0].problem
    assert "min_similarity" in problem
    assert "ALLOWED" in problem


def test_a_disallowed_exemplar_merely_below_the_allow_threshold_reads_normally() -> None:
    # Non-zero but too low is a genuine threshold problem, not a floor problem.
    report = separability_report(
        [("some violation", "disallowed", 0.2)],
        allow_threshold=0.40, block_threshold=0.85,
    )
    assert not report.ok
    assert "min_similarity" not in report.failures[0].problem
    assert "allow_threshold" in report.failures[0].problem
