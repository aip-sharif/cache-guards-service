"""Guard config + policy parsing — pure unit, no services.

The APP is the only source of a client's guard settings, so almost every test
here is about REJECTING a config rather than accepting one: a guard that
silently ignores a misspelled threshold is worse than one that refuses to start.
"""

from pathlib import Path

import pytest
import yaml

from semantic_cache.gateway import guard_config as gc
from semantic_cache.gateway.guard_config import (
    GuardConfigError,
    GuardLimits,
    GuardParams,
    derive_guard_config,
    parse_policy,
)

EMBED_URL = "https://embed.example.com"

POLICY = """
default_refusal: "I can't help with that."
categories:
  - category_id: competitor-mentions
    action: block
    description: Do not compare against named competitors.
    disallowed_exemplars:
      - "What do you think of Rivalco's product?"
      - "Is Rivalco better than you?"
    allowed_exemplars:
      - "What makes your product different?"
"""


def _guard(**overrides):
    block = {"enabled": True, "policy": POLICY}
    block.update(overrides)
    return block


def _config(**overrides):
    """A full APP config response with a guard block."""
    cfg = {
        "model": "gpt-x",
        "model_api_key": "sk-llm",
        "embed_model": "bge-m3",
        "embed_api_key": "sk-embed",
        "guard": _guard(),
    }
    cfg.update(overrides)
    return cfg


def _resolve(**guard_overrides):
    return derive_guard_config(
        _config(guard=_guard(**guard_overrides)), embed_base_url=EMBED_URL
    )


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [None, {}])
def test_absent_or_empty_guard_means_off(value) -> None:
    assert derive_guard_config(_config(guard=value), embed_base_url=EMBED_URL) is None


def test_no_guard_key_at_all_means_off() -> None:
    cfg = _config()
    del cfg["guard"]
    assert derive_guard_config(cfg, embed_base_url=EMBED_URL) is None


def test_enabled_false_means_off() -> None:
    assert _resolve(enabled=False) is None


def test_policy_without_an_explicit_enabled_is_an_error_not_an_off() -> None:
    # The whole point of the tri-state: ambiguity about whether a safety
    # feature is on must never resolve to "off".
    with pytest.raises(GuardConfigError, match="set it explicitly"):
        derive_guard_config(
            _config(guard={"policy": POLICY}), embed_base_url=EMBED_URL
        )


def test_enabled_true_without_a_policy_is_an_error() -> None:
    with pytest.raises(GuardConfigError, match="policy is missing or empty"):
        derive_guard_config(
            _config(guard={"enabled": True}), embed_base_url=EMBED_URL
        )
    with pytest.raises(GuardConfigError, match="policy is missing or empty"):
        derive_guard_config(
            _config(guard={"enabled": True, "policy": "   "}),
            embed_base_url=EMBED_URL,
        )


def test_guard_that_is_not_an_object_is_rejected_cleanly() -> None:
    # An APP that sends the block as a JSON *string* must produce a guard
    # error, not a pydantic error escaping as a bare 500.
    with pytest.raises(GuardConfigError, match="must be a JSON object"):
        derive_guard_config(_config(guard='{"enabled": true}'), embed_base_url=EMBED_URL)


# --------------------------------------------------------------------------- #
# Unknown / renamed fields
# --------------------------------------------------------------------------- #


def test_unknown_guard_field_is_rejected_and_names_the_allow_list() -> None:
    with pytest.raises(GuardConfigError) as excinfo:
        _resolve(bogus=1)
    message = str(excinfo.value)
    assert "bogus" in message
    assert "block_threshold" in message  # the allow-list is spelled out


def test_x_prefixed_keys_are_accepted_and_ignored() -> None:
    # So the APP can add x_notes without 502-ing every older gateway's clients
    # for a full config-TTL window.
    resolved = _resolve(x_notes="ticket-4417", x_version=3)
    assert resolved is not None
    assert not hasattr(resolved.params, "x_notes")


def test_fail_open_gets_a_rename_hint_not_a_generic_error() -> None:
    with pytest.raises(GuardConfigError) as excinfo:
        _resolve(fail_open=True)
    message = str(excinfo.value)
    assert "degrade_to_unguarded" in message
    assert "cache_config.fail_open" in message


# --------------------------------------------------------------------------- #
# Cross-field validation
# --------------------------------------------------------------------------- #


def test_inverted_thresholds_are_rejected() -> None:
    with pytest.raises(GuardConfigError, match="allow_threshold"):
        _resolve(allow_threshold=0.9, block_threshold=0.5)
    with pytest.raises(GuardConfigError, match="judge_allow_threshold"):
        _resolve(judge_allow_threshold=0.9, judge_block_threshold=0.5)


def test_equal_thresholds_are_allowed() -> None:
    assert _resolve(allow_threshold=0.5, block_threshold=0.5) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 0), ("top_k", 101),
        ("min_similarity", -0.1), ("min_similarity", 1.1),
        ("block_threshold", 1.5),
        ("judge_max_concurrency", 0),
        ("max_segments", 0),
    ],
)
def test_out_of_range_values_are_rejected(field, value) -> None:
    with pytest.raises(GuardConfigError):
        _resolve(**{field: value})


def test_mode_that_can_judge_requires_a_judge_model_and_key() -> None:
    for mode in ("cascade", "judge-only", "max"):
        with pytest.raises(GuardConfigError, match="judge_model"):
            _resolve(mode=mode)
        with pytest.raises(GuardConfigError, match="judge_model"):
            _resolve(mode=mode, judge_model="qwen")  # key still missing


def test_embedding_only_mode_needs_no_judge() -> None:
    resolved = _resolve(mode="embedding-only")
    assert resolved is not None
    assert resolved.judge_model is None


def test_a_configured_judge_resolves() -> None:
    resolved = _resolve(judge_model="qwen", judge_api_key="sk-judge")
    assert resolved.judge_model == "qwen"
    assert resolved.judge_api_key == "sk-judge"


def test_unknown_mode_and_prefix_style_are_rejected() -> None:
    with pytest.raises(GuardConfigError):
        _resolve(mode="telepathy")
    with pytest.raises(GuardConfigError, match="embed_prefix_style"):
        _resolve(embed_prefix_style="bge")


def test_unknown_unavailable_response_is_rejected() -> None:
    with pytest.raises(GuardConfigError, match="unavailable_response"):
        _resolve(unavailable_response="teapot")


# --------------------------------------------------------------------------- #
# check_roles — what gets guarded
# --------------------------------------------------------------------------- #


def test_check_roles_defaults_to_user_system_tool_with_assistant_off() -> None:
    resolved = _resolve()
    assert resolved.check_roles == ("user", "system", "tool")
    assert "assistant" not in resolved.check_roles


@pytest.mark.parametrize(
    "roles",
    [["user"], ["user", "system"], ["user", "system", "tool", "assistant"]],
)
def test_the_app_can_choose_what_is_guarded(roles) -> None:
    assert _resolve(check_roles=roles).check_roles == tuple(roles)


def test_empty_or_unknown_check_roles_are_rejected() -> None:
    with pytest.raises(GuardConfigError, match="must not be empty"):
        _resolve(check_roles=[])
    with pytest.raises(GuardConfigError, match="unknown roles"):
        _resolve(check_roles=["user", "developer"])


# --------------------------------------------------------------------------- #
# Policy document
# --------------------------------------------------------------------------- #


def test_invalid_yaml_is_rejected() -> None:
    with pytest.raises(GuardConfigError, match="not valid YAML"):
        _resolve(policy="categories: [ unclosed")


def test_policy_must_be_a_mapping_with_categories() -> None:
    with pytest.raises(GuardConfigError, match="must be a YAML mapping"):
        _resolve(policy="- just\n- a list\n")
    with pytest.raises(GuardConfigError, match="non-empty 'categories'"):
        _resolve(policy="default_refusal: hi\ncategories: []\n")


def test_yaml_aliases_and_anchors_are_rejected() -> None:
    aliased = """
categories:
  - category_id: a
    disallowed_exemplars: &shared ["bad"]
    allowed_exemplars: ["good"]
  - category_id: b
    disallowed_exemplars: *shared
    allowed_exemplars: ["fine"]
"""
    with pytest.raises(GuardConfigError, match="anchor|alias"):
        _resolve(policy=aliased)


def test_oversize_policy_is_rejected_before_the_parser_runs(monkeypatch) -> None:
    calls = []

    def _spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("yaml.load must not be reached for an oversize policy")

    monkeypatch.setattr(gc.yaml, "load", _spy)
    huge = "categories:\n" + ("# padding\n" * 40000)
    with pytest.raises(GuardConfigError, match="over the"):
        parse_policy(huge, GuardLimits(max_policy_bytes=1024))
    assert calls == []


def test_policy_with_no_disallowed_exemplars_is_rejected() -> None:
    with pytest.raises(GuardConfigError, match="nothing to catch"):
        _resolve(policy="""
categories:
  - category_id: a
    allowed_exemplars: ["fine"]
""")


def test_policy_with_no_allowed_exemplars_is_rejected() -> None:
    # An empty negative class makes the weighted vote 1.0 for EVERY input.
    with pytest.raises(GuardConfigError, match="every input scores 1.0"):
        _resolve(policy="""
categories:
  - category_id: a
    disallowed_exemplars: ["bad"]
""")


def test_all_flag_policy_is_rejected_with_the_right_advice() -> None:
    with pytest.raises(GuardConfigError, match="guard.enabled: false"):
        _resolve(policy="""
categories:
  - category_id: a
    action: flag
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
""")


def test_mixed_actions_are_fine_and_recorded() -> None:
    resolved = _resolve(policy="""
categories:
  - category_id: hard
    action: block
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
  - category_id: soft
    action: flag
    disallowed_exemplars: ["meh"]
    allowed_exemplars: ["ok"]
""")
    assert resolved.policy.category_actions == {"hard": "block", "soft": "flag"}


def test_per_category_severity_is_rejected() -> None:
    # GaaS carried it and nothing read it; accepting it would imply an effect.
    with pytest.raises(GuardConfigError, match="severity"):
        _resolve(policy="""
categories:
  - category_id: a
    severity: medium
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
""")


def test_duplicate_category_ids_are_rejected() -> None:
    with pytest.raises(GuardConfigError, match="duplicate category_id"):
        _resolve(policy="""
categories:
  - category_id: a
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
  - category_id: a
    disallowed_exemplars: ["worse"]
    allowed_exemplars: ["fine"]
""")


def test_blank_and_non_string_exemplars_are_rejected() -> None:
    for bad in ('["   "]', "[42]"):
        with pytest.raises(GuardConfigError, match="non-empty string"):
            _resolve(policy=f"""
categories:
  - category_id: a
    disallowed_exemplars: {bad}
    allowed_exemplars: ["good"]
""")


def test_exemplar_cap_is_enforced() -> None:
    many = "\n".join(f'      - "bad {i}"' for i in range(20))
    policy = f"""
categories:
  - category_id: a
    disallowed_exemplars:
{many}
    allowed_exemplars: ["good"]
"""
    with pytest.raises(GuardConfigError, match="over the"):
        parse_policy(policy, GuardLimits(max_exemplars=10))


# --------------------------------------------------------------------------- #
# The policy document is data, never identity
# --------------------------------------------------------------------------- #


def test_bookkeeping_keys_are_ignored_and_tenant_id_never_leaks() -> None:
    resolved = _resolve(policy="""
tenant_id: some-other-tenant
policy_version: 7
source_ref: "an email"
last_reviewed_by: someone
categories:
  - category_id: a
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
""")
    assert resolved is not None
    # The raw document is retained verbatim (its hash must match its text), but
    # nothing derived from it carries the tenant. Honouring tenant_id would let
    # whoever wrote the document choose which namespace it lands in.
    assert not hasattr(resolved.policy, "tenant_id")
    assert "some-other-tenant" not in resolved.index_key
    assert "some-other-tenant" not in resolved.params_hash
    assert resolved.policy.categories == ("a",)


def test_unknown_top_level_policy_key_is_rejected() -> None:
    with pytest.raises(GuardConfigError, match="unknown top-level keys"):
        _resolve(policy="""
mystery: 1
categories:
  - category_id: a
    disallowed_exemplars: ["bad"]
    allowed_exemplars: ["good"]
""")


# --------------------------------------------------------------------------- #
# Exemplar order — it indexes rows of the persisted matrix
# --------------------------------------------------------------------------- #


ORDER_POLICY = """
categories:
  - category_id: first
    disallowed_exemplars: ["d1", "d2"]
    allowed_exemplars: ["a1"]
  - category_id: second
    disallowed_exemplars: ["d3"]
    allowed_exemplars: ["a2", "a3"]
"""


def test_exemplar_order_is_document_order_disallowed_first_within_category() -> None:
    policy = parse_policy(ORDER_POLICY, GuardLimits())
    assert [e.text for e in policy.exemplars] == ["d1", "d2", "a1", "d3", "a2", "a3"]
    assert [e.ord for e in policy.exemplars] == [0, 1, 2, 3, 4, 5]
    assert [e.label for e in policy.exemplars] == [
        "disallowed", "disallowed", "allowed", "disallowed", "allowed", "allowed",
    ]
    assert [e.category_id for e in policy.exemplars] == [
        "first", "first", "first", "second", "second", "second",
    ]


def test_exemplar_order_is_stable_across_calls() -> None:
    a = parse_policy(ORDER_POLICY, GuardLimits()).exemplars
    b = parse_policy(ORDER_POLICY, GuardLimits()).exemplars
    assert a == b


def test_the_real_acme_policy_loads() -> None:
    """The shipped GaaS policy, minus per-category `severity`, still parses."""
    path = Path(__file__).resolve().parents[2] / "GaaS" / "policies" / "acme-corp.yaml"
    if not path.exists():  # pragma: no cover — GaaS is not part of the package
        pytest.skip("GaaS/policies/acme-corp.yaml not present")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for category in raw["categories"]:
        category.pop("severity", None)
    policy = parse_policy(yaml.safe_dump(raw, allow_unicode=True), GuardLimits())
    assert len(policy.exemplars) == 13
    assert sum(1 for e in policy.exemplars if e.label == "disallowed") == 8
    assert sum(1 for e in policy.exemplars if e.label == "allowed") == 5
    # The Persian exemplars survive the round trip — they are what exercises
    # the multilingual path the embedding-model choice turns on.
    assert any("ریوالکو" in e.text for e in policy.exemplars)


# --------------------------------------------------------------------------- #
# The two hashes — what costs money and what is free
# --------------------------------------------------------------------------- #


def test_index_key_changes_when_the_vectors_would_change() -> None:
    base = _resolve()
    variants = {
        "policy edit": _resolve(policy=POLICY + '      - "and another"\n'),
        "embed model": _resolve(embed_model="other-embedder"),
        "prefix style": _resolve(embed_prefix_style="e5"),
        "key rotation": _resolve(embed_api_key="sk-rotated"),
    }
    for name, variant in variants.items():
        assert variant.index_key != base.index_key, name


def test_index_key_changes_with_the_serving_endpoint() -> None:
    a = derive_guard_config(_config(), embed_base_url="https://a.example.com")
    b = derive_guard_config(_config(), embed_base_url="https://b.example.com")
    assert a.index_key != b.index_key


def test_retuning_a_threshold_is_free_but_invalidates_the_memo() -> None:
    judge = {"judge_model": "qwen", "judge_api_key": "sk-judge"}
    base = _resolve(**judge)
    for change in ({"block_threshold": 0.9}, {"top_k": 12},
                   {"mode": "embedding-only"}, {"min_similarity": 0.5},
                   {"judge_block_threshold": 0.7}, {"check_roles": ["user"]}):
        variant = _resolve(**{**judge, **change})
        assert variant.index_key == base.index_key, change   # no re-embed
        assert variant.params_hash != base.params_hash, change  # memo dropped


def test_editing_refusal_text_changes_neither_hash() -> None:
    # It changes the message, never the verdict — throwing away a memo full of
    # valid decisions for it would be pure waste.
    variant = _resolve(default_refusal="Sorry, no.")
    base = _resolve()
    assert variant.index_key == base.index_key
    assert variant.params_hash == base.params_hash


def test_hashes_are_full_length_sha256() -> None:
    resolved = _resolve()
    for value in (resolved.index_key, resolved.params_hash, resolved.policy_hash):
        assert len(value) == 64
        int(value, 16)  # hex


def test_api_keys_never_appear_in_any_hash_input_or_repr() -> None:
    resolved = _resolve(judge_model="qwen", judge_api_key="sk-judge-secret")
    assert "sk-judge-secret" not in resolved.index_key
    assert "sk-judge-secret" not in resolved.params_hash


# --------------------------------------------------------------------------- #
# Embedder inheritance
# --------------------------------------------------------------------------- #


def test_guard_inherits_the_caches_embedder_when_not_told_otherwise() -> None:
    resolved = _resolve()
    assert resolved.embed_model == "bge-m3"
    assert resolved.embed_api_key == "sk-embed"


def test_guard_can_use_a_different_embedder_than_the_cache() -> None:
    resolved = _resolve(embed_model="multilingual-e5-large", embed_api_key="sk-g")
    assert resolved.embed_model == "multilingual-e5-large"
    assert resolved.embed_api_key == "sk-g"


def test_missing_embedder_everywhere_is_an_error() -> None:
    cfg = _config()
    cfg["embed_model"] = None
    with pytest.raises(GuardConfigError, match="needs an embedding model"):
        derive_guard_config(cfg, embed_base_url=EMBED_URL)


def test_params_defaults_are_the_documented_ones() -> None:
    params = GuardParams(enabled=True, policy=POLICY,
                         judge_model="q", judge_api_key="k")
    assert params.mode == "cascade"
    assert params.top_k == 8
    assert params.min_similarity == 0.60
    assert params.block_threshold == 0.85
    assert params.allow_threshold == 0.40
    assert params.judge_block_threshold == 0.60
    assert params.judge_allow_threshold == 0.40
    assert params.degrade_to_unguarded is False
    assert params.unavailable_response == "error"
    assert params.log_turn_text is False
