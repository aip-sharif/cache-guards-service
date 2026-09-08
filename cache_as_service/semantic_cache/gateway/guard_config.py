"""Parse, validate and hash one client's guard configuration.

The APP's config response may carry a ``guard`` block (see
``Docs/APP_INTEGRATION.md``). This module turns that raw JSON — plus the inline
policy YAML inside it — into a frozen :class:`ResolvedGuard` that the rest of
the guard can rely on, or raises :class:`GuardConfigError`.

Three ideas are worth knowing before reading the code:

**Tri-state ``enabled``.** Absent guard block, ``{}``, ``null`` and
``enabled: false`` all mean OFF. But a block that carries a POLICY and omits
``enabled`` is an ERROR, not an off: ambiguity about whether a safety feature
is switched on must never resolve to "off".

**Two hashes, deliberately disjoint.** ``index_key`` covers only what changes
the VECTORS (policy text, embedding model, endpoint, key fingerprint, prefix
style) — changing it costs a re-embed, billed to the client. ``params_hash``
covers only what changes a DECISION given those vectors (thresholds, mode,
top_k, judge). So a client may retune every threshold for free, and every such
change still invalidates the per-segment memo by construction.

**The policy document is data, never identity.** ``tenant_id`` inside a policy
is accepted and ignored. Honouring it would let whatever wrote that document
choose which cache/exemplar namespace it lands in — a cross-scope primitive.
Identity comes only from ``_scope(config, api_key)`` in the router.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_cache.gateway.guard_logic import Mode

logger = logging.getLogger(__name__)

#: Roles a policy may ask the guard to check. ``assistant`` is legal but off by
#: default — see PER_GUARD_FIELDS / GuardParams.check_roles.
VALID_ROLES = frozenset({"user", "system", "assistant", "tool"})

DEFAULT_CHECK_ROLES: Tuple[str, ...] = ("user", "system", "tool")

#: Everything the APP may set inside ``guard``. Mirrors saas.registry's
#: PER_CACHE_FIELDS: an unknown key is a hard error, not a silent no-op, so a
#: typo in a safety setting can never look like it was applied.
PER_GUARD_FIELDS = frozenset(
    {
        "enabled",
        "policy",
        "embed_model",
        "embed_api_key",
        "embed_prefix_style",
        "judge_model",
        "judge_api_key",
        "judge_task_description",
        "judge_max_concurrency",
        "mode",
        "top_k",
        "min_similarity",
        "block_threshold",
        "allow_threshold",
        "judge_block_threshold",
        "judge_allow_threshold",
        "check_roles",
        "max_segments",
        "max_input_chars",
        "default_refusal",
        "unavailable_refusal",
        "unavailable_response",
        "degrade_to_unguarded",
        "log_turn_text",
    }
)

#: Policy keys that GaaS's format carried for human bookkeeping and nothing
#: reads. Accepted so an existing policy file loads unchanged, then dropped.
IGNORED_POLICY_KEYS = frozenset(
    {"tenant_id", "policy_version", "source_ref", "last_reviewed_by", "language"}
)


class GuardConfigError(Exception):
    """The APP's guard block, or the policy inside it, is unusable.

    Always surfaced to the client as an HTTP 502 — it is the APP's config that
    is wrong, not the client's request.
    """


# --------------------------------------------------------------------------- #
# Limits (operator-set, never client-set)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardLimits:
    """Operator ceilings from SC_GUARD_* env. Not negotiable by the APP."""

    max_policy_bytes: int = 262144
    max_exemplars: int = 5000


# --------------------------------------------------------------------------- #
# The client's guard parameters
# --------------------------------------------------------------------------- #


class GuardParams(BaseModel):
    """The ``guard`` block, validated.

    Every field except ``enabled`` and ``policy`` is optional: a minimal guard
    block is ``{"enabled": true, "policy": "..."}`` and everything else falls
    back to a default. Any default may be overridden per client by the APP.
    """

    model_config = ConfigDict(extra="forbid")

    # -- switch (tri-state: None means "the APP did not say") --------------- #
    enabled: Optional[bool] = None
    policy: Optional[str] = None

    # -- embedder (None → inherit the cache's embed_model/embed_api_key) ---- #
    embed_model: Optional[str] = None
    embed_api_key: Optional[str] = None
    #: One enum drives BOTH sides. Two independent prefix strings would make
    #: one-sided prefixing — a silent, unmeasurable quality loss — one omitted
    #: line away.
    embed_prefix_style: str = "none"

    # -- judge (required iff the mode can invoke one) ----------------------- #
    judge_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    #: Interpolated as a VALUE into our fixed template. A full template
    #: override is deliberately not offered: whoever wrote it could instruct
    #: the judge to always answer 0.0 while every dashboard reads healthy.
    judge_task_description: str = "a customer support assistant"
    judge_max_concurrency: int = Field(default=4, ge=1, le=64)

    # -- decision ----------------------------------------------------------- #
    #: Never None after validation: an OMITTED mode is filled in from whether a
    #: judge was supplied (see _screen_keys). An mode the APP states explicitly
    #: is honoured or rejected, never quietly downgraded.
    mode: Mode = "cascade"
    top_k: int = Field(default=8, ge=1, le=100)
    min_similarity: float = Field(default=0.60, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    allow_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    judge_block_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    judge_allow_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # -- what gets checked --------------------------------------------------- #
    check_roles: List[str] = Field(default_factory=lambda: list(DEFAULT_CHECK_ROLES))
    max_segments: int = Field(default=32, ge=1, le=500)
    max_input_chars: int = Field(default=16000, ge=1, le=1_000_000)

    # -- refusal text -------------------------------------------------------- #
    default_refusal: Optional[str] = None
    unavailable_refusal: Optional[str] = None

    # -- failure posture ----------------------------------------------------- #
    unavailable_response: str = "error"
    #: NOT named fail_open. cache_config.fail_open already exists in an adjacent
    #: block of the same APP response and means degrade BY DEFAULT; this
    #: defaults the other way and applies only to RUNTIME failures.
    degrade_to_unguarded: bool = False

    # -- privacy -------------------------------------------------------------- #
    log_turn_text: bool = False

    # ---------------------------------------------------------------------- #

    @model_validator(mode="before")
    @classmethod
    def _screen_keys(cls, data: Any) -> Any:
        """Drops ``x_*`` extensions, then rejects anything else unknown.

        The ``x_`` escape exists so the APP can add ``x_notes`` or
        ``x_version`` without taking every older gateway's clients to a hard
        502 for a full config-TTL window.
        """
        if not isinstance(data, dict):
            return data
        if "fail_open" in data:
            raise ValueError(
                "guard.fail_open is not a field; did you mean "
                "'degrade_to_unguarded'? (cache_config.fail_open is a "
                "different setting, and it defaults the other way)"
            )
        screened = {k: v for k, v in data.items() if not str(k).startswith("x_")}
        unknown = set(screened) - PER_GUARD_FIELDS
        if unknown:
            raise ValueError(
                f"Guard fields not customizable: {sorted(unknown)}. "
                f"Allowed: {sorted(PER_GUARD_FIELDS)}"
            )
        # An OMITTED mode is not a request for any particular one, so we pick
        # the only one the supplied models can serve: cascade when the APP gave
        # us a judge, embedding-only when it did not. This keeps the minimal
        # guard block {enabled, policy} valid without ever downgrading a mode
        # the APP actually asked for — a stated mode is honoured or rejected.
        if "mode" not in screened:
            has_judge = bool(
                screened.get("judge_model") and screened.get("judge_api_key")
            )
            screened["mode"] = "cascade" if has_judge else "embedding-only"
        return screened

    @model_validator(mode="after")
    def _check_coherence(self) -> "GuardParams":
        if self.embed_prefix_style not in ("none", "e5"):
            raise ValueError(
                f"guard.embed_prefix_style must be 'none' or 'e5', got "
                f"{self.embed_prefix_style!r}"
            )
        if self.unavailable_response not in ("error", "refusal"):
            raise ValueError(
                f"guard.unavailable_response must be 'error' or 'refusal', got "
                f"{self.unavailable_response!r}"
            )
        if self.allow_threshold > self.block_threshold:
            raise ValueError(
                f"guard.allow_threshold ({self.allow_threshold}) must not exceed "
                f"guard.block_threshold ({self.block_threshold})"
            )
        if self.judge_allow_threshold > self.judge_block_threshold:
            raise ValueError(
                f"guard.judge_allow_threshold ({self.judge_allow_threshold}) must "
                f"not exceed guard.judge_block_threshold "
                f"({self.judge_block_threshold})"
            )
        if not self.check_roles:
            raise ValueError("guard.check_roles must not be empty")
        bad_roles = sorted(set(self.check_roles) - VALID_ROLES)
        if bad_roles:
            raise ValueError(
                f"guard.check_roles contains unknown roles {bad_roles}. "
                f"Allowed: {sorted(VALID_ROLES)}"
            )
        # A judge that cannot run is never demanded; a judge that WILL run is
        # demanded up front rather than at the first ambiguous message.
        if self.mode != "embedding-only" and not (
            self.judge_model and self.judge_api_key
        ):
            raise ValueError(
                f"guard.mode={self.mode!r} can invoke the judge, so both "
                "guard.judge_model and guard.judge_api_key are required "
                "(use mode='embedding-only' for a judge-free guard)"
            )
        # Tri-state: a policy with no explicit switch is an error, never an off.
        if self.policy and self.enabled is None:
            raise ValueError(
                "guard.policy is set but guard.enabled is not; set it "
                "explicitly to true or false"
            )
        if self.enabled and not (self.policy or "").strip():
            raise ValueError(
                "guard.enabled is true but guard.policy is missing or empty"
            )
        return self


# --------------------------------------------------------------------------- #
# Policy YAML
# --------------------------------------------------------------------------- #


class SafeLoaderNoAlias(yaml.SafeLoader):
    """SafeLoader that refuses YAML anchors and aliases.

    An alias-expanding document can inflate to gigabytes from a few hundred
    bytes ("billion laughs"), which the byte cap alone does not stop. Anchors
    are inert without aliases, but rejecting both keeps the rule one sentence
    long for whoever authors a policy.

    Pure-Python SafeLoader on purpose: ``CSafeLoader`` composes inside C and
    gives no hook to intercept an alias event.
    """

    def compose_node(self, parent, index):  # type: ignore[no-untyped-def]
        if self.check_event(yaml.events.AliasEvent):
            raise GuardConfigError(
                "guard.policy uses a YAML alias (*name); anchors and aliases "
                "are not allowed in a policy document"
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise GuardConfigError(
                f"guard.policy defines a YAML anchor (&{event.anchor}); anchors "
                "and aliases are not allowed in a policy document"
            )
        return super().compose_node(parent, index)


@dataclass(frozen=True)
class FlatExemplar:
    """One labelled example sentence, at a fixed position in the policy."""

    ord: int
    category_id: str
    label: str  # "disallowed" | "allowed"
    text: str


@dataclass(frozen=True)
class Policy:
    """A parsed, validated policy document."""

    categories: Tuple[str, ...]
    category_actions: Mapping[str, str]
    category_refusals: Mapping[str, str]
    category_descriptions: Mapping[str, str]
    exemplars: Tuple[FlatExemplar, ...]
    default_refusal: Optional[str]


def parse_policy(text: str, limits: GuardLimits) -> Policy:
    """Parses the inline policy YAML into a :class:`Policy`.

    The exemplar ORDER is part of the contract: categories in document order,
    disallowed before allowed within a category, list order within each. That
    ordinal indexes rows of the persisted vector matrix, so a reordering would
    silently pair every exemplar with the wrong vector.
    """
    if not isinstance(text, str):
        raise GuardConfigError(
            f"guard.policy must be a YAML string, got {type(text).__name__}"
        )
    raw_bytes = len(text.encode("utf-8"))
    if raw_bytes > limits.max_policy_bytes:
        # Checked BEFORE parsing: a hostile document must not reach the parser.
        raise GuardConfigError(
            f"guard.policy is {raw_bytes} bytes, over the "
            f"{limits.max_policy_bytes}-byte limit"
        )

    try:
        doc = yaml.load(text, Loader=SafeLoaderNoAlias)
    except GuardConfigError:
        raise
    except yaml.YAMLError as e:
        raise GuardConfigError(f"guard.policy is not valid YAML: {e}") from e

    if not isinstance(doc, dict):
        raise GuardConfigError(
            "guard.policy must be a YAML mapping with a 'categories' key"
        )

    unknown_top = set(doc) - {"categories", "default_refusal"} - IGNORED_POLICY_KEYS
    if unknown_top:
        raise GuardConfigError(
            f"guard.policy has unknown top-level keys: {sorted(unknown_top)}"
        )
    ignored = sorted(set(doc) & IGNORED_POLICY_KEYS)
    if ignored:
        logger.warning(
            "guard.policy: ignoring bookkeeping keys %s. Note tenant_id is "
            "NEVER honoured — the guard's identity comes from the caller's own "
            "config, not from the policy document.",
            ignored,
        )

    default_refusal = doc.get("default_refusal")
    if default_refusal is not None and not isinstance(default_refusal, str):
        raise GuardConfigError("guard.policy default_refusal must be a string")

    raw_categories = doc.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise GuardConfigError(
            "guard.policy must define a non-empty 'categories' list"
        )

    names: List[str] = []
    actions: Dict[str, str] = {}
    refusals: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    exemplars: List[FlatExemplar] = []
    n_disallowed = 0
    n_allowed = 0

    for position, category in enumerate(raw_categories):
        if not isinstance(category, dict):
            raise GuardConfigError(
                f"guard.policy category #{position} is not a mapping"
            )
        unknown_cat = set(category) - {
            "category_id", "description", "refusal", "action",
            "disallowed_exemplars", "allowed_exemplars",
        }
        if unknown_cat:
            # `severity` lands here on purpose: GaaS carried it and nothing
            # ever read it, so accepting it would imply an effect it has not.
            raise GuardConfigError(
                f"guard.policy category #{position} has unknown keys: "
                f"{sorted(unknown_cat)}"
            )

        category_id = category.get("category_id")
        if not isinstance(category_id, str) or not category_id.strip():
            raise GuardConfigError(
                f"guard.policy category #{position} needs a non-empty "
                "'category_id'"
            )
        category_id = category_id.strip()
        if category_id in actions:
            raise GuardConfigError(
                f"guard.policy has duplicate category_id {category_id!r}"
            )

        action = category.get("action", "block")
        if action not in ("block", "flag"):
            raise GuardConfigError(
                f"guard.policy category {category_id!r} has action {action!r}; "
                "must be 'block' or 'flag'"
            )

        for key, target in (("refusal", refusals), ("description", descriptions)):
            value = category.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise GuardConfigError(
                        f"guard.policy category {category_id!r}: {key} must be "
                        "a string"
                    )
                target[category_id] = value

        names.append(category_id)
        actions[category_id] = action

        for label, key in (
            ("disallowed", "disallowed_exemplars"),
            ("allowed", "allowed_exemplars"),
        ):
            texts = category.get(key) or []
            if not isinstance(texts, list):
                raise GuardConfigError(
                    f"guard.policy category {category_id!r}: {key} must be a list"
                )
            for item in texts:
                if not isinstance(item, str) or not item.strip():
                    raise GuardConfigError(
                        f"guard.policy category {category_id!r}: every entry in "
                        f"{key} must be a non-empty string"
                    )
                exemplars.append(
                    FlatExemplar(
                        ord=len(exemplars),
                        category_id=category_id,
                        label=label,
                        text=item.strip(),
                    )
                )
                if label == "disallowed":
                    n_disallowed += 1
                else:
                    n_allowed += 1

    if all(action == "flag" for action in actions.values()):
        raise GuardConfigError(
            "no category in this policy blocks — every category has "
            "action: flag, so nothing would ever be refused. Use "
            "guard.enabled: false instead if that is the intent"
        )
    if n_disallowed == 0:
        raise GuardConfigError(
            "guard.policy has no disallowed_exemplars; there is nothing to catch"
        )
    if n_allowed == 0:
        # Without a negative class the weighted vote is disallowed/disallowed
        # == 1.0 for EVERY input, including "hi".
        raise GuardConfigError(
            "guard.policy has no allowed_exemplars; with an empty negative "
            "class every input scores 1.0 and would be blocked"
        )
    if len(exemplars) > limits.max_exemplars:
        raise GuardConfigError(
            f"guard.policy has {len(exemplars)} exemplars, over the "
            f"{limits.max_exemplars} limit"
        )

    return Policy(
        categories=tuple(names),
        category_actions=dict(actions),
        category_refusals=dict(refusals),
        category_descriptions=dict(descriptions),
        exemplars=tuple(exemplars),
        default_refusal=default_refusal,
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(secret: Optional[str]) -> str:
    """Short, non-reversible tag for a key. Never the key itself."""
    return _sha256(secret or "")[:16]


@dataclass(frozen=True)
class ResolvedGuard:
    """Everything the guard needs for one client, with nothing left to look up."""

    params: GuardParams
    policy: Policy
    #: FULL sha256 of the raw policy text. Truncate only for log lines.
    policy_hash: str
    #: Identity of the exemplar MATRIX. Changing it costs a re-embed.
    index_key: str
    #: Identity of the DECISION inputs. Changing it is free, and invalidates
    #: the per-segment memo.
    params_hash: str
    embed_model: str
    embed_api_key: str
    embed_base_url: str

    @property
    def check_roles(self) -> Tuple[str, ...]:
        return tuple(self.params.check_roles)

    @property
    def judge_model(self) -> Optional[str]:
        return self.params.judge_model

    @property
    def judge_api_key(self) -> Optional[str]:
        return self.params.judge_api_key


def derive_guard_config(
    config: Mapping[str, Any],
    *,
    embed_base_url: str,
    limits: Optional[GuardLimits] = None,
) -> Optional[ResolvedGuard]:
    """Turns one APP config response into a :class:`ResolvedGuard`, or ``None``.

    ``None`` means the guard is OFF for this client — the ``guard`` block was
    absent, empty, null, or explicitly ``enabled: false``. Everything else that
    is wrong raises :class:`GuardConfigError`.
    """
    limits = limits or GuardLimits()
    raw = config.get("guard")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise GuardConfigError(
            f"guard must be a JSON object, got {type(raw).__name__}"
        )

    try:
        params = GuardParams.model_validate(raw)
    except GuardConfigError:
        raise
    except Exception as e:  # pydantic ValidationError, or a ValueError from ours
        raise GuardConfigError(str(e)) from e

    if not params.enabled:
        logger.warning(
            "Guard is explicitly DISABLED for this client (guard.enabled=%r). "
            "Traffic is served unguarded.",
            params.enabled,
        )
        return None

    policy_text = params.policy or ""
    policy = parse_policy(policy_text, limits)
    policy_hash = _sha256(policy_text)

    # The guard's embedder defaults to the cache's — one model, one bill, and
    # one warm endpoint — but the APP may point it somewhere else entirely.
    embed_model = params.embed_model or config.get("embed_model")
    embed_api_key = params.embed_api_key or config.get("embed_api_key")
    if not embed_model or not embed_api_key:
        raise GuardConfigError(
            "the guard needs an embedding model and key: set guard.embed_model "
            "and guard.embed_api_key, or supply the top-level embed_model and "
            "embed_api_key it inherits from"
        )

    # WHAT CHANGES THE VECTORS. The key fingerprint is included because the
    # same model alias at the same URL can resolve to different deployments for
    # different keys — scoring a query against a matrix built somewhere else
    # is silent, not loud.
    index_key = _sha256(
        json.dumps(
            {
                "embed_model": str(embed_model),
                "embed_base_url": embed_base_url,
                "embed_key_fp": _fingerprint(str(embed_api_key)),
                "policy_hash": policy_hash,
                "prefix_style": params.embed_prefix_style,
            },
            sort_keys=True,
        )
    )

    # WHAT CHANGES A DECISION given those vectors. Refusal TEXT is deliberately
    # absent: it changes the message, never the verdict, so editing it must not
    # throw away a memo full of valid decisions.
    params_hash = _sha256(
        json.dumps(
            {
                "mode": params.mode,
                "top_k": params.top_k,
                "min_similarity": params.min_similarity,
                "block_threshold": params.block_threshold,
                "allow_threshold": params.allow_threshold,
                "judge_block_threshold": params.judge_block_threshold,
                "judge_allow_threshold": params.judge_allow_threshold,
                "judge_model": params.judge_model,
                "judge_key_fp": _fingerprint(params.judge_api_key),
                "judge_task_description": params.judge_task_description,
                "check_roles": sorted(params.check_roles),
                "max_segments": params.max_segments,
                "max_input_chars": params.max_input_chars,
            },
            sort_keys=True,
        )
    )

    return ResolvedGuard(
        params=params,
        policy=policy,
        policy_hash=policy_hash,
        index_key=index_key,
        params_hash=params_hash,
        embed_model=str(embed_model),
        embed_api_key=str(embed_api_key),
        embed_base_url=embed_base_url,
    )


__all__ = [
    "VALID_ROLES",
    "DEFAULT_CHECK_ROLES",
    "PER_GUARD_FIELDS",
    "IGNORED_POLICY_KEYS",
    "GuardConfigError",
    "GuardLimits",
    "GuardParams",
    "SafeLoaderNoAlias",
    "FlatExemplar",
    "Policy",
    "ResolvedGuard",
    "parse_policy",
    "derive_guard_config",
]
