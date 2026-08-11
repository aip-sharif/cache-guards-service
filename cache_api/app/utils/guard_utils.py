# guard_utils.py

import logging
import re
import yaml
from typing import Optional
from fastapi import HTTPException

logger = logging.getLogger("guard")

MAX_POLICY_BYTES = 256 * 1024  # 256 KiB
MAX_EXEMPLARS = 5000

_ANCHOR_OR_ALIAS_RE = re.compile(r"(^|\s)[&*][A-Za-z0-9_-]+")


def _has_yaml_anchors_or_aliases(raw_policy: str) -> bool:
    for line in raw_policy.splitlines():
        stripped = line.split("#", 1)[0]  
        if _ANCHOR_OR_ALIAS_RE.search(stripped):
            return True
    return False


def validate_policy(policy: str) -> None:
    if not policy or not policy.strip():
        raise HTTPException(status_code=502, detail="guard.policy is empty")

    if len(policy.encode("utf-8")) > MAX_POLICY_BYTES:
        raise HTTPException(
            status_code=502,
            detail=f"guard.policy exceeds max size of {MAX_POLICY_BYTES} bytes",
        )

    if _has_yaml_anchors_or_aliases(policy):
        raise HTTPException(
            status_code=502,
            detail="guard.policy must not contain YAML anchors or aliases (& / *)",
        )

    try:
        parsed = yaml.safe_load(policy)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=502, detail=f"guard.policy is not valid YAML: {e}")

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="guard.policy must be a YAML mapping")

    categories = parsed.get("categories")
    if not categories or not isinstance(categories, list):
        raise HTTPException(status_code=502, detail="guard.policy must have a non-empty 'categories' list")

    seen_ids = set()
    total_exemplars = 0
    total_disallowed = 0
    total_allowed = 0
    any_non_flag = False

    for category in categories:
        if not isinstance(category, dict):
            raise HTTPException(status_code=502, detail="each category must be a mapping")

        category_id = category.get("category_id")
        if not category_id or not isinstance(category_id, str):
            raise HTTPException(status_code=502, detail="every category needs a non-empty 'category_id'")
        if category_id in seen_ids:
            raise HTTPException(status_code=502, detail=f"duplicate category_id: '{category_id}'")
        seen_ids.add(category_id)

        action = category.get("action", "block")
        if action not in ("block", "flag"):
            raise HTTPException(
                status_code=502,
                detail=f"category '{category_id}' has invalid action '{action}' (must be block|flag)",
            )
        if action != "flag":
            any_non_flag = True

        disallowed = category.get("disallowed_exemplars") or []
        allowed = category.get("allowed_exemplars") or []

        if not disallowed:
            raise HTTPException(
                status_code=502,
                detail=f"category '{category_id}' has no disallowed_exemplars",
            )

        total_disallowed += len(disallowed)
        total_allowed += len(allowed)
        total_exemplars += len(disallowed) + len(allowed)

    if total_allowed == 0:
        raise HTTPException(
            status_code=502,
            detail="guard.policy has no allowed_exemplars anywhere - "
            "a policy with none scores 1.0 for every input and blocks everything",
        )

    if not any_non_flag:
        raise HTTPException(
            status_code=502,
            detail="guard.policy has every category set to action: flag - use \"enabled\": false instead",
        )

    if total_exemplars > MAX_EXEMPLARS:
        raise HTTPException(
            status_code=502,
            detail=f"guard.policy has {total_exemplars} exemplars, exceeding the max of {MAX_EXEMPLARS}",
        )


def resolve_guard(guard, client_id: str):
    if guard is None or (guard.enabled is None and not guard.policy):
        return None

    if guard.enabled is None and guard.policy:
        raise HTTPException(
            status_code=502,
            detail="guard.policy provided without 'enabled' - refusing to guess the intent",
        )

    if guard.enabled is False:
        logger.warning("guard explicitly disabled by client id_user=%s", client_id)
        return None


    if guard.enabled is True:
        validate_policy(guard.policy) 
        return guard

    return None
