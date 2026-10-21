"""Validated Agent Bricks deployment names and their workspace prefix."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from databricks_agentbricks.errors import AgentCliError

_DEPLOYMENT_PREFIX = "agent-bricks-"
_MAX_DEPLOYMENT_NAME_LEN = 30  # Databricks Apps name limit
# The room left for a base name once deploy adds the prefix.
MAX_BASE_NAME_LEN = _MAX_DEPLOYMENT_NAME_LEN - len(_DEPLOYMENT_PREFIX)
# Characters kept before the token when a base name has to be shortened (10 + "-" + 6 = 17).
_SHORT_NAME_HEAD_LEN = 10
_BASE_NAME_RE = re.compile(r"[a-z0-9-]+")


def name_budget_hint(suggestion: str) -> str:
    """The shared explanation of the base-name budget, ending with a suggested fix."""
    return (
        f"Names can be up to {MAX_BASE_NAME_LEN} characters (Agent Bricks adds "
        f"'{_DEPLOYMENT_PREFIX}', and app names cap at {_MAX_DEPLOYMENT_NAME_LEN}). Try: {suggestion}"
    )


def _strip_prefix(name: str) -> str:
    return name[len(_DEPLOYMENT_PREFIX) :] if name.startswith(_DEPLOYMENT_PREFIX) else name


def fit_base_name(name: str, token: Optional[str] = None) -> str:
    """A base name for ``name`` that deploys within the Databricks Apps name limit.

    Strips a leading ``agent-bricks-``, then sanitizes to lowercase letters, digits, and hyphens
    (falling back to ``agent``). A result over ``MAX_BASE_NAME_LEN`` keeps its first 10 characters
    plus ``-<token>``: ``init`` passes its per-scaffold token so the app shares it with the stores;
    otherwise the token is a stable hash of the sanitized name, so repeated suggestions agree.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", _strip_prefix(name.strip()).lower()).strip("-") or "agent"
    if len(slug) <= MAX_BASE_NAME_LEN:
        return slug
    suffix = token or hashlib.sha256(slug.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:_SHORT_NAME_HEAD_LEN].rstrip('-')}-{suffix}"


def base_name_problem(name: str) -> Optional[str]:
    """Why ``name`` can't be deployed as ``agent-bricks-<name>``, or None when it fits."""
    base = _strip_prefix(name)
    if not _BASE_NAME_RE.fullmatch(base):
        return f"'{name}' is not a valid Databricks Apps name (use lowercase letters, digits, and hyphens)"
    prefixed = _prefixed_name(name)
    if len(prefixed) > _MAX_DEPLOYMENT_NAME_LEN:
        return (
            f"'{name}' is too long to deploy ({prefixed} is "
            f"{len(prefixed)} > {_MAX_DEPLOYMENT_NAME_LEN})"
        )
    return None


class DeploymentName(str):
    """A deployment name validated when constructed."""

    __slots__ = ()

    @classmethod
    def _validate(cls, name: str) -> str:
        """Reject an empty or unsafe deployment name before it reaches a URL or workspace path."""
        if (
            not (name or "").strip()
            or name != name.strip()
            or any(token in name for token in ("/", "\\", ".."))
            or any(character.isspace() for character in name)
        ):
            raise AgentCliError(
                f"Invalid deployment name {name!r}.",
                hint="Use a non-empty name of letters, digits, and hyphens "
                "(no slashes, spaces, or '..').",
            )
        if len(name) > _MAX_DEPLOYMENT_NAME_LEN:
            raise AgentCliError(
                f"Deployment name {name!r} is too long ({len(name)} > {_MAX_DEPLOYMENT_NAME_LEN}).",
                hint=name_budget_hint(f"'{_prefixed_name(fit_base_name(name))}'"),
            )
        return name

    def __new__(cls, raw: str) -> "DeploymentName":
        return super().__new__(cls, cls._validate(raw))


def _prefixed_name(name: str) -> str:
    """Add the Agent Bricks deployment prefix unless it is already present."""
    return name if name.startswith(_DEPLOYMENT_PREFIX) else f"{_DEPLOYMENT_PREFIX}{name}"
