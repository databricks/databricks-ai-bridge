"""Validated Agent Bricks deployment names and their workspace prefix."""

from __future__ import annotations

from databricks_agentbricks.errors import AgentCliError

_DEPLOYMENT_PREFIX = "agent-bricks-"
_MAX_DEPLOYMENT_NAME_LEN = 30  # Databricks Apps name limit


def _validate_deployment_name(name: str, *, check_length: bool = True) -> str:
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
    if check_length and len(name) > _MAX_DEPLOYMENT_NAME_LEN:
        raise AgentCliError(
            f"Deployment name {name!r} is too long ({len(name)} > {_MAX_DEPLOYMENT_NAME_LEN}).",
            hint=f"Databricks app names cap at {_MAX_DEPLOYMENT_NAME_LEN} characters, including the "
            f"'{_DEPLOYMENT_PREFIX}' prefix Agent Bricks adds on deploy.",
        )
    return name


class DeploymentName(str):
    """A deployment name validated when constructed."""

    __slots__ = ()

    def __new__(cls, raw: str) -> "DeploymentName":
        return super().__new__(cls, _validate_deployment_name(raw))


def _prefixed_name(name: str) -> str:
    """Add the Agent Bricks deployment prefix unless it is already present."""
    return name if name.startswith(_DEPLOYMENT_PREFIX) else f"{_DEPLOYMENT_PREFIX}{name}"
