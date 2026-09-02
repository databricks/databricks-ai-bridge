"""Framework-neutral, inert specifications for Databricks agent integrations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from databricks_mason.errors import AgentCliError

ScopeKind: TypeAlias = Literal["table", "volume", "workspace"]
Permission: TypeAlias = Literal["read_only", "read_write"]

_SCOPE_KINDS = {"table", "volume", "workspace"}
_PERMISSIONS = {"read_only", "read_write"}
_INTEGRATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_UC_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


def _three_part_name(value: str, description: str) -> str:
    parts = value.split(".") if isinstance(value, str) else []
    if (
        not isinstance(value, str)
        or len(parts) != 3
        or any(not _UC_COMPONENT.fullmatch(part) for part in parts)
        or any(character.isspace() for character in value)
    ):
        raise AgentCliError(
            f"Invalid three-part {description} {value!r}.",
            hint=f"Use a three-part name: catalog.schema.{description.replace(' ', '_')}.",
        )
    return value


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not _INTEGRATION_ID.fullmatch(value):
        raise AgentCliError(f"Invalid integration id {value!r}.")


@dataclass(frozen=True)
class Scope:
    """One table, volume, or workspace resource exposed to Sandbox."""

    kind: ScopeKind
    value: str
    permission: Permission = "read_only"

    def __post_init__(self) -> None:
        if self.kind not in _SCOPE_KINDS:
            raise AgentCliError(f"Unsupported sandbox scope kind {self.kind!r}.")
        if self.permission not in _PERMISSIONS:
            raise AgentCliError(f"Unsupported sandbox permission {self.permission!r}.")
        if not isinstance(self.value, str):
            raise AgentCliError(f"Invalid {self.kind} scope {self.value!r}.")
        if self.kind == "workspace":
            if not self.value.startswith("/Workspace/") or any(
                character in self.value for character in ("\r", "\n", "\t")
            ):
                raise AgentCliError(
                    f"Invalid workspace scope {self.value!r}.",
                    hint="Workspace paths must begin with /Workspace/.",
                )
        else:
            _three_part_name(self.value, f"{self.kind} scope")

    @classmethod
    def table(cls, value: str, permission: Permission = "read_only") -> Scope:
        return cls("table", value, permission)

    @classmethod
    def volume(cls, value: str, permission: Permission = "read_only") -> Scope:
        return cls("volume", value, permission)

    @classmethod
    def workspace(cls, value: str, permission: Permission = "read_only") -> Scope:
        return cls("workspace", value, permission)

    @classmethod
    def parse(cls, value: str, permission: Permission = "read_only") -> Scope:
        original = value.strip()
        if not original:
            raise AgentCliError("Sandbox scopes cannot be empty.")
        if original.startswith("/Workspace/"):
            return cls.workspace(original, permission)
        prefix, separator, remainder = original.partition(":")
        if separator:
            if prefix not in _SCOPE_KINDS:
                raise AgentCliError(f"Unsupported sandbox scope kind {prefix!r}.")
            return cls(prefix, remainder.strip(), permission)  # type: ignore[arg-type]
        return cls.volume(original, permission)

    @property
    def resource(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class Sandbox:
    """A fixed-downscope binding to the managed ``system.ai.sandbox`` MCP service."""

    id: str
    scopes: tuple[Scope, ...]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.scopes:
            raise AgentCliError("Sandbox integrations require at least one scope.")
        if not isinstance(self.scopes, tuple) or any(
            not isinstance(scope, Scope) for scope in self.scopes
        ):
            raise AgentCliError("Sandbox integration scopes must be a tuple of Scope values.")

    @property
    def kind(self) -> str:
        return "sandbox"


@dataclass(frozen=True)
class MCPService:
    """A Databricks-managed MCP Service selected by three-part UC name."""

    id: str
    service: str

    def __post_init__(self) -> None:
        _validate_id(self.id)
        _three_part_name(self.service, "MCP service")

    @property
    def kind(self) -> str:
        return "mcp"


@dataclass(frozen=True)
class UCFunction:
    """A Unity Catalog function exposed through its managed MCP endpoint."""

    id: str
    function: str

    def __post_init__(self) -> None:
        _validate_id(self.id)
        _three_part_name(self.function, "UC function")

    @property
    def kind(self) -> str:
        return "uc_function"


Integration: TypeAlias = Sandbox | MCPService | UCFunction


def downscope_wire(sandbox: Sandbox) -> dict[str, list[dict[str, str]]]:
    """Convert a Sandbox specification to its protected MCP ``_meta`` payload."""

    fields = {
        "table": ("tables", "name"),
        "volume": ("volumes", "name"),
        "workspace": ("workspace_paths", "path"),
    }
    result: dict[str, list[dict[str, str]]] = {}
    for scope in sandbox.scopes:
        group, field = fields[scope.kind]
        result.setdefault(group, []).append({field: scope.value, "permission": scope.permission})
    return result
