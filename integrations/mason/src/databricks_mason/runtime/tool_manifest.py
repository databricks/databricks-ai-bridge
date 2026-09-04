"""Read the framework-neutral tool bindings in the project's ``agent.toml``."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Literal, cast

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib

# agent.toml table names for the managed store bindings (`mason memory/sessions bind`) and the env
# vars that override them at deploy time.
MEMORY_STORE_TABLE = "memory_store"
SESSION_STORE_TABLE = "session_store"
MEMORY_STORE_ENV = "AGENT_MEMORY_STORE"
SESSION_STORE_ENV = "AGENT_SESSION_STORE"


@dataclass(frozen=True)
class ScopeRecord:
    kind: str
    value: str
    permission: str


@dataclass(frozen=True)
class ToolRecord:
    id: str
    kind: str
    service: str | None = None
    function: str | None = None
    entrypoint: str | None = None
    downscope: tuple[ScopeRecord, ...] = ()


def project_root() -> pathlib.Path:
    """Resolve the agent project containing ``agent.toml``.

    This module ships in the databricks-mason package, not inside the agent project, so locate the
    project relative to where the agent runs (the current working directory) — not this file's
    location. ``MASON_PROJECT_ROOT`` overrides for cases where the process starts elsewhere.
    """
    configured = os.getenv("MASON_PROJECT_ROOT")
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if (root / "agent.toml").is_file():
            return root
        raise RuntimeError(f"MASON_PROJECT_ROOT has no agent.toml: {root}")

    cwd = pathlib.Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "agent.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate agent.toml; set MASON_PROJECT_ROOT to the project root.")


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"agent.toml must declare {description}.")
    return value


def _scope(value: object) -> ScopeRecord:
    if not isinstance(value, dict):
        raise RuntimeError("agent.toml downscope entries must be tables.")
    value = cast(dict[str, Any], value)
    resource = _required_string(value.get("resource"), "a downscope resource")
    kind, separator, resource_value = resource.partition(":")
    if not separator or kind not in {"table", "volume", "workspace"} or not resource_value:
        raise RuntimeError(f"Invalid agent.toml downscope resource: {resource!r}.")
    permission = _required_string(value.get("permission", "read_only"), "a permission")
    if permission not in {"read_only", "read_write"}:
        raise RuntimeError(f"Invalid agent.toml downscope permission: {permission!r}.")
    return ScopeRecord(kind=kind, value=resource_value, permission=permission)


def _tool(value: object) -> ToolRecord:
    if not isinstance(value, dict):
        raise RuntimeError("agent.toml tools must be tables.")
    value = cast(dict[str, Any], value)
    source = value.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Each agent.toml tool must declare a source table.")
    source = cast(dict[str, Any], source)
    policy = value.get("policy", {})
    if not isinstance(policy, dict):
        raise RuntimeError("agent.toml tool policy must be a table.")
    policy = cast(dict[str, Any], policy)
    raw_downscope = policy.get("downscope", [])
    if not isinstance(raw_downscope, list):
        raise RuntimeError("agent.toml policy.downscope must be an array.")
    record = ToolRecord(
        id=_required_string(value.get("id"), "a tool id"),
        kind=_required_string(source.get("kind"), "a tool source kind"),
        service=source.get("service") if isinstance(source.get("service"), str) else None,
        function=source.get("function") if isinstance(source.get("function"), str) else None,
        entrypoint=source.get("entrypoint") if isinstance(source.get("entrypoint"), str) else None,
        downscope=tuple(_scope(item) for item in raw_downscope),
    )
    if record.kind == "sandbox" and (record.service != "system.ai.sandbox" or not record.downscope):
        raise RuntimeError("Sandbox bindings require system.ai.sandbox and a downscope.")
    if record.kind == "mcp" and not record.service:
        raise RuntimeError("MCP bindings require source.service.")
    if record.kind == "uc_function" and not record.function:
        raise RuntimeError("UC function bindings require source.function.")
    if record.kind == "python" and not record.entrypoint:
        raise RuntimeError("Python bindings require source.entrypoint.")
    if record.kind not in {"sandbox", "mcp", "uc_function", "python"}:
        raise RuntimeError(f"Unsupported agent.toml tool kind: {record.kind!r}.")
    if record.kind != "sandbox" and record.downscope:
        raise RuntimeError("Only sandbox bindings accept policy.downscope.")
    return record


def load_tools(*, expected_framework: str) -> tuple[ToolRecord, ...]:
    """Load a fresh immutable view so direct manifest edits apply on the next request."""
    path = project_root() / "agent.toml"
    try:
        with path.open("rb") as input_file:
            document: dict[str, Any] = tomllib.load(input_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported agent.toml schema in {path}; expected schema_version = 1.")
    agent = document.get("agent")
    if not isinstance(agent, dict) or agent.get("framework") != expected_framework:
        actual = agent.get("framework") if isinstance(agent, dict) else None
        raise RuntimeError(
            f"agent.toml framework {actual!r} does not match runtime {expected_framework!r}."
        )
    raw_tools = document.get("tools", [])
    if not isinstance(raw_tools, list):
        raise RuntimeError("agent.toml tools must be an array of tables.")
    tools = tuple(_tool(item) for item in raw_tools)
    ids = [tool.id for tool in tools]
    if len(ids) != len(set(ids)):
        raise RuntimeError("agent.toml tool ids must be unique.")
    return tools


def store_binding(table: str, prefer: Literal["name", "id"] = "name") -> str | None:
    """The store binding declared in agent.toml's ``[memory_store]`` / ``[session_store]``, or None.

    Read at runtime so `mason memory/sessions bind` (which writes these tables) takes effect without
    any env plumbing. With ``prefer="id"`` the table's ``id`` is returned when present, falling back
    to ``name``; otherwise the ``name`` is returned. Returns None when the project has no agent.toml,
    no such table, or no value — the adapters treat that as "unbound" and fall back to their default.
    Never raises: a malformed or missing manifest just means "no binding here".
    """
    try:
        path = project_root() / "agent.toml"
        with path.open("rb") as input_file:
            document: dict[str, Any] = tomllib.load(input_file)
    except (RuntimeError, OSError, tomllib.TOMLDecodeError):
        return None
    section = document.get(table)
    if not isinstance(section, dict):
        return None
    keys = ("id", "name") if prefer == "id" else ("name",)
    for key in keys:
        value = section.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_memory_store(explicit: str | None = None) -> str | None:
    """The memory store id: ``explicit`` arg → ``AGENT_MEMORY_STORE`` env → agent.toml ``[memory_store]``.

    The one place the store-resolution precedence lives, shared by the framework adapters and the
    chat-app UI so they always agree on which store is in effect. None means "no memory store".

    The entries API is keyed by store id, not display name, so the binding's ``id`` is preferred
    (`mason memory bind` records it); the display ``name`` is only a fallback for a hand-written
    binding without an id.
    """
    if explicit:
        return explicit
    env = os.getenv(MEMORY_STORE_ENV)
    if env:
        return env
    return store_binding(MEMORY_STORE_TABLE, prefer="id")


def resolve_session_store(explicit: str | None = None) -> str | None:
    """The session store name: ``explicit`` arg → ``AGENT_SESSION_STORE`` env → agent.toml binding.

    Shared precedence for the framework adapters and the chat-app UI (see ``resolve_memory_store``).
    None means "no session store" (the in-memory default).
    """
    return explicit or os.getenv(SESSION_STORE_ENV) or store_binding(SESSION_STORE_TABLE)


def downscope_wire(tool: ToolRecord) -> dict[str, list[dict[str, str]]]:
    """Convert protected policy into the system.ai.sandbox MCP ``_meta`` shape."""
    fields = {
        "table": ("tables", "name"),
        "volume": ("volumes", "name"),
        "workspace": ("workspace_paths", "path"),
    }
    result: dict[str, list[dict[str, str]]] = {}
    for scope in tool.downscope:
        group, field = fields[scope.kind]
        result.setdefault(group, []).append({field: scope.value, "permission": scope.permission})
    return result
