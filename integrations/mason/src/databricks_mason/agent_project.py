"""Typed, comment-preserving access to a Mason project's ``agent.toml``."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import ParseError

from databricks_mason.errors import AgentCliError
from databricks_mason.runtime.tool_manifest import MEMORY_STORE_TABLE, SESSION_STORE_TABLE

_SCHEMA_VERSION = 1
_DURABILITY_TABLE = "durability"
_SUPPORTED_FRAMEWORKS = {"langgraph", "openai"}
_SUPPORTED_SCOPE_KINDS = {"table", "volume", "workspace"}
_SUPPORTED_PERMISSIONS = {"read_only", "read_write"}
_TOOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _three_part_name(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value.split(".")) != 3
        or any(not part for part in value.split("."))
        or any(character.isspace() for character in value)
    ):
        raise AgentCliError(
            f"Invalid {description} {value!r}.",
            hint=f"Use a three-part name: catalog.schema.{description.replace(' ', '_')}.",
        )
    return value


@dataclass(frozen=True)
class Scope:
    """One sandbox downscope resource."""

    kind: str
    value: str
    permission: str = "read_only"

    def __post_init__(self) -> None:
        if self.kind not in _SUPPORTED_SCOPE_KINDS:
            raise AgentCliError(
                f"Unsupported sandbox scope kind {self.kind!r}.",
                hint=f"Supported scope kinds: {', '.join(sorted(_SUPPORTED_SCOPE_KINDS))}.",
            )
        if self.permission not in _SUPPORTED_PERMISSIONS:
            raise AgentCliError(f"Unsupported sandbox permission {self.permission!r}.")
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
    def table(cls, value: str, permission: str = "read_only") -> "Scope":
        return cls(kind="table", value=value, permission=permission)

    @classmethod
    def volume(cls, value: str, permission: str = "read_only") -> "Scope":
        return cls(kind="volume", value=value, permission=permission)

    @classmethod
    def workspace(cls, value: str, permission: str = "read_only") -> "Scope":
        return cls(kind="workspace", value=value, permission=permission)

    @classmethod
    def parse(cls, value: str, permission: str = "read_only") -> "Scope":
        """Parse ``kind:value`` CLI form, defaulting dotted names to volumes."""
        original = value.strip()
        if not original:
            raise AgentCliError("Sandbox scopes cannot be empty.")
        prefix, separator, remainder = original.partition(":")
        if separator and prefix in _SUPPORTED_SCOPE_KINDS:
            return cls(prefix, remainder.strip(), permission)
        if original.startswith("/Workspace/"):
            return cls.workspace(original, permission)
        return cls.volume(original, permission)

    @property
    def resource(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class ToolSource:
    """Discriminated source for one tool binding."""

    kind: str
    service: str | None = None
    function: str | None = None
    entrypoint: str | None = None


@dataclass(frozen=True)
class ToolPolicy:
    """Protected runtime policy for a tool binding."""

    downscope: tuple[Scope, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """One framework-neutral tool binding from ``agent.toml``."""

    id: str
    source: ToolSource
    policy: ToolPolicy = field(default_factory=ToolPolicy)

    def __post_init__(self) -> None:
        if not _TOOL_ID.fullmatch(self.id):
            raise AgentCliError(f"Invalid tool id {self.id!r}.")
        kind = self.source.kind
        if kind == "sandbox":
            if self.source.service != "system.ai.sandbox":
                raise AgentCliError("Sandbox tools must bind system.ai.sandbox.")
            if not self.policy.downscope:
                raise AgentCliError("Sandbox tools require at least one scope.")
        elif kind == "mcp":
            if self.source.service is None:
                raise AgentCliError("MCP tools require a service.")
            _three_part_name(self.source.service, "MCP service")
            if self.policy.downscope:
                raise AgentCliError("Generic MCP tools do not accept sandbox scopes.")
        elif kind == "uc_function":
            if self.source.function is None:
                raise AgentCliError("UC function tools require a function name.")
            _three_part_name(self.source.function, "UC function")
            if self.policy.downscope:
                raise AgentCliError("UC function tools do not accept sandbox scopes.")
        elif kind == "python":
            entrypoint = self.source.entrypoint or ""
            module, separator, callable_name = entrypoint.partition(":")
            if not separator or not module or not callable_name.isidentifier():
                raise AgentCliError(
                    f"Invalid Python tool entrypoint {entrypoint!r}.",
                    hint="Use module.path:callable_name.",
                )
            if self.policy.downscope:
                raise AgentCliError("Python tools do not accept sandbox scopes.")
        else:
            raise AgentCliError(f"Unsupported tool source kind {kind!r}.")

    @classmethod
    def sandbox(
        cls,
        tool_id: str,
        *,
        scopes: Sequence[Scope],
    ) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="sandbox", service="system.ai.sandbox"),
            policy=ToolPolicy(tuple(scopes)),
        )

    @classmethod
    def mcp(cls, tool_id: str, *, service: str) -> "ToolSpec":
        return cls(id=tool_id, source=ToolSource(kind="mcp", service=service))

    @classmethod
    def uc_function(cls, tool_id: str, *, function: str) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="uc_function", function=function),
        )

    @classmethod
    def python(cls, tool_id: str, *, entrypoint: str) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="python", entrypoint=entrypoint),
        )


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentCliError(f"Tool manifest must declare {description}.")
    return value


def _store_name_from_manifest(value: object, table: str) -> str | None:
    """Read the ``name`` from a ``[memory_store]`` / ``[session_store]`` table, or None if absent."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AgentCliError(f"agent.toml [{table}] must be a table.")
    return _required_string(cast(Mapping[str, Any], value).get("name"), f"[{table}] name")


def _durability_from_manifest(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise AgentCliError("agent.toml [durability] must be a table.")
    if cast(Mapping[str, Any], value).get("enabled") is not True:
        raise AgentCliError("agent.toml [durability] must set enabled = true.")
    return True


def _scope_from_manifest(value: object) -> Scope:
    if not isinstance(value, Mapping):
        raise AgentCliError("Sandbox downscope entries must be TOML tables.")
    value = cast(Mapping[str, Any], value)
    resource = _required_string(value.get("resource"), "a downscope resource")
    permission = value.get("permission", "read_only")
    if not isinstance(permission, str):
        raise AgentCliError("Sandbox downscope permission must be a string.")
    prefix, separator, name = resource.partition(":")
    if not separator:
        raise AgentCliError(
            f"Invalid sandbox downscope resource {resource!r}.",
            hint="Use table:<name>, volume:<name>, or workspace:<path>.",
        )
    return Scope(prefix, name, permission)


def _tool_from_manifest(value: object) -> ToolSpec:
    if not isinstance(value, Mapping):
        raise AgentCliError("Each agent.toml tool must be a TOML table.")
    value = cast(Mapping[str, Any], value)
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise AgentCliError("Each agent.toml tool must declare a source table.")
    source = cast(Mapping[str, Any], source)
    kind = _required_string(source.get("kind"), "a source kind")
    policy_value = value.get("policy", {})
    if not isinstance(policy_value, Mapping):
        raise AgentCliError("Tool policy must be a TOML table.")
    policy_value = cast(Mapping[str, Any], policy_value)
    downscope_value = policy_value.get("downscope", [])
    if not isinstance(downscope_value, list):
        raise AgentCliError("Tool policy downscope must be an array.")
    return ToolSpec(
        id=_required_string(value.get("id"), "an id"),
        source=ToolSource(
            kind=kind,
            service=source.get("service") if isinstance(source.get("service"), str) else None,
            function=source.get("function") if isinstance(source.get("function"), str) else None,
            entrypoint=source.get("entrypoint")
            if isinstance(source.get("entrypoint"), str)
            else None,
        ),
        policy=ToolPolicy(tuple(_scope_from_manifest(item) for item in downscope_value)),
    )


def _inline_table(values: Mapping[str, str]) -> Any:
    table = tomlkit.inline_table()
    for key, value in values.items():
        table[key] = value
    return table


def _tool_table(spec: ToolSpec) -> Any:
    table = tomlkit.table()
    table.add("id", spec.id)
    source_values = {"kind": spec.source.kind}
    for key in ("service", "function", "entrypoint"):
        value = getattr(spec.source, key)
        if value is not None:
            source_values[key] = value
    table.add("source", _inline_table(source_values))
    if spec.policy.downscope:
        downscope = tomlkit.array()
        for scope in spec.policy.downscope:
            downscope.append(
                _inline_table({"resource": scope.resource, "permission": scope.permission})
            )
        policy = tomlkit.inline_table()
        policy["downscope"] = downscope
        table.add("policy", policy)
    return table


class AgentProject:
    """Loaded mutable view of a project's canonical tool manifest."""

    def __init__(
        self,
        root: pathlib.Path,
        document: TOMLDocument,
        framework: str,
        tools: list[ToolSpec],
        memory_store: str | None = None,
        session_store: str | None = None,
        durability_enabled: bool = False,
    ) -> None:
        self.root = root
        self.path = root / "agent.toml"
        self._document = document
        self.framework = framework
        self.tools = tools
        # Managed store bindings declared in agent.toml; None = unbound.
        self.memory_store = memory_store
        self.session_store = session_store
        self.durability_enabled = durability_enabled

    @classmethod
    def load(cls, root: pathlib.Path | str) -> "AgentProject":
        project_root = pathlib.Path(root).expanduser().resolve()
        path = project_root / "agent.toml"
        try:
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AgentCliError(
                f"Could not find agent.toml in {project_root}.",
                hint="This command needs a Mason project. Run `mason init` to create one, "
                "or point at an existing project with --source <dir>.",
            ) from exc
        except (OSError, ParseError) as exc:
            raise AgentCliError(f"Could not read agent manifest at {path}: {exc}.") from exc
        if document.get("schema_version") != _SCHEMA_VERSION:
            raise AgentCliError(
                f"Unsupported agent manifest schema in {path}.",
                hint=f"Expected schema_version = {_SCHEMA_VERSION}.",
            )
        agent = document.get("agent")
        if not isinstance(agent, Mapping):
            raise AgentCliError("agent.toml must declare an [agent] table.")
        framework = _required_string(agent.get("framework"), "agent.framework")
        if framework not in _SUPPORTED_FRAMEWORKS:
            raise AgentCliError(f"Unsupported Mason framework {framework!r}.")
        raw_tools = document.get("tools", [])
        if not isinstance(raw_tools, list):
            raise AgentCliError("agent.toml tools must be an array of tables.")
        tools = [_tool_from_manifest(item) for item in raw_tools]
        ids = [tool.id for tool in tools]
        if len(ids) != len(set(ids)):
            raise AgentCliError("agent.toml tool ids must be unique.")
        memory_store = _store_name_from_manifest(
            document.get(MEMORY_STORE_TABLE), MEMORY_STORE_TABLE
        )
        session_store = _store_name_from_manifest(
            document.get(SESSION_STORE_TABLE), SESSION_STORE_TABLE
        )
        durability_enabled = _durability_from_manifest(document.get(_DURABILITY_TABLE))
        return cls(
            project_root,
            document,
            framework,
            tools,
            memory_store,
            session_store,
            durability_enabled,
        )

    @classmethod
    def create(cls, root: pathlib.Path | str, *, framework: str) -> "AgentProject":
        if framework not in _SUPPORTED_FRAMEWORKS:
            raise AgentCliError(f"Unsupported Mason framework {framework!r}.")
        project_root = pathlib.Path(root).expanduser().resolve()
        document = tomlkit.document()
        document.add("schema_version", _SCHEMA_VERSION)
        document.add(tomlkit.nl())
        agent = tomlkit.table()
        agent.add("framework", framework)
        document.add("agent", agent)
        return cls(project_root, document, framework, [])

    def add_tool(self, spec: ToolSpec) -> bool:
        for existing in self.tools:
            if existing.id != spec.id:
                continue
            if existing == spec:
                return False

            def _summary(s: ToolSpec) -> str:
                src = s.source
                return src.service or src.function or src.entrypoint or src.kind

            raise AgentCliError(
                f"Tool id {spec.id!r} already exists with a different configuration "
                f"(existing: {_summary(existing)}; requested: {_summary(spec)}).",
                hint="Use --name to add it under a different id, or remove the existing "
                "tool from agent.toml first.",
            )
        raw_tools = self._document.get("tools")
        if raw_tools is None:
            raw_tools = tomlkit.aot()
            self._document.append("tools", raw_tools)
        elif not hasattr(raw_tools, "append"):
            raise AgentCliError("agent.toml tools must be an array of tables.")
        raw_tools.append(_tool_table(spec))
        self.tools.append(spec)
        return True

    def remove_tool(self, tool_id: str) -> bool:
        index = next((index for index, tool in enumerate(self.tools) if tool.id == tool_id), None)
        if index is None:
            return False
        raw_tools = self._document.get("tools")
        if not isinstance(raw_tools, list):
            raise AgentCliError("agent.toml tools must be an array of tables.")
        del raw_tools[index]
        del self.tools[index]
        return True

    def bind_memory_store(self, name: str) -> bool:
        """Declare the memory store binding in agent.toml. Returns True if it changed."""
        return self._set_store(MEMORY_STORE_TABLE, name)

    def bind_session_store(self, name: str) -> bool:
        """Declare the session store binding in agent.toml. Returns True if it changed."""
        return self._set_store(SESSION_STORE_TABLE, name)

    def unbind_memory_store(self) -> bool:
        """Remove the memory store binding from agent.toml. Returns True if it was present."""
        return self._clear_store(MEMORY_STORE_TABLE)

    def unbind_session_store(self) -> bool:
        """Remove the session store binding from agent.toml. Returns True if it was present."""
        return self._clear_store(SESSION_STORE_TABLE)

    def bind_durability(self) -> bool:
        """Declare durable invocation handling in agent.toml."""
        if self.durability_enabled:
            return False
        durability = tomlkit.table()
        durability.add("enabled", True)
        self._document.append(_DURABILITY_TABLE, durability)
        self.durability_enabled = True
        return True

    def unbind_durability(self) -> bool:
        """Remove durable invocation handling from agent.toml."""
        if not self.durability_enabled:
            return False
        del self._document[_DURABILITY_TABLE]
        self.durability_enabled = False
        return True

    def _set_store(self, table: str, name: str) -> bool:
        name = _required_string(name, f"[{table}] name")
        if getattr(self, table) == name:
            return False
        existing = self._document.get(table)
        if isinstance(existing, Mapping):
            existing["name"] = name
        else:
            store_table = tomlkit.table()
            store_table.add("name", name)
            self._document.append(table, store_table)
        setattr(self, table, name)
        return True

    def _clear_store(self, table: str) -> bool:
        if getattr(self, table) is None:
            return False
        if table in self._document:
            del self._document[table]
        setattr(self, table, None)
        return True

    def write(self) -> pathlib.Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = pathlib.Path(output.name)
                output.write(tomlkit.dumps(self._document))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise AgentCliError(f"Could not write agent manifest at {self.path}: {exc}.") from exc
        return self.path
