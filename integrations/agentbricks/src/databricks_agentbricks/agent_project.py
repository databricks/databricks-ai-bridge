"""Typed, comment-preserving access to an Agent Bricks project's ``agent.toml``."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import ParseError

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_types import (
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)
from databricks_agentkit.runtime import tool_manifest

# agent.toml resource-table names, read by the manifest parsing below (and by `agentbricks dev`/`deploy`).
MEMORY_STORE_TABLE = "memory_store"
SESSION_STORE_TABLE = "session_store"
# The tracing binding (`agentbricks tracing bind` / `unbind`): the `experiment_name` key under [tracing] is
# the bound MLflow experiment. Its presence means tracing is on; an absent binding means off. `agentbricks
# init` bootstraps a default name.
TRACING_TABLE = "tracing"
EXPERIMENT_NAME_KEY = "experiment_name"

_SCHEMA_VERSION = 1
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
    space_id: str | None = None


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
    auth: Literal["user", "app"] | None = None

    def __post_init__(self) -> None:
        source_values = {"kind": self.source.kind}
        for key in ("service", "function", "space_id"):
            value = getattr(self.source, key)
            if value is not None:
                source_values[key] = value
        try:
            tool_manifest.validate_genie_source(
                self.id, source_values, has_downscope=bool(self.policy.downscope)
            )
        except tool_manifest.ToolManifestError as exc:
            raise AgentCliError(str(exc)) from exc
        kind = self.source.kind
        if self.auth is not None and self.auth not in ("user", "app"):
            raise AgentCliError("Tool auth must be 'user' or 'app'.")
        if kind in {"genie_one", "genie_agent"}:
            return
        if not _TOOL_ID.fullmatch(self.id):
            raise AgentCliError(f"Invalid tool id {self.id!r}.")
        if kind == "uc_function" and self.auth == "user":
            raise AgentCliError("UC function auth supports only app/default identity.")
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
            raise AgentCliError(
                "Python tools are code-first and cannot be declared in agent.toml.",
                hint="Remove this entry and wire the tool through framework-native agent code.",
            )
        else:
            raise AgentCliError(f"Unsupported tool source kind {kind!r}.")

    @classmethod
    def sandbox(
        cls,
        tool_id: str,
        *,
        scopes: Sequence[Scope],
        auth: Literal["user", "app"] | None = None,
    ) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="sandbox", service="system.ai.sandbox"),
            policy=ToolPolicy(tuple(scopes)),
            auth=auth,
        )

    @classmethod
    def mcp(
        cls, tool_id: str, *, service: str, auth: Literal["user", "app"] | None = None
    ) -> "ToolSpec":
        return cls(id=tool_id, source=ToolSource(kind="mcp", service=service), auth=auth)

    @classmethod
    def uc_function(
        cls, tool_id: str, *, function: str, auth: Literal["app"] | None = None
    ) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="uc_function", function=function),
            auth=auth,
        )

    @classmethod
    def genie_one(
        cls,
        tool_id: str = "genie_one",
        *,
        auth: Literal["user", "app"] | None = None,
    ) -> "ToolSpec":
        return cls(id=tool_id, source=ToolSource(kind="genie_one"), auth=auth)

    @classmethod
    def genie_agent(
        cls,
        tool_id: str,
        *,
        space_id: str,
        auth: Literal["user", "app"] | None = None,
    ) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="genie_agent", space_id=space_id),
            auth=auth,
        )


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentCliError(f"agent.toml must declare {description}.")
    return value


def default_store_name(project_name: str, suffix: str, token: str | None = None) -> str:
    """A store display name derived from the project directory, e.g. ``my-agent`` -> ``my-agent-memory``.

    Sanitized to the store display-name charset (lower-case alphanumerics and hyphens); a name that
    reduces to nothing (e.g. only punctuation) falls back to ``agent``. An optional per-scaffold
    ``token`` is inserted before the suffix (``my-agent-<token>-memory``) so fresh scaffolds get
    distinct stores while the name still ends with the store kind.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", project_name.lower()).strip("-")
    middle = f"{token}-" if token else ""
    return f"{slug or 'agent'}-{middle}{suffix}"


def _store_name_from_manifest(value: object, table: str) -> str | None:
    """Read the ``name`` from a ``[memory_store]`` / ``[session_store]`` table, or None if absent."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AgentCliError(f"agent.toml [{table}] must be a table.")
    return _required_string(cast(Mapping[str, Any], value).get("name"), f"[{table}] name")


def _store_id_from_manifest(value: object) -> str | None:
    """Read the optional bare store ``id`` from a ``[memory_store]`` table, or None if absent."""
    if not isinstance(value, Mapping):
        return None
    store_id = cast(Mapping[str, Any], value).get("id")
    return store_id if isinstance(store_id, str) and store_id else None


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
    tool_id = _required_string(value.get("id"), "an id")
    try:
        tool_manifest.validate_genie_source(
            tool_id, source, has_downscope="downscope" in policy_value
        )
    except tool_manifest.ToolManifestError as exc:
        raise AgentCliError(str(exc)) from exc
    downscope_value = policy_value.get("downscope", [])
    if not isinstance(downscope_value, list):
        raise AgentCliError("Tool policy downscope must be an array.")
    return ToolSpec(
        id=tool_id,
        source=ToolSource(
            kind=kind,
            service=source.get("service") if isinstance(source.get("service"), str) else None,
            function=source.get("function") if isinstance(source.get("function"), str) else None,
            space_id=source.get("space_id") if isinstance(source.get("space_id"), str) else None,
        ),
        policy=ToolPolicy(tuple(_scope_from_manifest(item) for item in downscope_value)),
        auth=value.get("auth"),
    )


def _inline_table(values: Mapping[str, str]) -> Any:
    table = tomlkit.inline_table()
    for key, value in values.items():
        table[key] = value
    return table


def _tool_table(spec: ToolSpec) -> Any:
    table = tomlkit.table()
    table.add("id", spec.id)
    if spec.auth is not None:
        table.add("auth", spec.auth)
    source_values = {"kind": spec.source.kind}
    for key in ("service", "function", "space_id"):
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
    """Loaded mutable view of a project's canonical agent manifest."""

    def __init__(
        self,
        root: pathlib.Path,
        document: TOMLDocument,
        framework: AgentFramework,
        server: AgentServer,
        tools: list[ToolSpec],
        memory_store: str | None = None,
        session_store: str | None = None,
        memory_store_id: str | None = None,
        deployment_name: str | None = None,
        trace_experiment_name: str | None = None,
        user_auth: tool_manifest.UserAuthConfig | None = None,
    ) -> None:
        self.root = root
        self.path = root / "agent.toml"
        self._document = document
        self.framework = framework
        # Server selection is deployment behavior, so agent.toml—not hidden template metadata—is
        # the source of truth for whether Agent Bricks provisions and wires a Runtime Store.
        self.server = server
        self.tools = tools
        # Managed store bindings declared in agent.toml; None = unbound. memory_store_id is the bare
        # store id the runtime needs for the entries API (the display name can't be used there).
        self.memory_store = memory_store
        self.session_store = session_store
        self.memory_store_id = memory_store_id
        # Bare deployment names get the `agent-bricks-` prefix.
        self.deployment_name = deployment_name
        # Tracing config: the MLflow experiment NAME to trace to (a workspace path). Its presence IS
        # the enable switch: a bound name means tracing is on (deploy get-or-creates it); None means
        # unbound, i.e. off. Storing a name (not an id) keeps the binding valid across workspaces and
        # profiles, since an id is workspace-local. `agentbricks init` bootstraps a default name.
        self.trace_experiment_name = trace_experiment_name
        self.user_auth = user_auth or tool_manifest.UserAuthConfig()

    @classmethod
    def load(cls, root: pathlib.Path | str | None = None) -> "AgentProject":
        try:
            project_root = (
                tool_manifest.project_root()
                if root is None
                else pathlib.Path(root).expanduser().resolve()
            )
        except RuntimeError as exc:
            raise AgentCliError(str(exc)) from exc
        path = project_root / "agent.toml"
        try:
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AgentCliError(
                f"Could not find agent.toml in {project_root}.",
                hint="This command needs an Agent Bricks project. Run `agentbricks init` to create one, "
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
        framework = parse_framework(_required_string(agent.get("framework"), "agent.framework"))
        server = parse_server(_required_string(agent.get("server"), "agent.server"))
        try:
            user_auth = tool_manifest.parse_user_auth(document)
        except tool_manifest.ToolManifestError as exc:
            raise AgentCliError(str(exc)) from exc
        deployment_name = agent.get("deployment_name")
        if deployment_name is not None and not (
            isinstance(deployment_name, str) and deployment_name
        ):
            raise AgentCliError("agent.toml [agent] deployment_name must be a non-empty string.")
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
        memory_store_id = _store_id_from_manifest(document.get(MEMORY_STORE_TABLE))
        session_store = _store_name_from_manifest(
            document.get(SESSION_STORE_TABLE), SESSION_STORE_TABLE
        )
        tracing_table = document.get(TRACING_TABLE)
        trace_experiment_name: str | None = None
        if isinstance(tracing_table, Mapping):
            raw_experiment = tracing_table.get(EXPERIMENT_NAME_KEY)
            trace_experiment_name = (
                str(raw_experiment) if isinstance(raw_experiment, str) and raw_experiment else None
            )
        return cls(
            project_root,
            document,
            framework,
            server,
            tools,
            memory_store,
            session_store,
            memory_store_id,
            str(deployment_name) if deployment_name is not None else None,
            trace_experiment_name,
            user_auth,
        )

    @classmethod
    def create(
        cls,
        root: pathlib.Path | str,
        *,
        framework: str,
        server: str,
        memory_store: str | None = None,
        session_store: str | None = None,
        experiment_name: str | None = None,
    ) -> "AgentProject":
        selected_framework = parse_framework(framework)
        selected_server = parse_server(server)
        project_root = pathlib.Path(root).expanduser().resolve()
        document = tomlkit.document()
        document.add("schema_version", _SCHEMA_VERSION)
        document.add(tomlkit.nl())
        agent = tomlkit.table()
        agent.add("framework", selected_framework.value)
        agent.add("server", selected_server.value)
        document.add("agent", agent)
        project = cls(
            project_root,
            document,
            selected_framework,
            selected_server,
            [],
        )
        if memory_store:
            project.bind_memory_store(memory_store)
        if session_store:
            project.bind_session_store(session_store)
        if experiment_name:
            project.bind_tracing(experiment_name)
        return project

    def set_deployment_name(self, name: str) -> bool:
        """Record the deployment's base name under [agent].deployment_name. True if it changed."""
        name = _required_string(name, "[agent] deployment_name")
        if self.deployment_name == name:
            return False
        agent = self._document.get("agent")
        if not isinstance(agent, Mapping):
            raise AgentCliError("agent.toml must declare an [agent] table.")
        agent["deployment_name"] = name
        self.deployment_name = name
        return True

    def add_tool(self, spec: ToolSpec) -> bool:
        for existing in self.tools:
            if existing.id != spec.id:
                continue
            if existing == spec:
                return False

            def _summary(s: ToolSpec) -> str:
                src = s.source
                return src.service or src.function or src.space_id or src.kind

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

    def bind_memory_store(self, name: str, store_id: str | None = None) -> bool:
        """Declare the memory store binding in agent.toml. Returns True if it changed.

        ``store_id`` is the bare store id (``memory-stores/<id>`` minus the prefix). `agentbricks deploy`
        resolves the id fresh and injects it via ``AGENT_MEMORY_STORE``; the field is only recorded for
        legacy/hand-written bindings that pin the id in the manifest.
        """
        return self._set_store(MEMORY_STORE_TABLE, name, store_id)

    def bind_session_store(self, name: str) -> bool:
        """Declare the session store binding in agent.toml. Returns True if it changed."""
        return self._set_store(SESSION_STORE_TABLE, name)

    def unbind_memory_store(self) -> bool:
        """Remove the memory store binding from agent.toml. Returns True if it was present."""
        return self._clear_store(MEMORY_STORE_TABLE)

    def unbind_session_store(self) -> bool:
        """Remove the session store binding from agent.toml. Returns True if it was present."""
        return self._clear_store(SESSION_STORE_TABLE)

    def bind_tracing(self, experiment_name: str) -> bool:
        """Bind tracing to an MLflow experiment NAME (an experiment path). Returns True if changed.

        The binding's presence is the enable switch: a bound name means tracing is on (deploy
        get-or-creates it). Storing a name (not an id) keeps the binding portable across
        workspaces/profiles.
        """
        experiment_name = _required_string(
            experiment_name, f"[{TRACING_TABLE}] {EXPERIMENT_NAME_KEY}"
        )
        if self.trace_experiment_name == experiment_name:
            return False
        table = self._document.get(TRACING_TABLE)
        if not isinstance(table, Mapping):
            table = tomlkit.table()
            self._document.append(TRACING_TABLE, table)
        table[EXPERIMENT_NAME_KEY] = experiment_name
        self.trace_experiment_name = experiment_name
        return True

    def unbind_tracing(self) -> bool:
        """Unbind tracing: remove the experiment binding so tracing is off. Returns True if changed."""
        if self.trace_experiment_name is None:
            return False
        table = self._document.get(TRACING_TABLE)
        if isinstance(table, Mapping) and EXPERIMENT_NAME_KEY in table:
            del table[EXPERIMENT_NAME_KEY]
        if isinstance(table, Mapping) and not table:
            del self._document[TRACING_TABLE]
        self.trace_experiment_name = None
        return True

    def _set_store(self, table: str, name: str, store_id: str | None = None) -> bool:
        name = _required_string(name, f"[{table}] name")
        if getattr(self, table) == name and getattr(self, f"{table}_id", None) == store_id:
            return False
        existing = self._document.get(table)
        if isinstance(existing, Mapping):
            existing["name"] = name
            if store_id:
                existing["id"] = store_id
            elif "id" in existing:
                del existing["id"]
        else:
            store_table = tomlkit.table()
            store_table.add("name", name)
            if store_id:
                store_table.add("id", store_id)
            self._document.append(table, store_table)
        setattr(self, table, name)
        if hasattr(self, f"{table}_id"):
            setattr(self, f"{table}_id", store_id)
        return True

    def _clear_store(self, table: str) -> bool:
        if getattr(self, table) is None:
            return False
        if table in self._document:
            del self._document[table]
        setattr(self, table, None)
        if hasattr(self, f"{table}_id"):
            setattr(self, f"{table}_id", None)
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
