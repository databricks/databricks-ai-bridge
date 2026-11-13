"""Typed, comment-preserving access to an Agent Bricks project's ``agent.toml``."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, cast

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import ParseError
from tomlkit.items import Array, Table

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.types import (
    CUSTOM_FRAMEWORK,
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)
from databricks_agentkit.runtime import tool_manifest
from databricks_agentkit.runtime.cors import allowed_origins_from_manifest, validate_allowed_origins

# agent.toml resource-table names, read by the manifest parsing below (and by `agentbricks dev`/`deploy`).
MEMORY_STORE_TABLE = "memory_store"
SESSION_STORE_TABLE = "session_store"
# The tracing binding (`agentbricks tracing bind` / `unbind`): the `experiment_name` key under [tracing] is
# the bound MLflow experiment. Its presence means tracing is on; an absent binding means off. `agentbricks
# init` bootstraps a default name.
TRACING_TABLE = "tracing"
EXPERIMENT_NAME_KEY = "experiment_name"
# The `agentbricks experimental` bindings live under [experimental] until their interface settles, so
# they can graduate to top-level tables later without colliding with anything already there.
EXPERIMENTAL_TABLE = "experimental"
# The model-service bindings (`agentbricks experimental models bind`), one
# `[experimental.model_services.<role>]` table per
# LLM call site: `name` is the user-owned UC model service that call goes through
# (catalog.schema.name); `default` is the system.ai.* model deploy routes it to when it first
# creates the service. `agentbricks experimental models upgrade` / `set` repoint them afterwards. A single-model
# agent uses one role, DEFAULT_MODEL_ROLE.
MODEL_SERVICES_TABLE = "model_services"
MODEL_SERVICE_DEFAULT_KEY = "default"
DEFAULT_MODEL_ROLE = tool_manifest.DEFAULT_MODEL_ROLE
_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# The prompt bindings (`agentbricks experimental prompts bind`): MLflow Prompt Registry prompts
# the agent loads, as `<key> = "catalog.schema.name"` under [experimental.prompts]. `agentbricks deploy` grants
# the app access to their schemas, and `models upgrade` optimizes them by default.
PROMPTS_TABLE = "prompts"
MODEL_SERVICES_PATH = f"{EXPERIMENTAL_TABLE}.{MODEL_SERVICES_TABLE}"
PROMPTS_PATH = f"{EXPERIMENTAL_TABLE}.{PROMPTS_TABLE}"

_SCHEMA_VERSION = 1
_SUPPORTED_SCOPE_KINDS = {"volume"}
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


def model_role(value: str) -> str:
    """Validate a model-service role name (lowercase identifier, e.g. ``router``)."""
    if not isinstance(value, str) or not _ROLE_PATTERN.match(value):
        raise AgentCliError(
            f"Invalid model role {value!r}.",
            hint="Use a lowercase identifier such as `router` or `writer`.",
        )
    return value


@dataclass(frozen=True)
class ModelServiceBinding:
    """One `[experimental.model_services.<role>]` table: a call site's service and its first-deploy
    model."""

    name: str
    default: str | None = None


@dataclass(frozen=True)
class Scope:
    """One sandbox downscope resource."""

    kind: str
    value: str
    permission: str = "read_only"

    def __post_init__(self) -> None:
        if not tool_manifest.SANDBOX_DOWNSCOPE_ENABLED:
            raise AgentCliError(
                "Sandbox scopes are not supported.",
                hint="system.ai.sandbox does not accept resource scopes; sandbox code runs with "
                "the selected identity's own Unity Catalog access.",
            )
        if self.kind == "table" and not tool_manifest.SANDBOX_TABLE_SCOPES_ENABLED:
            raise AgentCliError(
                "Table sandbox scopes are not supported yet.",
                hint="Databricks Connect does not support table downscoping. "
                "Use volume:<name> instead.",
            )
        if self.kind == "workspace":
            raise AgentCliError(
                "Workspace sandbox scopes are not supported yet.",
                hint="Use volume:<name> instead.",
            )
        supported_kinds = _SUPPORTED_SCOPE_KINDS
        if tool_manifest.SANDBOX_TABLE_SCOPES_ENABLED:
            supported_kinds = supported_kinds | {"table"}
        if self.kind not in supported_kinds:
            raise AgentCliError(
                f"Unsupported sandbox scope kind {self.kind!r}.",
                hint=f"Supported scope kinds: {', '.join(sorted(supported_kinds))}.",
            )
        if self.permission not in _SUPPORTED_PERMISSIONS:
            raise AgentCliError(f"Unsupported sandbox permission {self.permission!r}.")
        _three_part_name(self.value, f"{self.kind} scope")

    @classmethod
    def table(cls, value: str, permission: str = "read_only") -> "Scope":
        return cls(kind="table", value=value, permission=permission)

    @classmethod
    def volume(cls, value: str, permission: str = "read_only") -> "Scope":
        return cls(kind="volume", value=value, permission=permission)

    @classmethod
    def parse(cls, value: str, permission: str = "read_only") -> "Scope":
        """Parse ``kind:value`` CLI form, defaulting dotted names to volumes."""
        original = value.strip()
        if not original:
            raise AgentCliError("Sandbox scopes cannot be empty.")
        if original.startswith("/Workspace/"):
            return cls("workspace", original, permission)
        prefix, separator, remainder = original.partition(":")
        if separator:
            return cls(prefix, remainder.strip(), permission)
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
    databricks_access_token_included: bool = False


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
        if kind != "sandbox" and self.policy.databricks_access_token_included:
            raise AgentCliError(
                "Only sandbox bindings accept policy.databricks_access_token_included."
            )
        if kind in {"genie_one", "genie_agent"}:
            return
        if not _TOOL_ID.fullmatch(self.id):
            raise AgentCliError(f"Invalid tool id {self.id!r}.")
        if kind == "sandbox":
            if self.source.service != "system.ai.sandbox":
                raise AgentCliError("Sandbox tools must bind system.ai.sandbox.")
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
        scopes: Sequence[Scope] = (),
        auth: Literal["user", "app"] | None = None,
        databricks_access_token_included: bool = True,
    ) -> "ToolSpec":
        return cls(
            id=tool_id,
            source=ToolSource(kind="sandbox", service="system.ai.sandbox"),
            policy=ToolPolicy(tuple(scopes), databricks_access_token_included),
            auth=auth,
        )

    @classmethod
    def mcp(
        cls, tool_id: str, *, service: str, auth: Literal["user", "app"] | None = None
    ) -> "ToolSpec":
        return cls(id=tool_id, source=ToolSource(kind="mcp", service=service), auth=auth)

    @classmethod
    def uc_function(
        cls, tool_id: str, *, function: str, auth: Literal["user", "app"] | None = None
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


def default_experiment_name(project: str | None, token: str | None = None) -> str:
    """Return a project-specific experiment path under ``/Shared/agentbricks_traces``."""
    if not project:
        raise AgentCliError("Cannot derive the default tracing experiment without a project name.")
    slug = re.sub(r"[^a-z0-9-]+", "-", project.lower()).strip("-") or "agent"
    middle = f"-{token}" if token else ""
    return f"/Shared/agentbricks_traces/{slug}{middle}"


def _store_name_from_manifest(value: object, table: str) -> str | None:
    """Read the ``name`` from a ``[memory_store]`` / ``[session_store]`` table, or None if absent."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AgentCliError(f"agent.toml [{table}] must be a table.")
    return _required_string(cast(Mapping[str, Any], value).get("name"), f"[{table}] name")


def _prompts_from_manifest(value: object, problems: list[ManifestProblem]) -> dict[str, str]:
    """Read ``[experimental.prompts]`` (``key = "catalog.schema.name"``), or {} if there is none.

    Each invalid key is recorded in ``problems`` under its own path; the rest are still returned.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        problems.append(
            ManifestProblem(
                PROMPTS_PATH, f"agent.toml [{PROMPTS_PATH}] must be a table of key = prompt name."
            )
        )
        return {}
    prompts: dict[str, str] = {}
    for key, name in cast(Mapping[str, Any], value).items():
        path = f"{PROMPTS_PATH}.{key}"
        try:
            model_role(key)
        except AgentCliError as exc:
            problems.append(ManifestProblem(path, _problem_text(exc)))
            continue
        prompt = _checked(problems, path, partial(_prompt_name, name, f"[{PROMPTS_PATH}] {key}"))
        if prompt is not None:
            # Plain str, not tomlkit's String (see _model_services_from_manifest).
            prompts[str(key)] = prompt
    return prompts


def _prompt_name(value: object, where: str) -> str:
    return str(_three_part_name(_required_string(value, where), "prompt"))


def _model_services_from_manifest(
    value: object, problems: list[ManifestProblem]
) -> dict[str, ModelServiceBinding]:
    """Read the ``[experimental.model_services.<role>]`` tables, or {} if there are none.

    Each invalid role or field is recorded in ``problems`` under its own path (for example
    ``experimental.model_services.router.name``); the valid bindings are still returned.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        problems.append(
            ManifestProblem(
                MODEL_SERVICES_PATH, f"agent.toml [{MODEL_SERVICES_PATH}] must be a table of roles."
            )
        )
        return {}
    bindings: dict[str, ModelServiceBinding] = {}
    for role, table in cast(Mapping[str, Any], value).items():
        path = f"{MODEL_SERVICES_PATH}.{role}"
        try:
            model_role(role)
        except AgentCliError as exc:
            problems.append(ManifestProblem(path, _problem_text(exc)))
            continue
        where = f"[{path}]"
        if not isinstance(table, Mapping):
            problems.append(ManifestProblem(path, f"agent.toml {where} must be a table."))
            continue
        table = cast(Mapping[str, Any], table)
        name = _checked(problems, f"{path}.name", partial(_model_service_name, table, where))
        default = _checked(
            problems,
            f"{path}.{MODEL_SERVICE_DEFAULT_KEY}",
            partial(_model_service_default, table, where),
        )
        if name is not None:
            # Plain str, not tomlkit's String: deploy writes these into app.yaml, and yaml.safe_dump
            # can't represent tomlkit types.
            bindings[role] = ModelServiceBinding(str(name), default)
    return bindings


def _model_service_name(table: Mapping[str, Any], where: str) -> str:
    return _three_part_name(_required_string(table.get("name"), f"{where} name"), "model service")


def _model_service_default(table: Mapping[str, Any], where: str) -> str | None:
    default = table.get(MODEL_SERVICE_DEFAULT_KEY)
    if default is not None and not (isinstance(default, str) and default):
        raise AgentCliError(
            f"agent.toml {where} {MODEL_SERVICE_DEFAULT_KEY} must be a non-empty string."
        )
    return str(default) if default else None


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
            hint="Use volume:<name>.",
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
    if downscope_value and not tool_manifest.SANDBOX_DOWNSCOPE_ENABLED:
        raise AgentCliError(tool_manifest.downscope_unsupported_message(tool_id))
    databricks_access_token_included = policy_value.get("databricks_access_token_included", False)
    if not isinstance(databricks_access_token_included, bool):
        raise AgentCliError("Tool policy databricks_access_token_included must be a boolean.")
    if kind != "sandbox" and "databricks_access_token_included" in policy_value:
        raise AgentCliError("Only sandbox bindings accept policy.databricks_access_token_included.")
    return ToolSpec(
        id=tool_id,
        source=ToolSource(
            kind=kind,
            service=source.get("service") if isinstance(source.get("service"), str) else None,
            function=source.get("function") if isinstance(source.get("function"), str) else None,
            space_id=source.get("space_id") if isinstance(source.get("space_id"), str) else None,
        ),
        policy=ToolPolicy(
            tuple(_scope_from_manifest(item) for item in downscope_value),
            databricks_access_token_included,
        ),
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
    if spec.source.kind == "sandbox":
        policy = tomlkit.inline_table()
        if spec.policy.downscope:
            downscope = tomlkit.array()
            for scope in spec.policy.downscope:
                downscope.append(
                    _inline_table({"resource": scope.resource, "permission": scope.permission})
                )
            policy["downscope"] = downscope
        policy["databricks_access_token_included"] = spec.policy.databricks_access_token_included
        table.add("policy", policy)
    return table


@dataclass(frozen=True)
class ManifestProblem:
    """One missing or invalid agent.toml field and what it accepts."""

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass(frozen=True)
class _ManifestFields:
    """The values `AgentProject.load` reads from a manifest that has no problems."""

    framework: AgentFramework | None = None
    server: AgentServer = AgentServer.AGENTBRICKS
    tools: list[ToolSpec] = field(default_factory=list)
    memory_store: str | None = None
    session_store: str | None = None
    memory_store_id: str | None = None
    deployment_name: str | None = None
    trace_experiment_name: str | None = None
    user_auth: tool_manifest.UserAuthConfig | None = None
    model_services: dict[str, ModelServiceBinding] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    allowed_origins: tuple[str, ...] = ()


def _choices(values: Sequence[str]) -> str:
    """``a or b`` / ``a, b, or c``."""
    if len(values) < 3:
        return " or ".join(values)
    return f"{', '.join(values[:-1])}, or {values[-1]}"


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _invalid(value: object, allowed: str) -> str:
    return (
        f"missing; use {allowed}"
        if value is None
        else f"{_rendered(value)} is not valid; use {allowed}"
    )


def _problem_text(exc: AgentCliError) -> str:
    return f"{exc.message} {exc.hint}" if exc.hint else exc.message


def _checked(
    problems: list[ManifestProblem], field_name: str, read: Callable[[], Any], fallback: Any = None
) -> Any:
    """Run ``read``; an invalid value is recorded in ``problems`` under ``field_name``."""
    try:
        return read()
    except AgentCliError as exc:
        problems.append(ManifestProblem(field_name, _problem_text(exc)))
    except (tool_manifest.ToolManifestError, ValueError) as exc:
        problems.append(ManifestProblem(field_name, str(exc)))
    return fallback


_FRAMEWORKS = tuple(framework.value for framework in AgentFramework)
_SERVERS = tuple(server.value for server in AgentServer)


def _read_manifest(document: Mapping[str, Any]) -> tuple[_ManifestFields, list[ManifestProblem]]:
    """Check every agent.toml field, collecting all problems rather than stopping at the first."""
    problems: list[ManifestProblem] = []

    def problem(field_name: str, message: str) -> None:
        problems.append(ManifestProblem(field_name, message))

    def checked(field_name: str, read: Callable[[], Any], fallback: Any = None) -> Any:
        return _checked(problems, field_name, read, fallback)

    schema_version = document.get("schema_version")
    if schema_version is None:
        problem("schema_version", f"missing; use {_SCHEMA_VERSION}")
    elif schema_version != _SCHEMA_VERSION:
        problem(
            "schema_version", f"{_rendered(schema_version)} is not supported; use {_SCHEMA_VERSION}"
        )

    framework: AgentFramework | None = None
    server: AgentServer | None = None
    deployment_name: str | None = None
    agent = document.get("agent")
    if not isinstance(agent, Mapping):
        problem("agent", "missing; add an [agent] table with framework and server")
    else:
        raw_server = agent.get("server")
        if isinstance(raw_server, str) and raw_server in _SERVERS:
            server = AgentServer(raw_server)
        raw_framework = agent.get("framework")
        if isinstance(raw_framework, str) and raw_framework in _FRAMEWORKS:
            framework = AgentFramework(raw_framework)
        elif raw_framework is None or raw_framework == CUSTOM_FRAMEWORK:
            # A hand-written agent on the Agent Bricks server needs no framework adapter.
            if server == AgentServer.CUSTOM:
                problem(
                    "agent.framework",
                    _invalid(None, _choices(_FRAMEWORKS))
                    if raw_framework is None
                    else f'"{CUSTOM_FRAMEWORK}" needs agent.server = "agentbricks"; '
                    f"use {_choices(_FRAMEWORKS)}",
                )
        else:
            # `custom` is only an allowed framework value where it is accepted, on the
            # Agent Bricks server; a custom-server app always needs a framework adapter.
            allowed = (
                _FRAMEWORKS if server == AgentServer.CUSTOM else (*_FRAMEWORKS, CUSTOM_FRAMEWORK)
            )
            problem("agent.framework", _invalid(raw_framework, _choices(allowed)))
        if server is None:
            problem("agent.server", _invalid(raw_server, _choices(_SERVERS)))
        raw_name = agent.get("deployment_name")
        if raw_name is not None and not (isinstance(raw_name, str) and raw_name):
            problem("agent.deployment_name", "must be a non-empty string")
        elif raw_name is not None:
            deployment_name = str(raw_name)

    user_auth, auth_problems = tool_manifest.user_auth_problems(document)
    for field_name, message in auth_problems:
        problem(field_name, message)
    allowed_origins: tuple[str, ...] = checked(
        "cors", lambda: allowed_origins_from_manifest(document), ()
    )
    if allowed_origins and server == AgentServer.CUSTOM:
        problem("cors", 'needs agent.server = "agentbricks"')

    tools: list[ToolSpec] = []
    raw_tools = document.get("tools", [])
    if not isinstance(raw_tools, list):
        problem("tools", "must be an array of tables ([[tools]])")
    else:
        for index, item in enumerate(raw_tools):
            tool = checked(f"tools[{index}]", lambda item=item: _tool_from_manifest(item))
            if tool is not None:
                tools.append(tool)
        ids = [tool.id for tool in tools]
        if len(ids) != len(set(ids)):
            problem("tools", "tool ids must be unique")

    memory_store = checked(
        MEMORY_STORE_TABLE,
        lambda: _store_name_from_manifest(document.get(MEMORY_STORE_TABLE), MEMORY_STORE_TABLE),
    )
    session_store = checked(
        SESSION_STORE_TABLE,
        lambda: _store_name_from_manifest(document.get(SESSION_STORE_TABLE), SESSION_STORE_TABLE),
    )
    tracing_table = document.get(TRACING_TABLE)
    trace_experiment_name: str | None = None
    if isinstance(tracing_table, Mapping):
        raw_experiment = tracing_table.get(EXPERIMENT_NAME_KEY)
        trace_experiment_name = (
            str(raw_experiment) if isinstance(raw_experiment, str) and raw_experiment else None
        )

    for moved in (MODEL_SERVICES_TABLE, PROMPTS_TABLE):
        if moved in document:
            problem(
                moved,
                f"agent.toml declares [{moved}] at the top level; move it under "
                f"[{EXPERIMENTAL_TABLE}]",
            )
    experimental = document.get(EXPERIMENTAL_TABLE)
    if experimental is not None and not isinstance(experimental, Mapping):
        problem(EXPERIMENTAL_TABLE, "must be a table")
        experimental = None
    experimental = cast(Mapping[str, Any], experimental or {})
    model_services = _model_services_from_manifest(experimental.get(MODEL_SERVICES_TABLE), problems)
    prompts = _prompts_from_manifest(experimental.get(PROMPTS_TABLE), problems)

    fields = _ManifestFields(
        framework=framework,
        server=server or AgentServer.AGENTBRICKS,
        tools=tools,
        memory_store=memory_store,
        session_store=session_store,
        memory_store_id=_store_id_from_manifest(document.get(MEMORY_STORE_TABLE)),
        deployment_name=deployment_name,
        trace_experiment_name=trace_experiment_name,
        user_auth=user_auth,
        model_services=model_services,
        prompts=prompts,
        allowed_origins=allowed_origins,
    )
    return fields, problems


def manifest_problems(document: Mapping[str, Any]) -> list[ManifestProblem]:
    """Every missing or invalid field in a parsed agent.toml, each with the values it accepts."""
    return _read_manifest(document)[1]


def manifest_error(problems: Sequence[ManifestProblem]) -> AgentCliError:
    """One error listing every problem, so a single run shows all of them."""
    if len(problems) == 1:
        return AgentCliError(f"agent.toml: {problems[0]}")
    lines = "".join(f"\n  • {problem}" for problem in problems)
    return AgentCliError(f"agent.toml has {len(problems)} problems:{lines}")


def _tool_inline_table(value: Any) -> Any:
    # For hand-written `tools = [{...}]` arrays, where a `[[tools]]` table can't go.
    if isinstance(value, Table):
        table = tomlkit.inline_table()
        for key, item in value.value.items():
            table[key] = _tool_inline_table(item)
        return table
    return value


def _tool_fields(spec: ToolSpec) -> dict[str, str]:
    scopes = " ".join(f"{scope.resource}:{scope.permission}" for scope in spec.policy.downscope)
    values = {
        "kind": spec.source.kind,
        "service": spec.source.service,
        "function": spec.source.function,
        "space_id": spec.source.space_id,
        "auth": spec.auth,
        "scope": scopes or None,
        "databricks_access_token_included": str(
            spec.policy.databricks_access_token_included
        ).lower(),
    }
    return {key: "none" if value is None else value for key, value in values.items()}


def _tool_conflict(existing: ToolSpec, requested: ToolSpec) -> AgentCliError:
    old, new = _tool_fields(existing), _tool_fields(requested)
    fields = [key for key in old if old[key] != new[key]]
    hint = f"Run `agentbricks tools remove {requested.id}` first, or use --name to add another binding."
    if not fields:
        return AgentCliError(
            f"Tool {requested.id!r} already exists with a different configuration.", hint=hint
        )
    return AgentCliError(
        f"Tool {requested.id!r} already exists with a different {', '.join(fields)} "
        f"(existing: {', '.join(old[key] for key in fields)}; "
        f"requested: {', '.join(new[key] for key in fields)}).",
        hint=hint,
    )


class AgentProject:
    """Loaded mutable view of a project's canonical agent manifest."""

    def __init__(
        self,
        root: pathlib.Path,
        document: TOMLDocument,
        framework: AgentFramework | None,
        server: AgentServer,
        tools: list[ToolSpec],
        memory_store: str | None = None,
        session_store: str | None = None,
        memory_store_id: str | None = None,
        deployment_name: str | None = None,
        trace_experiment_name: str | None = None,
        user_auth: tool_manifest.UserAuthConfig | None = None,
        model_services: dict[str, ModelServiceBinding] | None = None,
        prompts: dict[str, str] | None = None,
        allowed_origins: tuple[str, ...] = (),
    ) -> None:
        self.root = root
        self.path = root / "agent.toml"
        self._document = document
        # None for a hand-written agent (`framework = "custom"` or omitted) on the Agent Bricks server.
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
        # Model-service bindings by role: the UC model service each LLM call site goes through, and
        # the model deploy routes it to on first create. Empty = unbound (the agent calls its own
        # default models directly).
        self.model_services: dict[str, ModelServiceBinding] = dict(model_services or {})
        # Prompt bindings by key: the three-part Prompt Registry name of each prompt the agent loads.
        self.prompts: dict[str, str] = dict(prompts or {})
        self.allowed_origins = allowed_origins

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
        fields, problems = _read_manifest(document)
        if problems:
            raise manifest_error(problems)
        return cls(
            project_root,
            document,
            fields.framework,
            fields.server,
            fields.tools,
            fields.memory_store,
            fields.session_store,
            fields.memory_store_id,
            fields.deployment_name,
            fields.trace_experiment_name,
            fields.user_auth,
            fields.model_services,
            fields.prompts,
            fields.allowed_origins,
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

    def set_allowed_origins(self, origins: tuple[str, ...]) -> bool:
        try:
            validated = validate_allowed_origins(origins)
        except ValueError as exc:
            raise AgentCliError(str(exc)) from exc
        if validated and self.server != AgentServer.AGENTBRICKS:
            raise AgentCliError("CORS configuration requires [agent].server = 'agentbricks'.")
        if self.allowed_origins == validated:
            return False
        cors = self._document.setdefault("cors", tomlkit.table())
        cors["allowed_origins"] = list(validated)
        self.allowed_origins = validated
        return True

    def add_tool(self, spec: ToolSpec) -> bool:
        for existing in self.tools:
            if existing.id != spec.id:
                continue
            if existing == spec:
                return False
            raise _tool_conflict(existing, spec)
        raw_tools = self._document.get("tools")
        if isinstance(raw_tools, Array) and not raw_tools:
            # An empty `tools = []` can't hold a `[[tools]]` table; replace it with one.
            self._document.remove("tools")
            raw_tools = None
        if raw_tools is None:
            raw_tools = tomlkit.aot()
            self._document.append("tools", raw_tools)
        elif not hasattr(raw_tools, "append"):
            raise AgentCliError("agent.toml tools must be an array of tables.")
        if isinstance(raw_tools, Array):
            raw_tools.append(_tool_inline_table(_tool_table(spec)))
        else:
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
        if not raw_tools:
            # tomlkit renders an empty array of tables as `tools = []`.
            self._document.remove("tools")
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

    def bind_model_service(
        self, name: str, default: str | None = None, role: str = DEFAULT_MODEL_ROLE
    ) -> bool:
        """Declare the model service for ``role`` in agent.toml. Returns True if it changed.

        ``name`` is a three-part UC name (catalog.schema.name). ``default`` is the ``system.ai.*``
        model `agentbricks deploy` routes the service to when it creates it; an existing service is never
        repointed by deploy. Passing no ``default`` keeps any recorded one.
        """
        role = model_role(role)
        name = _three_part_name(
            _required_string(name, f"[{MODEL_SERVICES_TABLE}.{role}] name"), "model service"
        )
        for other_role, binding in self.model_services.items():
            if other_role != role and binding.name == name:
                raise AgentCliError(
                    f"Model service '{name}' is already bound to role '{other_role}'.",
                    hint="Give each LLM call site its own model service so each can be upgraded "
                    "on its own.",
                )
        current = self.model_services.get(role)
        default = default or (current.default if current else None)
        binding = ModelServiceBinding(name, default)
        if current == binding:
            return False
        experimental = self._experimental_table()
        tables = experimental.get(MODEL_SERVICES_TABLE)
        if not isinstance(tables, Mapping):
            tables = tomlkit.table(is_super_table=True)
            experimental.append(MODEL_SERVICES_TABLE, tables)
        existing = tables.get(role)
        created = not isinstance(existing, Mapping)
        table = cast(Any, tomlkit.table() if created else existing)
        if created:
            tables.append(role, table)
        table["name"] = name
        if default:
            table[MODEL_SERVICE_DEFAULT_KEY] = default
        elif MODEL_SERVICE_DEFAULT_KEY in table:
            del table[MODEL_SERVICE_DEFAULT_KEY]
        if created:
            # Keep a blank line before whatever table follows, e.g. [experimental.prompts].
            table.add(tomlkit.nl())
        self.model_services[role] = binding
        return True

    def bind_prompt(self, name: str, key: str | None = None) -> str:
        """Declare Prompt Registry prompt ``name`` (catalog.schema.name) in agent.toml; returns its key.

        ``key`` defaults to the prompt's own name (its last part).
        """
        name = _three_part_name(_required_string(name, f"[{PROMPTS_PATH}] prompt"), "prompt")
        key = model_role(key or name.rsplit(".", 1)[1])
        for other_key, other_name in self.prompts.items():
            if other_name == name and other_key != key:
                raise AgentCliError(f"Prompt '{name}' is already bound as '{other_key}'.")
        experimental = self._experimental_table()
        table = experimental.get(PROMPTS_TABLE)
        if not isinstance(table, Mapping):
            table = tomlkit.table()
            experimental.append(PROMPTS_TABLE, table)
        table[key] = name
        self.prompts[key] = name
        return key

    def unbind_prompt(self, key: str) -> bool:
        """Remove prompt binding ``key`` from agent.toml. True if it was present."""
        if key not in self.prompts:
            return False
        experimental = cast(Any, self._document.get(EXPERIMENTAL_TABLE))
        table = experimental.get(PROMPTS_TABLE) if isinstance(experimental, Mapping) else None
        if isinstance(table, Mapping) and key in table:
            del table[key]
            if not table:
                del experimental[PROMPTS_TABLE]
                self._drop_empty_experimental_table()
        del self.prompts[key]
        return True

    def unbind_model_service(self, role: str = DEFAULT_MODEL_ROLE) -> bool:
        """Remove ``role``'s model-service binding from agent.toml. True if it was present."""
        if role not in self.model_services:
            return False
        experimental = cast(Any, self._document.get(EXPERIMENTAL_TABLE))
        tables = (
            experimental.get(MODEL_SERVICES_TABLE) if isinstance(experimental, Mapping) else None
        )
        if isinstance(tables, Mapping) and role in tables:
            del tables[role]
            if not tables:
                del experimental[MODEL_SERVICES_TABLE]
                self._drop_empty_experimental_table()
        del self.model_services[role]
        return True

    def _experimental_table(self) -> Any:
        """The [experimental] table, created (as a super table, so it never renders as an empty
        `[experimental]` header) when missing."""
        experimental = cast(Any, self._document.get(EXPERIMENTAL_TABLE))
        if not isinstance(experimental, Mapping):
            experimental = tomlkit.table(is_super_table=True)
            self._document.append(EXPERIMENTAL_TABLE, experimental)
        return experimental

    def _drop_empty_experimental_table(self) -> None:
        experimental = self._document.get(EXPERIMENTAL_TABLE)
        if isinstance(experimental, Mapping) and not experimental:
            del self._document[EXPERIMENTAL_TABLE]

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
        rendered = tomlkit.dumps(self._document)
        try:
            tomlkit.parse(rendered)
        except ParseError as exc:
            raise AgentCliError(
                f"Refusing to write invalid agent manifest at {self.path}: {exc}.",
                hint="The file was not changed. Report this as an agentbricks bug.",
            ) from exc
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
                output.write(rendered)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise AgentCliError(f"Could not write agent manifest at {self.path}: {exc}.") from exc
        return self.path
