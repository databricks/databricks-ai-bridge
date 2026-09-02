"""Code-first ``mason tools`` commands."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import click
import tomli

from databricks_mason import render
from databricks_mason.attachment import Activation, activation_for
from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import IntegrationRegistry, registry_relative_path
from databricks_mason.integrations import (
    Integration,
    MCPService,
    Permission,
    Sandbox,
    Scope,
    UCFunction,
)
from databricks_mason.project_config import ProjectMetadata, load_project_metadata


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip()).strip("_").lower()
    if not normalized or normalized[0].isdigit():
        raise AgentCliError(f"Could not derive a Python identifier from {value!r}.")
    return normalized


def _default_id(resource: str) -> str:
    return _identifier(resource.rsplit(".", 1)[-1])


def _require_arg(value: str, label: str) -> str:
    """Reject an empty/whitespace positional argument with a clear message."""
    if value is None or not value.strip():
        raise AgentCliError(f"A {label} is required.")
    return value


def _source_value(spec: Integration) -> str:
    # For a sandbox tool, the useful detail is the allowed scopes, not the (constant)
    # 'system.ai.sandbox' service name that duplicates the KIND column.
    if isinstance(spec, Sandbox):
        return ", ".join(scope.resource for scope in spec.scopes)
    if isinstance(spec, MCPService):
        return spec.service
    return spec.function


def _tool_record(spec: Integration) -> dict[str, str]:
    return {
        "id": spec.id,
        "kind": spec.kind,
        "source": _source_value(spec),
    }


def _emit_change(
    obj: Any,
    registry: IntegrationRegistry,
    spec: Integration,
    changed_files: list[pathlib.Path],
    activation: Activation,
) -> None:
    definition = {"path": str(registry.path), "line": registry.definition_line(spec.id)}
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "tool": _tool_record(spec),
        "definition": definition,
        "activation": activation.as_dict(),
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        return
    if changed_files:
        click.echo(f"Added {spec.id}")
    else:
        click.echo(f"Tool {spec.id!r} is already configured")
    click.echo(f"Kind: {spec.kind}")
    click.echo(f"Definition: {registry.path}:{definition['line']}")
    if activation.status == "attached":
        for site in activation.sites:
            click.echo(f"Attached: {site.path}:{site.line} ({site.symbol})")
        click.echo("Status: Active after app restart")
    elif activation.status == "partial":
        for site in activation.sites:
            click.echo(f"Attached: {site.path}:{site.line} ({site.symbol})")
        click.echo("Status: Partially attached; not active on every agent path")
        click.echo("Next step:")
        click.echo("  Add at each remaining agent construction seam:")
        click.echo(f"    {activation.snippet}")
    else:
        click.echo("Status: Configured, not attached")
        click.echo("Next step:")
        if activation.imports:
            click.echo("  Add imports:")
            for import_line in activation.imports:
                click.echo(f"    {import_line}")
        click.echo("  Attach at the intended agent construction seam:")
        click.echo(f"    {activation.snippet}")


def _is_legacy_mason_manifest(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as input_file:
            document = tomli.load(input_file)
    except (OSError, tomli.TOMLDecodeError):
        return False
    agent = document.get("agent")
    return (
        document.get("schema_version") == 1
        and isinstance(agent, dict)
        and agent.get("framework") in {"langgraph", "openai"}
    )


def _registry(source: pathlib.Path, metadata: ProjectMetadata) -> IntegrationRegistry:
    relative_path = registry_relative_path(metadata.framework)
    path = source / relative_path
    manifest = source / "agent.toml"
    if (metadata.template is not None and manifest.exists()) or _is_legacy_mason_manifest(manifest):
        raise AgentCliError(
            f"agent.toml is retired for Mason projects at {source}.",
            hint=f"Move the selected integrations into {relative_path} as DATABRICKS_TOOLS, "
            "then remove agent.toml.",
        )
    if path.is_file():
        return IntegrationRegistry.load(source, relative_path=relative_path)
    return IntegrationRegistry.empty(source, relative_path=relative_path)


def _add_spec(
    obj: Any,
    source: pathlib.Path,
    spec: Integration,
    *,
    framework: str | None = None,
) -> None:
    metadata = load_project_metadata(source, framework_override=framework)
    registry = _registry(source, metadata)
    changed = registry.add(spec)
    changed_files = [registry.write()] if changed else []
    if not changed:
        registry = IntegrationRegistry.load(
            source,
            relative_path=registry_relative_path(metadata.framework),
        )
    _emit_change(
        obj,
        registry,
        spec,
        changed_files,
        activation_for(source, metadata),
    )


def _add_sandbox_to_registry(
    obj: Any,
    source: pathlib.Path,
    scopes: tuple[str, ...],
    permission: Permission,
    *,
    tool_id: str = "sandbox",
    framework: str | None = None,
) -> None:
    """Add a validated Sandbox descriptor to the framework's Python registry."""
    parsed: list[Scope] = []
    seen: set[tuple[str, str]] = set()
    for value in scopes:
        scope = Scope.parse(value, permission)
        identity = (scope.kind, scope.value)
        if identity not in seen:
            parsed.append(scope)
            seen.add(identity)
    _add_spec(
        obj,
        source,
        Sandbox(tool_id, scopes=tuple(parsed)),
        framework=framework,
    )


@click.group()
def tools() -> None:
    """Manage Databricks integrations selected in agent code."""


@tools.group("add")
def add() -> None:
    """Add a sandbox, MCP service, or UC function.

    Subcommands target the current directory by default.

    Pass --source PATH to target another project.
    """


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Mason agent project containing .mason/project.toml.",
    )(function)


def _framework_option(function):
    return click.option(
        "--framework",
        type=click.Choice(["langgraph", "openai"]),
        default=None,
        help="Framework adapter for BYO projects without .mason/project.toml metadata.",
    )(function)


@add.command("sandbox")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    required=True,
    help="Allowed table:, volume:, or workspace: resource. Repeat for multiple scopes.",
)
@click.option(
    "--permission",
    type=click.Choice(["read_only", "read_write"]),
    default="read_only",
    show_default=True,
)
@click.option("--name", "tool_id", default="sandbox", show_default=True)
@_source_option
@_framework_option
@click.pass_obj
def add_sandbox(
    obj: Any,
    scopes: tuple[str, ...],
    permission: Permission,
    tool_id: str,
    source: pathlib.Path,
    framework: str | None,
) -> None:
    """Bind system.ai.sandbox with protected downscoping."""
    _add_sandbox_to_registry(
        obj,
        source.resolve(),
        scopes,
        permission,
        tool_id=tool_id,
        framework=framework,
    )


@add.command("mcp")
@click.argument("service")
@click.option("--name", "tool_id", default=None)
@_source_option
@_framework_option
@click.pass_obj
def add_mcp(
    obj: Any,
    service: str,
    tool_id: str | None,
    source: pathlib.Path,
    framework: str | None,
) -> None:
    """Bind a Databricks managed MCP SERVICE."""
    _require_arg(service, "managed MCP service name (e.g. system.ai.web_search)")
    _add_spec(
        obj,
        source.resolve(),
        MCPService(tool_id or _default_id(service), service=service),
        framework=framework,
    )


@add.command("uc-function")
@click.argument("function_name")
@click.option("--name", "tool_id", default=None)
@_source_option
@_framework_option
@click.pass_obj
def add_uc_function(
    obj: Any,
    function_name: str,
    tool_id: str | None,
    source: pathlib.Path,
    framework: str | None,
) -> None:
    """Bind an existing three-part Unity Catalog function."""
    _require_arg(function_name, "Unity Catalog function name (catalog.schema.function)")
    _add_spec(
        obj,
        source.resolve(),
        UCFunction(
            tool_id or _default_id(function_name),
            function=function_name,
        ),
        framework=framework,
    )


@tools.command("list")
@_source_option
@_framework_option
@click.pass_obj
def list_tools(obj: Any, source: pathlib.Path, framework: str | None) -> None:
    """List Databricks integrations configured for this agent."""
    project = source.expanduser().resolve()
    metadata = load_project_metadata(project, framework_override=framework)
    registry = _registry(project, metadata)
    rows = [_tool_record(spec) for spec in registry.integrations]
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "tools": rows})
        return
    render.resource_table(
        "Agent tools",
        [("ID", "left"), ("KIND", "left"), ("SOURCE", "left")],
        [(row["id"], row["kind"], row["source"]) for row in rows],
    )
