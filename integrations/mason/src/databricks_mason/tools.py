"""Manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import require_managed_tool_support


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


def _source_value(spec: ToolSpec) -> str:
    # For a sandbox tool, the useful detail is the allowed scopes, not the (constant)
    # 'system.ai.sandbox' service name that duplicates the KIND column.
    if spec.source.kind == "sandbox" and spec.policy.downscope:
        return ", ".join(s.resource for s in spec.policy.downscope)
    return spec.source.service or spec.source.function or spec.source.kind


def _tool_record(spec: ToolSpec) -> dict[str, str]:
    return {
        "id": spec.id,
        "kind": spec.source.kind,
        "source": _source_value(spec),
    }


def _emit_change(
    obj: Any, project: AgentProject, spec: ToolSpec, changed_files: list[pathlib.Path]
) -> None:
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "tool": _tool_record(spec),
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        return
    if changed_files:
        render.success(
            f"Added {spec.id}",
            fields={"Kind": spec.source.kind, "Manifest": str(project.path)},
        )
    else:
        click.echo(f"Tool {spec.id!r} is already configured in {project.path}")


def _add_spec(obj: Any, source: pathlib.Path, spec: ToolSpec) -> None:
    # MCP / UC-function / sandbox bindings are framework-neutral agent.toml entries. Both Mason
    # server runtime adapters read them; custom-server projects wire tools directly in agent code.
    project = AgentProject.load(source)
    require_managed_tool_support(project.root)
    changed = project.add_tool(spec)
    changed_files = [project.write()] if changed else []
    _emit_change(obj, project, spec, changed_files)


def add_sandbox_to_manifest(
    obj: Any,
    source: pathlib.Path,
    scopes: tuple[str, ...],
    permission: str,
    *,
    tool_id: str = "sandbox",
) -> None:
    """Shared implementation for the nested command and compatibility alias."""
    parsed: list[Scope] = []
    seen: set[tuple[str, str]] = set()
    for value in scopes:
        scope = Scope.parse(value, permission)
        identity = (scope.kind, scope.value)
        if identity not in seen:
            parsed.append(scope)
            seen.add(identity)
    _add_spec(obj, source, ToolSpec.sandbox(tool_id, scopes=parsed))


@click.group()
def tools() -> None:
    """Manage Databricks-managed tools declared in agent.toml."""


@tools.group("add")
def add() -> None:
    """Add a managed sandbox, MCP service, or UC function.

    Subcommands target the current directory by default.

    Pass --source PATH to target another project.
    """


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Mason agent project containing agent.toml.",
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
@click.pass_obj
def add_sandbox(
    obj: Any,
    scopes: tuple[str, ...],
    permission: str,
    tool_id: str,
    source: pathlib.Path,
) -> None:
    """Bind system.ai.sandbox with protected downscoping."""
    add_sandbox_to_manifest(obj, source.resolve(), scopes, permission, tool_id=tool_id)


@add.command("mcp")
@click.argument("service")
@click.option("--name", "tool_id", default=None)
@_source_option
@click.pass_obj
def add_mcp(
    obj: Any,
    service: str,
    tool_id: str | None,
    source: pathlib.Path,
) -> None:
    """Bind a Databricks managed MCP SERVICE."""
    _require_arg(service, "managed MCP service name (e.g. system.ai.web_search)")
    _add_spec(
        obj,
        source.resolve(),
        ToolSpec.mcp(tool_id or _default_id(service), service=service),
    )


@add.command("uc-function")
@click.argument("function_name")
@click.option("--name", "tool_id", default=None)
@_source_option
@click.pass_obj
def add_uc_function(
    obj: Any,
    function_name: str,
    tool_id: str | None,
    source: pathlib.Path,
) -> None:
    """Bind an existing three-part Unity Catalog function."""
    _require_arg(function_name, "Unity Catalog function name (catalog.schema.function)")
    _add_spec(
        obj,
        source.resolve(),
        ToolSpec.uc_function(
            tool_id or _default_id(function_name),
            function=function_name,
        ),
    )


@tools.command("list")
@_source_option
@click.pass_obj
def list_tools(obj: Any, source: pathlib.Path) -> None:
    """List managed tool bindings for this agent."""
    project = AgentProject.load(source)
    rows = [_tool_record(spec) for spec in project.tools]
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "tools": rows})
        return
    render.resource_table(
        "Agent tools",
        [("ID", "left"), ("KIND", "left"), ("SOURCE", "left")],
        [(row["id"], row["kind"], row["source"]) for row in rows],
    )


@tools.command("remove")
@click.argument("tool_id")
@click.argument("mcp_service", required=False)
@_source_option
@click.pass_obj
def remove_tool(
    obj: Any,
    tool_id: str,
    mcp_service: str | None,
    source: pathlib.Path,
) -> None:
    """Remove a managed tool binding from this agent."""
    project = AgentProject.load(source)
    if mcp_service is not None:
        if tool_id != "mcp":
            raise AgentCliError("A second argument is supported only for `tools remove mcp`.")
        ToolSpec.mcp(_default_id(mcp_service), service=mcp_service)
        matches = [
            tool
            for tool in project.tools
            if tool.source.kind == "mcp" and tool.source.service == mcp_service
        ]
        if len(matches) > 1:
            raise AgentCliError(
                f"Multiple bindings use MCP service {mcp_service!r}.",
                hint="Run `mason tools list`, then remove the intended binding by ID.",
            )
        tool_id = matches[0].id if matches else _default_id(mcp_service)
    changed = project.remove_tool(tool_id)
    changed_files = [project.write()] if changed else []
    if getattr(obj, "output", "text") == "json":
        render.emit_json(
            {
                "schema_version": 1,
                "changed": changed,
                "changed_files": [str(path) for path in changed_files],
                "tool_id": tool_id,
            }
        )
        return
    if changed:
        render.success("Removed " + tool_id, fields={"Manifest": str(project.path)})
    else:
        click.echo(f"Tool {tool_id!r} is not configured in {project.path}")
