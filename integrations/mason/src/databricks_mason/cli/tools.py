"""Discover available integrations and manage manifest-backed tool bindings."""

from __future__ import annotations

import pathlib
import re
from typing import Any, Literal

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.cli.help import _example_epilog
from databricks_mason.cli.mcp import _add_command, _list_services, _validate_schema
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
    return spec.source.service or spec.source.function or spec.source.space_id or spec.source.kind


def _tool_record(spec: ToolSpec) -> dict[str, str]:
    record = {
        "id": spec.id,
        "kind": spec.source.kind,
        "source": _source_value(spec),
    }
    if spec.source.kind in ("mcp", "sandbox", "uc_function", "genie_one", "genie_agent"):
        record["auth"] = spec.auth or (
            "app/default" if spec.source.kind == "uc_function" else "unspecified"
        )
    return record


def _emit_change(
    obj: Any, project: AgentProject, spec: ToolSpec, changed_files: list[pathlib.Path]
) -> None:
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "manifest": str(project.path),
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
    click.echo(f"Review {project.path} to check configured managed tools and MCP bindings.")


def _add_spec(obj: Any, source: pathlib.Path, spec: ToolSpec) -> None:
    # Managed bindings are framework-neutral agent.toml entries. Both Mason server runtime
    # adapters read them; custom-server projects wire tools directly in agent code.
    project = AgentProject.load(source)
    require_managed_tool_support(project.root)
    changed = project.add_tool(spec)
    if spec.source.kind == "mcp":
        service = spec.source.service
        try:
            obj.client().get_mcp_service(service)
        except AgentCliError as exc:
            raise AgentCliError(
                f"Could not validate MCP service {service!r}: {exc.message}",
                error_code=exc.error_code,
                hint=exc.hint
                or "Run `mason tools list --kind mcp` with the same --profile to find "
                "available services.",
            ) from exc
    changed_files = [project.write()] if changed else []
    _emit_change(obj, project, spec, changed_files)


def add_sandbox_to_manifest(
    obj: Any,
    source: pathlib.Path,
    scopes: tuple[str, ...],
    permission: str,
    *,
    tool_id: str = "sandbox",
    auth: Literal["user", "app"] = "user",
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
    _add_spec(obj, source, ToolSpec.sandbox(tool_id, scopes=parsed, auth=auth))


@click.group()
def tools() -> None:
    """Discover available integrations and manage an agent's tool bindings.

    Tools are what let an agent act beyond the language model itself — query governed data, call a
    service, or run a function — and each one is recorded in agent.toml so `mason dev` / `mason
    deploy` wire it in. `mason tools add` manages these Databricks-managed tool types:

    \b
      sandbox       Query Unity Catalog data via system.ai.sandbox, scoped
                    to the tables, volumes, or paths you choose.
      mcp           A Databricks-managed MCP service (see `mason tools list --kind mcp`),
                    e.g. system.ai.web_search.
      uc-function   An existing Unity Catalog function (catalog.schema.function).
      genie-one     Workspace-wide Genie One MCP tools.
      genie-agent   Native Genie conversation tools for a configured space ID.

    Browse available integrations with `mason tools list`, add one with `mason tools add <type>`,
    and drop a binding with `mason tools remove`. Review agent.toml for configured managed tools
    and MCP bindings. The list shows addable integrations, not configured bindings or individual
    operations inside an MCP service. Custom Python tools are code-first — write them directly
    in your project's code rather than through the CLI.
    """


@tools.group("add")
def add() -> None:
    """Add a managed sandbox, MCP service, UC function, or Genie tool binding.

    Subcommands target the current directory by default.

    Pass --source PATH to target another project.

    Review that project's agent.toml to check configured managed tools and MCP bindings.
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
@click.option("--auth", type=click.Choice(["user", "app"]), default="user", show_default=True)
@_source_option
@click.pass_obj
def add_sandbox(
    obj: Any,
    scopes: tuple[str, ...],
    permission: str,
    tool_id: str,
    source: pathlib.Path,
    auth: Literal["user", "app"],
) -> None:
    """Add a data sandbox tool (system.ai.sandbox), scoped to specific Unity Catalog resources.

    Review the target project's agent.toml to check configured managed tools and MCP bindings.
    """
    add_sandbox_to_manifest(obj, source.resolve(), scopes, permission, tool_id=tool_id, auth=auth)


@add.command("mcp")
@click.argument("service")
@click.option("--name", "tool_id", default=None)
@click.option("--auth", type=click.Choice(["user", "app"]), default="user", show_default=True)
@_source_option
@click.pass_obj
def add_mcp(
    obj: Any,
    service: str,
    tool_id: str | None,
    source: pathlib.Path,
    auth: Literal["user", "app"],
) -> None:
    """Validate and add a Databricks-managed MCP service as a tool.

    Use `mason tools list --kind mcp` for available services. Review the target project's
    agent.toml to check configured managed tools and MCP bindings.
    """
    _require_arg(service, "managed MCP service name (e.g. system.ai.web_search)")
    _add_spec(
        obj,
        source.resolve(),
        ToolSpec.mcp(tool_id or _default_id(service), service=service, auth=auth),
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
    """Add an existing Unity Catalog function (catalog.schema.function) as a tool.

    Review the target project's agent.toml to check configured managed tools and MCP bindings.
    """
    _require_arg(function_name, "Unity Catalog function name (catalog.schema.function)")
    _add_spec(
        obj,
        source.resolve(),
        ToolSpec.uc_function(
            tool_id or _default_id(function_name),
            function=function_name,
        ),
    )


@add.command("genie-one", epilog=_example_epilog(("mason tools add genie-one",)))
@click.option("--name", "tool_id", default="genie_one", show_default=True)
@click.option("--auth", type=click.Choice(["user", "app"]), default="user", show_default=True)
@_source_option
@click.pass_obj
def add_genie_one(
    obj: Any, tool_id: str, auth: Literal["user", "app"], source: pathlib.Path
) -> None:
    """Add workspace-wide Genie One MCP tools."""
    _add_spec(obj, source.resolve(), ToolSpec.genie_one(tool_id, auth=auth))


@add.command(
    "genie-agent",
    epilog=_example_epilog(
        ("mason tools add genie-agent 0123456789abcdef0123456789abcdef --name sales",)
    ),
)
@click.argument("space_id")
@click.option("--name", "tool_id", default="genie_agent", show_default=True)
@click.option("--auth", type=click.Choice(["user", "app"]), default="user", show_default=True)
@_source_option
@click.pass_obj
def add_genie_agent(
    obj: Any,
    space_id: str,
    tool_id: str,
    auth: Literal["user", "app"],
    source: pathlib.Path,
) -> None:
    """Add native Genie conversation tools for a 32-character lowercase hexadecimal SPACE_ID."""
    _add_spec(
        obj,
        source.resolve(),
        ToolSpec.genie_agent(tool_id, space_id=space_id, auth=auth),
    )


@tools.command("list")
@click.option(
    "--kind",
    type=click.Choice(["sandbox", "mcp", "uc-function", "genie-one", "genie-agent"]),
    help=(
        "Show one integration kind. Sandbox, uc-function, genie-one, and genie-agent show local "
        "add recipes only."
    ),
)
@click.option(
    "--schema",
    help="Two-part UC schema: catalog.schema (default: system.ai). Requires --kind mcp.",
)
@click.pass_obj
def list_tools(obj: Any, kind: str | None, schema: str | None) -> None:
    """List available integrations to add, not configured agent bindings.

    By default, show built-in add recipes plus caller-visible MCP Services in system.ai.
    --kind mcp limits discovery to MCP Services; --schema catalog.schema replaces system.ai.
    Sandbox recipes require scopes; UC-function and Genie Agent recipes require concrete resource
    identifiers. Genie One needs no additional argument.

    No agent project is required. MCP discovery uses your Databricks profile; local recipes do
    not authenticate. This does not scan every workspace schema or list individual MCP operations.
    API failures return a nonzero exit status and mark discovery incomplete, not empty.

    Review agent.toml to check configured managed tools and MCP bindings. The former configured
    list and --source option are removed. JSON discovery uses schema_version 2 and available_tools.
    """
    if schema is not None and kind != "mcp":
        raise AgentCliError("--schema requires --kind mcp.")
    mcp_schema = (
        _validate_schema("system.ai" if schema is None else schema)
        if kind in (None, "mcp")
        else None
    )
    sandbox_command = _add_command("system.ai.sandbox")
    recipes = [
        {"name": "sandbox", "kind": "sandbox", "add_command": sandbox_command},
        {
            "name": "uc-function",
            "kind": "uc-function",
            "add_command": "mason tools add uc-function catalog.schema.function",
        },
        {
            "name": "genie-one",
            "kind": "genie-one",
            "add_command": "mason tools add genie-one",
        },
        {
            "name": "genie-agent",
            "kind": "genie-agent",
            "add_command": "mason tools add genie-agent SPACE_ID",
        },
    ]
    rows = [recipe for recipe in recipes if kind is None or recipe["kind"] == kind]
    discovery_error = None
    if mcp_schema is not None:
        try:
            services = _list_services(obj.client(), mcp_schema)
        except AgentCliError as exc:
            discovery_error = exc
        else:
            for service in services:
                name = service["name"]
                if name == "system.ai.sandbox" and kind is None:
                    continue
                rows.append(
                    {
                        "name": name,
                        "kind": "mcp",
                        "add_command": _add_command(name),
                    }
                )
    if getattr(obj, "output", "text") == "json":
        errors = []
        if discovery_error is not None:
            error = {"message": discovery_error.message}
            if discovery_error.error_code:
                error["code"] = discovery_error.error_code
            if discovery_error.hint:
                error["hint"] = discovery_error.hint
            errors.append(error)
        render.emit_json(
            {
                "schema_version": 2,
                "available_tools": rows,
                "mcp_schema": mcp_schema,
                "complete": discovery_error is None,
                "errors": errors,
            }
        )
    else:
        if kind != "mcp":
            render.resource_table(
                "Built-in add recipes",
                [("Kind", "left"), ("Add command (replace example resources)", "left")],
                [(row["kind"], row["add_command"]) for row in rows if row["kind"] != "mcp"],
            )
        if mcp_schema is not None:
            render.resource_table(
                "Available MCP Services",
                [("Service", "left"), ("Add command", "left")],
                [(row["name"], row["add_command"]) for row in rows if row["kind"] == "mcp"],
                subtitle=f"Caller-visible in {mcp_schema}"
                + (" — discovery incomplete" if discovery_error is not None else ""),
            )
        click.echo("Tip: focus on MCP services with `mason tools list --kind mcp`.")
        click.echo(
            "Discover services in another UC schema with "
            "`mason tools list --kind mcp --schema catalog.schema`."
        )
        click.echo("Review agent.toml to check configured managed tools and MCP bindings.")
        if discovery_error is not None:
            discovery_error.show()
    if discovery_error is not None:
        raise click.exceptions.Exit(1)


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
                hint=f"Review {project.path}, then remove the intended binding by ID.",
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
