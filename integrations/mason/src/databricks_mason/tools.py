"""Code-first ``mason tools`` commands."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from importlib import resources
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

_PYTHON_TOOL_TEMPLATE = "python_tool_langgraph.py"
_PYTHON_TEST_TEMPLATE = "python_tool_test.py"
_TOOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PYTHON_MARKER = re.compile(
    r"^# mason:python-tool id=(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*) "
    r"entrypoint=(?P<entrypoint>agent\.tools\.(?P<module>[A-Za-z_][A-Za-z0-9_]*):"
    r"(?P<function>[A-Za-z_][A-Za-z0-9_]*))$",
    re.MULTILINE,
)


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


def _require_tool_id(value: str) -> str:
    if not _TOOL_ID.fullmatch(value):
        raise AgentCliError(f"Invalid tool id {value!r}.")
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
    if any(record["id"] == spec.id for record in _python_records(source)):
        raise AgentCliError(
            f"Tool id {spec.id!r} is already used by a local Python tool.",
            hint="Use --name to choose a different integration id.",
        )
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


def _read_template(name: str) -> str:
    try:
        return (
            resources.files("databricks_mason")
            .joinpath("templates")
            .joinpath(name)
            .read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise AgentCliError(f"Could not read packaged Python tool template {name!r}.") from exc


def _render_python_template(
    name: str,
    *,
    tool_id: str,
    module: str,
    function: str,
) -> str:
    return (
        _read_template(name)
        .replace("__MASON_TOOL_ID__", tool_id)
        .replace("__MASON_TOOL_MODULE__", module)
        .replace("__MASON_TOOL_FUNCTION__", function)
    )


def _python_record(path: pathlib.Path) -> dict[str, str] | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = _PYTHON_MARKER.search(source)
    if match is None:
        return None
    return {
        "id": match.group("id"),
        "kind": "python",
        "source": match.group("entrypoint"),
    }


def _python_records(project: pathlib.Path) -> list[dict[str, str]]:
    directory = project / "agent" / "tools"
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.py")):
        record = _python_record(path)
        if record is None:
            continue
        module = record["source"].partition(":")[0].rsplit(".", 1)[-1]
        if module == path.stem:
            records.append(record)
    return records


def _require_unique_records(records: list[dict[str, str]]) -> None:
    ids = [record["id"] for record in records]
    duplicates = sorted({tool_id for tool_id in ids if ids.count(tool_id) > 1})
    if duplicates:
        rendered = ", ".join(repr(tool_id) for tool_id in duplicates)
        raise AgentCliError(f"Tool ids must be unique across generated code: {rendered}.")


def _emit_python_change(
    obj: Any,
    *,
    tool_id: str,
    source_path: pathlib.Path,
    changed_files: list[pathlib.Path],
) -> None:
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "tool": {
            "id": tool_id,
            "kind": "python",
            "source": f"agent.tools.{source_path.stem}:{source_path.stem}",
        },
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
    elif changed_files:
        render.success(f"Added {tool_id}", fields={"Source": str(source_path)})
    else:
        click.echo(f"Tool {tool_id!r} is already configured in {source_path}")


def _write_new_files(files: dict[pathlib.Path, str]) -> list[pathlib.Path]:
    for path in files:
        if path.exists():
            raise AgentCliError(f"Refusing to overwrite user-owned file {path}; it already exists.")

    temporary: dict[pathlib.Path, pathlib.Path] = {}
    created: list[pathlib.Path] = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
                temporary[path] = pathlib.Path(output.name)
        for target, staged in temporary.items():
            os.replace(staged, target)
            created.append(target)
        return created
    except OSError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise AgentCliError(f"Could not create Python tool scaffold: {exc}.") from exc
    finally:
        for staged in temporary.values():
            staged.unlink(missing_ok=True)


@click.group()
def tools() -> None:
    """Manage code-selected tools for a Mason agent project."""


@tools.group("add")
def add() -> None:
    """Add a sandbox, MCP service, UC function, or Python tool.

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


@add.command("python")
@click.argument("name")
@_source_option
@_framework_option
@click.pass_obj
def add_python(
    obj: Any,
    name: str,
    source: pathlib.Path,
    framework: str | None,
) -> None:
    """Scaffold a framework-native local Python tool and starter test."""
    tool_id = _require_tool_id(_require_arg(name, "tool name"))
    project = source.expanduser().resolve()
    metadata = load_project_metadata(project, framework_override=framework)
    if metadata.framework != "langgraph":
        raise AgentCliError(
            "Mason's generated Python tool scaffold currently supports only LangGraph."
        )
    function = _identifier(name)
    registry = _registry(project, metadata)
    if any(integration.id == tool_id for integration in registry.integrations):
        raise AgentCliError(
            f"Tool id {tool_id!r} is already used by a Databricks integration.",
            hint="Choose a different Python tool name.",
        )
    source_path = project / "agent" / "tools" / f"{function}.py"
    test_path = project / "tests" / "tools" / f"test_{function}.py"
    entrypoint = f"agent.tools.{function}:{function}"
    expected = {"id": tool_id, "kind": "python", "source": entrypoint}
    existing_paths = [path for path in (source_path, test_path) if path.exists()]
    if any(_python_record(path) != expected for path in existing_paths):
        existing_path = next(path for path in existing_paths if _python_record(path) != expected)
        raise AgentCliError(
            f"Refusing to overwrite user-owned file {existing_path}; it already exists."
        )

    if len(existing_paths) == 2:
        _emit_python_change(
            obj,
            tool_id=tool_id,
            source_path=source_path,
            changed_files=[],
        )
        return

    if len(existing_paths) == 1:
        missing_path = test_path if source_path.exists() else source_path
        template = _PYTHON_TEST_TEMPLATE if missing_path == test_path else _PYTHON_TOOL_TEMPLATE
        created = _write_new_files(
            {
                missing_path: _render_python_template(
                    template,
                    tool_id=tool_id,
                    module=function,
                    function=function,
                )
            }
        )
        _emit_python_change(
            obj,
            tool_id=tool_id,
            source_path=source_path,
            changed_files=created,
        )
        return

    files = {
        source_path: _render_python_template(
            _PYTHON_TOOL_TEMPLATE,
            tool_id=tool_id,
            module=function,
            function=function,
        ),
        test_path: _render_python_template(
            _PYTHON_TEST_TEMPLATE,
            tool_id=tool_id,
            module=function,
            function=function,
        ),
    }
    created = _write_new_files(files)
    _emit_python_change(
        obj,
        tool_id=tool_id,
        source_path=source_path,
        changed_files=created,
    )


@tools.command("list")
@_source_option
@_framework_option
@click.pass_obj
def list_tools(obj: Any, source: pathlib.Path, framework: str | None) -> None:
    """List tools configured for this agent."""
    project = source.expanduser().resolve()
    metadata = load_project_metadata(project, framework_override=framework)
    registry = _registry(project, metadata)
    rows = [*(_tool_record(spec) for spec in registry.integrations), *_python_records(project)]
    _require_unique_records(rows)
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "tools": rows})
        return
    render.resource_table(
        "Agent tools",
        [("ID", "left"), ("KIND", "left"), ("SOURCE", "left")],
        [(row["id"], row["kind"], row["source"]) for row in rows],
    )
