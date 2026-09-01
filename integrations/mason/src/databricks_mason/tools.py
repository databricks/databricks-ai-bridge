"""Manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from importlib import resources
from typing import Any

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.errors import AgentCliError

_PYTHON_TOOL_TEMPLATE = "python_tool_langgraph.py"
_PYTHON_TEST_TEMPLATE = "python_tool_test.py"


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
    return spec.source.service or spec.source.function or spec.source.entrypoint or spec.source.kind


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
    project = AgentProject.load(source)
    _require_runtime_adapter(project)
    changed = project.add_tool(spec)
    changed_files = [project.write()] if changed else []
    _emit_change(obj, project, spec, changed_files)


def _require_runtime_adapter(project: AgentProject) -> None:
    if project.framework != "langgraph":
        raise AgentCliError(
            f"Mason tools supports only the 'langgraph' framework; found {project.framework!r}."
        )


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


def _render_python_template(name: str, *, module: str, function: str) -> str:
    return (
        _read_template(name)
        .replace("__MASON_TOOL_MODULE__", module)
        .replace("__MASON_TOOL_FUNCTION__", function)
    )


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
    """Add, inspect, and test agent tools declared in agent.toml."""


@tools.group("add")
def add() -> None:
    """Add a tool binding to agent.toml."""


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


@add.command("python")
@click.argument("name")
@_source_option
@click.pass_obj
def add_python(obj: Any, name: str, source: pathlib.Path) -> None:
    """Scaffold a framework-native local Python tool and starter test."""
    _require_arg(name, "tool name")
    project = AgentProject.load(source)
    _require_runtime_adapter(project)
    function = _identifier(name)
    spec = ToolSpec.python(
        name,
        entrypoint=f"agent.tools.{function}:{function}",
    )
    existing = next((tool for tool in project.tools if tool.id == spec.id), None)
    source_path = project.root / "agent" / "tools" / f"{function}.py"
    test_path = project.root / "tests" / "tools" / f"test_{function}.py"
    if existing == spec:
        # Manifest already declares this tool; recreate any scaffold files that went
        # missing (deleted by the user) rather than silently no-op'ing.
        missing = {}
        if not source_path.exists():
            missing[source_path] = _render_python_template(
                _PYTHON_TOOL_TEMPLATE, module=function, function=function
            )
        if not test_path.exists():
            missing[test_path] = _render_python_template(
                _PYTHON_TEST_TEMPLATE, module=function, function=function
            )
        recreated = _write_new_files(missing) if missing else []
        _emit_change(obj, project, spec, recreated)
        return
    if source_path.exists() or test_path.exists():
        existing_path = source_path if source_path.exists() else test_path
        raise AgentCliError(
            f"Refusing to overwrite user-owned file {existing_path}; it already exists."
        )

    project.add_tool(spec)
    files = {
        source_path: _render_python_template(
            _PYTHON_TOOL_TEMPLATE,
            module=function,
            function=function,
        ),
        test_path: _render_python_template(
            _PYTHON_TEST_TEMPLATE,
            module=function,
            function=function,
        ),
    }
    created = _write_new_files(files)
    try:
        project.write()
    except AgentCliError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    _emit_change(obj, project, spec, [project.path, *created])


@tools.command("list")
@_source_option
@click.pass_obj
def list_tools(obj: Any, source: pathlib.Path) -> None:
    """List tools declared in agent.toml."""
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
