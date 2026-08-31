"""Manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.errors import AgentCliError

_RUNTIME_TIMEOUT_SECONDS = 60.0
_RUNTIME_LOG_LIMIT_BYTES = 64 * 1024
_PIPE_READ_BYTES = 8 * 1024


@dataclass(frozen=True)
class _BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class _BoundedCapture:
    def __init__(self, limit_bytes: int) -> None:
        self._limit = max(1, limit_bytes)
        self._head_limit = (self._limit + 1) // 2
        self._tail_limit = self._limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0

    def add(self, chunk: bytes) -> None:
        self._total += len(chunk)
        head_room = self._head_limit - len(self._head)
        if head_room > 0:
            self._head.extend(chunk[:head_room])
            chunk = chunk[head_room:]
        if chunk and self._tail_limit:
            self._tail.extend(chunk)
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]

    def text(self) -> str:
        captured = len(self._head) + len(self._tail)
        if self._total <= captured:
            data = bytes(self._head + self._tail)
            return data.decode("utf-8", errors="replace")
        dropped = self._total - captured
        head = bytes(self._head).decode("utf-8", errors="replace")
        tail = bytes(self._tail).decode("utf-8", errors="replace")
        return f"{head}\n... [{dropped} bytes truncated] ...\n{tail}"


def _drain_pipe(stream: Any, capture: _BoundedCapture) -> None:
    try:
        while chunk := stream.read(_PIPE_READ_BYTES):
            capture.add(chunk)
    except OSError:
        pass
    finally:
        stream.close()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        if process.poll() is None:
            process.kill()


def _run_bounded_process(
    command: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str],
    timeout_seconds: float,
    log_limit_bytes: int,
) -> _BoundedProcessResult:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _BoundedCapture(log_limit_bytes)
    stderr = _BoundedCapture(log_limit_bytes)
    readers = [
        threading.Thread(target=_drain_pipe, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain_pipe, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    else:
        for reader in readers:
            reader.join(timeout=max(0.0, deadline - time.monotonic()))
        timed_out = any(reader.is_alive() for reader in readers)

    if timed_out:
        _kill_process_group(process)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join(timeout=1)

    return _BoundedProcessResult(
        returncode=process.returncode,
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
    )


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip()).strip("_").lower()
    if not normalized or normalized[0].isdigit():
        raise AgentCliError(f"Could not derive a Python identifier from {value!r}.")
    return normalized


def _default_id(resource: str) -> str:
    return _identifier(resource.rsplit(".", 1)[-1])


def _source_value(spec: ToolSpec) -> str:
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


def _literal_tool_decorator(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id == "tool"
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "tool"
        and isinstance(target.value, ast.Name)
        and target.value.id == "mason"
    )


def _module_name(project: AgentProject, path: pathlib.Path) -> str:
    parts = list(path.relative_to(project.root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _discover_python_tools(
    project: AgentProject,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    discovery_root = project.root / "agent" / "tools"
    if not discovery_root.is_dir():
        return (), []
    project_root = project.root.resolve()
    candidates: list[str] = []
    warnings: list[dict[str, str]] = []
    for path in sorted(discovery_root.rglob("*.py")):
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(project_root):
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            relative_path = path.relative_to(project.root).as_posix()
            warnings.append(
                {
                    "code": "MASON002",
                    "path": relative_path,
                    "message": (
                        f"{relative_path} could not be scanned for Python tools: "
                        f"{exc.msg} (line {exc.lineno})."
                    ),
                }
            )
            continue
        except (OSError, UnicodeError) as exc:
            relative_path = path.relative_to(project.root).as_posix()
            warnings.append(
                {
                    "code": "MASON002",
                    "path": relative_path,
                    "message": f"{relative_path} could not be scanned for Python tools: {exc}.",
                }
            )
            continue
        module = _module_name(project, path)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_literal_tool_decorator(decorator) for decorator in node.decorator_list):
                candidates.append(f"{module}:{node.name}")
    return tuple(sorted(set(candidates))), warnings


def discover_python_tool_candidates(project: AgentProject) -> tuple[str, ...]:
    """Find likely undeclared Python tools without importing project modules."""
    candidates, _ = _discover_python_tools(project)
    declared = {
        spec.source.entrypoint
        for spec in project.python_tools()
        if spec.source.entrypoint is not None
    }
    return tuple(candidate for candidate in candidates if candidate not in declared)


def _discovery_warnings(project: AgentProject) -> list[dict[str, str]]:
    candidates, warnings = _discover_python_tools(project)
    declared = {
        spec.source.entrypoint
        for spec in project.python_tools()
        if spec.source.entrypoint is not None
    }
    warnings.extend(
        {
            "code": "MASON001",
            "entrypoint": entrypoint,
            "message": (
                f"{entrypoint} appears to be a decorated Python tool but is not active in agent.toml."
            ),
        }
        for entrypoint in candidates
        if entrypoint not in declared
    )
    return warnings


def _runtime_command(project: AgentProject, arguments: list[str]) -> dict[str, object]:
    result_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="mason-python-tool-",
            suffix=".json",
            delete=False,
        ) as result_file:
            result_path = pathlib.Path(result_file.name)
        command = [
            "uv",
            "run",
            "--project",
            str(project.root),
            "python",
            "-m",
            "databricks_mason.python_runtime",
            *arguments,
            "--result-path",
            str(result_path),
        ]
        child_env = os.environ.copy()
        child_env["MASON_PROJECT_ROOT"] = str(project.root)
        completed = _run_bounded_process(
            command,
            cwd=project.root,
            env=child_env,
            timeout_seconds=_RUNTIME_TIMEOUT_SECONDS,
            log_limit_bytes=_RUNTIME_LOG_LIMIT_BYTES,
        )
        payload: dict[str, object]
        if completed.timed_out:
            payload = {
                "schema_version": 1,
                "ok": False,
                "error": (
                    "Python tool runtime timed out after "
                    f"{_RUNTIME_TIMEOUT_SECONDS:g} seconds and was terminated."
                ),
            }
        else:
            try:
                decoded: object = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                payload = {
                    "schema_version": 1,
                    "ok": False,
                    "error": f"Python tool runtime did not write valid control JSON: {exc}.",
                }
            else:
                payload = (
                    cast(dict[str, object], decoded)
                    if isinstance(decoded, dict)
                    else {
                        "schema_version": 1,
                        "ok": False,
                        "error": "Python tool runtime control result must be a JSON object.",
                    }
                )
        if payload.get("schema_version") != 1:
            payload = {
                "schema_version": 1,
                "ok": False,
                "error": "Python tool runtime returned an unsupported control schema.",
            }
        if completed.returncode != 0 and payload.get("ok") is not False:
            payload = {
                "schema_version": 1,
                "ok": False,
                "error": f"Python tool runtime exited with status {completed.returncode}.",
            }
        logs: dict[str, str] = {}
        if completed.stdout:
            logs["stdout"] = completed.stdout
        if completed.stderr:
            logs["stderr"] = completed.stderr
        payload["logs"] = logs
        return payload
    except OSError as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "error": f"Could not start the Python tool runtime: {exc}.",
            "logs": {},
        }
    finally:
        if result_path is not None:
            result_path.unlink(missing_ok=True)


def _parent_error_payload(message: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ok": False,
        "error": message,
        "logs": {},
    }


def check_python_tools(
    source: pathlib.Path | str, *, tool_id: str | None = None
) -> dict[str, object]:
    """Validate manifest-declared Python tools and report best-effort discovery warnings."""
    try:
        project = AgentProject.load(source)
        warnings = _discovery_warnings(project)
        python_tools = project.python_tools()
        if tool_id is not None:
            python_tools = tuple(spec for spec in python_tools if spec.id == tool_id)
            if not python_tools:
                return {
                    "schema_version": 1,
                    "ok": False,
                    "error": f"Python tool {tool_id!r} is not declared in agent.toml.",
                    "tools": [],
                    "warnings": warnings,
                    "logs": {},
                }
        if not python_tools:
            return {
                "schema_version": 1,
                "ok": True,
                "tools": [],
                "warnings": warnings,
                "logs": {},
            }
        _require_runtime_adapter(project)
        arguments = ["check"]
        if tool_id is not None:
            arguments.append(tool_id)
        payload = _runtime_command(project, arguments)
        payload["warnings"] = warnings
        return payload
    except AgentCliError as exc:
        return _parent_error_payload(exc.message)


@click.group()
def tools() -> None:
    """Attach, inspect, validate, and run tools declared in agent.toml."""


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


def _emit_logs(payload: dict[str, object]) -> None:
    logs = payload.get("logs")
    if not isinstance(logs, dict):
        return
    typed_logs = cast(dict[str, object], logs)
    for stream in ("stdout", "stderr"):
        value = typed_logs.get(stream)
        if isinstance(value, str) and value:
            click.echo(f"{stream}:\n{value}", nl=not value.endswith("\n"))


def _emit_warnings(payload: dict[str, object]) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        typed_warning = cast(dict[str, object], warning)
        click.echo(
            f"Warning [{typed_warning.get('code', 'MASON')}]: {typed_warning.get('message', '')}"
        )


def render_python_tool_diagnostics(payload: dict[str, object]) -> None:
    """Render child logs and non-blocking warnings, then fail only on hard validation errors."""
    _emit_logs(payload)
    _emit_warnings(payload)
    if payload.get("ok") is not True:
        raise AgentCliError(str(payload.get("error") or "Python tool validation failed."))


def _emit_result(obj: Any, payload: dict[str, object], *, success_message: str) -> None:
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        if payload.get("ok") is not True:
            raise click.exceptions.Exit(1)
        return
    render_python_tool_diagnostics(payload)
    click.echo(success_message)


def _fail_parent_validation(obj: Any, message: str) -> NoReturn:
    if getattr(obj, "output", "text") == "json":
        render.emit_json(_parent_error_payload(message))
        raise click.exceptions.Exit(1)
    raise AgentCliError(message)


@tools.command("check")
@click.argument("name", required=False)
@_source_option
@click.pass_obj
def check_tools(obj: Any, name: str | None, source: pathlib.Path) -> None:
    """Validate declared Python tools without contacting a model or MCP endpoint."""
    payload = check_python_tools(source, tool_id=name)
    tool_rows = payload.get("tools")
    count = len(tool_rows) if isinstance(tool_rows, list) else 0
    _emit_result(obj, payload, success_message=f"Checked {count} Python tool(s).")


@tools.command("run")
@click.argument("name")
@click.option("--input", "input_json", required=True, help="Tool arguments as one JSON object.")
@_source_option
@click.pass_obj
def run_tool(obj: Any, name: str, input_json: str, source: pathlib.Path) -> None:
    """Invoke one manifest-declared Python tool directly."""
    try:
        arguments = json.loads(input_json)
    except json.JSONDecodeError as exc:
        _fail_parent_validation(obj, f"--input must be valid JSON: {exc}.")
    if not isinstance(arguments, dict):
        _fail_parent_validation(obj, "--input must be a JSON object.")
    try:
        project = AgentProject.load(source)
        _require_runtime_adapter(project)
    except AgentCliError as exc:
        _fail_parent_validation(obj, exc.message)
    if not any(spec.id == name for spec in project.python_tools()):
        _fail_parent_validation(obj, f"Python tool {name!r} is not declared in agent.toml.")
    payload = _runtime_command(
        project,
        ["run", name, "--input", json.dumps(arguments, separators=(",", ":"))],
    )
    if getattr(obj, "output", "text") == "json":
        _emit_result(obj, payload, success_message="")
        return
    _emit_logs(payload)
    if payload.get("ok") is not True:
        raise AgentCliError(str(payload.get("error") or "Python tool invocation failed."))
    click.echo(json.dumps(payload.get("result"), ensure_ascii=False))
