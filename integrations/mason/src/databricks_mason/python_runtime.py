"""Resolve and invoke manifest-declared LangGraph Python tools."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from databricks_mason.agent_project import AgentProject, ToolSpec


@dataclass(frozen=True)
class ResolvedPythonTool:
    """One validated Python tool and its stable model-facing contract."""

    record: ToolSpec
    tool: BaseTool
    schema: dict[str, Any]
    fingerprint: str


def _normalized_name(value: str) -> str:
    return value.replace("-", "_")


def _implementation_source(tool: BaseTool) -> str:
    implementation = getattr(tool, "func", None)
    if implementation is None:
        return ""
    try:
        return inspect.getsource(implementation)
    except (OSError, TypeError):
        return ""


def _resolve(record: ToolSpec) -> ResolvedPythonTool:
    entrypoint = record.source.entrypoint
    if not entrypoint:
        raise RuntimeError(f"Python tool {record.id!r} has no entry point.")
    module_name, _, attribute_name = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import module {module_name!r} for Python tool {record.id!r}: {exc}"
        ) from exc
    try:
        value = getattr(module, attribute_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Module {module_name!r} has no attribute {attribute_name!r} "
            f"for Python tool {record.id!r}."
        ) from exc
    if not isinstance(value, BaseTool):
        raise RuntimeError(f"Python tool {record.id!r} entry point must resolve to a BaseTool.")
    if _normalized_name(value.name) != _normalized_name(record.id):
        raise RuntimeError(
            f"Python tool {record.id!r} resolves to runtime name {value.name!r}; "
            "the names must match (hyphens and underscores are equivalent)."
        )
    description = value.description.strip() if isinstance(value.description, str) else ""
    if not description:
        raise RuntimeError(f"Python tool {record.id!r} must have a non-empty description.")
    try:
        schema = value.get_input_schema().model_json_schema()
    except Exception as exc:
        raise RuntimeError(
            f"Python tool {record.id!r} produced an invalid input schema: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise RuntimeError(f"Python tool {record.id!r} produced an invalid input schema.")
    contract = {
        "id": record.id,
        "entrypoint": entrypoint,
        "description": description,
        "input_schema": schema,
        "implementation": _implementation_source(value),
    }
    try:
        canonical = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Python tool {record.id!r} produced an invalid input schema: {exc}"
        ) from exc
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ResolvedPythonTool(
        record=record,
        tool=value,
        schema=schema,
        fingerprint=fingerprint,
    )


def _project_root() -> pathlib.Path:
    configured = os.getenv("MASON_PROJECT_ROOT")
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if (root / "agent.toml").is_file():
            return root
        raise RuntimeError(f"MASON_PROJECT_ROOT has no agent.toml: {root}")
    for candidate in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        if (candidate / "agent.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate agent.toml; set MASON_PROJECT_ROOT to the project root.")


def _python_records() -> tuple[ToolSpec, ...]:
    project = AgentProject.load(_project_root())
    if project.framework != "langgraph":
        raise RuntimeError(
            f"agent.toml framework {project.framework!r} does not match runtime 'langgraph'."
        )
    return project.python_tools()


def resolve_python_tools() -> tuple[ResolvedPythonTool, ...]:
    """Resolve every Python entry point explicitly activated in ``agent.toml``."""
    resolved = tuple(_resolve(record) for record in _python_records())
    names = [_normalized_name(item.tool.name) for item in resolved]
    if len(names) != len(set(names)):
        raise RuntimeError("Python tool runtime name collision after hyphen normalization.")
    return resolved


def python_tools() -> list[BaseTool]:
    """Return model-ready Python tools in manifest declaration order."""
    return [item.tool for item in resolve_python_tools()]


def _declared_tool(tool_id: str) -> ResolvedPythonTool:
    for record in _python_records():
        if record.id == tool_id:
            return _resolve(record)
    raise RuntimeError(f"Python tool {tool_id!r} is not declared in agent.toml.")


def _tool_contract(item: ResolvedPythonTool) -> dict[str, object]:
    return {
        "id": item.record.id,
        "entrypoint": item.record.source.entrypoint,
        "description": item.tool.description.strip(),
        "input_schema": item.schema,
        "fingerprint": item.fingerprint,
    }


def describe_python_tool(tool_id: str) -> dict[str, object]:
    """Return the stable model-facing contract for one declared tool."""
    return _tool_contract(_declared_tool(tool_id))


def invoke_python_tool(tool_id: str, arguments: dict[str, object]) -> object:
    """Invoke one declared tool and require a JSON-serializable result."""
    item = _declared_tool(tool_id)
    try:
        if getattr(item.tool, "func", None) is None and getattr(item.tool, "coroutine", None):
            result = asyncio.run(item.tool.ainvoke(arguments))
        else:
            result = item.tool.invoke(arguments)
    except Exception as exc:
        raise RuntimeError(f"Tool {tool_id!r} failed: {exc}") from exc
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Tool {tool_id!r} returned a result that is not JSON serializable: {exc}"
        ) from exc
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or invoke manifest-declared Python tools."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("tool_id", nargs="?")
    check.add_argument("--result-path", required=True)

    run = commands.add_parser("run")
    run.add_argument("tool_id")
    run.add_argument("--input", required=True)
    run.add_argument("--result-path", required=True)
    return parser


def _check_payload(tool_id: str | None) -> dict[str, object]:
    resolved = resolve_python_tools() if tool_id is None else (_declared_tool(tool_id),)
    return {
        "schema_version": 1,
        "ok": True,
        "tools": [_tool_contract(item) for item in resolved],
    }


def _run_payload(tool_id: str, input_json: str) -> dict[str, object]:
    try:
        arguments = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Tool input must be valid JSON: {exc}.") from exc
    if not isinstance(arguments, dict):
        raise RuntimeError("Tool input must be a JSON object.")
    return {
        "schema_version": 1,
        "ok": True,
        "tool": tool_id,
        "result": invoke_python_tool(tool_id, arguments),
    }


def main(argv: list[str] | None = None) -> int:
    """Write machine control data to ``--result-path`` and leave stdout/stderr for logs."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "check":
            payload = _check_payload(arguments.tool_id)
        else:
            payload = _run_payload(arguments.tool_id, arguments.input)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - child errors cross the process boundary as JSON
        payload = {"schema_version": 1, "ok": False, "error": str(exc)}
        exit_code = 1
    pathlib.Path(arguments.result_path).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
