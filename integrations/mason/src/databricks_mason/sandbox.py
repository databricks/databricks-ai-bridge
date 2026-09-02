"""``mason add-sandbox`` — attach a downscoped ``system.ai.sandbox`` MCP server."""

from __future__ import annotations

import ast
import json
import pathlib
import stat
import tempfile
import textwrap
from collections.abc import Sequence
from importlib import resources
from typing import Protocol

import click

from databricks_mason import render
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import load_project_metadata

_BEGIN_MARKER = "# BEGIN: mason add-sandbox"
_END_MARKER = "# END: mason add-sandbox"
_DOWNSCOPE_PLACEHOLDER = "# __MASON_SANDBOX_DOWNSCOPE__"
_SERVER_FACTORY_CALL = "_build_sandbox_mcp_server()"
_OPENAI_TEMPLATE_NAME = "sandbox_mcp.py"
_LANGGRAPH_TEMPLATE_NAME = "sandbox_mcp_langgraph.py"


def _validate_uc_name(value: str, resource_type: str) -> str:
    if (
        len(value.split(".")) != 3
        or any(not part for part in value.split("."))
        or any(character.isspace() for character in value)
    ):
        raise AgentCliError(
            f"Invalid {resource_type} scope '{value}'.",
            hint=f"Use a three-part Unity Catalog name: catalog.schema.{resource_type}.",
        )
    return value


def _parse_scopes(scopes: Sequence[str], permission: str) -> dict[str, list[dict[str, str]]]:
    """Convert CLI scope values to the MCP ``_meta.downscope`` wire shape."""
    parsed: dict[str, list[dict[str, str]]] = {
        "volumes": [],
        "tables": [],
        "workspace_paths": [],
    }
    seen: set[tuple[str, str]] = set()

    for original in scopes:
        value = original.strip()
        if not value:
            raise AgentCliError("Sandbox scopes cannot be empty.")

        prefix, separator, remainder = value.partition(":")
        explicit_type = prefix if separator and prefix in {"volume", "table", "workspace"} else None
        if explicit_type:
            value = remainder.strip()

        if explicit_type == "workspace" or (
            explicit_type is None and value.startswith("/Workspace/")
        ):
            if not value.startswith("/Workspace/") or any(
                character in value for character in ("\r", "\n", "\t")
            ):
                raise AgentCliError(
                    f"Invalid workspace scope '{value}'.",
                    hint="Workspace paths must begin with /Workspace/ and cannot contain tabs or newlines.",
                )
            key, field = "workspace_paths", "path"
        else:
            resource_type = explicit_type or "volume"
            if resource_type == "workspace":  # Handled above; keeps the type narrow below.
                raise AssertionError("unreachable")
            value = _validate_uc_name(value, resource_type)
            key, field = ("tables", "name") if resource_type == "table" else ("volumes", "name")

        identity = (key, value)
        if identity not in seen:
            parsed[key].append({field: value, "permission": permission})
            seen.add(identity)

    return {key: values for key, values in parsed.items() if values}


def _offset(lines: list[str], line: int, column: int) -> int:
    try:
        line_prefix = lines[line - 1].encode("utf-8")[:column].decode("utf-8")
    except (IndexError, UnicodeDecodeError) as exc:
        raise AgentCliError(
            "Could not locate the generated Python syntax in agent/mcps.py."
        ) from exc
    return sum(len(part) for part in lines[: line - 1]) + len(line_prefix)


def _end_position(node: ast.expr | ast.stmt) -> tuple[int, int]:
    line = node.end_lineno
    column = node.end_col_offset
    if line is None or column is None:
        raise AgentCliError("Could not locate the generated Python syntax in agent/mcps.py.")
    return line, column


def _find_server_list(tree: ast.Module) -> tuple[ast.FunctionDef, ast.List]:
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_mcp_servers"
    ]
    if len(functions) != 1:
        raise AgentCliError(
            "Could not find one build_mcp_servers() function in agent/mcps.py.",
            hint="Run this command from a Mason agent project generated from a supported template.",
        )

    direct_returns = [node for node in functions[0].body if isinstance(node, ast.Return)]
    all_returns = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Return)]
    if (
        len(direct_returns) != 1
        or len(all_returns) != 1
        or not isinstance(direct_returns[0].value, ast.List)
    ):
        raise AgentCliError(
            "build_mcp_servers() must directly return a list before Mason can add the sandbox.",
            hint="Change the function to `return [...]`, then retry.",
        )
    return functions[0], direct_returns[0].value


def _append_server_to_list(source: str) -> str:
    """Append the generated server factory while preserving the rest of ``mcps.py``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AgentCliError(
            f"Could not parse agent/mcps.py: {exc.msg} (line {exc.lineno})."
        ) from exc

    function, server_list = _find_server_list(tree)
    lines = source.splitlines(keepends=True)
    start = _offset(lines, server_list.lineno, server_list.col_offset)
    list_end_line, list_end_column = _end_position(server_list)
    end = _offset(lines, list_end_line, list_end_column)
    item_indent = " " * (function.col_offset + 8)
    closing_indent = " " * (function.col_offset + 4)

    if not server_list.elts:
        replacement = f"[\n{item_indent}{_SERVER_FACTORY_CALL},\n{closing_indent}]"
        return source[:start] + replacement + source[end:]

    if server_list.lineno == server_list.end_lineno:
        existing = []
        for element in server_list.elts:
            segment = ast.get_source_segment(source, element)
            if segment is None:
                raise AgentCliError("Could not preserve the existing MCP server list.")
            existing.append(textwrap.indent(textwrap.dedent(segment), item_indent))
        replacement = (
            "[\n"
            + ",\n".join(existing)
            + f",\n{item_indent}{_SERVER_FACTORY_CALL},\n{closing_indent}]"
        )
        return source[:start] + replacement + source[end:]

    closing_line_start = _offset(lines, list_end_line, 0)
    closing_prefix = source[closing_line_start : end - 1]
    if closing_prefix.strip():
        raise AgentCliError(
            "The closing bracket of build_mcp_servers() must be on its own line.",
            hint="Format the returned list as a standard multi-line Python list, then retry.",
        )

    last = server_list.elts[-1]
    last_end_line, last_end_column = _end_position(last)
    last_end = _offset(lines, last_end_line, last_end_column)
    suffix = source[last_end:closing_line_start]
    comma = "" if suffix.lstrip().startswith(",") else ","
    with_item = (
        source[:last_end]
        + comma
        + source[last_end:closing_line_start]
        + f"{item_indent}{_SERVER_FACTORY_CALL},\n"
        + source[closing_line_start:]
    )
    return with_item


def _read_template(template_name: str = _OPENAI_TEMPLATE_NAME) -> str:
    try:
        return (
            resources.files("databricks_mason")
            .joinpath("templates")
            .joinpath(template_name)
            .read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise AgentCliError("Could not read the packaged sandbox MCP template.") from exc


def _insert_imports(
    source: str,
    template: str,
    *,
    required_import: tuple[str, str] = ("agents.mcp", "MCPServer"),
) -> str:
    tree = ast.parse(source)
    try:
        template_tree = ast.parse(template)
    except SyntaxError as exc:
        raise AgentCliError("The packaged sandbox MCP template is invalid.") from exc
    lines = source.splitlines(keepends=True)

    def has_from_import(module: str, name: str) -> bool:
        return any(
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any((alias.asname or alias.name) == name for alias in node.names)
            for node in tree.body
        )

    template_imports = [
        node
        for node in template_tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    if not template_imports or any(
        node.level != 0 or len(node.names) != 1 or node.names[0].asname is not None
        for node in template_imports
    ):
        raise AgentCliError("The packaged sandbox MCP template has invalid imports.")

    missing_imports: list[tuple[str, str]] = []
    for node in template_imports:
        module = node.module
        if module is None:
            raise AgentCliError("The packaged sandbox MCP template has invalid imports.")
        alias = node.names[0]
        if not has_from_import(module, alias.name):
            segment = ast.get_source_segment(template, node)
            if segment is None:
                raise AgentCliError("The packaged sandbox MCP template has invalid imports.")
            missing_imports.append((module, segment))

    required_module, required_name = required_import
    protocol_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == required_module
        and any((alias.asname or alias.name) == required_name for alias in node.names)
    ]
    if len(protocol_imports) != 1:
        raise AgentCliError(
            f"agent/mcps.py must import {required_name} from {required_module} before Mason can "
            "add the sandbox.",
            hint="Start from the Mason agent template, then retry.",
        )

    insert_after: ast.stmt | None = None
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_after = tree.body[0]

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_after = node
        elif node is not insert_after and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            break

    with_runtime_imports = source
    runtime_imports = sorted(
        ((module, segment) for module, segment in missing_imports if module != "typing"),
        key=lambda item: item[0],
    )
    if runtime_imports:
        import_start = _offset(lines, protocol_imports[0].lineno, 0)
        import_end_line, import_end_column = _end_position(protocol_imports[0])
        import_end = _offset(lines, import_end_line, import_end_column)
        before = [segment for module, segment in runtime_imports if module < required_module]
        after = [segment for module, segment in runtime_imports if module > required_module]
        with_runtime_imports = (
            source[:import_start]
            + ("\n".join(before) + "\n" if before else "")
            + source[import_start:import_end]
            + ("\n" + "\n".join(after) if after else "")
            + source[import_end:]
        )

    typing_imports = [segment for module, segment in missing_imports if module == "typing"]
    if not typing_imports:
        return with_runtime_imports

    if insert_after is not None:
        end_line, end_column = _end_position(insert_after)
        insert_at = _offset(lines, end_line, end_column)
        following = with_runtime_imports[insert_at:].lstrip("\n")
        return (
            with_runtime_imports[:insert_at]
            + "\n\n"
            + "\n".join(typing_imports)
            + "\n\n"
            + following
        )
    return "\n".join(typing_imports) + "\n\n" + with_runtime_imports.lstrip("\n")


def _format_policy(policy: dict[str, list[dict[str, str]]]) -> str:
    lines = ["_SANDBOX_DOWNSCOPE: dict[str, list[dict[str, str]]] = {"]
    for key, values in policy.items():
        lines.append(f'    "{key}": [')
        for value in values:
            fields = ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in value.items())
            lines.append(f"        {{{fields}}},")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def _existing_policy(
    source: str,
    *,
    template_name: str = _OPENAI_TEMPLATE_NAME,
) -> dict[str, list[dict[str, str]]] | None:
    begin_count = source.count(_BEGIN_MARKER)
    end_count = source.count(_END_MARKER)
    if begin_count == 0 and end_count == 0:
        return None
    if (
        begin_count != 1
        or end_count != 1
        or source.index(_BEGIN_MARKER) > source.index(_END_MARKER)
    ):
        raise AgentCliError(
            "The existing mason add-sandbox block has invalid markers.",
            hint="Repair or remove the generated block in agent/mcps.py, then retry.",
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AgentCliError(
            f"Could not parse agent/mcps.py: {exc.msg} (line {exc.lineno})."
        ) from exc

    _, server_list = _find_server_list(tree)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_SANDBOX_DOWNSCOPE"
        and node.value is not None
    ]
    if len(assignments) != 1:
        raise AgentCliError(
            "The existing mason add-sandbox block is incomplete.",
            hint="Repair or remove the generated block in agent/mcps.py, then retry.",
        )

    assignment_value = assignments[0].value
    if assignment_value is None:
        raise AgentCliError("The existing sandbox downscope has no policy value.")
    try:
        raw_policy = ast.literal_eval(assignment_value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise AgentCliError(
            "The existing sandbox downscope is not a fixed literal policy.",
            hint="Repair or remove the generated block in agent/mcps.py, then retry.",
        ) from exc

    if not isinstance(raw_policy, dict):
        raise AgentCliError("The existing sandbox downscope is not a valid policy object.")

    policy: dict[str, list[dict[str, str]]] = {}
    for key, entries in raw_policy.items():
        if not isinstance(key, str) or not isinstance(entries, list):
            raise AgentCliError("The existing sandbox downscope has an invalid resource list.")
        parsed_entries: list[dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not all(
                isinstance(field, str) and isinstance(value, str) for field, value in entry.items()
            ):
                raise AgentCliError("The existing sandbox downscope has an invalid resource entry.")
            parsed_entries.append(entry)
        policy[key] = parsed_entries

    expected_fields = {
        "volumes": "name",
        "tables": "name",
        "workspace_paths": "path",
    }
    if not policy or any(key not in expected_fields for key in policy):
        raise AgentCliError("The existing sandbox downscope has an invalid resource type.")
    for resource_type, entries in policy.items():
        expected_field = expected_fields[resource_type]
        if not entries:
            raise AgentCliError("The existing sandbox downscope has an empty resource list.")
        for entry in entries:
            if set(entry) != {expected_field, "permission"}:
                raise AgentCliError("The existing sandbox downscope has an invalid resource entry.")
            if entry["permission"] not in {"read_only", "read_write"}:
                raise AgentCliError("The existing sandbox downscope has an invalid permission.")
            if resource_type == "workspace_paths":
                path = entry[expected_field]
                if not path.startswith("/Workspace/") or any(
                    character in path for character in ("\r", "\n", "\t")
                ):
                    raise AgentCliError(
                        "The existing sandbox downscope has an invalid workspace path."
                    )
            else:
                _validate_uc_name(entry[expected_field], resource_type.removesuffix("s"))

    block_start = source.index(_BEGIN_MARKER)
    block_end = source.index(_END_MARKER) + len(_END_MARKER)
    try:
        actual_tree = ast.parse(source[block_start:block_end] + "\n")
        expected_tree = ast.parse(_generated_block(policy, template_name=template_name))
    except SyntaxError as exc:
        raise AgentCliError("The existing mason add-sandbox block is incomplete.") from exc
    if ast.dump(actual_tree, include_attributes=False) != ast.dump(
        expected_tree, include_attributes=False
    ):
        raise AgentCliError(
            "The existing mason add-sandbox block does not match the generated security wrapper.",
            hint="Repair or remove the generated block in agent/mcps.py, then retry.",
        )

    factory_calls = [
        element
        for element in server_list.elts
        if isinstance(element, ast.Call)
        and isinstance(element.func, ast.Name)
        and element.func.id == "_build_sandbox_mcp_server"
    ]
    if len(factory_calls) != 1 or factory_calls[0].args or factory_calls[0].keywords:
        raise AgentCliError(
            "build_mcp_servers() must register exactly one generated sandbox server.",
            hint="Repair the generated factory entry in agent/mcps.py, then retry.",
        )
    return policy


def _policy_signature(
    policy: dict[str, list[dict[str, str]]],
) -> frozenset[tuple[str, str, str]]:
    return frozenset(
        (resource_type, entry.get("name", entry.get("path", "")), entry.get("permission", ""))
        for resource_type, entries in policy.items()
        for entry in entries
    )


def _policy_cli_values(
    policy: dict[str, list[dict[str, str]]],
) -> tuple[list[str], str]:
    scopes: list[str] = []
    permissions: set[str] = set()
    for resource_type, entries in policy.items():
        for entry in entries:
            value = entry.get("path") or entry.get("name", "")
            scopes.append(f"table:{value}" if resource_type == "tables" else value)
            permissions.add(entry.get("permission", ""))
    if len(permissions) != 1:
        raise AgentCliError("The existing sandbox downscope has inconsistent permissions.")
    return scopes, permissions.pop()


def _write_text_atomic(target: pathlib.Path, content: str) -> None:
    temporary: pathlib.Path | None = None
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as output:
            temporary = pathlib.Path(output.name)
            output.write(content)
        temporary.chmod(mode)
        temporary.replace(target)
        temporary = None
    except OSError as exc:
        raise AgentCliError(f"Could not update {target}: {exc}.") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _generated_block(
    policy: dict[str, list[dict[str, str]]],
    template: str | None = None,
    *,
    template_name: str = _OPENAI_TEMPLATE_NAME,
) -> str:
    template = template if template is not None else _read_template(template_name)
    try:
        tree = ast.parse(template)
    except SyntaxError as exc:
        raise AgentCliError("The packaged sandbox MCP template is invalid.") from exc
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_SANDBOX_DOWNSCOPE"
    ]
    lines = template.splitlines(keepends=True)
    if (
        len(assignments) != 1
        or assignments[0].lineno != assignments[0].end_lineno
        or _DOWNSCOPE_PLACEHOLDER not in lines[assignments[0].lineno - 1]
    ):
        raise AgentCliError("The packaged sandbox MCP template is invalid.")
    line_start = _offset(lines, assignments[0].lineno, 0)
    line_end = line_start + len(lines[assignments[0].lineno - 1])
    rendered = template[:line_start] + _format_policy(policy) + "\n" + template[line_end:]
    if rendered.count(_BEGIN_MARKER) != 1 or rendered.count(_END_MARKER) != 1:
        raise AgentCliError("The packaged sandbox MCP template has invalid markers.")
    block_start = rendered.index(_BEGIN_MARKER)
    block_end = rendered.index(_END_MARKER) + len(_END_MARKER)
    return rendered[block_start:block_end] + "\n"


def _add_sandbox_to_source(
    source: str,
    policy: dict[str, list[dict[str, str]]],
    *,
    template_name: str = _OPENAI_TEMPLATE_NAME,
    required_import: tuple[str, str] = ("agents.mcp", "MCPServer"),
) -> str:
    template = _read_template(template_name)
    with_server = _append_server_to_list(source)
    with_imports = _insert_imports(with_server, template, required_import=required_import)
    generated = with_imports.rstrip() + "\n\n\n" + _generated_block(policy, template)
    try:
        ast.parse(generated)
    except SyntaxError as exc:
        raise AgentCliError(
            f"Refusing to write invalid generated Python: {exc.msg} (line {exc.lineno})."
        ) from exc
    return generated


def _read_source(target: pathlib.Path) -> str:
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentCliError(f"Could not read {target}: {exc}.") from exc


def _effective_policy(
    existing_policy: dict[str, list[dict[str, str]]] | None,
    requested_policy: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    if existing_policy is not None and _policy_signature(existing_policy) != _policy_signature(
        requested_policy
    ):
        raise AgentCliError(
            "system.ai.sandbox is already configured with a different downscope.",
            hint="Edit or remove the generated add-sandbox block in agent/mcps.py before changing scopes.",
        )
    return existing_policy or requested_policy


def _configure_langgraph_runtime(source: str) -> str:
    """Pass the generated fixed-meta interceptor into the LangGraph MCP client."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AgentCliError(
            f"Could not parse agent/mason/mcp_runtime.py: {exc.msg} (line {exc.lineno})."
        ) from exc

    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "agent.mcps"
        and any(alias.name == "build_mcp_servers" for alias in node.names)
    ]
    if len(imports) != 1 or imports[0].lineno != imports[0].end_lineno:
        raise AgentCliError(
            "agent/mason/mcp_runtime.py must import build_mcp_servers from agent.mcps.",
            hint="Start from the Mason LangGraph agent template, then retry.",
        )

    configured = any(alias.name == "_sandbox_tool_interceptor" for alias in imports[0].names)
    with_import = source
    if not configured:
        lines = source.splitlines(keepends=True)
        import_start = _offset(lines, imports[0].lineno, imports[0].col_offset)
        import_end_line, import_end_column = _end_position(imports[0])
        import_end = _offset(lines, import_end_line, import_end_column)
        if any(alias.asname is not None for alias in imports[0].names):
            raise AgentCliError(
                "agent/mason/mcp_runtime.py must import build_mcp_servers without an alias."
            )
        names = sorted({alias.name for alias in imports[0].names} | {"_sandbox_tool_interceptor"})
        replacement = f"from agent.mcps import {', '.join(names)}"
        with_import = source[:import_start] + replacement + source[import_end:]

    try:
        updated_tree = ast.parse(with_import)
    except SyntaxError as exc:
        raise AgentCliError("Could not update the LangGraph MCP runtime import.") from exc
    client_calls = [
        node
        for node in ast.walk(updated_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DatabricksMultiServerMCPClient"
    ]
    if len(client_calls) != 1:
        raise AgentCliError(
            "Could not find one DatabricksMultiServerMCPClient() call in "
            "agent/mason/mcp_runtime.py.",
            hint="Start from the Mason LangGraph agent template, then retry.",
        )
    client_call = client_calls[0]
    interceptor_keywords = [
        keyword for keyword in client_call.keywords if keyword.arg == "tool_interceptors"
    ]
    if interceptor_keywords:
        value = interceptor_keywords[0].value
        if (
            len(interceptor_keywords) != 1
            or not isinstance(value, ast.List)
            or len(value.elts) != 1
            or not isinstance(value.elts[0], ast.Name)
            or value.elts[0].id != "_sandbox_tool_interceptor"
        ):
            raise AgentCliError(
                "The LangGraph MCP runtime already has a different tool_interceptors setting."
            )
        return with_import

    lines = with_import.splitlines(keepends=True)
    call_end_line, call_end_column = _end_position(client_call)
    call_end = _offset(lines, call_end_line, call_end_column)
    if call_end == 0 or with_import[call_end - 1] != ")":
        raise AgentCliError("Could not update DatabricksMultiServerMCPClient().")
    configured_source = (
        with_import[: call_end - 1]
        + ", tool_interceptors=[_sandbox_tool_interceptor]"
        + with_import[call_end - 1 :]
    )
    try:
        ast.parse(configured_source)
    except SyntaxError as exc:
        raise AgentCliError("Could not update the LangGraph MCP runtime.") from exc
    return configured_source


class _SandboxAdapter(Protocol):
    def apply(
        self,
        project: pathlib.Path,
        policy: dict[str, list[dict[str, str]]],
    ) -> tuple[pathlib.Path, dict[str, list[dict[str, str]]], bool]: ...


class _OpenAISandboxAdapter:
    def apply(
        self,
        project: pathlib.Path,
        policy: dict[str, list[dict[str, str]]],
    ) -> tuple[pathlib.Path, dict[str, list[dict[str, str]]], bool]:
        target = project / "agent" / "mcps.py"
        current = _read_source(target)
        existing_policy = _existing_policy(current, template_name=_OPENAI_TEMPLATE_NAME)
        effective_policy = _effective_policy(existing_policy, policy)
        if existing_policy is None:
            _write_text_atomic(
                target,
                _add_sandbox_to_source(
                    current,
                    policy,
                    template_name=_OPENAI_TEMPLATE_NAME,
                    required_import=("agents.mcp", "MCPServer"),
                ),
            )
        return target, effective_policy, existing_policy is None


class _LangGraphSandboxAdapter:
    def apply(
        self,
        project: pathlib.Path,
        policy: dict[str, list[dict[str, str]]],
    ) -> tuple[pathlib.Path, dict[str, list[dict[str, str]]], bool]:
        target = project / "agent" / "mcps.py"
        runtime = project / "agent" / "mason" / "mcp_runtime.py"
        if not runtime.is_file():
            raise AgentCliError(
                f"Could not find {runtime}.",
                hint="Start from the Mason LangGraph agent template, then retry.",
            )

        current = _read_source(target)
        runtime_source = _read_source(runtime)
        existing_policy = _existing_policy(current, template_name=_LANGGRAPH_TEMPLATE_NAME)
        effective_policy = _effective_policy(existing_policy, policy)
        configured_runtime = _configure_langgraph_runtime(runtime_source)
        if existing_policy is not None:
            if configured_runtime != runtime_source:
                raise AgentCliError(
                    "The existing LangGraph sandbox configuration is missing its runtime adapter.",
                    hint="Repair or remove the generated add-sandbox block, then retry.",
                )
            return target, effective_policy, False

        generated = _add_sandbox_to_source(
            current,
            policy,
            template_name=_LANGGRAPH_TEMPLATE_NAME,
            required_import=("databricks_langchain", "DatabricksMCPServer"),
        )
        _write_text_atomic(target, generated)
        try:
            _write_text_atomic(runtime, configured_runtime)
        except AgentCliError:
            _write_text_atomic(target, current)
            raise
        return target, effective_policy, True


_SANDBOX_ADAPTERS: dict[str, _SandboxAdapter] = {
    "langgraph": _LangGraphSandboxAdapter(),
    "openai": _OpenAISandboxAdapter(),
}


@click.command("add-sandbox")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    required=True,
    help=(
        "Allowed volume (catalog.schema.volume) or /Workspace/ path. Repeat for multiple scopes; "
        "use table:catalog.schema.table for a table."
    ),
)
@click.option(
    "--permission",
    type=click.Choice(["read_only", "read_write"]),
    default="read_only",
    show_default=True,
)
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    default=pathlib.Path("."),
    show_default=True,
    help="Mason agent project containing agent/mcps.py.",
)
@click.option(
    "--framework",
    type=click.Choice(sorted(_SANDBOX_ADAPTERS)),
    default=None,
    help="Override framework detection for projects created before Mason metadata existed.",
)
@click.pass_obj
def add_sandbox(
    obj,
    scopes: tuple[str, ...],
    permission: str,
    source: pathlib.Path,
    framework: str | None,
) -> None:
    """Add the system.ai.sandbox MCP with a fixed downscope to an agent project."""
    project = source.resolve()
    policy = _parse_scopes(scopes, permission)
    target = project / "agent" / "mcps.py"
    if not target.is_file():
        raise AgentCliError(
            f"Could not find {target}.",
            hint="Pass --source pointing to a Mason agent project containing agent/mcps.py.",
        )

    metadata = load_project_metadata(project, framework_override=framework)
    adapter = _SANDBOX_ADAPTERS[metadata.framework]
    target, effective_policy, changed = adapter.apply(project, policy)
    effective_scopes, effective_permission = _policy_cli_values(effective_policy)

    status = "added" if changed else "already_configured"
    result = {
        "mcp_server": "system.ai.sandbox",
        "path": str(target),
        "permission": effective_permission,
        "scopes": effective_scopes,
        "status": status,
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(result)
    elif changed:
        click.echo(f"Added system.ai.sandbox with fixed downscoping to {target}")
    else:
        click.echo(f"system.ai.sandbox is already configured in {target}")
