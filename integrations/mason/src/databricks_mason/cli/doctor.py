"""`mason doctor` — inspect an existing agent repository without running it.

The command deliberately uses bounded, static reads.  In particular, it never imports target
source: Python files are parsed as ASTs solely to find the Mason runtime wiring that migration adds.
"""

from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import tokenize
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import click
import tomli
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError

_SUPPORTED_FRAMEWORKS = ("langgraph", "openai")
_MAX_CONFIG_BYTES = 512 * 1024
_MAX_SOURCE_BYTES = 256 * 1024
_MAX_SOURCE_FILES = 1_000
_MAX_SOURCE_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_SCAN_DIRECTORIES = 2_000
_MAX_SCAN_ENTRIES = 20_000
_MAX_HUMAN_PATH_CHARS = 240
_MAX_HUMAN_DETAIL_CHARS = 500
_IGNORED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        "build",
        "dist",
        "env",
        "example",
        "examples",
        "mason-migrate",
        "node_modules",
        "old",
        "site-packages",
        "stale",
        "venv",
    }
)
_TEST_DIRECTORIES = frozenset({"test", "tests", "testdata"})
# Keep this synchronized with the public adapter calls exercised by the generated framework
# templates. Prefix matching is intentionally forbidden: a made-up symbol is not migration proof.
_FRAMEWORK_ADAPTER_CALLS = {
    "langgraph": frozenset(
        {
            "databricks_mason.langgraph.checkpointer",
            "databricks_mason.langgraph.configure_tracing",
            "databricks_mason.langgraph.genie_tools",
            "databricks_mason.langgraph.mcp_tools",
            "databricks_mason.langgraph.memory_tools",
            "databricks_mason.langgraph.start_trace",
            "databricks_mason.langgraph.thread_config",
        }
    ),
    "openai": frozenset(
        {
            "databricks_mason.openai.configure_tracing",
            "databricks_mason.openai.genie_tools",
            "databricks_mason.openai.mcp_servers",
            "databricks_mason.openai.memory_tools",
            "databricks_mason.openai.session_store",
            "databricks_mason.openai.start_trace",
        }
    ),
}
_AGENT_APP_EXPORTS = frozenset({"databricks_mason.AgentApp", "databricks_mason.runtime.AgentApp"})


@dataclass(frozen=True)
class DoctorCheck:
    """One stable, machine-readable onboarding check."""

    id: str
    status: str
    detail: str


@dataclass(frozen=True)
class _TomlResult:
    data: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class _SourceEvidence:
    agent_app: pathlib.Path | None
    adapters: dict[str, pathlib.Path]
    parse_failures: int
    unreadable_files: int
    truncated: bool


@dataclass(frozen=True)
class _PythonEvidence:
    agent_app_invoked: bool
    adapters: frozenset[str]


@dataclass(frozen=True)
class _QualifiedSymbol:
    name: str


@dataclass(frozen=True)
class _AgentAppInstance:
    id: int


_SourceSymbol = _QualifiedSymbol | _AgentAppInstance | None


@dataclass
class _LexicalScope:
    parent: _LexicalScope | None
    bindings: dict[str, _SourceSymbol]
    global_names: set[str]
    nonlocal_names: set[str]
    kind: str


def _check(check_id: str, passed: bool, detail: str) -> DoctorCheck:
    return DoctorCheck(check_id, "pass" if passed else "fail", detail)


def _bounded_bytes(
    root: pathlib.Path, path: pathlib.Path, *, description: str, limit: int
) -> tuple[bytes | None, str | None]:
    """Read one in-project regular file, rejecting escapes and oversized evidence."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None, f"{description} is missing at {path}."
    except (OSError, RuntimeError) as exc:
        return None, f"Could not resolve {description} at {path}: {exc}."

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"Refused {description} symlink outside the inspected directory: {path}."
    if not resolved.is_file():
        return None, f"{description} at {path} is not a regular file."

    try:
        with resolved.open("rb") as input_file:
            content = input_file.read(limit + 1)
    except OSError as exc:
        return None, f"Could not read {description} at {path}: {exc}."
    if len(content) > limit:
        return None, f"{description} at {path} exceeds the {limit // 1024} KiB inspection limit."
    if b"\x00" in content:
        return None, f"{description} at {path} appears to be binary."
    return content, None


def _toml_result(root: pathlib.Path, path: pathlib.Path, description: str) -> _TomlResult:
    content, error = _bounded_bytes(root, path, description=description, limit=_MAX_CONFIG_BYTES)
    if error:
        return _TomlResult(None, error)
    try:
        data = tomli.loads(content.decode("utf-8")) if content is not None else None
    except (UnicodeDecodeError, tomli.TOMLDecodeError) as exc:
        return _TomlResult(None, f"Could not parse {description} at {path}: {exc}.")
    if not isinstance(data, dict):
        return _TomlResult(None, f"{description.capitalize()} at {path} must be a TOML table.")
    return _TomlResult(data, None)


def _framework(value: object) -> str | None:
    return value if isinstance(value, str) and value in _SUPPORTED_FRAMEWORKS else None


def _agent_manifest(
    root: pathlib.Path,
) -> tuple[DoctorCheck, DoctorCheck, str | None]:
    path = root / "agent.toml"
    result = _toml_result(root, path, "agent manifest")
    if result.error:
        return (
            _check("agent_manifest", False, result.error),
            _check("mason_server", False, "Cannot verify the server without a valid agent.toml."),
            None,
        )

    assert result.data is not None
    if result.data.get("schema_version") != 1:
        detail = f"{path} must declare schema_version = 1."
        return (
            _check("agent_manifest", False, detail),
            _check("mason_server", False, "Cannot verify the server without a valid agent.toml."),
            None,
        )
    agent = result.data.get("agent")
    if not isinstance(agent, dict):
        detail = f"{path} must declare an [agent] table."
        return (
            _check("agent_manifest", False, detail),
            _check("mason_server", False, "Cannot verify the server without a valid agent.toml."),
            None,
        )
    framework = _framework(agent.get("framework"))
    if framework is None:
        detail = f"{path} must declare agent.framework as langgraph or openai."
        return (
            _check("agent_manifest", False, detail),
            _check("mason_server", False, "Cannot verify the server without a valid agent.toml."),
            None,
        )
    server = agent.get("server")
    if server not in {"mason", "custom"}:
        detail = f"{path} must declare agent.server as mason or custom."
        return (
            _check("agent_manifest", False, detail),
            _check("mason_server", False, "Cannot verify the server without a valid agent.toml."),
            framework,
        )

    server_check = _check(
        "mason_server",
        server == "mason",
        (
            "agent.server is mason."
            if server == "mason"
            else "agent.server is custom; Agent Bricks onboarding requires mason."
        ),
    )
    # The bounded read above protects this authoritative loader from unbounded or escaping input.
    # Reuse the same validation as every manifest-consuming Mason command so malformed tools,
    # stores, deployment names, and other semantic errors cannot produce a false pass.
    try:
        project = AgentProject.load(root)
    except AgentCliError as exc:
        return (
            _check("agent_manifest", False, f"Invalid agent.toml: {exc.message}"),
            server_check,
            framework,
        )
    return (
        _check(
            "agent_manifest",
            True,
            f"Valid schema v1 manifest for {project.framework.value}.",
        ),
        server_check,
        project.framework.value,
    )


def _project_metadata(
    root: pathlib.Path, expected_framework: str | None
) -> tuple[DoctorCheck, str | None]:
    path = root / ".mason" / "project.toml"
    result = _toml_result(root, path, "Mason project config")
    if result.error:
        return _check("project_metadata", False, result.error), None

    assert result.data is not None
    framework = _framework(result.data.get("framework"))
    if result.data.get("schema_version") != 1:
        return _check(
            "project_metadata", False, f"{path} must declare schema_version = 1."
        ), framework
    if framework is None:
        return _check(
            "project_metadata", False, f"{path} must declare framework as langgraph or openai."
        ), None
    template = result.data.get("template")
    if not isinstance(template, str) or not template.strip():
        return _check(
            "project_metadata", False, f"{path} must declare a non-empty template."
        ), framework
    if expected_framework is not None and framework != expected_framework:
        return _check(
            "project_metadata",
            False,
            f"Framework {framework!r} does not match agent.toml ({expected_framework!r}).",
        ), framework
    return _check(
        "project_metadata", True, f"Valid schema v1 metadata for {framework} ({template})."
    ), framework


def _project_dependencies(
    root: pathlib.Path,
) -> tuple[list[Requirement] | None, str | None, set[str]]:
    path = root / "pyproject.toml"
    result = _toml_result(root, path, "pyproject")
    if result.error:
        return None, result.error, set()
    assert result.data is not None
    project = result.data.get("project")
    if not isinstance(project, dict):
        return None, f"{path} must declare a [project] table.", set()
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(requirement, str) for requirement in dependencies
    ):
        return None, f"{path} [project].dependencies must be an array of strings.", set()

    parsed: list[Requirement] = []
    for requirement_text in dependencies:
        try:
            parsed.append(Requirement(requirement_text))
        except InvalidRequirement as exc:
            return (
                None,
                f"{path} contains an invalid PEP 508 requirement {requirement_text!r}: {exc}.",
                set(),
            )

    inferred: set[str] = set()
    for requirement in parsed:
        normalized = canonicalize_name(requirement.name)
        if normalized in {"langgraph", "databricks-langchain"}:
            inferred.add("langgraph")
        if normalized == "openai-agents":
            inferred.add("openai")
        if normalized == "databricks-mason":
            extras = {str(canonicalize_name(extra)) for extra in requirement.extras}
            inferred.update(extras & set(_SUPPORTED_FRAMEWORKS))
    return parsed, None, inferred


def _dependency_check(
    dependencies: list[Requirement] | None,
    error: str | None,
    framework: str | None,
) -> DoctorCheck:
    if error:
        return _check("mason_dependency", False, error)
    if framework is None:
        return _check(
            "mason_dependency",
            False,
            "Could not determine the framework needed for the databricks-mason extra.",
        )
    assert dependencies is not None
    mason_requirements: list[tuple[Requirement, set[str]]] = []
    for requirement in dependencies:
        if canonicalize_name(requirement.name) != "databricks-mason":
            continue
        extras = {str(canonicalize_name(extra)) for extra in requirement.extras}
        mason_requirements.append((requirement, extras))
    if not mason_requirements:
        return _check(
            "mason_dependency",
            False,
            f"pyproject.toml needs a databricks-mason[{framework}] project dependency.",
        )
    if not any(framework in extras for _, extras in mason_requirements):
        found = ", ".join(str(requirement) for requirement, _ in mason_requirements)
        return _check(
            "mason_dependency",
            False,
            f"Mason dependency lacks the {framework!r} extra (found: {found}).",
        )
    return _check("mason_dependency", True, f"Project depends on databricks-mason[{framework}].")


def _app_command(root: pathlib.Path) -> DoctorCheck:
    path = root / "app.yaml"
    content, error = _bounded_bytes(
        root, path, description="Databricks Apps manifest", limit=_MAX_CONFIG_BYTES
    )
    if error:
        return _check("app_command", False, error)
    try:
        data = yaml.safe_load(content.decode("utf-8")) if content is not None else None
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return _check("app_command", False, f"Could not parse app.yaml at {path}: {exc}.")
    if not isinstance(data, dict):
        return _check("app_command", False, f"{path} must contain a YAML object.")
    command = data.get("command")
    valid_string = isinstance(command, str) and bool(command.strip())
    valid_list = (
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and part.strip() for part in command)
    )
    if not (valid_string or valid_list):
        return _check(
            "app_command", False, f"{path} must declare a non-empty command string or list."
        )
    rendered = command if isinstance(command, str) else " ".join(command)
    normalized = rendered.casefold()
    if "replace_with" in normalized or "# todo: set your run command" in normalized:
        return _check("app_command", False, f"{path} still contains a placeholder command.")
    return _check("app_command", True, "app.yaml declares a non-empty startup command.")


def _ignored_directory(name: str) -> bool:
    normalized = name.casefold()
    return (
        name.startswith(".")
        or normalized in _IGNORED_DIRECTORIES
        or normalized in _TEST_DIRECTORIES
        or normalized.startswith(("test_", "tests_"))
        or normalized.endswith(("_test", "_tests"))
    )


def _ignored_python_file(name: str) -> bool:
    normalized = name.casefold()
    return (
        normalized in {"conftest.py", "test.py", "tests.py"}
        or normalized.startswith(("test_", "tests_"))
        or normalized.endswith(("_test.py", "_tests.py"))
    )


def _expression_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_key(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


class _LocalNameCollector(ast.NodeVisitor):
    """Collect names local to one function without descending into nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self.names.add(imported.asname or imported.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            if imported.name != "*":
                self.names.add(imported.asname or imported.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    }
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


class _SourceAnalyzer:
    """Interpret imports, bindings, and calls in lexical source order."""

    def __init__(self) -> None:
        self.scope = _LexicalScope(None, {}, set(), set(), "module")
        self.agent_app_invoked = False
        self.adapter_calls: set[str] = set()
        self._next_instance_id = 1

    def analyze(self, tree: ast.Module) -> _PythonEvidence:
        self._visit_statements(tree.body)
        return _PythonEvidence(self.agent_app_invoked, frozenset(self.adapter_calls))

    def _visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._visit_statement(statement)

    def _lookup_name(self, name: str) -> tuple[bool, _SourceSymbol]:
        scope = self.scope
        if name in scope.global_names:
            if name in scope.bindings:
                return True, scope.bindings[name]
            while scope.parent is not None:
                scope = scope.parent
            return (True, scope.bindings[name]) if name in scope.bindings else (False, None)
        if name in scope.nonlocal_names:
            if name in scope.bindings:
                return True, scope.bindings[name]
            scope = scope.parent
            while scope is not None:
                if name in scope.bindings:
                    return True, scope.bindings[name]
                scope = scope.parent
            return False, None

        while scope is not None:
            if name in scope.bindings:
                return True, scope.bindings[name]
            scope = scope.parent
        return False, None

    def _lookup_key(self, key: str) -> tuple[bool, _SourceSymbol]:
        if "." not in key:
            return self._lookup_name(key)
        root = key.split(".", 1)[0]
        scope: _LexicalScope | None = self.scope
        while scope is not None:
            if key in scope.bindings:
                return True, scope.bindings[key]
            if root in scope.bindings:
                break
            scope = scope.parent
        return False, None

    def _resolve(self, node: ast.AST) -> _SourceSymbol:
        if isinstance(node, ast.Name):
            found, symbol = self._lookup_name(node.id)
            return symbol if found else None
        if isinstance(node, ast.Attribute):
            if key := _expression_key(node):
                found, symbol = self._lookup_key(key)
                if found:
                    return symbol
            parent = self._resolve(node.value)
            if isinstance(parent, _QualifiedSymbol):
                return _QualifiedSymbol(f"{parent.name}.{node.attr}")
        return None

    def _bound_instance(self, node: ast.AST) -> _AgentAppInstance | None:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        symbol = self._resolve(node)
        return symbol if isinstance(symbol, _AgentAppInstance) else None

    def _bind_target(self, target: ast.AST, symbol: _SourceSymbol) -> None:
        if key := _expression_key(target):
            self.scope.bindings[key] = symbol
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_target(element, None)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, None)

    def _new_agent_app(self) -> _AgentAppInstance:
        instance = _AgentAppInstance(self._next_instance_id)
        self._next_instance_id += 1
        return instance

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        if not node.generators:
            return

        # Python evaluates the outermost iterable in the enclosing scope, then runs the rest in an
        # implicit function scope whose targets shadow outer names throughout the comprehension.
        self._eval_expr(node.generators[0].iter)
        parent = self.scope
        lexical_parent = parent.parent if parent.kind == "class" else parent
        self.scope = _LexicalScope(lexical_parent, {}, set(), set(), "comprehension")
        for generator in node.generators:
            self._bind_target(generator.target, None)

        for index, generator in enumerate(node.generators):
            if index:
                self._eval_expr(generator.iter)
            for condition in generator.ifs:
                self._eval_expr(condition)

        if isinstance(node, ast.DictComp):
            self._eval_expr(node.key)
            self._eval_expr(node.value)
        else:
            self._eval_expr(node.elt)
        self.scope = parent

    def _eval_expr(self, node: ast.AST | None) -> _SourceSymbol:
        if node is None:
            return None
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self._resolve(node)
        if isinstance(node, ast.Call):
            invoke_receiver: _AgentAppInstance | None = None
            if isinstance(node.func, ast.Attribute) and node.func.attr == "invoke":
                invoke_receiver = self._bound_instance(node.func.value)
                self._eval_expr(node.func.value)
                function = None
            else:
                function = self._eval_expr(node.func)

            for argument in node.args:
                self._eval_expr(argument)
            for keyword in node.keywords:
                self._eval_expr(keyword.value)

            if invoke_receiver is not None and (node.args or node.keywords):
                self.agent_app_invoked = True
            if isinstance(function, _QualifiedSymbol):
                if any(function.name in allowed for allowed in _FRAMEWORK_ADAPTER_CALLS.values()):
                    self.adapter_calls.add(function.name)
                if function.name in _AGENT_APP_EXPORTS:
                    return self._new_agent_app()
            return None
        if isinstance(node, ast.NamedExpr):
            value = self._eval_expr(node.value)
            self._bind_target(node.target, value)
            return value
        if isinstance(node, ast.Lambda):
            self._visit_lambda(node)
            return None
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self._visit_comprehension(node)
            return None

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._eval_expr(child)
        return None

    def _visit_decorator(self, decorator: ast.expr) -> None:
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "invoke"
            and self._bound_instance(decorator.value) is not None
        ):
            self.agent_app_invoked = True
        self._eval_expr(decorator)

    def _function_scope(self, arguments: ast.arguments, body: list[ast.stmt]) -> _LexicalScope:
        collector = _LocalNameCollector()
        for statement in body:
            collector.visit(statement)
        local_names = (collector.names | _argument_names(arguments)) - (
            collector.global_names | collector.nonlocal_names
        )
        parent = self.scope.parent if self.scope.kind == "class" else self.scope
        return _LexicalScope(
            parent,
            {name: None for name in local_names},
            collector.global_names,
            collector.nonlocal_names,
            "function",
        )

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, bind_name: bool = True
    ) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            self._eval_expr(default)
        for decorator in node.decorator_list:
            self._visit_decorator(decorator)
        if bind_name:
            self.scope.bindings[node.name] = None

        parent = self.scope
        self.scope = self._function_scope(node.args, node.body)
        self._visit_statements(node.body)
        self.scope = parent

    def _visit_lambda(self, node: ast.Lambda) -> None:
        parent = self.scope
        self.scope = self._function_scope(node.args, [])
        self._eval_expr(node.body)
        self.scope = parent

    def _visit_class(self, node: ast.ClassDef) -> None:
        for expression in [*node.bases, *(keyword.value for keyword in node.keywords)]:
            self._eval_expr(expression)
        for decorator in node.decorator_list:
            self._visit_decorator(decorator)
        self.scope.bindings[node.name] = None

        parent = self.scope
        self.scope = _LexicalScope(parent, {}, set(), set(), "class")
        self._visit_statements(node.body)
        self.scope = parent

    def _visit_import(self, node: ast.Import) -> None:
        for imported in node.names:
            bound_name = imported.asname or imported.name.split(".", 1)[0]
            qualified = imported.name if imported.asname else imported.name.split(".", 1)[0]
            self.scope.bindings[bound_name] = _QualifiedSymbol(qualified)

    def _visit_import_from(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            if imported.name == "*":
                continue
            bound_name = imported.asname or imported.name
            qualified = (
                f"{node.module}.{imported.name}" if node.level == 0 and node.module else None
            )
            self.scope.bindings[bound_name] = (
                _QualifiedSymbol(qualified) if qualified is not None else None
            )

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self._eval_expr(node.iter)
        self._bind_target(node.target, None)
        self._visit_statements(node.body)
        self._visit_statements(node.orelse)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self._eval_expr(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars, None)
        self._visit_statements(node.body)

    def _visit_match(self, node: ast.Match) -> None:
        self._eval_expr(node.subject)
        for case in node.cases:
            captures = _LocalNameCollector()
            captures.visit(case.pattern)
            for name in captures.names:
                self.scope.bindings[name] = None
            for pattern_node in ast.walk(case.pattern):
                if isinstance(pattern_node, ast.MatchValue):
                    self._eval_expr(pattern_node.value)
                elif isinstance(pattern_node, ast.MatchClass):
                    self._eval_expr(pattern_node.cls)
                elif isinstance(pattern_node, ast.MatchMapping):
                    for key in pattern_node.keys:
                        self._eval_expr(key)
            self._eval_expr(case.guard)
            self._visit_statements(case.body)

    def _visit_statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Import):
            self._visit_import(node)
            return
        if isinstance(node, ast.ImportFrom):
            self._visit_import_from(node)
            return
        if isinstance(node, ast.Assign):
            value = self._eval_expr(node.value)
            for target in node.targets:
                self._bind_target(target, value)
            return
        if isinstance(node, ast.AnnAssign):
            self._eval_expr(node.annotation)
            value = self._eval_expr(node.value)
            self._bind_target(node.target, value)
            return
        if isinstance(node, ast.AugAssign):
            self._eval_expr(node.target)
            self._eval_expr(node.value)
            self._bind_target(node.target, None)
            return
        if isinstance(node, ast.Expr):
            self._eval_expr(node.value)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(node)
            return
        if isinstance(node, ast.ClassDef):
            self._visit_class(node)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self._visit_for(node)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            self._visit_with(node)
            return
        if isinstance(node, ast.Match):
            self._visit_match(node)
            return
        if isinstance(node, ast.Delete):
            for target in node.targets:
                self._bind_target(target, None)
            return

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._eval_expr(child)
            elif isinstance(child, ast.stmt):
                self._visit_statement(child)
            elif isinstance(child, ast.ExceptHandler):
                if child.type:
                    self._eval_expr(child.type)
                if child.name:
                    self.scope.bindings[child.name] = None
                self._visit_statements(child.body)


def _python_evidence(content: bytes) -> tuple[_PythonEvidence | None, str | None]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
        source = content.decode(encoding)
        tree = ast.parse(source)
        return _SourceAnalyzer().analyze(tree), None
    except (LookupError, RecursionError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        return None, str(exc)


def _scan_sources(root: pathlib.Path) -> _SourceEvidence:
    agent_app: pathlib.Path | None = None
    adapters: dict[str, pathlib.Path] = {}
    parse_failures = 0
    unreadable_files = 0
    source_files = 0
    source_bytes = 0
    scanned_directories = 0
    scanned_entries = 0
    truncated = False
    pending = [root]

    while pending and not truncated:
        if scanned_directories >= _MAX_SCAN_DIRECTORIES:
            truncated = True
            break
        current_path = pending.pop()
        scanned_directories += 1

        try:
            entries = os.scandir(current_path)
        except OSError:
            unreadable_files += 1
            continue

        with entries:
            for entry in entries:
                if scanned_entries >= _MAX_SCAN_ENTRIES:
                    truncated = True
                    break
                scanned_entries += 1
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if not _ignored_directory(entry.name):
                            pending.append(pathlib.Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    unreadable_files += 1
                    continue

                if not entry.name.casefold().endswith(".py") or _ignored_python_file(entry.name):
                    continue
                if source_files >= _MAX_SOURCE_FILES:
                    truncated = True
                    break

                path = pathlib.Path(entry.path)
                source_files += 1
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    unreadable_files += 1
                    continue
                if size > _MAX_SOURCE_BYTES or source_bytes + size > _MAX_SOURCE_TOTAL_BYTES:
                    truncated = True
                    break

                content, error = _bounded_bytes(
                    root, path, description="Python source", limit=_MAX_SOURCE_BYTES
                )
                if error or content is None:
                    unreadable_files += 1
                    continue
                if source_bytes + len(content) > _MAX_SOURCE_TOTAL_BYTES:
                    truncated = True
                    break
                source_bytes += len(content)
                parsed, parse_error = _python_evidence(content)
                if parse_error or parsed is None:
                    parse_failures += 1
                    continue

                relative = path.relative_to(root)
                if agent_app is None and parsed.agent_app_invoked:
                    agent_app = relative
                for framework, allowed in _FRAMEWORK_ADAPTER_CALLS.items():
                    if framework not in adapters and parsed.adapters & allowed:
                        adapters[framework] = relative

    return _SourceEvidence(agent_app, adapters, parse_failures, unreadable_files, truncated)


def _source_suffix(evidence: _SourceEvidence) -> str:
    notes = []
    if evidence.parse_failures:
        notes.append(f"{evidence.parse_failures} Python file(s) did not parse")
    if evidence.unreadable_files:
        notes.append(f"{evidence.unreadable_files} Python file(s) could not be inspected")
    if evidence.truncated:
        notes.append("the bounded source scan exceeded its limit")
    return f" ({'; '.join(notes)})" if notes else ""


def _source_checks(
    evidence: _SourceEvidence, framework: str | None
) -> tuple[DoctorCheck, DoctorCheck]:
    suffix = _source_suffix(evidence)
    agent_app = _check(
        "agent_app",
        evidence.agent_app is not None,
        (
            f"AgentApp is constructed and has invoke registered in {evidence.agent_app}.{suffix}"
            if evidence.agent_app is not None
            else (
                "No production Python source constructs databricks_mason.AgentApp and registers "
                f"invoke on that instance.{suffix}"
            )
        ),
    )
    if framework is None:
        adapter = _check(
            "framework_adapter",
            False,
            f"Cannot select a Mason framework adapter because the framework is unknown.{suffix}",
        )
    elif framework in evidence.adapters:
        adapter = _check(
            "framework_adapter",
            True,
            f"A recognized databricks_mason.{framework} adapter call is present in "
            f"{evidence.adapters[framework]}.{suffix}",
        )
    else:
        adapter = _check(
            "framework_adapter",
            False,
            f"No Python source calls a databricks_mason.{framework} adapter.{suffix}",
        )
    return agent_app, adapter


def inspect_project(directory: pathlib.Path) -> dict[str, Any]:
    """Return the complete offline doctor report for ``directory``."""
    root = directory.resolve()
    manifest_check, server_check, manifest_framework = _agent_manifest(root)

    # Metadata and dependencies can identify a framework even when agent.toml is absent or broken.
    provisional_metadata, metadata_framework = _project_metadata(root, manifest_framework)
    dependencies, dependency_error, dependency_frameworks = _project_dependencies(root)
    evidence = _scan_sources(root)

    framework = manifest_framework or metadata_framework
    if framework is None and len(dependency_frameworks) == 1:
        framework = next(iter(dependency_frameworks))
    if framework is None and not evidence.truncated and len(evidence.adapters) == 1:
        framework = next(iter(evidence.adapters))

    # Re-evaluate metadata once fallback inference identifies the expected framework.
    metadata_check, _ = _project_metadata(root, framework)
    dependency_check = _dependency_check(dependencies, dependency_error, framework)
    agent_app_check, adapter_check = _source_checks(evidence, framework)
    checks = [
        manifest_check,
        server_check,
        metadata_check if framework is not None else provisional_metadata,
        dependency_check,
        _app_command(root),
        agent_app_check,
        adapter_check,
    ]
    passed = sum(check.status == "pass" for check in checks)
    failed = len(checks) - passed
    onboarded = failed == 0
    next_step = None
    if not onboarded:
        framework_option = framework or "<langgraph|openai>"
        next_step = f"mason init --framework {framework_option} --existing <directory>"
    return {
        "directory": str(root),
        "framework": framework,
        "onboarded": onboarded,
        "checks": [asdict(check) for check in checks],
        "summary": {"total": len(checks), "passed": passed, "failed": failed},
        "next_step": next_step,
    }


def _terminal_text(value: object, *, limit: int) -> str:
    """Make dynamic report content one-line, terminal-safe, and bounded."""
    escaped: list[str] = []
    for character in str(value):
        if character == "\n":
            escaped.append(r"\n")
        elif character == "\r":
            escaped.append(r"\r")
        elif character == "\t":
            escaped.append(r"\t")
        elif not character.isprintable() or unicodedata.category(character).startswith("C"):
            codepoint = ord(character)
            escaped.append(f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}")
        else:
            escaped.append(character)
    rendered = "".join(escaped)
    if len(rendered) > limit:
        return f"{rendered[: limit - 1]}…"
    return rendered


def _emit_text(report: dict[str, Any]) -> None:
    directory = _terminal_text(report["directory"], limit=_MAX_HUMAN_PATH_CHARS)
    framework = _terminal_text(report["framework"] or "unknown", limit=32)
    click.echo(f"Mason doctor: {directory}")
    click.echo(f"Framework: {framework}")
    for check in report["checks"]:
        status = _terminal_text(check["status"], limit=16)
        check_id = _terminal_text(check["id"], limit=64)
        detail = _terminal_text(check["detail"], limit=_MAX_HUMAN_DETAIL_CHARS)
        click.echo(f"[{status}] {check_id}: {detail}")
    summary = report["summary"]
    result = "onboarded" if report["onboarded"] else "not onboarded"
    click.echo(f"Summary: {summary['passed']}/{summary['total']} checks passed — {result}.")
    if not report["onboarded"]:
        next_step = _terminal_text(report["next_step"], limit=_MAX_HUMAN_PATH_CHARS)
        click.echo(
            f"Next: use `{next_step}` to prepare migration instructions"
            " (it does not modify application source)."
        )


@click.command()
@click.argument(
    "directory",
    required=False,
    default=pathlib.Path("."),
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
)
@click.pass_obj
def doctor(obj: Any, directory: pathlib.Path) -> None:
    """Check whether an existing agent repository is onboarded to Agent Bricks through Mason.

    DIRECTORY defaults to the current directory. Doctor reads local configuration and Python source
    without importing it, making network calls, or changing any files.
    """
    try:
        report = inspect_project(directory)
    except AgentCliError:
        raise
    except Exception as exc:
        raise AgentCliError(
            f"mason doctor could not finish inspecting {directory}.",
            hint=(
                "This is unexpected — doctor should always produce a report instead of failing. "
                "Please report the repository layout that triggered it."
            ),
        ) from exc
    if obj.output == "json":
        click.echo(json.dumps(report, indent=2))
    else:
        _emit_text(report)
    if not report["onboarded"]:
        raise click.exceptions.Exit(1)
