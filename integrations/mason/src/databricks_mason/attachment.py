"""Report explicit integration seams in templates Mason owns and recognizes."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass

from databricks_mason.project_config import ProjectMetadata


@dataclass(frozen=True)
class AttachmentSite:
    path: pathlib.Path
    line: int
    symbol: str

    def as_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "line": self.line, "symbol": self.symbol}


@dataclass(frozen=True)
class Activation:
    status: str
    sites: tuple[AttachmentSite, ...]
    snippet: str
    imports: tuple[str, ...] = ()

    @property
    def path(self) -> pathlib.Path | None:
        return self.sites[0].path if self.sites else None

    @property
    def line(self) -> int | None:
        return self.sites[0].line if self.sites else None

    @property
    def symbol(self) -> str | None:
        return self.sites[0].symbol if self.sites else None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"status": self.status}
        if len(self.sites) == 1:
            result.update(self.sites[0].as_dict())
        elif self.sites:
            result["sites"] = [site.as_dict() for site in self.sites]
        if self.status != "attached":
            result["snippet"] = self.snippet
            result["imports"] = list(self.imports)
        return result


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def _uses_selection(node: ast.Call) -> bool:
    values = [*node.args, *(keyword.value for keyword in node.keywords)]
    return any(isinstance(value, ast.Name) and value.id == "DATABRICKS_TOOLS" for value in values)


class _AwaitedCallFinder(ast.NodeVisitor):
    def __init__(self, expected_call: str) -> None:
        self.expected_call = expected_call
        self.lines: list[int] = []

    def visit_Await(self, node: ast.Await) -> None:
        value = node.value
        if (
            isinstance(value, ast.Call)
            and _call_name(value) == self.expected_call
            and _uses_selection(value)
        ):
            self.lines.append(value.lineno)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            branch = node.body if node.test.value else node.orelse
            for statement in branch:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None


def _imports(tree: ast.Module, module: str, name: str) -> bool:
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == module
        and any(alias.name == name and alias.asname is None for alias in statement.names)
        for statement in tree.body
    )


def _attached_site(
    tree: ast.Module,
    path: pathlib.Path,
    symbol: str,
    expected_call: str,
) -> AttachmentSite | None:
    function = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, ast.AsyncFunctionDef) and statement.name == symbol
        ),
        None,
    )
    if function is None:
        return None
    finder = _AwaitedCallFinder(expected_call)
    for statement in function.body:
        finder.visit(statement)
    if not finder.lines:
        return None
    return AttachmentSite(path=path, line=finder.lines[0], symbol=symbol)


def _required_template(
    project: pathlib.Path,
    metadata: ProjectMetadata,
) -> tuple[pathlib.Path, str, str, str, tuple[str, ...], str] | None:
    if metadata.framework == "langgraph" and metadata.template == "agent-langgraph":
        return (
            project / "agent" / "agent.py",
            "databricks_mason.langgraph",
            "agent.databricks_tools",
            "load_tools",
            ("create_agent_graph",),
            "*await load_tools(DATABRICKS_TOOLS)",
        )
    if metadata.framework == "openai" and metadata.template == "agent-openai":
        return (
            project / "agent" / "agent.py",
            "databricks_mason.openai",
            "agent.databricks_tools",
            "bind_tools",
            ("stream_handler",),
            "agent = await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)",
        )
    return None


def activation_for(project: pathlib.Path, metadata: ProjectMetadata) -> Activation:
    required = _required_template(project, metadata)
    if required is None:
        call = "load_tools" if metadata.framework == "langgraph" else "bind_tools"
        snippet = (
            "*await load_tools(DATABRICKS_TOOLS)"
            if call == "load_tools"
            else "agent = await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)"
        )
        imports = (
            (
                "from agent.databricks_tools import DATABRICKS_TOOLS",
                "from databricks_mason.langgraph import load_tools",
            )
            if metadata.framework == "langgraph"
            else (
                "from agent.databricks_tools import DATABRICKS_TOOLS",
                "from databricks_mason.openai import bind_tools",
            )
        )
        return Activation("action_required", (), snippet, imports)

    path, adapter_module, selection_module, expected_call, symbols, snippet = required
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return Activation(
            "action_required",
            (),
            snippet,
            (
                f"from {selection_module} import DATABRICKS_TOOLS",
                f"from {adapter_module} import {expected_call}",
            ),
        )
    if not (
        _imports(tree, adapter_module, expected_call)
        and _imports(tree, selection_module, "DATABRICKS_TOOLS")
    ):
        return Activation(
            "action_required",
            (),
            snippet,
            (
                f"from {selection_module} import DATABRICKS_TOOLS",
                f"from {adapter_module} import {expected_call}",
            ),
        )

    sites = tuple(
        site
        for symbol in symbols
        if (site := _attached_site(tree, path, symbol, expected_call)) is not None
    )
    if len(sites) == len(symbols):
        return Activation("attached", sites, snippet)
    if sites:
        return Activation("partial", sites, snippet)
    return Activation("action_required", (), snippet)
