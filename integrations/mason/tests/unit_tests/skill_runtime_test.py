"""Behavior tests for UC Skills manifest loading and LangGraph injection."""

from __future__ import annotations

import asyncio
import importlib
import pathlib
from types import SimpleNamespace

import pytest

UC_SKILL = """[[skills]]
id = "quarter-close"
source = { kind = "uc", name = "main.finance.quarter-close" }
"""


def _write_manifest(root: pathlib.Path, *skills: str, framework: str = "langgraph") -> None:
    (root / "agent.toml").write_text(
        f'''schema_version = 1

[agent]
framework = "{framework}"

{"".join(skills)}''',
        encoding="utf-8",
    )


def test_load_skills_preserves_exact_uc_binding(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, UC_SKILL)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    manifest = importlib.import_module("databricks_mason.runtime.skill_manifest")

    assert manifest.load_skills(expected_framework="langgraph") == (
        manifest.SkillRecord(
            id="quarter-close",
            kind="uc",
            name="main.finance.quarter-close",
        ),
    )


@pytest.mark.parametrize(
    "source",
    [
        '{ kind = "local", path = "skills/review" }',
        '{ kind = "uc", name = "main.finance" }',
        '{ kind = "uc", name = "main.finance.review", path = "skills/review" }',
    ],
)
def test_load_skills_rejects_non_uc_or_malformed_sources(
    monkeypatch, tmp_path: pathlib.Path, source: str
):
    _write_manifest(
        tmp_path,
        f'[[skills]]\nid = "review"\nsource = {source}\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    manifest = importlib.import_module("databricks_mason.runtime.skill_manifest")

    with pytest.raises(RuntimeError, match="[Ss]kill|UC"):
        manifest.load_skills(expected_framework="langgraph")


def test_load_skills_rejects_duplicate_ids(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, UC_SKILL, UC_SKILL)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    manifest = importlib.import_module("databricks_mason.runtime.skill_manifest")

    with pytest.raises(RuntimeError, match="skill ids must be unique"):
        manifest.load_skills(expected_framework="langgraph")


class _FakeStructuredTool:
    def __init__(self, *, name, description, coroutine):
        self.name = name
        self.description = description
        self.coroutine = coroutine

    @classmethod
    def from_function(cls, *, coroutine, name, description):
        return cls(name=name, description=description, coroutine=coroutine)

    async def ainvoke(self, arguments):
        return await self.coroutine(**arguments)


class _FakeSessions:
    def __init__(self, tools):
        self.tools = tools
        self.connections = []
        self.calls = []

    def create(self, connection):
        self.connections.append(connection)
        parent = self

        class Context:
            async def __aenter__(self):
                return Session()

            async def __aexit__(self, *args):
                return False

        class Session:
            async def initialize(self):
                return None

            async def list_tools(self):
                return SimpleNamespace(tools=parent.tools)

            async def call_tool(self, name, arguments):
                parent.calls.append((name, arguments))
                text = (
                    "=== references/close.md ===\nREFERENCE_MARKER"
                    if name == "get_skill_files"
                    else "BODY_MARKER"
                )
                return SimpleNamespace(
                    isError=False,
                    content=[SimpleNamespace(type="text", text=text)],
                )

        return Context()


def _uc_tool(
    name: str = "skill_main.finance.quarter-close",
    description: str = "[main.finance.quarter-close] Close the quarter consistently.",
):
    return SimpleNamespace(name=name, description=description, inputSchema={})


def _load_runtime(monkeypatch, tmp_path: pathlib.Path, tools):
    _write_manifest(tmp_path, UC_SKILL)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    runtime = importlib.import_module("databricks_mason.langgraph.skills")
    sessions = _FakeSessions(tools)

    class FakeDatabricksMCPServer:
        created = []

        def __init__(self, name, url, **kwargs):
            self.name = name
            self.url = url
            self.kwargs = kwargs
            self.created.append(self)

        def to_connection_dict(self):
            return {
                "transport": "streamable_http",
                "url": self.url,
                "headers": self.kwargs.get("headers"),
            }

    client = SimpleNamespace(config=SimpleNamespace(host="https://df1.example.com/"))
    monkeypatch.setattr(runtime, "DatabricksMCPServer", FakeDatabricksMCPServer)
    monkeypatch.setattr(runtime, "StructuredTool", _FakeStructuredTool)
    monkeypatch.setattr(runtime, "create_session", sessions.create)
    monkeypatch.setattr(runtime, "workspace_client", lambda: client)
    monkeypatch.setattr(runtime, "workspace_headers", lambda: {"X-Databricks-Org-Id": "123"})
    return runtime, sessions, FakeDatabricksMCPServer, client


def test_uc_skill_injects_metadata_and_loads_body_and_reference_lazily(
    monkeypatch, tmp_path: pathlib.Path
):
    runtime, sessions, server_type, client = _load_runtime(
        monkeypatch,
        tmp_path,
        [_uc_tool(), _uc_tool("get_skill_files", "Read referenced files.")],
    )

    context, tools = asyncio.run(runtime.build_skill_context())

    assert context == (
        "Available skills:\n"
        "- [quarter-close] (uc:main.finance.quarter-close) "
        "Close the quarter consistently.\n\n"
        "Call load_skill with an ID when a task matches. "
        "Read referenced files only with read_skill_file."
    )
    assert "BODY_MARKER" not in context
    assert "REFERENCE_MARKER" not in context
    assert [tool.name for tool in tools] == ["load_skill", "read_skill_file"]
    assert [server.url for server in server_type.created] == [
        "https://df1.example.com/api/2.0/ai-gateway/skills/?schema=main.finance"
    ]
    assert server_type.created[0].kwargs["workspace_client"] is client
    assert server_type.created[0].kwargs["headers"] == {"X-Databricks-Org-Id": "123"}
    assert sessions.calls == []

    load, read = tools
    assert asyncio.run(load.ainvoke({"skill_id": "quarter-close"})) == "BODY_MARKER"
    assert (
        asyncio.run(read.ainvoke({"skill_id": "quarter-close", "path": "references/close.md"}))
        == "=== references/close.md ===\nREFERENCE_MARKER"
    )
    assert sessions.calls == [
        ("skill_main.finance.quarter-close", {}),
        (
            "get_skill_files",
            {"full_name": "main.finance.quarter-close", "paths": ["references/close.md"]},
        ),
    ]


def test_uc_skill_rejects_missing_dynamic_tool(monkeypatch, tmp_path: pathlib.Path):
    runtime, _, _, _ = _load_runtime(
        monkeypatch,
        tmp_path,
        [_uc_tool("skill_main.finance.other", "Another skill.")],
    )

    with pytest.raises(RuntimeError, match="quarter-close.*not found.*UC Skills endpoint"):
        asyncio.run(runtime.build_skill_context())


@pytest.mark.parametrize("path", ["../secret.md", "/tmp/secret.md", r"C:\\secret.md"])
def test_uc_reference_reader_requires_relative_contained_path(
    monkeypatch, tmp_path: pathlib.Path, path: str
):
    runtime, sessions, _, _ = _load_runtime(monkeypatch, tmp_path, [_uc_tool()])
    _, tools = asyncio.run(runtime.build_skill_context())

    with pytest.raises(RuntimeError, match="relative path"):
        asyncio.run(tools[1].ainvoke({"skill_id": "quarter-close", "path": path}))
    assert sessions.calls == []


def test_langgraph_package_exports_skill_context_builder():
    langgraph = importlib.import_module("databricks_mason.langgraph")
    runtime = importlib.import_module("databricks_mason.langgraph.skills")

    assert langgraph.build_skill_context is runtime.build_skill_context
