"""Unit tests for ``mason add-sandbox`` source generation."""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import sys
import types
from importlib import resources
from typing import Any

import pytest
from click.testing import CliRunner

from databricks_mason import sandbox as sandbox_mod
from databricks_mason.sandbox import add_sandbox

_EMPTY_MCPS = '''"""MCP servers to offer the agent."""

from agents.mcp import MCPServer


def build_mcp_servers() -> list[MCPServer]:
    """Return the configured MCP servers."""
    return []
'''


class _TextCtx:
    output = "text"


class _JsonCtx:
    output = "json"


def _project(
    tmp_path: pathlib.Path, content: str = _EMPTY_MCPS
) -> tuple[pathlib.Path, pathlib.Path]:
    project = tmp_path / "agent-project"
    agent = project / "agent"
    agent.mkdir(parents=True)
    mcps = agent / "mcps.py"
    mcps.write_text(content)
    return project, mcps


def test_generated_block_reads_valid_python_template_at_render_time(
    tmp_path: pathlib.Path, monkeypatch
):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "sandbox_mcp.py").write_text(
        "# BEGIN: mason add-sandbox\n"
        "_SANDBOX_DOWNSCOPE: dict[str, list[dict[str, str]]] = {}  "
        "# __MASON_SANDBOX_DOWNSCOPE__\n"
        'TEMPLATE_SOURCE = "loaded"\n'
        "# END: mason add-sandbox\n"
    )
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)

    rendered = sandbox_mod._generated_block(
        {
            "volumes": [
                {"name": "catalog.schema.volume", "permission": "read_only"},
            ]
        }
    )

    assert 'TEMPLATE_SOURCE = "loaded"' in rendered
    namespace: dict[str, Any] = {}
    exec(compile(rendered, "sandbox_mcp.py", "exec"), namespace)
    assert namespace["_SANDBOX_DOWNSCOPE"] == {
        "volumes": [
            {"name": "catalog.schema.volume", "permission": "read_only"},
        ]
    }


def test_add_sandbox_inserts_imports_declared_by_template(tmp_path: pathlib.Path, monkeypatch):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "sandbox_mcp.py").write_text(
        "from custom_sandbox import CustomMcpServer\n\n"
        "# BEGIN: mason add-sandbox\n"
        "_SANDBOX_DOWNSCOPE: dict[str, list[dict[str, str]]] = {}  "
        "# __MASON_SANDBOX_DOWNSCOPE__\n\n"
        "def _build_sandbox_mcp_server():\n"
        "    return CustomMcpServer(_SANDBOX_DOWNSCOPE)\n\n"
        "# END: mason add-sandbox\n"
    )
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)

    generated = sandbox_mod._add_sandbox_to_source(
        _EMPTY_MCPS,
        {
            "volumes": [
                {"name": "catalog.schema.volume", "permission": "read_only"},
            ]
        },
    )

    assert generated.index("from custom_sandbox import CustomMcpServer") < generated.index(
        "def build_mcp_servers"
    )
    assert generated.count("from custom_sandbox import CustomMcpServer") == 1


def test_add_sandbox_generates_fixed_volume_downscope(tmp_path: pathlib.Path):
    project, mcps = _project(tmp_path)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    generated = mcps.read_text()
    ast.parse(generated)
    assert '"name": "catalog.schema.volume"' in generated
    assert '"permission": "read_only"' in generated
    assert "/ai-gateway/mcp-services/system.ai.sandbox" in generated
    assert "from_uc_function" not in generated
    assert 'tool_filter={"allowed_tool_names": ["run_code"]}' in generated
    assert "return [\n        _build_sandbox_mcp_server(),\n    ]" in generated
    assert 'meta["downscope"] = _SANDBOX_DOWNSCOPE' in generated
    assert "super().call_tool(tool_name, arguments, meta=meta, **kwargs)" in generated
    assert "arguments[" not in generated


def test_generated_server_overrides_caller_downscope_without_changing_arguments(
    tmp_path: pathlib.Path, monkeypatch
):
    project, mcps = _project(tmp_path)
    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )
    assert result.exit_code == 0, result.output

    class FakeMcpServer:
        connection: dict[str, object]

        def __init__(self, **kwargs):
            self.connection = kwargs

        async def call_tool(self, tool_name, arguments, **kwargs):
            return tool_name, arguments, kwargs

    class FakeWorkspaceClient:
        def __init__(self):
            self.config = types.SimpleNamespace(host="https://tilefood.example.com")

    agents = types.ModuleType("agents")
    agents_mcp = types.ModuleType("agents.mcp")
    agents_mcp.__dict__["MCPServer"] = object
    databricks_openai = types.ModuleType("databricks_openai")
    databricks_openai_agents = types.ModuleType("databricks_openai.agents")
    databricks_openai_agents.__dict__["McpServer"] = FakeMcpServer
    databricks_sdk = types.ModuleType("databricks.sdk")
    databricks_sdk.__dict__["WorkspaceClient"] = FakeWorkspaceClient
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.mcp", agents_mcp)
    monkeypatch.setitem(sys.modules, "databricks_openai", databricks_openai)
    monkeypatch.setitem(sys.modules, "databricks_openai.agents", databricks_openai_agents)
    monkeypatch.setitem(sys.modules, "databricks.sdk", databricks_sdk)

    namespace: dict[str, Any] = {}
    exec(compile(mcps.read_text(), str(mcps), "exec"), namespace)
    server = namespace["build_mcp_servers"]()[0]
    arguments = {"code": 'print("hello")'}
    _, forwarded_arguments, kwargs = asyncio.run(
        server.call_tool(
            "sandbox",
            arguments,
            meta={"downscope": {"volumes": []}, "trace_id": "123"},
        )
    )

    assert forwarded_arguments is arguments
    assert kwargs["meta"] == {
        "downscope": {
            "volumes": [
                {"name": "catalog.schema.volume", "permission": "read_only"},
            ]
        },
        "trace_id": "123",
    }
    assert server.connection["url"] == (
        "https://tilefood.example.com/ai-gateway/mcp-services/system.ai.sandbox"
    )
    assert isinstance(server.connection["workspace_client"], FakeWorkspaceClient)
    assert server.connection["timeout"] == 120.0
    assert server.connection["tool_filter"] == {"allowed_tool_names": ["run_code"]}


def test_add_sandbox_supports_workspace_table_and_read_write_scopes(tmp_path: pathlib.Path):
    project, mcps = _project(tmp_path)

    result = CliRunner().invoke(
        add_sandbox,
        [
            "--source",
            str(project),
            "--scope",
            "/Workspace/Users/alice@example.com",
            "--scope",
            "table:catalog.schema.records",
            "--permission",
            "read_write",
        ],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    generated = mcps.read_text()
    assert '"workspace_paths": [' in generated
    assert '"path": "/Workspace/Users/alice@example.com"' in generated
    assert '"tables": [' in generated
    assert '"name": "catalog.schema.records"' in generated
    assert generated.count('"permission": "read_write"') == 2


def test_add_sandbox_appends_to_existing_server_list(tmp_path: pathlib.Path):
    project, mcps = _project(
        tmp_path,
        _EMPTY_MCPS.replace(
            "return []",
            'return [\n        ExistingServer(name="existing"),\n    ]',
        ),
    )

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    generated = mcps.read_text()
    assert 'ExistingServer(name="existing"),' in generated
    assert generated.index('ExistingServer(name="existing"),') < generated.index(
        "_build_sandbox_mcp_server(),"
    )


def test_add_sandbox_preserves_non_ascii_existing_server(tmp_path: pathlib.Path):
    project, mcps = _project(
        tmp_path,
        _EMPTY_MCPS.replace(
            "return []",
            'return [\n        ExistingServer(name="café")\n    ]',
        ),
    )

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    generated = mcps.read_text()
    ast.parse(generated)
    assert 'ExistingServer(name="café"),' in generated


def test_add_sandbox_keeps_future_import_at_top(tmp_path: pathlib.Path):
    project, mcps = _project(
        tmp_path,
        _EMPTY_MCPS.replace(
            "from agents.mcp import MCPServer",
            "from __future__ import annotations\n\nfrom agents.mcp import MCPServer",
        ),
    )

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    ast.parse(mcps.read_text())


def test_add_sandbox_does_not_treat_aliased_imports_as_required_bindings(tmp_path: pathlib.Path):
    original = _EMPTY_MCPS.replace(
        "from agents.mcp import MCPServer",
        "from typing import Any as T\n\n"
        "from agents.mcp import MCPServer\n"
        "from databricks_openai.agents import McpServer as ExistingMcpServer",
    )
    project, mcps = _project(tmp_path, original)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code == 0, result.output
    generated = mcps.read_text()
    assert "from typing import Any\n" in generated
    assert "from databricks_openai.agents import McpServer\n" in generated


def test_add_sandbox_is_idempotent(tmp_path: pathlib.Path):
    project, mcps = _project(tmp_path)
    runner = CliRunner()
    args = ["--source", str(project), "--scope", "catalog.schema.volume"]

    first = runner.invoke(add_sandbox, args, obj=_TextCtx())
    first_content = mcps.read_text()
    second = runner.invoke(add_sandbox, args, obj=_TextCtx())

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already configured" in second.output
    assert mcps.read_text() == first_content


def test_add_sandbox_rejects_policy_change_instead_of_reporting_stale_values(
    tmp_path: pathlib.Path,
):
    project, mcps = _project(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        add_sandbox,
        [
            "--source",
            str(project),
            "--scope",
            "catalog.schema.volume",
            "--permission",
            "read_write",
        ],
        obj=_TextCtx(),
    )
    first_content = mcps.read_text()

    second = runner.invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "other.schema.volume"],
        obj=_JsonCtx(),
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "different downscope" in second.output
    assert mcps.read_text() == first_content


def test_add_sandbox_rejects_conditional_builder_without_touching_it(tmp_path: pathlib.Path):
    original = _EMPTY_MCPS.replace(
        "return []",
        "if disabled:\n        return []\n    return []",
    )
    project, mcps = _project(tmp_path, original)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "directly return a list" in result.output
    assert mcps.read_text() == original


def test_add_sandbox_rejects_invalid_scope_without_touching_file(tmp_path: pathlib.Path):
    project, mcps = _project(tmp_path)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "not-a-volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "catalog.schema.volume" in result.output
    assert mcps.read_text() == _EMPTY_MCPS


@pytest.mark.parametrize(
    "scope",
    ["catalog .schema.volume", "workspace:/Workspace/Users/alice\tinvalid"],
)
def test_add_sandbox_rejects_downstream_invalid_scope_without_touching_file(
    tmp_path: pathlib.Path, scope: str
):
    project, mcps = _project(tmp_path)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", scope],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert mcps.read_text() == _EMPTY_MCPS


def test_add_sandbox_reports_missing_mason_project(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(tmp_path), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "agent/mcps.py" in result.output


def test_add_sandbox_rejects_unsupported_mcps_file_without_touching_it(tmp_path: pathlib.Path):
    original = _EMPTY_MCPS.replace("from agents.mcp import MCPServer", "import os")
    project, mcps = _project(tmp_path, original)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "from agents.mcp" in result.output
    assert mcps.read_text() == original


def test_add_sandbox_rejects_partial_generated_block(tmp_path: pathlib.Path):
    original = _EMPTY_MCPS + "\n# BEGIN: mason add-sandbox\n"
    project, mcps = _project(tmp_path, original)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "invalid markers" in result.output
    assert mcps.read_text() == original


@pytest.mark.parametrize(
    ("original_fragment", "tampered_fragment"),
    [
        (
            '        meta["downscope"] = _SANDBOX_DOWNSCOPE',
            '        # meta["downscope"] intentionally removed',
        ),
        (
            '{"name": "catalog.schema.volume", "permission": "read_only"}',
            '{"path": "catalog.schema.volume", "permission": "read_only"}',
        ),
        (
            "        _build_sandbox_mcp_server(),",
            "        _build_sandbox_mcp_server(),\n        _build_sandbox_mcp_server(),",
        ),
    ],
)
def test_add_sandbox_rejects_tampered_generated_security_block(
    tmp_path: pathlib.Path, original_fragment: str, tampered_fragment: str
):
    project, mcps = _project(tmp_path)
    first = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )
    assert first.exit_code == 0, first.output
    tampered = mcps.read_text().replace(original_fragment, tampered_fragment)
    assert tampered != mcps.read_text()
    mcps.write_text(tampered)

    second = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert second.exit_code != 0
    assert mcps.read_text() == tampered


def test_add_sandbox_reports_atomic_write_failure_without_touching_file(
    tmp_path: pathlib.Path, monkeypatch
):
    project, mcps = _project(tmp_path)

    def fail_to_create_temporary(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sandbox_mod.tempfile, "NamedTemporaryFile", fail_to_create_temporary)
    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_TextCtx(),
    )

    assert result.exit_code != 0
    assert "disk full" in result.output
    assert mcps.read_text() == _EMPTY_MCPS


def test_add_sandbox_honors_json_output(tmp_path: pathlib.Path):
    project, _ = _project(tmp_path)

    result = CliRunner().invoke(
        add_sandbox,
        ["--source", str(project), "--scope", "catalog.schema.volume"],
        obj=_JsonCtx(),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "mcp_server": "system.ai.sandbox",
        "path": str(project / "agent" / "mcps.py"),
        "permission": "read_only",
        "scopes": ["catalog.schema.volume"],
        "status": "added",
    }
