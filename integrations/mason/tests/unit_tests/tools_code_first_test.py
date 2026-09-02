from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_mason.integration_codegen import IntegrationRegistry, registry_relative_path
from databricks_mason.integrations import MCPService, Sandbox, UCFunction
from databricks_mason.project_config import write_project_metadata
from databricks_mason.tools import tools


class _Ctx:
    def __init__(self, output: str = "text") -> None:
        self.output = output


def _project(
    tmp_path: pathlib.Path,
    *,
    framework: str,
    attached: bool = False,
) -> pathlib.Path:
    project = tmp_path / framework
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    template = "agent-langgraph" if framework == "langgraph" else "agent-openai"
    write_project_metadata(project, framework=framework, template=template)
    IntegrationRegistry.empty(
        project,
        relative_path=registry_relative_path(framework),
    ).write()
    if framework == "langgraph":
        attachment = (
            "from databricks_mason.langgraph import load_tools\n"
            "from agent.databricks_tools import DATABRICKS_TOOLS\n\n"
            "async def create_agent_graph():\n"
            "    return await load_tools(DATABRICKS_TOOLS)\n"
            if attached
            else "\nasync def create_agent_graph():\n    return []\n"
        )
        attachment_path = project / "agent" / "agent.py"
    else:
        attachment = (
            "from databricks_mason.openai import bind_tools\n"
            "from agent.databricks_tools import DATABRICKS_TOOLS\n\n"
            "async def stream_handler(agent, stack):\n"
            "    return await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)\n"
            if attached
            else "\ndef create_agent():\n    return object()\n"
        )
        attachment_path = project / "agent" / "agent.py"
    attachment_path.parent.mkdir(parents=True, exist_ok=True)
    attachment_path.write_text(attachment, encoding="utf-8")
    return project


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_add_sandbox_generates_code_for_both_supported_frameworks(
    tmp_path: pathlib.Path, framework: str
) -> None:
    project = _project(tmp_path, framework=framework, attached=True)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "sandbox",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    relative_registry = registry_relative_path(framework)
    integrations = IntegrationRegistry.load(
        project,
        relative_path=relative_registry,
    ).integrations
    assert len(integrations) == 1
    assert isinstance(integrations[0], Sandbox)
    assert not (project / "agent.toml").exists()
    assert f"{project / relative_registry}:" in result.output
    attachment_path = (
        project / "agent" / "agent.py"
        if framework == "langgraph"
        else project / "agent" / "agent.py"
    )
    assert f"{attachment_path}:" in result.output
    assert "Attached" in result.output


def test_add_sandbox_reports_manual_action_instead_of_guessing_byo_location(
    tmp_path: pathlib.Path,
) -> None:
    project = _project(tmp_path, framework="openai", attached=False)

    result = CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "main.data.files", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Configured, not attached" in result.output
    assert "from agent.databricks_tools import DATABRICKS_TOOLS" in result.output
    assert "from databricks_mason.openai import bind_tools" in result.output
    assert "bind_tools" in result.output
    assert "agent/agent.py:" not in result.output


def test_attachment_detection_does_not_scan_byo_or_dead_code(tmp_path: pathlib.Path) -> None:
    byo = tmp_path / "byo"
    (byo / "agent").mkdir(parents=True)
    (byo / "agent" / "agent.py").write_text(
        "from databricks_mason.openai import bind_tools\n"
        "from agent.databricks_tools import DATABRICKS_TOOLS\n\n"
        "async def stream_handler(agent, stack):\n"
        "    return await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)\n",
        encoding="utf-8",
    )
    byo_result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--framework",
            "openai",
            "--source",
            str(byo),
        ],
        obj=_Ctx(),
    )

    template = _project(tmp_path / "template", framework="langgraph", attached=False)
    (template / "agent" / "agent.py").write_text(
        "from databricks_mason.langgraph import load_tools\n"
        "from agent.databricks_tools import DATABRICKS_TOOLS\n\n"
        "async def create_agent_graph():\n"
        "    if False:\n"
        "        return await load_tools(DATABRICKS_TOOLS)\n"
        "    return []\n",
        encoding="utf-8",
    )
    dead_result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--source",
            str(template),
        ],
        obj=_Ctx(),
    )

    assert byo_result.exit_code == 0, byo_result.output
    assert "Configured, not attached" in byo_result.output
    assert "Attached:" not in byo_result.output
    assert dead_result.exit_code == 0, dead_result.output
    assert "Configured, not attached" in dead_result.output
    assert "Attached:" not in dead_result.output


def test_openai_requires_the_template_streaming_seam(tmp_path: pathlib.Path) -> None:
    project = _project(tmp_path, framework="openai", attached=True)
    agent_file = project / "agent" / "agent.py"
    agent_file.write_text(
        agent_file.read_text(encoding="utf-8").replace(
            "    return await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)\n",
            "    return agent\n",
            1,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Configured, not attached" in result.output
    assert "Active after app restart" not in result.output


def test_add_sandbox_json_reports_definition_and_attachment_lines(tmp_path: pathlib.Path) -> None:
    project = _project(tmp_path, framework="langgraph", attached=True)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "sandbox",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx("json"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["definition"] == {
        "path": str(project / "agent" / "databricks_tools.py"),
        "line": IntegrationRegistry.load(project).definition_line("sandbox"),
    }
    assert payload["activation"] == {
        "status": "attached",
        "path": str(project / "agent" / "agent.py"),
        "line": 5,
        "symbol": "create_agent_graph",
    }


def test_existing_mcp_and_uc_function_commands_generate_the_shared_registry(
    tmp_path: pathlib.Path,
) -> None:
    project = _project(tmp_path, framework="openai")
    runner = CliRunner()

    mcp = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--name", "web", "--source", str(project)],
        obj=_Ctx(),
    )
    function = runner.invoke(
        tools,
        ["add", "uc-function", "main.tools.lookup", "--source", str(project)],
        obj=_Ctx(),
    )

    assert mcp.exit_code == 0, mcp.output
    assert function.exit_code == 0, function.output
    integrations = IntegrationRegistry.load(
        project,
        relative_path=registry_relative_path("openai"),
    ).integrations
    assert isinstance(integrations[0], MCPService)
    assert isinstance(integrations[1], UCFunction)


def test_identical_add_is_byte_stable_and_conflict_does_not_write(tmp_path: pathlib.Path) -> None:
    project = _project(tmp_path, framework="langgraph")
    runner = CliRunner()
    command = [
        "add",
        "sandbox",
        "--scope",
        "main.data.files",
        "--source",
        str(project),
    ]

    first = runner.invoke(tools, command, obj=_Ctx())
    path = project / "agent" / "databricks_tools.py"
    original = path.read_bytes()
    second = runner.invoke(tools, command, obj=_Ctx())
    conflict = runner.invoke(
        tools,
        [
            "add",
            "sandbox",
            "--scope",
            "main.data.other",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already configured" in second.output
    assert f"Definition: {path}:" in second.output
    assert "Configured, not attached" in second.output
    assert conflict.exit_code != 0
    assert "main.data.files" in conflict.output
    assert "main.data.other" in conflict.output
    assert path.read_bytes() == original


def test_legacy_agent_toml_is_not_silently_replaced(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "legacy"
    project.mkdir()
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    legacy = project / "agent.toml"
    legacy.write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "agent.toml is retired" in result.output
    assert "DATABRICKS_TOOLS" in result.output
    assert not (project / "agent" / "databricks_tools.py").exists()
    assert legacy.read_text(encoding="utf-8").startswith("schema_version = 1")


def test_malformed_agent_toml_in_mason_project_blocks_registry_write(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "legacy"
    project.mkdir()
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    legacy = project / "agent.toml"
    malformed = 'schema_version = 1\n\n[agent]\nframework = "langgraph"\n\n[[tools]\nid = "web"\n'
    legacy.write_text(malformed, encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "agent.toml is retired" in result.output
    assert not (project / "agent" / "databricks_tools.py").exists()
    assert legacy.read_text(encoding="utf-8") == malformed


def test_legacy_agent_toml_is_rejected_even_when_code_registry_exists(
    tmp_path: pathlib.Path,
) -> None:
    project = _project(tmp_path, framework="langgraph")
    registry = project / "agent" / "databricks_tools.py"
    before = registry.read_bytes()
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "agent.toml is retired" in result.output
    assert registry.read_bytes() == before


def test_byo_project_can_select_framework_without_mason_metadata(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "existing-agent"
    project.mkdir()
    (project / "server.py").write_text("agent = object()\n", encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--framework",
            "openai",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert IntegrationRegistry.load(
        project,
        relative_path=registry_relative_path("openai"),
    ).integrations == [MCPService(id="web_search", service="system.ai.web_search")]
    assert "Configured, not attached" in result.output
    assert "bind_tools" in result.output


def test_byo_project_ignores_unrelated_agent_toml(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "existing-agent"
    project.mkdir()
    unrelated = project / "agent.toml"
    unrelated.write_text('[application]\nname = "customer-owned"\n', encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--framework",
            "langgraph",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert unrelated.read_text(encoding="utf-8") == '[application]\nname = "customer-owned"\n'
    assert IntegrationRegistry.load(project).integrations == [
        MCPService(id="web_search", service="system.ai.web_search")
    ]
