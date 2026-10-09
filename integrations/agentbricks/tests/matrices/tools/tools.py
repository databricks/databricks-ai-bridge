"""The tools under test: how each is added, prompted, verified, and what a deploy grants for it."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping

from agentbricks_cli import Agent, AgentbricksCli, Project
from uc_objects import UcFunction
from workspace_client import GrantTuple

PYTHON_MARKER_FILE = "agent/tools/matrix_marker.py"
PYTHON_MARKER_SOURCE = (
    "from langchain_core.tools import tool\n\n\n"
    "@tool\n"
    "def matrix_marker(value: str) -> str:\n"
    '    """Return the deterministic AgentBricks E2E marker."""\n'
    "    return 'AGENTBRICKS_PYTHON_OK'\n"
)
UC_MARKER = "AGENTBRICKS_UC_OK:matrix"
PYTHON_MARKER = "AGENTBRICKS_PYTHON_OK"


Check = Callable[[str], None]


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    prompt: str
    # Asserts the serialized response proves the tool ran.
    check: Check
    # `agentbricks tools add` arguments; None when the tool is a user-owned file instead.
    add_args: tuple[str, ...] | None
    # The [[tools]] entry for direct authoring, and any user-owned files the tool needs.
    toml: str
    files: Mapping[str, str]
    # The id agent.toml declares it under, and whether it must be App-authed.
    tool_id: str | None
    app_auth: bool
    # The App resource a deploy grants for it; None when it needs no UC grant.
    grant: GrantTuple | None = None
    # A function the tool calls that Agent Bricks deliberately does not grant; it works only after
    # a manual grant.
    transitive_function: str | None = None


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """A tool and the fixtures it needs; ``build`` takes them as keyword arguments, by fixture name."""

    name: str
    requires: tuple[str, ...]
    build: Callable[..., Tool]


def contains(marker: str) -> Check:
    def check(serialized: str) -> None:
        assert marker in serialized, f"Missing exact marker {marker!r}: {serialized[:2000]}"

    return check


def check_web_search(serialized: str) -> None:
    lowered = serialized.lower()
    tool_evidence = any(v in lowered for v in ("web_search", "web search", "search"))
    assert tool_evidence and "https" in lowered and len(serialized) >= 80, (
        f"Missing web-search execution/result evidence: {serialized[:2000]}"
    )


def check_genie(serialized: str) -> None:
    lowered = serialized.lower()
    assert (
        any(v in lowered for v in ("genie_ask", "genie", "conversation_id"))
        and len(serialized) >= 80
    ), f"Missing Genie execution/result evidence: {serialized[:2000]}"


def sandbox_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="sandbox",
        prompt=(
            "You must call the sandbox tool. In the sandbox, use Python to read the "
            f"entire text file at {uc_function.volume_file_path}. Return only its exact contents; "
            "do not fabricate them."
        ),
        check=contains(uc_function.volume_marker),
        add_args=("sandbox", "--scope", f"volume:{uc_function.volume}", "--auth", "app"),
        toml=(
            '[[tools]]\nid = "sandbox"\n'
            'source = { kind = "sandbox", service = "system.ai.sandbox" }\n'
            "policy = { downscope = [\n"
            f'  {{ resource = "volume:{uc_function.volume}", permission = "read_only" }},\n'
            "] }\n"
        ),
        files={},
        tool_id="sandbox",
        app_auth=True,
        grant=("uc_securable", uc_function.volume, "VOLUME", "READ_VOLUME"),
    )


def web_search_tool(**_resources: object) -> Tool:
    return Tool(
        name="mcp",
        prompt=(
            "You must use a tool from the configured system.ai.web_search MCP server. "
            "Search official Databricks documentation for Model Context Protocol, then "
            "return the title and https URL of one result. Do not answer from memory."
        ),
        check=check_web_search,
        add_args=("mcp", "system.ai.web_search", "--auth", "app"),
        toml=(
            '[[tools]]\nid = "web_search"\n'
            'source = { kind = "mcp", service = "system.ai.web_search" }\n'
        ),
        files={},
        tool_id="web_search",
        app_auth=True,
    )


def python_tool() -> Tool:
    return Tool(
        name="python",
        prompt=(
            "You must call the matrix_marker Python tool with value 'matrix'. "
            "Return its exact result."
        ),
        check=contains(PYTHON_MARKER),
        add_args=None,
        toml="",
        files={PYTHON_MARKER_FILE: PYTHON_MARKER_SOURCE},
        tool_id=None,
        app_auth=False,
    )


def uc_function_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="uc_function",
        prompt=(
            f"You must call the tool named {uc_function.function.replace('.', '__')} with value "
            "'matrix'. Return the called tool's exact result."
        ),
        check=contains(UC_MARKER),
        add_args=("uc-function", uc_function.function, "--name", "agentbricks_uc_marker"),
        toml=(
            '[[tools]]\nid = "agentbricks_uc_marker"\n'
            f'source = {{ kind = "uc_function", function = "{uc_function.function}" }}\n'
        ),
        files={},
        tool_id="agentbricks_uc_marker",
        app_auth=False,
        grant=("uc_securable", uc_function.function, "FUNCTION", "EXECUTE"),
        transitive_function=uc_function.nested_function,
    )


def genie_tool(genie_space: str) -> Tool:
    return Tool(
        name="genie",
        prompt=(
            "You must call the genie_ask tool and ask what data is available in the "
            "configured Genie space. Return a one-sentence summary based only on the tool "
            "response."
        ),
        check=check_genie,
        add_args=("genie-agent", genie_space, "--name", "genie", "--auth", "app"),
        toml=(
            '[[tools]]\nid = "genie"\nauth = "app"\n'
            f'source = {{ kind = "genie_agent", space_id = "{genie_space}" }}\n'
        ),
        files={},
        tool_id="genie",
        app_auth=True,
        grant=("genie_space", genie_space, "GENIE_SPACE", "CAN_RUN"),
    )


# MCP needs a workspace that serves system.ai.web_search; Python needs nothing.
TOOL_DEFINITIONS = (
    ToolDefinition("sandbox", ("uc_function",), sandbox_tool),
    ToolDefinition("mcp", ("workspace_client",), web_search_tool),
    ToolDefinition("python", (), python_tool),
    ToolDefinition("uc_function", ("uc_function",), uc_function_tool),
    ToolDefinition("genie", ("genie_space",), genie_tool),
)


def bind(cli: AgentbricksCli, project: Project, tool: Tool) -> None:
    """CLI authoring: add one tool and confirm agent.toml declares it as intended."""
    if tool.add_args is not None:
        cli.tools_add(project, *tool.add_args)
    for relative_path, content in tool.files.items():
        cli.write_file(project, relative_path, content)
    if tool.tool_id is None:
        return
    declared = {t["id"]: t for t in cli.manifest(project).get("tools", [])}
    assert tool.tool_id in declared, (
        f"tools add did not declare {tool.tool_id!r}: {sorted(declared)}"
    )
    if tool.app_auth:
        assert declared[tool.tool_id].get("auth") == "app", (
            f"CLI-authored {tool.tool_id} binding is not App-auth."
        )


def reject_unavailable_mcp(cli: AgentbricksCli, project: Project) -> None:
    """`tools add` must refuse an MCP service that does not exist and leave agent.toml untouched."""
    manifest = project.path / "agent.toml"
    before = manifest.read_bytes()
    rejected = cli.tools_add(
        project,
        "mcp",
        "system.ai.missing_service",
        "--name",
        "broken_mcp",
        check=False,
        json_output=True,
    )
    assert rejected.returncode != 0 and manifest.read_bytes() == before, (
        "agentbricks tools add accepted an unavailable MCP service or changed agent.toml"
    )
    code = json.loads(rejected.stderr).get("error", {}).get("code")
    assert code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}, (
        f"Unexpected MCP validation error: {rejected.stderr}"
    )
    cli.tools_remove(project, "mcp", "system.ai.missing_service")
    assert not any(tool["id"] == "broken_mcp" for tool in cli.manifest(project).get("tools", [])), (
        "agentbricks tools remove left the broken MCP binding in agent.toml"
    )


def invoke_and_check(agent: Agent, tool: Tool) -> None:
    tool.check(agent.invoke(tool.prompt, f"{agent.runtime}-{tool.name}"))


def assert_endpoint_invoke(agentbricks_cli: AgentbricksCli, agent: Agent) -> None:
    """The CLI's own invoke command reaches the running agent and gets a reply."""
    output = agentbricks_cli.endpoint_invoke(agent, "Reply with the single word OK.")
    assert "ok" in output.lower(), f"endpoint invoke returned no reply: {output[:500]}"
