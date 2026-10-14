"""The tools under test: how each is added, prompted, verified, and what a deploy grants for it."""

from __future__ import annotations

import dataclasses
import json
import shlex
import uuid
from collections.abc import Callable, Mapping

from harness.agent_endpoint import AgentEndpoint
from harness.cli import AgentbricksCli
from harness.project import Project
from harness.workspace import GrantTuple
from uc_objects import UcFunction

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
    # What follows `agentbricks tools add`; None when the tool is a user-owned file instead.
    add_command: str | None
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


def excerpt(serialized: str, head: int = 1000, tail: int = 3000) -> str:
    """The response for a failure message: tool results come last, so keep the end, not just the start."""
    if len(serialized) <= head + tail:
        return serialized
    omitted = len(serialized) - head - tail
    return f"{serialized[:head]} ...[{omitted} chars omitted]... {serialized[-tail:]}"


def contains(marker: str) -> Check:
    def check(serialized: str) -> None:
        assert marker in serialized, f"Missing exact marker {marker!r}: {excerpt(serialized)}"

    return check


def check_web_search(serialized: str) -> None:
    lowered = serialized.lower()
    tool_evidence = any(v in lowered for v in ("web_search", "web search", "search"))
    assert tool_evidence and "https" in lowered and len(serialized) >= 80, (
        f"Missing web-search execution/result evidence: {excerpt(serialized)}"
    )


def check_genie(serialized: str) -> None:
    lowered = serialized.lower()
    assert (
        any(v in lowered for v in ("genie_ask", "genie", "conversation_id"))
        and len(serialized) >= 80
    ), f"Missing Genie execution/result evidence: {excerpt(serialized)}"


def sandbox_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="sandbox",
        prompt=(
            "You must call the sandbox tool. In the sandbox, use Python to read the "
            f"entire text file at {uc_function.volume_file_path}. Return only its exact contents; "
            "do not fabricate them."
        ),
        check=contains(uc_function.volume_marker),
        # The CLI's default auth: the sandbox reads the volume as the user invoking the agent, so
        # deploy adds no App grant. A manifest entry without `auth` would run as the App instead.
        add_command=f"sandbox --scope volume:{uc_function.volume}",
        toml=(
            '[[tools]]\nid = "sandbox"\nauth = "user"\n'
            'source = { kind = "sandbox", service = "system.ai.sandbox" }\n'
            "policy = { downscope = [\n"
            f'  {{ resource = "volume:{uc_function.volume}", permission = "read_only" }},\n'
            "] }\n"
        ),
        files={},
        tool_id="sandbox",
        app_auth=False,
        grant=None,
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
        add_command="mcp system.ai.web_search --auth app",
        toml=(
            '[[tools]]\nid = "web_search"\nauth = "app"\n'
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
        add_command=None,
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
        add_command=f"uc-function {uc_function.function} --name agentbricks_uc_marker",
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
        add_command=f"genie-agent {genie_space} --name genie --auth app",
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
    if tool.add_command is not None:
        cli.run(f"tools add {tool.add_command} --source {project.path}")
    for relative_path, content in tool.files.items():
        project.write_file(relative_path, content)
    if tool.tool_id is None:
        return
    declared = {t["id"]: t for t in project.manifest().get("tools", [])}
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
    rejected = cli.run(
        "--output json tools add mcp system.ai.missing_service --name broken_mcp "
        f"--source {project.path}",
        check=False,
    )
    assert rejected.returncode != 0 and manifest.read_bytes() == before, (
        "agentbricks tools add accepted an unavailable MCP service or changed agent.toml"
    )
    code = json.loads(rejected.stderr).get("error", {}).get("code")
    assert code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}, (
        f"Unexpected MCP validation error: {rejected.stderr}"
    )
    cli.run(f"tools remove mcp system.ai.missing_service --source {project.path}")
    assert not any(tool["id"] == "broken_mcp" for tool in project.manifest().get("tools", [])), (
        "agentbricks tools remove left the broken MCP binding in agent.toml"
    )


def invoke_and_check(endpoint: AgentEndpoint, tool: Tool) -> None:
    tool.check(endpoint.invoke(tool.prompt, f"{endpoint.runtime}-{tool.name}"))


def assert_endpoint_invoke(cli: AgentbricksCli, endpoint: AgentEndpoint) -> None:
    """`agentbricks endpoint invoke`, the call the CLI tells users to run after dev/deploy, gets a reply."""
    invocation_id = str(uuid.uuid4())
    body = json.dumps(
        {
            "id": invocation_id,
            "session_id": invocation_id,
            "input": [{"role": "user", "content": "Reply with the single word OK."}],
        }
    )
    target = endpoint.app_name or f"--url {endpoint.base_url} --no-auth"
    output = cli.run(
        f"endpoint invoke {target} --path /api/invocations --json {shlex.quote(body)}",
        retry_unavailable=True,
    ).stdout
    assert "ok" in output.lower(), f"endpoint invoke returned no reply: {output[:500]}"
