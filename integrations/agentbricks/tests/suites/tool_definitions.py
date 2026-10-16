"""The tools under test: how each is added, prompted and verified."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from harness.agentbricks_cli import excerpt, tool_results
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
    # Asserts the `endpoint invoke --output json` output shows the tool ran and returned its result.
    check: Check
    # What follows `agentbricks tools add`; None when the tool is a user-owned file instead.
    add_command: str | None
    # User-owned files the tool needs.
    files: Mapping[str, str]
    # The id `tools add` binds it under, which `tools remove` takes.
    tool_id: str | None


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """A tool and the fixtures it needs; ``build`` takes them as keyword arguments, by fixture name."""

    name: str
    requires: tuple[str, ...]
    build: Callable[..., Tool]


def _successful(serialized: str, matches: Callable[[dict[str, Any]], bool], what: str) -> None:
    results = tool_results(serialized)
    assert any(r.get("status") == "success" and matches(r) for r in results), (
        f"No successful tool result {what}: {excerpt(serialized)}"
    )


def returned(marker: str) -> Check:
    """A tool returned ``marker``, which only the tool can produce."""

    def check(serialized: str) -> None:
        _successful(serialized, lambda r: marker in str(r.get("content")), f"containing {marker!r}")

    return check


def check_web_search(serialized: str) -> None:
    _successful(
        serialized,
        lambda r: r.get("name") == "web_search" and "https://" in str(r.get("content")),
        "from web_search with an https result",
    )


def check_genie(serialized: str) -> None:
    _successful(
        serialized,
        lambda r: "genie" in str(r.get("name")) and bool(r.get("content")),
        "from the Genie tool",
    )


def sandbox_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="sandbox",
        prompt=(
            "You must call the sandbox tool. In the sandbox, use Python to read the "
            f"entire text file at {uc_function.volume_file_path}. Return only its exact contents; "
            "do not fabricate them."
        ),
        check=returned(uc_function.volume_marker),
        add_command=f"sandbox --scope volume:{uc_function.volume}",
        files={},
        tool_id="sandbox",
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
        files={},
        tool_id="web_search",
    )


def python_tool() -> Tool:
    return Tool(
        name="python",
        prompt=(
            "You must call the matrix_marker Python tool with value 'matrix'. "
            "Return its exact result."
        ),
        check=returned(PYTHON_MARKER),
        add_command=None,
        files={PYTHON_MARKER_FILE: PYTHON_MARKER_SOURCE},
        tool_id=None,
    )


def uc_function_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="uc_function",
        prompt=(
            "You must call the Unity Catalog function tool that computes the Agent Bricks marker, "
            "with value 'matrix'. Return the called tool's exact result."
        ),
        check=returned(UC_MARKER),
        add_command=f"uc-function {uc_function.function} --name agentbricks_uc_marker",
        files={},
        tool_id="agentbricks_uc_marker",
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
        files={},
        tool_id="genie",
    )


# MCP needs a workspace that serves system.ai.web_search; Python needs nothing.
TOOL_DEFINITIONS = (
    ToolDefinition("sandbox", ("uc_function",), sandbox_tool),
    ToolDefinition("mcp", ("workspace_client",), web_search_tool),
    ToolDefinition("python", (), python_tool),
    ToolDefinition("uc_function", ("uc_function",), uc_function_tool),
    ToolDefinition("genie", ("genie_space",), genie_tool),
)
