"""The tools under test: how each is added, prompted and verified."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any

from harness.agentbricks_cli import excerpt, tool_results
from harness.workspace import Workspace
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


# Whether one tool step of an `endpoint invoke` response is this tool returning its result.
Called = Callable[[dict[str, Any]], bool]


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    prompt: str
    called: Called
    # What follows `agentbricks tools add`; None when the tool is a user-owned file instead.
    add_command: str | None
    # User-owned files the tool needs.
    files: Mapping[str, str]
    # The id `tools add` binds it under, which `tools remove` takes.
    tool_id: str | None

    def assert_called(self, serialized: str) -> None:
        """Assert the `endpoint invoke --output json` output shows the tool ran and returned its result."""
        assert self._calls(serialized), f"No successful {self.name} result: {excerpt(serialized)}"

    def assert_not_called(self, serialized: str) -> None:
        """Assert the agent answered without the tool returning its result."""
        assert not self._calls(serialized), f"{self.name} still ran: {excerpt(serialized)}"

    def _calls(self, serialized: str) -> list[dict[str, Any]]:
        return [
            r for r in tool_results(serialized) if r.get("status") == "success" and self.called(r)
        ]


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """A tool and the fixtures it needs; ``build`` takes them as keyword arguments, by fixture name."""

    name: str
    requires: tuple[str, ...]
    build: Callable[..., Tool]


def returned(marker: str) -> Called:
    """The step returned ``marker``, which only the tool can produce."""
    return lambda r: marker in str(r.get("content"))


def web_search_called(r: dict[str, Any]) -> bool:
    return r.get("name") == "web_search" and "https://" in str(r.get("content"))


def genie_called(r: dict[str, Any]) -> bool:
    # Genie reports a failed or timed-out answer inside a successful tool step, as its status.
    if r.get("name") not in ("genie_ask", "genie_poll"):
        return False
    content = r.get("content")
    try:
        message = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return False
    return (
        isinstance(message, dict)
        and message.get("status") == "COMPLETED"
        and bool(message.get("attachments"))
    )


def sandbox_tool(uc_function: UcFunction) -> Tool:
    return Tool(
        name="sandbox",
        prompt=(
            "You must call the sandbox tool. In the sandbox, use Python to read the "
            f"entire text file at {uc_function.volume_file_path}. Return only its exact contents; "
            "do not fabricate them."
        ),
        called=returned(uc_function.volume_marker),
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
        called=web_search_called,
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
        called=returned(PYTHON_MARKER),
        add_command=None,
        files={PYTHON_MARKER_FILE: PYTHON_MARKER_SOURCE},
        tool_id=None,
    )


def uc_function_tool(uc_function: UcFunction, workspace_client: Workspace) -> Tool:
    return Tool(
        name="uc_function",
        # Name the tool: its description matches the Python marker's, so the model may pick either.
        prompt=(
            f"You must call the {uc_function.tool_name} Unity Catalog function tool with value "
            "'matrix'. Return its exact result."
        ),
        # The function returns `current_user()`. Bound with `--auth user`, it runs as whoever
        # invoked the agent, not the deployed App's service principal.
        called=returned(f"{UC_MARKER}:{workspace_client.user_name}"),
        add_command=f"uc-function {uc_function.function} --name agentbricks_uc_marker --auth user",
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
        called=genie_called,
        add_command=f"genie-agent {genie_space} --name genie --auth app",
        files={},
        tool_id="genie",
    )


# MCP needs a workspace that serves system.ai.web_search; Python needs nothing.
TOOL_DEFINITIONS = (
    ToolDefinition("sandbox", ("uc_function",), sandbox_tool),
    ToolDefinition("mcp", ("workspace_client",), web_search_tool),
    ToolDefinition("python", (), python_tool),
    ToolDefinition("uc_function", ("uc_function", "workspace_client"), uc_function_tool),
    ToolDefinition("genie", ("genie_space",), genie_tool),
)
