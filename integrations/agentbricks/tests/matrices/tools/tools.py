"""The tools under test: how each is added, prompted, verified, and what a deploy grants for it."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Iterator, Mapping, Sequence

from agentbricks_cli import Agent, AgentbricksCli, Project
from common import EXPECTED, Evidence, Inputs, MatrixError, Outcome, recorded_row
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


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    prompt: str
    # The exact text the response must contain; None for tools checked by their tool-call evidence.
    marker: str | None
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
    # The UC function wraps a nested one that Agent Bricks deliberately does not grant.
    needs_transitive_grant: bool = False

    def check(self, serialized: str) -> None:
        lowered = serialized.lower()
        if self.marker is not None:
            if self.marker not in serialized:
                raise AssertionError(f"Missing exact marker {self.marker!r}: {serialized[:2000]}")
        elif self.name == "mcp":
            tool_evidence = any(v in lowered for v in ("web_search", "web search", "search"))
            if not tool_evidence or "https" not in lowered or len(serialized) < 80:
                raise AssertionError(
                    f"Missing web-search execution/result evidence: {serialized[:2000]}"
                )
        elif self.name == "genie":
            if (
                not any(v in lowered for v in ("genie_ask", "genie", "conversation_id"))
                or len(serialized) < 80
            ):
                raise AssertionError(
                    f"Missing Genie execution/result evidence: {serialized[:2000]}"
                )
        else:
            raise MatrixError(f"No semantic assertion is defined for tool {self.name!r}.")


def tools_for(inputs: Inputs) -> tuple[Tool, ...]:
    """The TOOLS table, resolved against this run's scratch volume, function and Genie space."""
    uc_tool_name = inputs.uc_function.replace(".", "__")
    return (
        Tool(
            name="sandbox",
            prompt=(
                "You must call the sandbox tool. In the sandbox, use Python to read the "
                f"entire text file at {inputs.volume_file_path}. Return only its exact contents; "
                "do not fabricate them."
            ),
            marker=inputs.volume_marker,
            add_args=("sandbox", "--scope", f"volume:{inputs.uc_volume}", "--auth", "app"),
            toml=(
                '[[tools]]\nid = "sandbox"\n'
                'source = { kind = "sandbox", service = "system.ai.sandbox" }\n'
                "policy = { downscope = [\n"
                f'  {{ resource = "volume:{inputs.uc_volume}", permission = "read_only" }},\n'
                "] }\n"
            ),
            files={},
            tool_id="sandbox",
            app_auth=True,
            grant=("uc_securable", inputs.uc_volume, "VOLUME", "READ_VOLUME"),
        ),
        Tool(
            name="mcp",
            prompt=(
                "You must use a tool from the configured system.ai.web_search MCP server. "
                "Search official Databricks documentation for Model Context Protocol, then "
                "return the title and https URL of one result. Do not answer from memory."
            ),
            marker=None,
            add_args=("mcp", "system.ai.web_search", "--auth", "app"),
            toml=(
                '[[tools]]\nid = "web_search"\n'
                'source = { kind = "mcp", service = "system.ai.web_search" }\n'
            ),
            files={},
            tool_id="web_search",
            app_auth=True,
        ),
        Tool(
            name="python",
            prompt=(
                "You must call the matrix_marker Python tool with value 'matrix'. "
                "Return its exact result."
            ),
            marker=PYTHON_MARKER,
            add_args=None,
            toml="",
            files={PYTHON_MARKER_FILE: PYTHON_MARKER_SOURCE},
            tool_id=None,
            app_auth=False,
        ),
        Tool(
            name="uc_function",
            prompt=(
                f"You must call the tool named {uc_tool_name} with value 'matrix'. "
                "Return the called tool's exact result."
            ),
            marker=UC_MARKER,
            add_args=("uc-function", inputs.uc_function, "--name", "agentbricks_uc_marker"),
            toml=(
                '[[tools]]\nid = "agentbricks_uc_marker"\n'
                f'source = {{ kind = "uc_function", function = "{inputs.uc_function}" }}\n'
            ),
            files={},
            tool_id="agentbricks_uc_marker",
            app_auth=False,
            grant=("uc_securable", inputs.uc_function, "FUNCTION", "EXECUTE"),
            needs_transitive_grant=True,
        ),
        Tool(
            name="genie",
            prompt=(
                "You must call the genie_ask tool and ask what data is available in the "
                "configured Genie space. Return a one-sentence summary based only on the tool "
                "response."
            ),
            marker=None,
            add_args=("genie-agent", inputs.genie_space_id, "--name", "genie", "--auth", "app"),
            toml=(
                '[[tools]]\nid = "genie"\nauth = "app"\n'
                f'source = {{ kind = "genie_agent", space_id = "{inputs.genie_space_id}" }}\n'
            ),
            files={},
            tool_id="genie",
            app_auth=True,
            grant=("genie_space", inputs.genie_space_id, "GENIE_SPACE", "CAN_RUN"),
        ),
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
    if tool.tool_id not in declared:
        raise MatrixError(f"tools add did not declare {tool.tool_id!r}: {sorted(declared)}")
    if tool.app_auth and declared[tool.tool_id].get("auth") != "app":
        raise MatrixError(f"CLI-authored {tool.tool_id} binding is not App-auth.")


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
    if rejected.returncode == 0 or manifest.read_bytes() != before:
        raise MatrixError(
            "agentbricks tools add accepted an unavailable MCP service or changed agent.toml"
        )
    if json.loads(rejected.stderr).get("error", {}).get("code") not in {
        "NOT_FOUND",
        "RESOURCE_DOES_NOT_EXIST",
    }:
        raise MatrixError(f"Unexpected MCP validation error: {rejected.stderr}")
    cli.tools_remove(project, "mcp", "system.ai.missing_service")
    if any(tool["id"] == "broken_mcp" for tool in cli.manifest(project).get("tools", [])):
        raise MatrixError("agentbricks tools remove left the broken MCP binding in agent.toml")


def tool_row(evidence: Evidence, authoring: str, runtime: str, tool: Tool):
    """One evidence row for one tool in one authoring x runtime cell."""
    return recorded_row(
        evidence,
        authoring,
        runtime,
        tool.name,
        command=f"agentbricks {'dev' if runtime == 'dev' else 'deploy'}",
        expected=tool.marker or EXPECTED[tool.name],
        marker=tool.marker,
    )


def invoke_and_check(outcome: Outcome, agent: Agent, tool: Tool) -> None:
    outcome.command = agent.curl(tool.prompt)
    outcome.log_path = agent.log_path
    outcome.app_name = agent.app_name
    outcome.app_url = agent.base_url if agent.app_name else None
    outcome.serialized = agent.invoke(tool.prompt, f"{agent.runtime}-{tool.name}")
    tool.check(outcome.serialized)


@contextlib.contextmanager
def setup_failures(
    evidence: Evidence,
    authoring: str,
    runtime: str,
    tools: Sequence[Tool],
    app_name: str | None = None,
) -> Iterator[None]:
    """Fail every tool's cell when the shared agent they all need cannot be set up."""
    try:
        yield
    except Exception as exc:
        evidence.record_setup_failure(
            authoring, runtime, exc, None, [tool.name for tool in tools], app_name
        )
        raise
