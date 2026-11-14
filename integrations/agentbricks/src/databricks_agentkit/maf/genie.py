"""Native Microsoft Agent Framework tools for managed Genie Agent bindings."""

from agent_framework import FunctionTool, tool

from databricks_agentkit.runtime.genie import GenieAgent
from databricks_agentkit.runtime.tool_manifest import ToolManifestError, load_tools


def genie_tools() -> list[FunctionTool]:
    """Build ask/poll/result tools with space IDs bound outside model arguments."""
    tools = []
    for record in load_tools(expected_framework="maf"):
        if record.kind != "genie_agent":
            continue
        if record.auth == "user":
            raise ToolManifestError(
                "Microsoft Agent Framework currently supports only App-authenticated tool bindings."
            )
        agent = GenieAgent(record.space_id or "", auth="app")
        tools.extend(
            tool(method, name=f"{record.id}_{method.__name__}")
            for method in (agent.ask, agent.poll, agent.query_result)
        )
    return tools
