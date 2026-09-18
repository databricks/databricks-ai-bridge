"""Native Genie Agent tools for LangGraph, configured through ``agent.toml``."""

from langchain_core.tools import BaseTool, tool

from databricks_mason.runtime.genie import GenieAgent
from databricks_mason.runtime.tool_manifest import load_tools


def genie_tools() -> list[BaseTool]:
    """Build ask, poll and query-result tools for each configured Genie Agent.

    Space IDs are bound from the manifest, not exposed as model arguments. Construction performs
    no authentication or network requests; an agent with no Genie Agent bindings gets no tools.
    """
    tools = []
    for record in load_tools(expected_framework="langgraph"):
        if record.kind != "genie_agent":
            continue
        agent = GenieAgent(record.space_id or "")
        tools.extend(
            tool(f"{record.id}_{method.__name__}")(method)
            for method in (agent.ask, agent.poll, agent.query_result)
        )
    return tools
