"""Create a Claude agent that answers data questions with Databricks Genie One.

The agent declares Genie One's tools as custom tools. By default that is
``genie_ask_and_wait`` (the worker polls Genie inside the sandbox) and
``genie_get_query_result``. The tool list is read with your own Databricks login, so you
need EXECUTE on ``system.ai.genie_one_mcp``.

    uv run --extra mcp examples/genie_agent/setup_agent.py \
        --profile my-workspace
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shared  # noqa: E402

from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server  # noqa: E402

SYSTEM = (
    "You are a data analyst for this Databricks workspace. Answer data questions with "
    "genie_ask_and_wait; it returns Genie's final answer, the SQL it ran and a link to explore it. "
    "Use genie_get_query_result only when you need the full result rows. "
    "Lead with the answer, keep it short, and include the 'Explore in Databricks' link when Genie gives one."
)


async def create_agent(args: argparse.Namespace, *, settings=None, w=None) -> str:
    settings = settings or _shared.load_settings(args)
    w = w or _shared.databricks_login(args)
    tools = await alist_mcp_custom_tools(w, genie_one_server(w))
    print(f"Genie One tools: {[t['name'] for t in tools]}")
    agent = await _shared.anthropic_client(settings).beta.agents.create(
        name="databricks-genie-one-analyst",
        model=args.model,
        system=SYSTEM,
        tools=[{"type": "agent_toolset_20260401"}, *tools],
    )
    print(f"Created agent {agent.id}\nRun: examples/genie_agent/main.py --agent-id {agent.id}")
    return agent.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _shared.add_args(parser)
    asyncio.run(create_agent(parser.parse_args()))
