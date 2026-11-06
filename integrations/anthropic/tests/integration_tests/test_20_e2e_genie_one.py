"""End to end, with Genie One MCP: Claude asks Genie; the worker calls Genie One from inside the sandbox.

The agent sees ``genie_ask_and_wait``; the worker calls Genie One's ``genie_ask`` and polls
``genie_poll_response`` inside the sandbox with the sandbox's built-in credential (the dispatcher identity).

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_20_e2e_genie_one.py -v -s
"""

from __future__ import annotations

import os
import re

import pytest
from _support import MAX_REPLY_WORDS, find_or_create_agent, make_launcher

from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

QUESTION = os.environ.get("GENIE_E2E_QUESTION", "How many trips are in samples.nyctaxi.trips?")
SYSTEM = (
    "You answer data questions with Databricks Genie. Call genie_ask_and_wait once with the user's question; "
    f"it returns the final answer. Do not use bash. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
@pytest.mark.genie
def test_genie_one_tools_run_inside_the_sandbox(api, settings, workspace_client, run_session):
    import asyncio

    server = genie_one_server(workspace_client)
    tools = asyncio.run(alist_mcp_custom_tools(workspace_client, server))
    assert [t["name"] for t in tools][0] == "genie_ask_and_wait"
    agent_id = find_or_create_agent(
        api, variant="genie-one", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}, *tools]
    )

    launcher = make_launcher(workspace_client, settings, mcp_servers=[server])
    run = run_session(
        launcher, agent_id, f"{QUESTION} Reply in fewer than {MAX_REPLY_WORDS} words."
    )
    print(f"\nlaunched={run.launched} tools={run.tool_calls} reply={run.reply!r} usage={run.usage}")

    assert not run.failures, f"launch failed: {run.failures}"
    assert run.launched == [run.sandbox_name]
    assert run.stop_reason != "budget_reached", "session hit its budget"
    assert run.tool_calls.count("genie_ask_and_wait") == 1, (
        f"expected one Genie call, got {run.tool_calls}"
    )
    assert "bash" not in run.tool_calls
    assert run.reply and len(run.reply.split()) < MAX_REPLY_WORDS, (
        f"reply too long or empty: {run.reply!r}"
    )
    assert re.search(r"\d", run.reply), f"Genie's answer should contain a number: {run.reply!r}"
