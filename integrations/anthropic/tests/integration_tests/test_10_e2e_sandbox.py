"""End to end, without Genie One: Claude's tool call runs in the session's Databricks Sandbox.

Uses the default lifecycle (ephemeral): the sandbox is tagged with its Claude session and is
deleted by the worker once the session goes idle.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_10_e2e_sandbox.py -v -s
"""

from __future__ import annotations

import pytest
from _support import MAX_REPLY_WORDS, find_or_create_agent, make_launcher
from databricks.sdk.errors import NotFound

from databricks_anthropic.sandbox import _ops

WORKDIR = "/home/sandbox-agent/agent-workspace"
MARKER = "written-by-claude-in-databricks-sandbox"
SYSTEM = (
    f"You run inside a Databricks Sandbox; your working directory is {WORKDIR}. Use as few tool calls as "
    f"possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)
PROMPT = (
    f"Write '{MARKER}' and the output of `whoami` to {WORKDIR}/e2e.txt using one bash command. "
    f"Reply in fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
def test_session_tools_run_in_databricks_sandbox(api, settings, workspace_client, run_session):
    agent_id = find_or_create_agent(
        api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
    )
    launcher = make_launcher(workspace_client, settings)
    seen: dict = {}

    def inspect(sandbox_name: str) -> None:
        # Before the session is deleted: deleting it ends the worker, which deletes the ephemeral sandbox.
        seen["content"] = (
            _ops.run(workspace_client, sandbox_name, f"cat {WORKDIR}/e2e.txt", check=False).stdout
            or ""
        )
        sandbox = workspace_client.sandbox.get_sandbox(sandbox_name)
        seen["tags"] = dict(sandbox.metadata.custom_tags or {}) if sandbox.metadata else {}

    run = run_session(launcher, agent_id, PROMPT, while_open=inspect)
    print(f"\nlaunched={run.launched} tools={run.tool_calls} reply={run.reply!r} usage={run.usage}")

    assert not run.failures, f"launch failed: {run.failures}"
    assert run.launched == [run.sandbox_name], (
        "the dispatcher did not launch this session's sandbox"
    )
    assert run.stop_reason != "budget_reached", "session hit its budget"
    assert "bash" in run.tool_calls, f"expected a bash tool call, got {run.tool_calls}"
    assert run.reply and len(run.reply.split()) < MAX_REPLY_WORDS, (
        f"reply too long or empty: {run.reply!r}"
    )

    content = seen.get("content", "")
    assert MARKER in content, f"file not written inside the sandbox: {content!r}"
    assert "sandbox-agent" in content, "the command did not run as the sandbox user"

    tags = seen.get("tags", {})
    assert (
        tags.get("claude_session_id") == run.session_id and tags.get("lifecycle") == "ephemeral"
    ), tags

    # Ephemeral: the worker deletes its own sandbox once the session ends (or after max_idle).
    import time

    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        try:
            workspace_client.sandbox.get_sandbox(run.sandbox_name)
        except NotFound:
            break
        time.sleep(5)
    else:
        pytest.fail(
            f"ephemeral sandbox {run.sandbox_name} was not deleted after the session went idle"
        )
