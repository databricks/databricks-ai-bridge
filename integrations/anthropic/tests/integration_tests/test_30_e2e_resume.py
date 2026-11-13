"""End to end, resumable lifecycle: the sandbox stops when idle and the next message restarts it.

Turn 1 writes a file; the worker goes idle and stops its own sandbox; turn 2 relaunches the worker in the
*same* sandbox (not recreated), where the file and the worker venv are still present.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_30_e2e_resume.py -v -s
"""

from __future__ import annotations

import asyncio

import pytest
from _support import MAX_REPLY_WORDS, SessionDriver, find_or_create_agent, make_launcher

from databricks_anthropic.sandbox import _ops

WORKDIR = "/home/sandbox-agent/agent-workspace"
MARKER = "persisted-across-sandbox-restart"
SYSTEM = (
    f"You run inside a Databricks Sandbox; your working directory is {WORKDIR}. Use as few tool calls as "
    f"possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
def test_second_message_resumes_the_stopped_sandbox(api, settings, workspace_client):
    agent_id = find_or_create_agent(
        api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
    )
    launcher = make_launcher(workspace_client, settings, max_idle_s=30, lifecycle="resumable")

    async def scenario():
        async with SessionDriver(
            settings, launcher, agent_id, title="databricks-sandbox resume test"
        ) as driver:
            try:
                first = await driver.turn(
                    f"Write '{MARKER}' to {WORKDIR}/resume.txt using one bash command. "
                    f"Reply in fewer than {MAX_REPLY_WORDS} words."
                )
                assert [r.created for r in first.launches] == [True], (
                    f"turn 1 launches: {first.launches}"
                )

                # Worker idles (max_idle_s) and stops its own sandbox; home directory is kept.
                await asyncio.to_thread(
                    _ops.wait_for_state,
                    workspace_client,
                    driver.sandbox_name,
                    _ops.SandboxState.SANDBOX_STATE_STOPPED,
                    timeout_s=300,
                    poll_s=5,
                )

                second = await driver.turn(
                    f"Print the contents of {WORKDIR}/resume.txt with one bash command. "
                    f"Reply with that text in fewer than {MAX_REPLY_WORDS} words."
                )
                return driver.sandbox_name, first, second
            finally:
                launcher.delete_session_sandbox(driver.session_id)

    sandbox, first, second = asyncio.run(scenario())
    print(
        f"\nturn1 launch={first.launches} reply={first.reply!r}\nturn2 launch={second.launches} "
        f"reply={second.reply!r}"
    )
    assert not first.failures and not second.failures
    assert [r.sandbox_name for r in second.launches] == [sandbox], (
        "turn 2 did not use the session's sandbox"
    )
    assert [r.created for r in second.launches] == [False], (
        "turn 2 recreated the sandbox instead of resuming"
    )
    assert MARKER in second.reply, f"file did not survive the stop/start: {second.reply!r}"
    assert len(second.reply.split()) < MAX_REPLY_WORDS
