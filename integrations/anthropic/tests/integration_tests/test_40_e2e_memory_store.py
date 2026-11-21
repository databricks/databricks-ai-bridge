"""End to end, memory store: the worker mounts a Managed Agents memory store inside the Databricks Sandbox.

A memory store seeded with a fact is attached to the session. The agent reads it from
``/mnt/memory/<store>/`` inside the sandbox and appends a line, which the worker syncs back to the store.
Requires the work item's per-session secret (forwarded by the launcher) and a writable /mnt/memory.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_40_e2e_memory_store.py -v -s
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from _support import MAX_REPLY_WORDS, SessionDriver, find_or_create_agent, make_launcher

CODENAME = f"BLUEHERON-{uuid.uuid4().hex[:6].upper()}"
APPENDED = "verified-from-databricks-sandbox"
SYSTEM = (
    f"Use as few tool calls as possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
def test_memory_store_mounts_and_syncs_from_the_sandbox(api, settings, workspace_client):
    store = api.beta.memory_stores.create(
        name=f"cma-e2e-{uuid.uuid4().hex[:8]}",
        description="Test facts for the Databricks Sandbox integration test.",
    )
    try:
        memory = api.beta.memory_stores.memories.create(
            store.id, path="/facts.md", content=f"Project codename: {CODENAME}\n"
        )
        agent_id = find_or_create_agent(
            api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
        )
        launcher = make_launcher(workspace_client, settings)
        resources = [
            {
                "type": "memory_store",
                "memory_store_id": store.id,
                "access": "read_write",
                "instructions": "Project facts.",
            }
        ]

        def store_content() -> str:
            return (
                api.beta.memory_stores.memories.retrieve(
                    memory.id, memory_store_id=store.id
                ).content
                or ""
            )

        async def scenario():
            async with SessionDriver(
                settings,
                launcher,
                agent_id,
                resources=resources,
                title="databricks-sandbox memory test",
            ) as driver:
                try:
                    turn = await driver.turn(
                        "Read facts.md in your memory store. Then append the line "
                        f"'{APPENDED}' to that same file using one bash command. "
                        f"Reply with the project codename in fewer than {MAX_REPLY_WORDS} words."
                    )
                    # Wait for the worker's sync (every ~15 s, plus a final flush when it goes idle) while the
                    # session and its sandbox are still alive; tearing them down early would lose the upload.
                    deadline, content = time.monotonic() + 150, ""
                    while time.monotonic() < deadline and APPENDED not in content:
                        await asyncio.sleep(5)
                        content = await asyncio.to_thread(store_content)
                    return turn, content
                finally:
                    launcher.delete_session_sandbox(driver.session_id)

        turn, content = asyncio.run(scenario())
        print(f"\nlaunch={turn.launches} tools={turn.tool_calls} reply={turn.reply!r}")
        print(f"store content after sync: {content!r}")
        assert not turn.failures, turn.failures
        assert CODENAME in turn.reply, f"agent did not read the mounted memory: {turn.reply!r}"
        assert APPENDED in content, (
            "the agent's edit inside the sandbox was not synced back to the memory store"
        )
    finally:
        api.beta.memory_stores.delete(store.id)  # test-owned store
