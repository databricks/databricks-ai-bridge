"""Lifecycle cleanup against real sandboxes (no model calls).

- resumable retention: a stopped sandbox is kept within the retention window and deleted after it
- session gone: a sandbox whose Claude session was deleted is deleted (even if still running)

Uses its own sandbox id prefix so cleanup here can only touch sandboxes this test created.

    RUN_SANDBOX_INTEGRATION_TESTS=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_60_lifecycle_cleanup.py -v -s
"""

from __future__ import annotations

import uuid

from _support import make_launcher

from databricks_anthropic import DatabricksSandboxDispatcher
from databricks_anthropic.sandbox import _ops

PREFIX = f"cmac{uuid.uuid4().hex[:6]}-"  # this test's own prefix (cleanup only sees these)


def test_retention_and_session_gone_cleanup(api, settings, workspace_client):
    launcher = make_launcher(
        workspace_client, settings, lifecycle="resumable", sandbox_id_prefix=PREFIX
    )
    dispatcher = DatabricksSandboxDispatcher(launcher)
    session = api.beta.sessions.create(
        agent=_any_agent(api),
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        title="databricks-sandbox cleanup test",
    )  # never sent a message: no cost
    stale_session = f"sesn_cleanup_test_{uuid.uuid4().hex[:8]}"
    names = []
    try:
        live, _ = launcher.ensure_sandbox(session.id)
        stale, _ = launcher.ensure_sandbox(stale_session)
        names += [live, stale]
        tags = workspace_client.sandbox.get_sandbox(live).metadata.custom_tags
        assert tags["claude_session_id"] == session.id and tags["lifecycle"] == "resumable"

        workspace_client.sandbox.stop_sandbox(stale)
        _ops.wait_for_state(workspace_client, stale, _ops.SandboxState.SANDBOX_STATE_STOPPED)

        plan = {a.sandbox_name: a for a in dispatcher.plan_cleanup()}  # default 7-day retention
        assert plan[stale].action == "keep" and plan[live].action == "keep", plan

        plan = {a.sandbox_name: a for a in dispatcher.cleanup(retention_days=0)}
        print("\nretention=0:", {n: (a.action, a.reason) for n, a in plan.items()})
        assert (
            plan[stale].action == "delete" and plan[live].action == "keep"
        )  # running is never retention-deleted

        api.beta.sessions.delete(session.id)
        plan = {a.sandbox_name: a for a in dispatcher.cleanup(sessions_api=api)}
        print("session gone:", {n: (a.action, a.reason) for n, a in plan.items()})
        assert plan[live].action == "delete" and "deleted or archived" in plan[live].reason
        assert dispatcher.list_session_sandboxes() == []
    finally:
        for name in names:
            _ops.delete_quietly(workspace_client, name)
        try:
            api.beta.sessions.delete(session.id)
        except Exception:  # noqa: BLE001 - already deleted above
            pass


def _any_agent(api) -> str:
    for agent in api.beta.agents.list(limit=100):
        if (agent.metadata or {}).get(
            "purpose"
        ) == "databricks-sandbox-e2e" and not agent.archived_at:
            return agent.id
    agent = api.beta.agents.create(
        name="databricks-sandbox-e2e-cleanup",
        model="claude-haiku-5-5",
        tools=[{"type": "agent_toolset_20260401"}],
        metadata={"purpose": "databricks-sandbox-e2e", "variant": "cleanup"},
    )
    return agent.id
