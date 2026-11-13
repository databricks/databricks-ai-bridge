"""End to end, hosted dispatcher (M1, service mode): the deployed Databricks App serves a session.

No local dispatcher runs here. A session is created on the App's environment; the App claims it and starts
the worker in a sandbox as the App's service principal; Genie One answers from that sandbox with the
principal's Unity Catalog grants. The test then reads the App's status and, if this identity can see the
sandbox, checks its tags and name prefix.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 SANDBOX_E2E_APP=cma-dispatcher-dev \\
        DATABRICKS_SANDBOX_PROFILE=<profile> uv run pytest tests/integration_tests/test_90_e2e_app_service_mode.py -v -s

  SANDBOX_E2E_APP              deployed App name (test skipped if unset)
  SANDBOX_E2E_APP_ENV_ID_KEY   key in ANTHROPIC_ENV_FILE with the App's environment id
                               (default ANTHROPIC_ENVIRONMENT_ID_2)
  SANDBOX_E2E_APP_QUESTION     Genie question (default: a count over samples.nyctaxi.trips)
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import pytest
import requests
from _support import MAX_REPLY_WORDS, budget_cents, find_or_create_agent

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

APP = os.environ.get("SANDBOX_E2E_APP")
QUESTION = os.environ.get(
    "SANDBOX_E2E_APP_QUESTION", "How many trips are in samples.nyctaxi.trips?"
)
SYSTEM = (
    "You answer data questions with Databricks Genie. Call genie_ask_and_wait once with the user's question; "
    f"it returns the final answer. Do not use bash. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


def session_of(sandbox) -> str | None:
    return ((sandbox.metadata.custom_tags if sandbox.metadata else None) or {}).get(
        "claude_session_id"
    )


def app_get(w, url: str, path: str) -> requests.Response:
    return requests.get(url.rstrip("/") + path, headers=w.config.authenticate(), timeout=30)


@pytest.mark.e2e
@pytest.mark.genie
@pytest.mark.skipif(not APP, reason="set SANDBOX_E2E_APP to the deployed App's name")
def test_app_serves_a_session_as_its_service_principal(api, settings, workspace_client):
    w = workspace_client
    environment_id = settings[
        os.environ.get("SANDBOX_E2E_APP_ENV_ID_KEY", "ANTHROPIC_ENVIRONMENT_ID_2")
    ]
    app = w.apps.get(APP)
    assert app.url and app.service_principal_client_id
    before = app_get(w, app.url, "/api/status")
    assert before.status_code == 200, f"/api/status -> {before.status_code}"
    status0 = before.json()
    assert status0["healthy"] and status0["environment_id"] == environment_id
    claimed0 = status0["dispatcher"]["claimed"]

    server = genie_one_server(w)
    tools = asyncio.run(alist_mcp_custom_tools(w, server))
    agent_id = find_or_create_agent(
        api, variant="genie-one", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}, *tools]
    )
    session = api.beta.sessions.create(
        agent=agent_id,
        environment_id=environment_id,
        title="databricks-sandbox App (M1) test",
        budget={
            "type": "limit",
            "max_list_cost": {"amount": str(budget_cents()), "currency": "USD"},
        },
    )
    reply, tool_calls, stop_reason, sandbox = "", [], None, None
    try:
        with api.beta.sessions.events.stream(session.id) as events:
            api.beta.sessions.events.send(
                session.id,
                events=[
                    {
                        "type": "user.message",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{QUESTION} Reply in fewer than {MAX_REPLY_WORDS} words.",
                            }
                        ],
                    }
                ],
            )
            deadline = time.monotonic() + 420
            for event in events:
                if event.type in ("agent.tool_use", "agent.custom_tool_use"):
                    tool_calls.append(getattr(event, "name", ""))
                    if sandbox is None:  # the worker is running: find its sandbox while it exists
                        sandbox = next(
                            (s for s in w.sandbox.list_sandboxes() if session_of(s) == session.id),
                            "not visible",
                        )
                elif event.type == "agent.message":
                    reply = "".join(
                        getattr(b, "text", "") for b in (getattr(event, "content", None) or [])
                    )
                elif event.type == "session.status_idle":
                    stop_reason = getattr(getattr(event, "stop_reason", None), "type", None)
                    if stop_reason == "budget_reached" or (
                        stop_reason != "requires_action" and reply
                    ):
                        break
                assert time.monotonic() < deadline, "session did not finish in time"

        identity = None
        if sandbox not in (None, "not visible"):
            try:
                identity = (
                    _ops.run(
                        w,
                        sandbox.name,
                        'databricks current-user me -o json | grep -o \'"userName":"[^"]*"\'',
                        check=False,
                    ).stdout
                    or ""
                ).strip()
            except Exception as e:  # noqa: BLE001 - visibility is a finding, not a failure
                identity = f"exec refused: {type(e).__name__}"
        after = app_get(w, app.url, "/api/status").json()
    finally:
        api.beta.sessions.delete(session.id)

    tags = (
        sandbox.metadata.custom_tags
        if sandbox not in (None, "not visible") and sandbox.metadata
        else {}
    )
    print(f"\nreply={reply!r} tools={tool_calls} stop={stop_reason}")
    print(
        f"app status: claimed {claimed0} -> {after['dispatcher']['claimed']}, launched "
        f"{after['dispatcher']['launched']}, failed {after['dispatcher']['failed']}, healthy {after['healthy']}"
    )
    print(
        f"sandbox: {getattr(sandbox, 'name', sandbox)} tags={tags} in-sandbox identity={identity}"
    )
    print(f"app service principal client id: {app.service_principal_client_id}")

    assert after["dispatcher"]["claimed"] > claimed0, "the App did not claim the session"
    assert after["dispatcher"]["failed"] == status0["dispatcher"]["failed"], (
        f"launch failed: {after}"
    )
    assert tool_calls.count("genie_ask_and_wait") == 1 and "bash" not in tool_calls, tool_calls
    assert stop_reason != "budget_reached"
    assert re.search(r"\d", reply), f"Genie's answer should contain a number: {reply!r}"
    if tags:
        assert getattr(sandbox, "name", "").startswith("sandboxes/cmaapp-")
        assert tags.get("claude_environment_id") == environment_id
