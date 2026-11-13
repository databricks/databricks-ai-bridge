"""Prerequisite check: is this Databricks workspace and Claude account set up for the integration?

No model calls. Creates one short-lived sandbox (deleted at the end). Each failure message says how to fix it.

    RUN_SANDBOX_INTEGRATION_TESTS=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_00_prerequisites.py -v
"""

from __future__ import annotations

import uuid

import anthropic
import pytest

from databricks_anthropic.sandbox import _ops


def test_databricks_login(workspace_client):
    me = workspace_client.current_user.me()
    assert me.user_name, "login returned no user"
    print(f"\nsandboxes will act as {me.user_name} ({workspace_client.config.auth_type})")


def test_sandbox_preview_enabled(workspace_client):
    try:
        list(workspace_client.sandbox.list_sandboxes(page_size=1))
    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"Sandbox API not available ({type(e).__name__}: {e}). A workspace admin must enable the "
            "'Databricks Sandboxes' preview (Settings -> Previews); allow ~1 minute after enabling."
        )


def test_sandbox_lifecycle_and_image(workspace_client):
    """Create a sandbox, check what the worker needs (bash, python3, uv, outbound HTTPS, identity), delete it."""
    w = workspace_client
    sandbox_id = f"cma-prereq-{uuid.uuid4().hex[:8]}"
    name = _ops.create(w, sandbox_id, inactivity_timeout_s=300, image_uri=None)
    try:
        _ops.wait_for_state(w, name, _ops.SandboxState.SANDBOX_STATE_RUNNING, timeout_s=300)
        out = (
            _ops.run(
                w,
                name,
                (
                    "python3 --version; command -v uv >/dev/null && echo uv=ok; "
                    "curl -s -o /dev/null -w 'anthropic=%{http_code}\\n' --max-time 10 https://api.anthropic.com/; "
                    "databricks current-user me -o json"
                    ' | python3 -c \'import json,sys; print("user=" + json.load(sys.stdin)["userName"])\''
                ),
            ).stdout
            or ""
        )
        assert "Python 3" in out, f"python3 missing in the sandbox image:\n{out}"
        assert "uv=ok" in out, (
            f"uv missing in the sandbox image (needed to bootstrap the worker):\n{out}"
        )
        assert "anthropic=000" not in out, f"sandbox cannot reach api.anthropic.com:\n{out}"
        assert f"user={w.current_user.me().user_name}" in out, (
            f"in-sandbox identity is not the creator:\n{out}"
        )
    finally:
        _ops.delete_quietly(w, name)


def test_anthropic_environment_is_self_hosted(api, settings):
    try:
        environment = api.beta.environments.retrieve(settings["ANTHROPIC_ENVIRONMENT_ID"])
    except anthropic.APIStatusError as e:
        pytest.fail(
            f"Cannot read environment {settings['ANTHROPIC_ENVIRONMENT_ID']}: HTTP {e.status_code} "
            f"{e.message}. Is the API key in the same Console workspace as the environment?"
        )
    hosting = getattr(environment.config, "type", None)
    assert hosting == "self_hosted", (
        f"{environment.name} is {hosting!r}; create an environment with Hosting = Self-hosted in the Claude Console"
    )
    api.beta.environments.work.stats(environment.id)  # the queue is readable


def test_anthropic_environment_key_is_accepted(settings):
    """The environment key (not the API key) authenticates the worker to the queue."""
    client = anthropic.Anthropic(
        auth_token=settings["ANTHROPIC_ENVIRONMENT_KEY"],
        base_url=settings.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
    )
    try:
        client.beta.environments.work.stats(settings["ANTHROPIC_ENVIRONMENT_ID"])
    except anthropic.APIStatusError as e:
        if e.status_code in (401, 403):
            pytest.fail(
                "The environment key was rejected: generate a new one on the environment's page in the "
                "Claude Console (it is shown once) and update ANTHROPIC_ENVIRONMENT_KEY."
            )
        raise


@pytest.mark.genie
def test_genie_one_access(workspace_client):
    import asyncio

    from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

    try:
        server = genie_one_server(workspace_client)
        names = [t["name"] for t in asyncio.run(alist_mcp_custom_tools(workspace_client, server))]
    except Exception as e:  # noqa: BLE001 - includes anyio ExceptionGroups
        pytest.fail(
            f"Genie One MCP not reachable ({type(e).__name__}: {e}). Needs Genie One enabled and "
            "EXECUTE on system.ai.genie_one_mcp for this identity. Set SKIP_GENIE=1 to skip."
        )
    assert "genie_ask_and_wait" in names
