"""End to end, custom image: the session's sandbox is created from an Artifact Registry image.

Skipped unless ``SANDBOX_E2E_IMAGE_URI`` names an image (``catalog.schema.artifact:tag``) that is pushed
to Databricks Artifact Registry and finished acceleration. The image needs glibc >= 2.25 (no Alpine),
``/bin/bash``, ``python3`` with ``venv``, ``uv`` (the worker bootstrap uses it), and passwordless ``sudo``
if sessions attach memory stores. Optionally set ``SANDBOX_E2E_IMAGE_MARKER`` to a command whose output
proves the custom image is in use (default: ``cat /etc/os-release``).

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 SANDBOX_E2E_IMAGE_URI=main.agents.agent-base:v1 \\
        DATABRICKS_SANDBOX_PROFILE=<profile> uv run pytest tests/integration_tests/test_50_e2e_custom_image.py -v -s
"""

from __future__ import annotations

import os

import pytest
from _support import MAX_REPLY_WORDS, find_or_create_agent, make_launcher

from databricks_anthropic.sandbox import _ops

IMAGE_URI = os.environ.get("SANDBOX_E2E_IMAGE_URI")
MARKER_CMD = os.environ.get("SANDBOX_E2E_IMAGE_MARKER", "cat /etc/os-release")
WORKDIR = "/home/sandbox-agent/agent-workspace"
SYSTEM = (
    f"You run inside a Databricks Sandbox; your working directory is {WORKDIR}. Use as few tool calls as "
    f"possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
@pytest.mark.skipif(not IMAGE_URI, reason="set SANDBOX_E2E_IMAGE_URI to an Artifact Registry image")
def test_session_runs_in_a_custom_image_sandbox(api, settings, workspace_client, run_session):
    agent_id = find_or_create_agent(
        api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
    )
    launcher = make_launcher(workspace_client, settings, image_uri=IMAGE_URI)
    run = run_session(
        launcher,
        agent_id,
        f"Run `echo custom-image-ok` with bash. "
        f"Reply with its output in fewer than {MAX_REPLY_WORDS} words.",
    )
    print(f"\nimage={IMAGE_URI} launched={run.launched} reply={run.reply!r}")
    assert not run.failures, run.failures
    assert run.launched == [run.sandbox_name]
    assert "custom-image-ok" in run.reply

    spec = workspace_client.sandbox.get_sandbox(run.sandbox_name).spec
    assert spec and spec.environment and spec.environment.image_uri == IMAGE_URI, (
        f"sandbox spec: {spec}"
    )
    print(_ops.run(workspace_client, run.sandbox_name, MARKER_CMD, check=False).stdout)
