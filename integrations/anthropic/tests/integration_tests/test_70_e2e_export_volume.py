"""End to end, ephemeral + export: outputs are copied to a UC Volume, then the sandbox is deleted.

Skipped unless ``SANDBOX_E2E_EXPORT_VOLUME`` is a UC Volume folder the dispatcher identity can write,
e.g. ``/Volumes/main/agents/session_outputs``. Files the agent writes under ``agent-workspace/outputs``
land in ``<volume folder>/<session id>/``.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 SANDBOX_E2E_EXPORT_VOLUME=/Volumes/<c>/<s>/<v>/claude \\
        DATABRICKS_SANDBOX_PROFILE=<profile> uv run pytest tests/integration_tests/test_70_e2e_export_volume.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest
from _support import MAX_REPLY_WORDS, find_or_create_agent, make_launcher
from databricks.sdk.errors import NotFound

VOLUME = os.environ.get("SANDBOX_E2E_EXPORT_VOLUME", "").rstrip("/")
WORKDIR = "/home/sandbox-agent/agent-workspace"
MARKER = "exported-from-databricks-sandbox"
SYSTEM = (
    f"You run inside a Databricks Sandbox; your working directory is {WORKDIR}. Use as few tool calls as "
    f"possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)


@pytest.mark.e2e
@pytest.mark.skipif(
    not VOLUME, reason="set SANDBOX_E2E_EXPORT_VOLUME to a writable UC Volume folder"
)
def test_outputs_exported_to_volume_then_sandbox_deleted(
    api, settings, workspace_client, run_session
):
    agent_id = find_or_create_agent(
        api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
    )
    launcher = make_launcher(
        workspace_client, settings, export_volume_path=VOLUME
    )  # ephemeral (default)
    run = run_session(
        launcher,
        agent_id,
        f"Write '{MARKER}' to {WORKDIR}/outputs/result.txt using one bash "
        f"command (create the folder). Reply in fewer than {MAX_REPLY_WORDS} words.",
    )
    assert not run.failures and run.launched == [run.sandbox_name], run

    deadline, deleted = time.monotonic() + 240, False
    while time.monotonic() < deadline and not deleted:
        try:
            workspace_client.sandbox.get_sandbox(run.sandbox_name)
            time.sleep(5)
        except NotFound:
            deleted = True
    assert deleted, "the ephemeral sandbox was not deleted after export"

    target = f"{VOLUME}/{run.session_id}/result.txt"
    content = workspace_client.files.download(target).contents.read().decode()
    print(f"\n{target}: {content!r}")
    assert MARKER in content
    workspace_client.files.delete(target)
