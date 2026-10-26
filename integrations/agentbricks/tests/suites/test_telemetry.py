"""CLI telemetry: a command uploads its event to the workspace it authenticated against.

Only the fake workspace can receive the upload; Lumberjack ingestion needs a workspace with the
registered schema deployed. The event's fields are the unit tests' job (`telemetry_test.py`).
"""

from __future__ import annotations

import json
import pathlib
import time

from harness.agentbricks_cli import AgentbricksCli
from harness.fake_workspace import TELEMETRY_PATH, TelemetryUpload
from harness.target import TargetWorkspace, require_fake_workspace


def test_telemetry(
    agentbricks_cli: AgentbricksCli, target_workspace: TargetWorkspace, tmp_path: pathlib.Path
) -> None:
    workspace = require_fake_workspace(target_workspace, "a telemetry receiver")
    uploaded = len(workspace.telemetry)
    # The harness opts every command out of telemetry; this is the one that must send it.
    agentbricks_cli.run(
        f"init --framework openai --server custom {tmp_path / 'agent'}",
        env={"AGENTBRICKS_DISABLE_TELEMETRY": ""},
    )

    upload = _next_upload(workspace.telemetry, uploaded)

    assert upload.authorization == "Bearer fake-token"
    (proto_log,) = upload.body["protoLogs"]
    log = json.loads(proto_log)["entry"]["agentbricks_cli_log"]
    assert log["command_path"] == "agentbricks init"
    assert log["exit_code"] == 0


def _next_upload(uploads: list[TelemetryUpload], already_seen: int) -> TelemetryUpload:
    """The upload is sent on a daemon thread the CLI waits for only briefly, so allow it to land."""
    deadline = time.monotonic() + 10
    while len(uploads) <= already_seen:
        assert time.monotonic() < deadline, f"no upload to {TELEMETRY_PATH}"
        time.sleep(0.1)
    return uploads[already_seen]
