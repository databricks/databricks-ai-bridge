"""CLI telemetry: a command uploads its FrontendLog event to the workspace it authenticated against.

Only the fake workspace can receive the upload and show its payload; Lumberjack ingestion needs a
workspace with the registered schema deployed.
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid

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
        f"-o json init --framework openai --server custom {tmp_path / 'agent'}",
        env={"AGENTBRICKS_DISABLE_TELEMETRY": ""},
    )

    upload = _next_upload(workspace.telemetry, uploaded)

    assert upload.authorization == "Bearer fake-token"
    assert isinstance(upload.body["uploadTime"], int)
    assert upload.body["items"] == []
    assert len(upload.body["protoLogs"]) == 1
    frontend_log = json.loads(upload.body["protoLogs"][0])
    assert uuid.UUID(frontend_log["frontend_log_event_id"])
    log = frontend_log["entry"]["agentbricks_cli_log"]
    assert log["command_path"] == "agentbricks init"
    assert log["exit_code"] == 0
    assert {parameter["name"]: parameter for parameter in log["parameters"]} == {
        "agentbricks.output": {
            "name": "agentbricks.output",
            "choice_value": "AGENTBRICKS_CLI_JSON",
        },
        "agentbricks init.framework": {
            "name": "agentbricks init.framework",
            "choice_value": "AGENTBRICKS_CLI_OPENAI",
        },
        "agentbricks init.server": {
            "name": "agentbricks init.server",
            "choice_value": "AGENTBRICKS_CLI_CUSTOM",
        },
        "agentbricks init.existing": {"name": "agentbricks init.existing", "bool_value": False},
        "agentbricks init.disable_chat_app": {
            "name": "agentbricks init.disable_chat_app",
            "bool_value": False,
        },
        "agentbricks init.enable_chat_app": {
            "name": "agentbricks init.enable_chat_app",
            "bool_value": False,
        },
    }


def _next_upload(uploads: list[TelemetryUpload], already_seen: int) -> TelemetryUpload:
    """The upload is sent on a daemon thread the CLI waits for only briefly, so allow it to land."""
    deadline = time.monotonic() + 10
    while len(uploads) <= already_seen:
        assert time.monotonic() < deadline, f"no upload to {TELEMETRY_PATH}"
        time.sleep(0.1)
    return uploads[already_seen]
