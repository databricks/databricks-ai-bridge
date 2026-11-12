from unittest.mock import MagicMock

from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.jobs import RunManager


def setup_job():
    # The job manager is real; its DB and remote execution boundaries are mocked.
    control, orchestrator = MagicMock(), MagicMock()
    bundle = bundle_files({"pyproject.toml": b"test", "uv.lock": b"lock"})
    control.agent.return_value = {
        "id": "agent-a",
        "bundle_data": bundle.data,
        "source_hash": bundle.digest,
        "file_count": bundle.file_count,
        "entrypoint": "custom.agent:invoke",
    }
    row = {
        "id": "run-a",
        "agent_id": "agent-a",
        "session_id": "session-a",
        "invocation_id": "call-a",
        "input": {"message": "hello"},
        "attempt_id": "attempt-a",
    }
    return RunManager(control, orchestrator), control, orchestrator, row, bundle


def test_job_executes_the_saved_revision_with_its_entrypoint_and_session_scope():
    manager, control, orchestrator, row, bundle = setup_job()
    orchestrator.invoke.return_value = {"result": {"reply": "hello"}}
    manager.execute(row)
    args, kwargs = orchestrator.invoke.call_args
    assert args == ("session-a", "call-a", {"message": "hello"})
    assert kwargs["registry_session_id"] == scoped_session("agent-a", "session-a")
    assert kwargs["bundle"] == bundle
    assert kwargs["entrypoint"] == "custom.agent:invoke"
    kwargs["progress"]("Installing")
    control.progress.assert_called_once_with("run-a", "Installing", attempt_id="attempt-a")
    control.track.assert_called_once_with("run-a", "attempt-a")
    assert kwargs["lifecycle"] is control.track.return_value.__enter__.return_value
    control.finish.assert_called_once_with(
        "run-a",
        result={
            "result": {"reply": "hello"},
            "session_id": "session-a",
            "agent_id": "agent-a",
        },
        attempt_id="attempt-a",
    )


def test_job_failure_is_persisted_for_the_ui():
    manager, control, orchestrator, row, _ = setup_job()
    orchestrator.invoke.side_effect = RuntimeError("Dependency installation failed")
    manager.execute(row)
    control.finish.assert_called_once_with(
        "run-a", error="RuntimeError: Dependency installation failed", attempt_id="attempt-a"
    )
