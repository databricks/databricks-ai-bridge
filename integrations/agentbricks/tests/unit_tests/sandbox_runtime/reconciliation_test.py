"""Recovery inspects an existing attempt; it never invokes an agent again."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.sandbox.reconciliation import INSPECT_ATTEMPT, SandboxReconciler
from databricks_agentkit.sandbox.runner import CALL_SCRIPT


def inspect(source):
    result = subprocess.run(
        [sys.executable, "-c", INSPECT_ATTEMPT],
        input=json.dumps(source),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def test_attempt_journal_inspection_and_revocation(tmp_path):
    request = {"session_id": "s", "invocation_id": "i", "input": {}}
    namespace = {}
    exec(CALL_SCRIPT, namespace)
    path = str(tmp_path / "attempt.jsonl")
    source = {
        "path": path,
        "identity": "worker",
        "request_digest": namespace["request_digest"](request),
    }
    assert inspect(source)["status"] == "not_started"
    namespace["begin_attempt"](request, path, "worker")
    assert inspect(source)["status"] == "running"
    with pytest.raises(FileExistsError):
        namespace["begin_attempt"](request, path, "worker")
    namespace["finish_attempt"](request, path, {"ok": True, "result": "saved"}, "worker")
    assert inspect(source) == {"status": "completed", "response": {"ok": True, "result": "saved"}}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        inspect({**source, "identity": "different-worker"})
    revoked = {**source, "path": str(tmp_path / "revoked.jsonl"), "revoke_if_missing": True}
    assert inspect(revoked)["status"] == "not_started"
    with pytest.raises(RuntimeError, match="revoked"):
        namespace["begin_attempt"](request, revoked["path"], "worker")


def setup_recovery():
    request = {"agent_id": "a", "session_id": "s", "invocation_id": "i", "input": {}}
    worker_request = {key: request[key] for key in ("session_id", "invocation_id", "input")}
    source = {
        "protocol_version": 1,
        "sandbox_id": "box",
        "path": "/attempt",
        "source_hash": "hash",
        "identity": "worker",
        "request_digest": hashlib.sha256(
            json.dumps(worker_request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    control, orchestrator = MagicMock(), MagicMock()
    control.call.side_effect = lambda value: value
    control.runtime_store.events.return_value = [
        SimpleNamespace(attempt=1, event={"type": "sandbox.stream.ready", "source": source})
    ]
    orchestrator.sandboxes.runtime.return_value = {"sandbox": {"sandbox_id": "box"}}
    orchestrator.lakebox.get.return_value = {"sandboxId": "box", "status": "Running"}
    emit = MagicMock(return_value=None)
    context = SimpleNamespace(invocation_id="i", attempt=2, emit=emit)
    remote = orchestrator.lakebox.connect.return_value.__enter__.return_value
    reconciler = SandboxReconciler(control, orchestrator, recovery_timeout=0.2)
    return request, context, remote, reconciler, orchestrator, source


def test_recovery_reattaches_then_reads_completed_result_without_execution():
    request, context, remote, reconciler, orchestrator, source = setup_recovery()
    remote.run.side_effect = [
        json.dumps({"status": "running"}),
        json.dumps(
            {
                "status": "completed",
                "response": {"ok": True, "result": {"reply": "done"}, "worker": {"id": "worker"}},
            }
        ),
    ]
    reconciler.recovery_timeout = 1
    result = reconciler.recover(request, context)
    assert result["result"] == {"reply": "done"}
    assert result["recovered"] is True
    assert context.emit.call_args.args[0]["source"] == source
    orchestrator.invoke.assert_not_called()
    orchestrator.lakebox.create.assert_not_called()
    orchestrator.lakebox.start.assert_not_called()


@pytest.mark.parametrize("failure", ["lost", "stopped", "missing", "reassigned", "wrong-request"])
def test_uncertain_outcomes_are_not_replayed(failure):
    request, context, remote, reconciler, orchestrator, _ = setup_recovery()
    remote.run.return_value = json.dumps({"status": "lost"})
    if failure == "stopped":
        orchestrator.lakebox.get.return_value["status"] = "Stopped"
    elif failure == "missing":
        orchestrator.lakebox.get.return_value = None
    elif failure == "reassigned":
        orchestrator.sandboxes.runtime.return_value["sandbox"]["sandbox_id"] = "new-box"
    elif failure == "wrong-request":
        request["input"] = {"changed": True}
    with pytest.raises(RuntimeError):
        reconciler.recover(request, context)
    orchestrator.invoke.assert_not_called()
    orchestrator.lakebox.create.assert_not_called()
    orchestrator.lakebox.start.assert_not_called()


def test_idle_reconciliation_only_observes_owned_mappings():
    _, _, _, reconciler, orchestrator, _ = setup_recovery()
    owned = {"session_key": "a", "sandbox_id": "box", "execution_principal": "app"}
    other = {"session_key": "b", "sandbox_id": "other", "execution_principal": "creator"}
    orchestrator.execution_principal.return_value = "app"
    orchestrator.registry.mapping_page.side_effect = [[owned, other], []]
    orchestrator.lakebox.statuses.return_value = {"box": "stopped"}
    reconciler.reconcile()
    orchestrator.registry.observe.assert_called_once_with(owned, "stopped")
    orchestrator.lakebox.create.assert_not_called()
    orchestrator.lakebox.start.assert_not_called()
    orchestrator.lakebox.connect.assert_not_called()
