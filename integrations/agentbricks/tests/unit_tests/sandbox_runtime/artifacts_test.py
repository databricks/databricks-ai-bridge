"""Artifact integrity, revision binding and builder cleanup at real boundaries."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import InternalError

from databricks_agentbricks.services.environment_artifact import build_artifact, stop_builder
from databricks_agentkit.sandbox.artifact_scripts import COMMON, INSTALL
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.environment import ARTIFACT_FORMAT, bind_environment, sandbox_scope


def fingerprint():
    scope = {}
    exec(COMMON, scope)
    return scope["target"]()


def descriptor(**updates):
    return {
        "format": "agentbricks.wheelhouse.v1",
        "path": "/env/artifacts/run.tar",
        "sha256": "a" * 64,
        "bytes": 10240,
        "source_hash": "b" * 64,
        "entrypoint": "agent:invoke",
        "target": fingerprint(),
        **updates,
    }


def run_install(tmp_path, archive, artifact):
    (tmp_path / "execution.tar").write_bytes(archive)
    artifact.update(bytes=len(archive), sha256=hashlib.sha256(archive).hexdigest())
    return subprocess.run(
        [sys.executable, "-c", INSTALL, str(tmp_path), json.dumps(artifact)],
        capture_output=True,
        text=True,
    )


def tar_bytes(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return out.getvalue()


def test_runtime_rejects_target_mismatch_before_extracting(tmp_path):
    artifact = descriptor(target={"python": "incompatible"})
    result = run_install(tmp_path, tar_bytes({}), artifact)
    assert result.returncode != 0 and "incompatible" in result.stderr
    assert not (tmp_path / "artifact").exists()
    assert not (tmp_path / "ready.json").exists()


def test_runtime_rejects_changed_archive_before_extracting(tmp_path):
    (tmp_path / "execution.tar").write_bytes(b"tampered")
    result = subprocess.run(
        [sys.executable, "-c", INSTALL, str(tmp_path), json.dumps(descriptor())],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "archive digest mismatch" in result.stderr
    assert not (tmp_path / "ready.json").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_runtime_rejects_unsafe_archive_paths(tmp_path, name):
    result = run_install(tmp_path, tar_bytes({name: b"unsafe"}), descriptor())
    assert result.returncode != 0 and "Unsafe execution artifact member" in result.stderr
    assert not (tmp_path / "ready.json").exists()


def test_runtime_verifies_inner_inventory_before_running_installer(tmp_path):
    artifact = descriptor()
    manifest = {
        "format": artifact["format"],
        "source_hash": artifact["source_hash"],
        "entrypoint": artifact["entrypoint"],
        "target": artifact["target"],
        "files": {"wheels/agent.whl": {"bytes": 4, "sha256": "0" * 64}},
    }
    archive = tar_bytes(
        {"artifact.json": json.dumps(manifest).encode(), "wheels/agent.whl": b"oops"}
    )
    result = run_install(tmp_path, archive, artifact)
    assert result.returncode != 0 and "file digest mismatch" in result.stderr
    assert not (tmp_path / "tools").exists()


def test_ready_marker_cannot_be_reused_for_a_different_artifact(tmp_path):
    (tmp_path / "ready.json").write_text(json.dumps({"artifact_hash": "f" * 64}))
    result = run_install(tmp_path, tar_bytes({}), descriptor())
    assert result.returncode != 0 and "differs from the pinned revision" in result.stderr


def builder_setup(monkeypatch, *, ok=True):
    from databricks_agentbricks.services import environment_artifact as service

    monkeypatch.setattr(service.time, "sleep", lambda _: None)
    lakebox, workspace, remote = MagicMock(), MagicMock(), MagicMock()
    sandbox = {"sandboxId": "builder-one", "status": "Running"}
    lakebox.create.return_value = sandbox
    lakebox.get.return_value = {**sandbox, "status": "Stopped"}
    artifact = descriptor()

    def run(command, *args, **kwargs):
        if command.startswith('printf "%s"'):
            return "/home/sandbox"
        if command.startswith("if test"):
            return json.dumps({"ok": ok, "artifact": artifact, "error": "missing binary wheel"})
        return "bounded build logs"

    remote.run.side_effect = run
    lakebox.connect.return_value.__enter__.return_value = remote
    manifest = {"manifest_path": "/env/environment.json", "entrypoint": "agent:invoke"}
    states = []
    save = lambda: states.append(copy.deepcopy(manifest))
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    return service, lakebox, workspace, remote, manifest, states, save, bundle


def test_successful_builder_is_stopped_before_result_is_returned(monkeypatch):
    _, lakebox, workspace, _, manifest, states, save, bundle = builder_setup(monkeypatch)
    artifact = build_artifact(workspace, manifest, bundle, lakebox, save)
    assert artifact["sha256"] == "a" * 64
    assert states[0]["builder"]["status"] == "creating"
    assert "sandbox_id" not in states[0]["builder"]
    assert states[1]["builder"]["sandbox_id"] == "builder-one"
    assert states[-1]["builder"]["status"] == "stopped"
    lakebox.api.assert_called_once_with(
        "POST", "/sandboxes/builder-one/stop", {"sandbox_id": "builder-one"}
    )


def test_failed_build_stops_compute_and_does_not_produce_an_artifact(monkeypatch):
    _, lakebox, workspace, _, manifest, states, save, bundle = builder_setup(monkeypatch, ok=False)
    with pytest.raises(RuntimeError, match="missing binary wheel"):
        build_artifact(workspace, manifest, bundle, lakebox, save)
    assert states[-1]["builder"]["status"] == "stopped"
    assert "execution_artifact" not in manifest
    lakebox.api.assert_called_once()


def test_ambiguous_create_reconciles_owned_name_without_second_create(monkeypatch):
    _, lakebox, workspace, _, manifest, _, save, bundle = builder_setup(monkeypatch)
    lakebox.create.side_effect = InternalError("lost create reply")
    lakebox.find_named.return_value = {"sandboxId": "builder-one", "status": "Running"}
    build_artifact(workspace, manifest, bundle, lakebox, save)
    lakebox.create.assert_called_once()
    lakebox.find_named.assert_called_once_with(manifest["builder"]["name"])


def test_cleanup_failure_is_saved_and_blocks_success(monkeypatch):
    service, lakebox, workspace, _, manifest, states, save, bundle = builder_setup(monkeypatch)
    monkeypatch.setattr(service, "stop_builder", MagicMock(side_effect=RuntimeError("cannot stop")))
    with pytest.raises(RuntimeError, match="cannot stop"):
        build_artifact(workspace, manifest, bundle, lakebox, save)
    assert states[-1]["builder"]["status"] == "cleanup_failed"
    assert manifest["status"] != "published"


def test_cancelled_stop_is_reconciled_without_replaying_mutation():
    lakebox = MagicMock()
    lakebox.api.side_effect = InternalError("cancelled")
    lakebox.get.return_value = {"status": "Stopped"}
    stop_builder(lakebox, "builder")
    lakebox.api.assert_called_once()


def test_v4_binds_per_session_compute_without_claiming_builder_ownership():
    orchestrator = MagicMock(namespace="test")
    manifest = {
        "namespace": "test",
        "format": ARTIFACT_FORMAT,
        "status": "published",
        "runtime_key": "environment:test",
    }
    bind_environment(orchestrator, manifest)
    assert sandbox_scope(manifest) == "session"
    assert orchestrator.sandbox_scope == "session"
    assert orchestrator.prepared_session_id is None
    orchestrator.execution_principal.assert_not_called()


def test_reused_installation_checks_runtime_fingerprint(tmp_path):
    from databricks_agentkit.sandbox.artifact_scripts import CHECK_READY

    artifact = descriptor(target={"python": "incompatible"})
    (tmp_path / "ready.json").write_text(json.dumps({"artifact_hash": artifact["sha256"]}))
    for script in (CHECK_READY, INSTALL):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), json.dumps(artifact)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "incompatible" in result.stderr


def test_unresolved_builder_creation_is_never_reported_as_stopped(monkeypatch):
    _, lakebox, workspace, _, manifest, states, save, bundle = builder_setup(monkeypatch)
    lakebox.create.side_effect = InternalError("lost create reply")
    lakebox.find_named.return_value = None
    with pytest.raises(InternalError, match="lost create reply"):
        build_artifact(workspace, manifest, bundle, lakebox, save)
    assert states[-1]["builder"]["status"] == "cleanup_failed"
    assert "reconciliation" in states[-1]["builder"]["cleanup_error"]
    with pytest.raises(RuntimeError, match="unresolved"):
        build_artifact(workspace, manifest, bundle, lakebox, save)
    lakebox.create.assert_called_once()


def test_v4_cannot_fall_back_to_source_when_artifact_binding_is_missing():
    from databricks_agentkit.sandbox.environment import saved_bundle

    control = MagicMock()
    control.agent.return_value = {"source_hash": "source"}
    with pytest.raises(ValueError, match="artifact does not match"):
        saved_bundle(
            control, {"format": ARTIFACT_FORMAT, "agent_id": "agent", "source_hash": "source"}
        )


def test_job_uses_its_own_revisions_artifact():
    from jobs_test import setup_job

    manager, control, orchestrator, row, _ = setup_job()
    artifact = descriptor()
    control.agent.return_value["execution_artifact"] = artifact
    orchestrator.invoke.return_value = {"result": {}}
    manager.execute(row)
    assert orchestrator.invoke.call_args.kwargs["execution_artifact"] == artifact


def test_source_archive_lost_upload_reply_is_reconciled_by_checksum():
    from databricks_agentbricks.prepared_environment import _source_archive

    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    workspace = MagicMock()
    workspace.workspace.upload.side_effect = InternalError("reply lost")
    workspace.workspace.download.return_value.__enter__.return_value = io.BytesIO(bundle.data)
    result = _source_archive(
        workspace, {"manifest_path": "/env/environment.json", "source_hash": bundle.digest}, bundle
    )
    assert result == bundle
    workspace.workspace.upload.assert_called_once()


def test_rejected_prepare_lock_never_overwrites_active_builder(monkeypatch):
    from databricks_agentbricks import prepared_environment as service
    from databricks_agentkit.sandbox.store import SessionBusy

    workspace, registry = MagicMock(), MagicMock()
    registry.session.return_value.__enter__.side_effect = SessionBusy("already building")
    manifest = {
        "manifest_path": "/env/environment.json",
        "namespace": "env",
        "status": "building",
        "builder": {"name": "active-builder"},
    }
    monkeypatch.setattr(service, "_source_archive", lambda *a: "frozen-source")
    monkeypatch.setattr(service, "services", lambda *a: (registry, None, None, None))
    write = MagicMock()
    monkeypatch.setattr(service, "write_manifest", write)
    with pytest.raises(SessionBusy):
        service.publish_artifact(workspace, manifest)
    write.assert_not_called()
    assert manifest["builder"]["name"] == "active-builder"


def test_retry_recovers_interrupted_source_upload_only_when_hash_matches(monkeypatch):
    from databricks.sdk.errors import NotFound

    from databricks_agentbricks.prepared_environment import _source_archive
    from databricks_agentkit.sandbox.sources import SourceLoader

    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    workspace = MagicMock()
    workspace.workspace.download.side_effect = NotFound("upload was interrupted")
    load = MagicMock(return_value=bundle)
    monkeypatch.setattr(SourceLoader, "load", load)
    manifest = {
        "manifest_path": "/env/environment.json",
        "source_path": "/env/source",
        "source_hash": bundle.digest,
    }
    assert _source_archive(workspace, manifest) == bundle
    workspace.workspace.upload.assert_called_once()
    load.return_value = bundle_files({"pyproject.toml": b"changed", "uv.lock": b"locked"})
    with pytest.raises(Exception, match="Source changed"):
        _source_archive(workspace, manifest)
    assert workspace.workspace.upload.call_count == 1


def test_initial_creator_reloads_builder_intent_after_acquiring_lock(monkeypatch):
    from databricks_agentbricks import prepared_environment as service

    registry = MagicMock()
    stale = {"manifest_path": "/env/environment.json", "namespace": "env", "status": "uploading"}
    current = {
        **stale,
        "status": "failed",
        "builder": {"name": "earlier-builder", "status": "cleanup_failed"},
    }
    monkeypatch.setattr(service, "_source_archive", lambda *a: "frozen-source")
    monkeypatch.setattr(service, "services", lambda *a: (registry, None, None, None))
    monkeypatch.setattr(service, "read_manifest", lambda *a: copy.deepcopy(current))
    publish = MagicMock(return_value="built")
    monkeypatch.setattr(service, "_publish_artifact", publish)
    assert service.publish_artifact(None, stale, bundle="already-uploaded") == "built"
    assert publish.call_args.args[1]["builder"] == current["builder"]


@pytest.mark.parametrize("exit_code", [0, 1, None, "missing"])
def test_long_artifact_commands_use_streaming_rpc_and_require_completion(monkeypatch, exit_code):
    from databricks_agentkit.sandbox import _command_pb2 as wire
    from databricks_agentkit.sandbox import streaming
    from databricks_agentkit.sandbox.lakebox import HttpsRemote, RemoteError

    client = MagicMock()
    responses = [
        wire.ExecuteCommandResponse(
            process_event=wire.ProcessEvent(data=wire.DataEvent(stdout=b'{"installed":true}'))
        )
    ]
    if exit_code == "missing":
        responses.append(
            wire.ExecuteCommandResponse(process_event=wire.ProcessEvent(end=wire.EndEvent()))
        )
    elif exit_code is not None:
        responses.append(
            wire.ExecuteCommandResponse(
                process_event=wire.ProcessEvent(end=wire.EndEvent(exit_code=exit_code))
            )
        )
    client.execute.return_value = iter(responses)
    monkeypatch.setattr(streaming, "ExecuteCommand", lambda *a: client)
    workspace = MagicMock()
    remote = HttpsRemote(workspace, "sandbox")
    if exit_code == 0:
        assert remote.run("install", timeout=240, stream=True) == '{"installed":true}'
    else:
        with pytest.raises(RemoteError, match="artifact command failed"):
            remote.run("install", timeout=240, stream=True)
    workspace.api_client.do.assert_not_called()
    client.close.assert_called_once()
    assert "240s" in client.execute.call_args.args[1][-1]
