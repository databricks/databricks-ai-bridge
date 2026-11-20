"""Synchronous invocation orchestration; all framework code runs remotely."""

import hashlib
import json
import os
import re
import shlex
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from databricks_agentkit.sandbox.artifact_scripts import CHECK_READY, FETCH, INSTALL
from databricks_agentkit.sandbox.bootstrap import SCRIPT
from databricks_agentkit.sandbox.bundle import build_bundle
from databricks_agentkit.sandbox.lifecycle import RunLifecycle
from databricks_agentkit.sandbox.persistent import CLIENT_SCRIPT
from databricks_agentkit.sandbox.runner import SCRIPT as RUNNER_SCRIPT
from databricks_agentkit.sandbox.session_client import RevisionConflict, SessionSandboxClient, timed
from databricks_agentkit.sandbox.sources import DEFAULT_ENTRYPOINT, validate_entrypoint
from databricks_agentkit.sandbox.streaming import FileFollower


class Orchestrator:
    def __init__(self, registry, lakebox, source: Path, namespace: str, *, worker_mode=None):
        self.registry, self.lakebox = registry, lakebox
        self.source, self.namespace = source, namespace
        self.prepared_session_id = None
        self.environment_session_id = None
        self.sandbox_scope = "session"
        self.sandboxes = SessionSandboxClient(registry, lakebox, namespace)
        self.worker_mode = worker_mode or os.getenv("SANDBOX_WORKER_MODE", "persistent")
        if self.worker_mode not in {"persistent", "oneshot"}:
            raise ValueError("SANDBOX_WORKER_MODE must be persistent or oneshot")

    def execution_session_id(self, registry_session_id):
        """Use the same mapping for invocation, status reads, and stop requests."""
        if self.prepared_session_id:
            return self.prepared_session_id
        if self.environment_session_id and self.sandbox_scope == "environment":
            return self.environment_session_id
        return registry_session_id

    def execution_principal(self):
        return self.sandboxes.execution_principal()

    def prepare(self, preparation_id, *, bundle, entrypoint, progress=None):
        """Install and validate source without submitting an agent invocation."""
        return self.invoke(
            preparation_id,
            "",
            {},
            bundle=bundle,
            entrypoint=entrypoint,
            progress=progress,
            _prepare_only=True,
        )

    def prepare_session(self, session_id, **kwargs):
        """Install source and initialize the resident handler, without invoking it."""
        if self.worker_mode != "persistent":
            raise ValueError("Session preparation requires persistent workers")
        return self.invoke(session_id, "", {}, _prepare_only=True, _warm_worker=True, **kwargs)

    def invoke(
        self,
        session_id,
        invocation_id,
        value,
        *,
        bundle=None,
        execution_artifact=None,
        entrypoint=DEFAULT_ENTRYPOINT,
        progress=None,
        registry_session_id=None,
        source_revision=None,
        emit=None,
        stream_ready=None,
        lifecycle=None,
        _prepare_only=False,
        _warm_worker=False,
        _allow_install=True,
    ):
        started = time.perf_counter()
        metrics = {}
        entrypoint = validate_entrypoint(entrypoint)
        if execution_artifact and (
            bundle is None
            or execution_artifact["source_hash"] != bundle.digest
            or execution_artifact["entrypoint"] != entrypoint
        ):
            raise RevisionConflict("Execution artifact does not match the agent revision")
        report = progress or (lambda stage: None)
        with RunLifecycle(lifecycle) as timeline:
            timeline.start("sandbox_start")
            with timed(metrics, "source_pack"):
                bundle = bundle or build_bundle(self.source)
            report("Looking up session")
            registry_key = registry_session_id if registry_session_id is not None else session_id
            state_key = hashlib.sha256(f"{self.namespace}\0{registry_key}".encode()).hexdigest()
            execution_key = self.execution_session_id(registry_key)
            with self.sandboxes.invocation(execution_key):
                allocation = self.sandboxes.get_or_create(
                    execution_key,
                    source_hash=bundle.digest,
                    source_revision=source_revision,
                    allow_create=_allow_install and (not self.prepared_session_id or _prepare_only),
                    progress=report,
                    metrics=metrics,
                )
                sandbox, sandbox_id = allocation.sandbox, allocation.id
                created, replaced = allocation.created, allocation.replaced
                try:
                    connect_started = time.perf_counter()
                    with self.lakebox.connect(sandbox) as remote:
                        metrics["remote_connect"] = round(
                            (time.perf_counter() - connect_started) * 1000, 2
                        )
                        with timed(metrics, "environment_probe"):
                            home = remote.run('printf "%s" "$HOME"').strip()
                            deployment = hashlib.sha256(self.namespace.encode()).hexdigest()[:16]
                            root = f"{home}/.agentbricks-prototype/{deployment}/{bundle.digest}"
                            if execution_artifact:
                                artifact_hash = execution_artifact["sha256"]
                                if not re.fullmatch(r"[a-f0-9]{64}", artifact_hash):
                                    raise ValueError("Invalid execution artifact hash")
                                root += "/" + artifact_hash
                                ready = remote.run(
                                    f"python3 -c {shlex.quote(CHECK_READY)} {shlex.quote(root)} "
                                    f"{shlex.quote(json.dumps(execution_artifact))}"
                                ).strip()
                            else:
                                ready = remote.run(
                                    f"if test -f {shlex.quote(root + '/ready.json')}; then "
                                    f"cat {shlex.quote(root + '/ready.json')}; fi"
                                ).strip()
                        bootstrap_reused = bool(ready)
                        if ready:
                            timeline.reuse_installation()
                            bootstrap = json.loads(ready)
                            if (
                                execution_artifact
                                and bootstrap.get("artifact_hash") != execution_artifact["sha256"]
                            ):
                                raise RevisionConflict(
                                    "Installed artifact does not match the agent revision"
                                )
                        else:
                            if not _allow_install or self.prepared_session_id and not _prepare_only:
                                raise RuntimeError(
                                    "Prepared installation is missing; prepare a new environment before invoking"
                                )
                            timeline.start("source_copy")
                            if execution_artifact:
                                report("Downloading verified execution artifact")
                                encoded = shlex.quote(json.dumps(execution_artifact))
                                with timed(metrics, "artifact_transfer"):
                                    output = remote.run(
                                        f"python3 -c {shlex.quote(FETCH)} {shlex.quote(root)} {encoded}",
                                        timeout=240,
                                        stream=True,
                                    )
                                    transfer = json.loads(output.strip().splitlines()[-1])
                                timeline.complete(details=transfer)
                                timeline.start("install_build")
                                report("Installing prebuilt wheels offline")
                                with timed(metrics, "bootstrap"):
                                    output = remote.run(
                                        f"python3 -c {shlex.quote(INSTALL)} {shlex.quote(root)} {encoded}",
                                        timeout=240,
                                        stream=True,
                                    )
                                    bootstrap = json.loads(output.strip().splitlines()[-1])
                            else:
                                report("Uploading source")
                                with timed(metrics, "source_transfer"):
                                    remote.run(
                                        f"mkdir -p {shlex.quote(root)} && cat > "
                                        f"{shlex.quote(root + '/source.tar.gz')}",
                                        bundle.data,
                                    )
                                timeline.start("install_build")
                                with timed(metrics, "bootstrap"):
                                    report("Installing dependencies and building agent")
                                    output = remote.run(
                                        f"python3 -c {shlex.quote(SCRIPT)} {shlex.quote(root)} "
                                        f"{shlex.quote(bundle.digest)} {shlex.quote(entrypoint)}",
                                        timeout=900,
                                    )
                                    bootstrap = json.loads(output.strip().splitlines()[-1])
                            timeline.complete(details=bootstrap["timings_ms"])
                        if not bootstrap_reused or allocation.installation_status != "ready":
                            self.sandboxes.record(
                                execution_key, bundle.digest, sandbox_id, "ready", bootstrap
                            )
                        state_dir = (
                            f"{home}/.agentbricks-prototype/{deployment}/sessions/{state_key}"
                        )
                        request = {
                            "session_id": session_id,
                            "invocation_id": invocation_id,
                            "input": value,
                        }
                        metrics["environment_startup"] = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                        report("Environment ready" if _prepare_only else "Running agent")
                        # cwd is outside the source tree: imports must come from the built wheel.
                        result = None
                        worker = None
                        if _warm_worker:
                            timeline.start("handler_prepare")
                            report("Initializing resident handler")
                            result, worker = self._invoke_worker(
                                remote,
                                home,
                                bootstrap,
                                state_dir,
                                entrypoint,
                                request,
                                None,
                                metrics,
                                prepare=True,
                            )
                            timeline.complete(
                                details=(result.get("details") or {}).get("timings_ms")
                            )
                            metrics["environment_startup"] = round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                        timeline.ready(metrics["environment_startup"], prepare_only=_prepare_only)
                        if not _prepare_only:
                            timeline.start("agent_run")
                            result, worker = self._invoke_worker(
                                remote,
                                home,
                                bootstrap,
                                state_dir,
                                entrypoint,
                                request,
                                emit,
                                metrics,
                                stream_ready=stream_ready,
                            )
                            timeline.complete(
                                details=result.get("worker_timings_ms")
                                if isinstance(result, dict)
                                else None
                            )
                except Exception:
                    self.sandboxes.record(execution_key, bundle.digest, sandbox_id, "failed")
                    raise
            metrics["total"] = round((time.perf_counter() - started) * 1000, 2)
            return {
                "session_id": session_id,
                "invocation_id": invocation_id,
                "sandbox_id": sandbox_id,
                "sandbox_scope": self.sandbox_scope,
                "execution_transport": self.lakebox.transport,
                "stream_transport": "ExecuteCommand(tail -F)" if emit or stream_ready else None,
                "sandbox_created": created,
                "sandbox_replaced": replaced,
                "environment_reused": bootstrap_reused,
                "source_hash": bundle.digest,
                "artifact_hash": execution_artifact["sha256"] if execution_artifact else None,
                "entrypoint": entrypoint,
                "bundle_bytes": len(bundle.data),
                "bundle_files": bundle.file_count,
                "timings_ms": metrics,
                "lifecycle": timeline.snapshot,
                "initial_bootstrap_timings_ms": bootstrap["timings_ms"],
                "result": result,
                "worker": worker,
            }

    def _invoke_worker(
        self,
        remote,
        home,
        bootstrap,
        state_dir,
        entrypoint,
        request,
        emit,
        metrics,
        *,
        prepare=False,
        stream_ready=None,
    ):
        with timed(metrics, "worker_process_and_execution"):
            command = (
                f"cd {shlex.quote(home)} && {shlex.quote(bootstrap['python'])} "
                f"-c {shlex.quote(RUNNER_SCRIPT)} {shlex.quote(state_dir)} "
                f"{shlex.quote(entrypoint)}"
            )
            if self.worker_mode == "persistent":
                # The controller uses stock Python; only the resident process
                # imports the agent and framework in the installed environment.
                command = (
                    f"cd {shlex.quote(home)} && python3 -c {shlex.quote(CLIENT_SCRIPT)} "
                    f"{shlex.quote(state_dir)} {shlex.quote(entrypoint)} "
                    f"{shlex.quote(bootstrap['python'])}"
                )
            if prepare:
                command += " --prepare"
            data = json.dumps(request).encode()
            if emit or stream_ready:
                invocation = hashlib.sha256(request["invocation_id"].encode()).hexdigest()
                directory = state_dir + "/streams"
                # A fresh attempt file is never truncated or reused after retry.
                path = f"{directory}/{invocation}.{uuid4().hex}.jsonl"
                remote.run(f"mkdir -p -m 700 -- {shlex.quote(directory)}")
                if stream_ready:
                    # Publish once before execution. Subscribers can independently
                    # follow this attempt; their lifetime never owns the worker.
                    stream_ready({"sandbox_id": remote.sandbox_id, "path": path})
                reader = FileFollower(remote, path, emit) if emit else nullcontext()
                with reader as follower:
                    try:
                        output = remote.run(
                            command + f" --stream-file {shlex.quote(path)}", data, timeout=300
                        )
                    except Exception:
                        # Preserve the original worker error if a crash also left
                        # an incomplete file. Reader failure never cancels the agent.
                        try:
                            if follower:
                                follower.finish()
                        except Exception:
                            pass
                        raise
                    if follower:
                        follower.finish()
            else:
                output = remote.run(command, data, timeout=300)
            result = json.loads(output.strip().splitlines()[-1])
            if self.worker_mode == "persistent":
                if result.get("ok") is not True:
                    raise RuntimeError("Persistent worker returned an incomplete result")
                return result["result"], result["worker"]
        return result, {"mode": "oneshot"}
