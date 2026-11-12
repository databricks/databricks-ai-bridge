"""Build an execution artifact under the creator; session compute remains App-owned."""

import json
import logging
import shlex
import time
from pathlib import Path
from uuid import uuid4

from databricks.sdk.errors import DatabricksError

from databricks_agentkit.sandbox.artifact_scripts import BUILD

LOG = logging.getLogger(__name__)


def stop_builder(lakebox, sandbox_id):
    """Stop once, reconcile an uncertain response, and verify terminal state."""
    failure = None
    try:
        lakebox.api("POST", f"/sandboxes/{sandbox_id}/stop", {"sandbox_id": sandbox_id})
    except DatabricksError as exc:
        failure = exc
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        current = lakebox.get(sandbox_id)
        if current is None or current.get("status", "").lower() in {"stopped", "terminated"}:
            return
        if current.get("status", "").lower() not in {"stopping", "running"}:
            break
        time.sleep(1)
    raise RuntimeError(f"Builder {sandbox_id} did not stop; cleanup must be retried") from failure


def build_artifact(workspace, manifest, bundle, lakebox, save, *, progress=None):
    """Persist intent before create; publish only after build and cleanup succeed.

    Failed attempts retain source and bounded builder logs. `environment prepare`
    retries with a new owned staging path after reconciling the previous builder.
    """
    report = progress or (lambda stage: None)
    previous = manifest.get("builder")
    if previous and previous.get("status") != "stopped":
        sandbox = (
            lakebox.get(previous["sandbox_id"])
            if previous.get("sandbox_id")
            else lakebox.find_named(previous["name"])
        )
        if sandbox:
            stop_builder(lakebox, sandbox["sandboxId"])
        elif not previous.get("sandbox_id"):
            raise RuntimeError(
                "Earlier builder create is unresolved; inspect its saved name before retrying"
            )
        previous["status"] = "stopped"
        save()
    attempt = uuid4().hex
    builder = {"name": "ab-build-" + attempt, "status": "creating"}
    manifest.update(status="building", builder=builder, error=None)
    save()
    sandbox = None
    primary_error = None
    create_attempted = False
    artifact_path = str(Path(manifest["manifest_path"]).parent / "artifacts" / (attempt + ".tar"))
    start = time.perf_counter()
    try:
        workspace.workspace.mkdirs(str(Path(artifact_path).parent))
        report("Creating artifact build sandbox")
        try:
            create_attempted = True
            sandbox = lakebox.create(builder["name"])
        except Exception:
            sandbox = lakebox.find_named(builder["name"])
            if sandbox is None:
                raise
        builder.update(sandbox_id=sandbox["sandboxId"], status="building")
        save()
        lakebox.set_idle_timeout(sandbox["sandboxId"])
        deadline = time.monotonic() + 240
        while sandbox.get("status", "").lower() != "running":
            if time.monotonic() > deadline:
                raise TimeoutError("Build sandbox did not become running")
            time.sleep(1)
            sandbox = lakebox.get(sandbox["sandboxId"])
        with lakebox.connect(sandbox) as remote:
            home = remote.run('printf "%s" "$HOME"').strip()
            root = home + "/.agentbricks-build/" + attempt
            request = {
                "source_hash": bundle.digest,
                "entrypoint": manifest["entrypoint"],
                "artifact_path": artifact_path,
            }
            for name, data in (
                ("source.tar.gz", bundle.data),
                ("build.py", BUILD.encode()),
                ("request.json", json.dumps(request).encode()),
            ):
                remote.run(
                    f"mkdir -p {shlex.quote(root)} && cat > {shlex.quote(root + '/' + name)}", data
                )
            report("Building locked wheelhouse and validating an offline installation")
            launch = "import subprocess,sys; from pathlib import Path; p=Path(sys.argv[1]); f=(p/'build.log').open('wb'); subprocess.Popen([sys.executable,str(p/'build.py'),str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)"
            remote.run(f"python3 -c {shlex.quote(launch)} {shlex.quote(root)}")
            deadline = time.monotonic() + 900
            while True:
                output = remote.run(
                    f"if test -f {shlex.quote(root + '/build-result.json')}; then cat {shlex.quote(root + '/build-result.json')}; fi"
                )
                if output.strip():
                    result = json.loads(output)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Artifact build exceeded 15 minutes")
                time.sleep(2)
            # Keep bounded diagnostics, never environment/credential dumps.
            builder["log_tail"] = remote.run("tail -c 12000 " + shlex.quote(root + "/build.log"))
            if not result["ok"]:
                raise RuntimeError(result["error"] + "\n" + builder["log_tail"][-4000:])
            artifact = result["artifact"]
            manifest["execution_artifact"] = artifact
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        builder["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        # A lost create reply may still have created compute. Retain intent if
        # lookup also fails so the next explicit prepare can reconcile it.
        try:
            if sandbox is None and create_attempted:
                sandbox = lakebox.find_named(builder["name"])
            if sandbox:
                report("Stopping build sandbox")
                stop_builder(lakebox, sandbox["sandboxId"])
                builder["sandbox_id"] = sandbox["sandboxId"]
            elif create_attempted:
                raise RuntimeError(
                    "Builder creation is unresolved; retained intent requires reconciliation"
                )
            builder["status"] = "stopped"
        except Exception as exc:
            builder.update(status="cleanup_failed", cleanup_error=f"{type(exc).__name__}: {exc}")
            if primary_error is None:
                raise
            LOG.warning("Artifact builder cleanup failed; retained for reconciliation")
        finally:
            save()
    return artifact
