"""Observe session compute and reattach fenced invocations without replaying user code.

The App owns sandbox reconciliation. The resident worker owns execution and its
result journal. Monitoring idle sessions never starts compute or imports an agent.
"""

import hashlib
import json
import logging
import shlex
import threading
import time

from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.store import SessionBusy

LOG = logging.getLogger(__name__)

INSPECT_ATTEMPT = r"""
import fcntl
import json
import os
from pathlib import Path
import sys

source = json.load(sys.stdin)
path = Path(source["path"])
result = Path(str(path) + ".result.json")
started = Path(str(path) + ".started.json")
def read(file):
    with file.open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise RuntimeError("Attempt journal exceeds 16 MiB")
    value = json.loads(data)
    if value.get("request_digest") != source["request_digest"] or value.get("identity") != source.get("identity"):
        raise RuntimeError("Attempt journal identity mismatch")
    return value
with Path(str(path) + ".lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if result.exists():
        value = {"status": "completed", "response": read(result)["response"]}
    elif started.exists():
        value = read(started)
        try:
            stat = Path(f'/proc/{int(value["pid"])}/stat').read_text().rsplit(") ", 1)[1].split()
            alive = stat[19] == value["process_start"] and stat[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        value = {"status": "running" if alive else "lost"}
    else:
        # Atomic with begin_attempt: delayed dispatch cannot start after this.
        if source.get("revoke_if_missing"):
            Path(str(path) + ".revoked").touch(mode=0o600)
        value = {"status": "not_started"}
print(json.dumps(value), flush=True)
"""


class SandboxReconciler:
    def __init__(self, control, orchestrator, *, interval=10, recovery_timeout=300):
        self.control, self.orchestrator = control, orchestrator
        self.registry, self.lakebox = orchestrator.registry, orchestrator.lakebox
        self.interval, self.recovery_timeout = interval, recovery_timeout
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._worker, name="sandbox-reconciler", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=1)

    def reconcile(self):
        with self.registry.session("sandbox-reconciler"):
            after = ""
            while not self.stopping.is_set():
                rows = self.registry.mapping_page(after=after)
                if not rows:
                    return
                after = rows[-1]["session_key"]
                owned = [
                    row
                    for row in rows
                    if row["sandbox_id"]
                    and row["execution_principal"] == self.orchestrator.execution_principal()
                ]
                statuses = (
                    self.lakebox.statuses({row["sandbox_id"] for row in owned}) if owned else {}
                )
                for row in owned:
                    self.registry.observe(row, statuses.get(row["sandbox_id"], "missing"))

    def _worker(self):
        while not self.stopping.is_set():
            try:
                self.reconcile()
            except SessionBusy:
                pass  # Another App replica owns this sweep.
            except Exception:
                LOG.exception("Sandbox reconciliation unavailable")
            self.stopping.wait(self.interval)

    def recover(self, request, context):
        events = self.control.call(self.control.runtime_store.events(context.invocation_id))
        source = next(
            (
                event.event["source"]
                for event in reversed(events)
                if event.attempt < context.attempt
                and event.event.get("type") == "sandbox.stream.ready"
            ),
            None,
        )
        if not source or source.get("protocol_version") != 1:
            raise RuntimeError(
                "Interrupted invocation has no recoverable attempt journal; automatic replay is disabled. Inspect the session before retrying."
            )
        worker_request = {key: request[key] for key in ("session_id", "invocation_id", "input")}
        digest = hashlib.sha256(
            json.dumps(worker_request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != source["request_digest"]:
            raise RuntimeError("Recovered attempt does not match the invocation request")
        key = self.orchestrator.execution_session_id(
            scoped_session(request["agent_id"], request["session_id"])
        )
        mapped = self.orchestrator.sandboxes.runtime(key)["sandbox"]
        if not mapped or mapped["sandbox_id"] != source["sandbox_id"]:
            raise RuntimeError(
                "Session sandbox changed; the previous invocation outcome is unknown. Automatic replay is disabled."
            )
        sandbox = self.lakebox.get(source["sandbox_id"])
        if not sandbox:
            raise RuntimeError(
                "Session sandbox was lost; invocation outcome is unknown. Automatic replay is disabled."
            )
        if sandbox.get("status", "").lower() != "running":
            raise RuntimeError(
                "Session sandbox stopped before its result was confirmed. Resume it for inspection; automatic replay is disabled."
            )
        self.control.call(
            context.emit({"type": "sandbox.stage", "stage": "Reattaching to sandbox attempt"})
        )
        self.control.call(context.emit({"type": "sandbox.stream.ready", "source": source}))
        deadline = time.monotonic() + self.recovery_timeout
        startup_deadline = time.monotonic() + 15
        with self.lakebox.connect(sandbox) as remote:
            while not self.stopping.is_set() and time.monotonic() < deadline:
                observed = json.loads(
                    remote.run(
                        f"python3 -c {shlex.quote(INSPECT_ATTEMPT)}",
                        json.dumps(
                            {**source, "revoke_if_missing": time.monotonic() >= startup_deadline}
                        ).encode(),
                        timeout=15,
                    )
                    .strip()
                    .splitlines()[-1]
                )
                if observed["status"] == "completed":
                    response = observed["response"]
                    if response.get("ok") is not True:
                        raise RuntimeError(response.get("error", "Sandbox invocation failed"))
                    return {
                        "session_id": request["session_id"],
                        "invocation_id": request["invocation_id"],
                        "agent_id": request["agent_id"],
                        "sandbox_id": source["sandbox_id"],
                        "source_hash": source["source_hash"],
                        "result": response["result"],
                        "worker": response["worker"],
                        "recovered": True,
                    }
                if observed["status"] == "lost":
                    raise RuntimeError(
                        "Sandbox worker died without a confirmed result; outcome is unknown. Automatic replay is disabled."
                    )
                if observed["status"] == "not_started" and time.monotonic() >= startup_deadline:
                    raise RuntimeError(
                        "Invocation dispatch was interrupted; automatic replay is disabled. Submit a new invocation explicitly."
                    )
                self.stopping.wait(0.25)
        raise RuntimeError(
            "Could not confirm the sandbox invocation outcome; automatic replay is disabled."
        )
