"""Bounded pool of initialized, unassigned sandboxes for one deployed revision.

Only allocation is pooled. An assigned sandbox never returns to the pool, and no
agent invocation is submitted by this controller. Interrupted startup is retired
rather than replayed. The platform's idle timeout remains the fallback if the App
disappears; active controllers ping unused workers until their one-hour expiry.
"""

import hashlib
import json
import logging
import shlex
import threading

from databricks_agentkit.sandbox.persistent import CLIENT_SCRIPT, WORKER_VERSION
from databricks_agentkit.sandbox.store import SessionBusy
from databricks_agentkit.sandbox.warm_pool_store import WarmPoolStore

LOG = logging.getLogger(__name__)


class WarmPool:
    def __init__(self, orchestrator, manifest, bundle, *, size, generation):
        if not 0 <= size <= 10:
            raise ValueError("Sandbox warm pool size must be 0–10")
        if orchestrator.sandbox_scope != "session" or orchestrator.worker_mode != "persistent":
            raise ValueError("Warm pools require session sandboxes and persistent workers")
        self.orchestrator, self.manifest, self.bundle = orchestrator, manifest, bundle
        self.registry, self.sandboxes = orchestrator.registry, orchestrator.sandboxes
        key = hashlib.sha256(
            json.dumps(
                [
                    manifest["agent_id"],
                    bundle.digest,
                    manifest["entrypoint"],
                    manifest.get("execution_artifact"),
                    self.sandboxes.execution_principal(),
                    WORKER_VERSION,
                    orchestrator.lakebox.idle_timeout,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.store = WarmPoolStore(self.registry, key, generation, size)
        self.stopping = threading.Event()
        self.wake = threading.Event()
        self.thread = None

    def start(self):
        self.store.initialize()
        self.sandboxes.warm_pool = self
        self.thread = threading.Thread(target=self._worker, name="sandbox-warm-pool", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=1)

    def claim(self, session, source_hash, source_revision):
        if source_hash != self.bundle.digest or source_revision != self.manifest["agent_id"]:
            return False
        result = self.store.claim(session, self.healthy)
        self.wake.set()
        return result

    def healthy(self, slot):
        try:
            details = slot["metadata"]
            lakebox = self.orchestrator.lakebox
            sandbox = lakebox.get(details["sandbox_id"])
            if not sandbox or sandbox.get("status", "").lower() != "running":
                return False
            handle = details["pool_worker"]
            with lakebox.connect(sandbox) as remote:
                output = remote.run(
                    f"python3 -c {shlex.quote(CLIENT_SCRIPT)} {shlex.quote(handle['state_dir'])} "
                    f"{shlex.quote(details['entrypoint'])} {shlex.quote(handle['python'])} --probe",
                    timeout=10,
                )
            worker = json.loads(output.strip().splitlines()[-1])
            return bool(
                worker
                and worker.get("prepared")
                and worker.get("invocation_count") == 0
                and worker["id"] == details["worker"]["id"]
                and worker["identity"] == details["worker"]["identity"]
            )
        except Exception:
            return False

    def retire(self, slot):
        key = slot["slot_id"]
        row = self.sandboxes.runtime(key)["sandbox"]
        if not row and (slot.get("metadata") or {}).get("sandbox_id"):
            raise RuntimeError("Pool sandbox mapping is missing; reserved capacity retained")
        if row and not row["sandbox_id"] and row["status"] == "creating":
            label = (
                f"ab-proto-{hashlib.sha256(self.registry.namespace.encode()).hexdigest()[:8]}-"
                f"{self.registry.session_key(key)[:16]}"
            )
            sandbox = self.orchestrator.lakebox.find_named(label)
            if sandbox is None:
                raise RuntimeError("Pool create outcome is unresolved; reserved capacity retained")
            self.sandboxes.record(key, row["source_hash"], sandbox["sandboxId"], "bootstrapping")
            row = self.sandboxes.runtime(key)["sandbox"]
        if row and row["sandbox_id"]:
            if self.orchestrator.lakebox.get(row["sandbox_id"]) is not None:
                self.sandboxes.stop(key)
        self.store.finish_retirement(key)

    def reconcile(self):
        # One replenisher across replicas; allocation claims use independent row
        # locks and stay available while a replacement spends time installing.
        with self.registry.session("warm-pool:controller"):
            if not self.store.current():
                return
            retained = 0
            for slot in self.store.slots():
                if self.stopping.is_set() or not self.store.current():
                    return
                if slot["status"] == "preparing":
                    self.store.mark(
                        slot["slot_id"],
                        "retiring",
                        expected="preparing",
                        error="Previous pool preparation was interrupted",
                    )
                    slot["status"] = "retiring"
                if slot["status"] == "ready":
                    keep = (
                        slot["pool_key"] == self.store.key
                        and not slot["expired"]
                        and retained < self.store.size
                        and self.healthy(slot)
                    )
                    if keep:
                        retained += 1
                        self.store.mark(slot["slot_id"], "ready", expected="ready", checked=True)
                    elif self.store.mark(slot["slot_id"], "retiring", expected="ready"):
                        slot["status"] = "retiring"
                if slot["status"] == "retiring":
                    try:
                        self.retire(slot)
                    except Exception as error:
                        self.store.mark(
                            slot["slot_id"], "retiring", expected="retiring", error=str(error)
                        )
                        self.store.cooldown(error)
            if self.stopping.is_set():
                return
            slot = self.store.reserve()
            if slot:
                try:
                    result = self.orchestrator.prepare_session(
                        slot["slot_id"],
                        bundle=self.bundle,
                        entrypoint=self.manifest["entrypoint"],
                        source_revision=self.manifest["agent_id"],
                        _pool_fill=True,
                        execution_artifact=self.manifest.get("execution_artifact"),
                    )
                    details = {
                        key: result[key]
                        for key in (
                            "sandbox_id",
                            "worker",
                            "pool_worker",
                            "entrypoint",
                            "timings_ms",
                        )
                    }
                    if not self.store.current() or self.stopping.is_set():
                        self.store.mark(
                            slot["slot_id"], "retiring", expected="preparing", metadata=details
                        )
                    elif not self.healthy({"metadata": details}):
                        raise RuntimeError("Prepared pool worker failed its readiness probe")
                    else:
                        self.store.mark(
                            slot["slot_id"],
                            "ready",
                            expected="preparing",
                            metadata=details,
                            checked=True,
                        )
                        self.store.recovered()
                        self.wake.set()  # Continue filling without waiting a poll interval.
                except Exception as error:
                    LOG.exception("Warm pool preparation failed")
                    self.store.mark(
                        slot["slot_id"], "retiring", expected="preparing", error=str(error)
                    )
                    self.store.cooldown(error)
                    self.wake.set()
            self.store.prune()

    def _worker(self):
        while not self.stopping.is_set():
            self.wake.clear()
            try:
                self.reconcile()
            except SessionBusy:
                pass
            except Exception:
                LOG.exception("Warm pool controller unavailable")
            self.wake.wait(30)
