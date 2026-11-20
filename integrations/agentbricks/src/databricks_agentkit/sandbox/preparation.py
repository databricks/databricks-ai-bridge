"""Explicit session preparation, persisted alongside the sandbox mapping.

Two bounded workers scan durable requests. A Postgres advisory lock prevents
replicas from preparing the same session. An interrupted initialization requires
an explicit retry: custom initialization hooks may have side effects.
"""

import hashlib
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.session_client import RevisionConflict
from databricks_agentkit.sandbox.store import SessionBusy

ACTIVE = {"queued", "preparing"}


class SessionPreparation:
    def __init__(self, control, orchestrator):
        self.control, self.orchestrator = control, orchestrator
        self.registry = orchestrator.registry
        self.stopping = threading.Event()
        self.threads = []

    def key(self, agent_id, session_id):
        return self.orchestrator.execution_session_id(scoped_session(agent_id, session_id))

    def get(self, agent_id, session_id):
        row = self.orchestrator.sandboxes.runtime(self.key(agent_id, session_id))["sandbox"]
        preparation = ((row or {}).get("bootstrap_metrics") or {}).get("preparation")
        return {
            **(preparation or {"status": "not_requested"}),
            "agent_id": agent_id,
            "session_id": session_id,
            "sandbox_id": (row or {}).get("sandbox_id"),
        }

    def rows(self, *, agent_id=None):
        with self.registry.connect() as db:
            where, args = (
                ("= %s", (agent_id,))
                if agent_id
                else (
                    "IS NOT NULL AND bootstrap_metrics->'preparation'->>'status' IN ('queued','preparing')",
                    (),
                )
            )
            return db.execute(
                "SELECT bootstrap_metrics->'preparation' AS preparation FROM sessions "
                "WHERE namespace=%s AND bootstrap_metrics->'preparation'->>'agent_id' "
                + where
                + " ORDER BY updated_at ASC LIMIT 100",
                (self.registry.namespace, *args),
            ).fetchall()

    def _write(self, key, value):
        digest = hashlib.sha256(f"{self.registry.namespace}\0{key}".encode()).hexdigest()
        with self.registry.connect() as db:
            db.execute(
                "UPDATE sessions SET bootstrap_metrics=jsonb_set("
                "COALESCE(bootstrap_metrics, '{}'::jsonb), '{preparation}', %s), updated_at=now() "
                "WHERE namespace=%s AND session_key=%s",
                (Jsonb(value), self.registry.namespace, digest),
            )

    def create(self, agent_id, session_id, *, retry=False):
        if self.orchestrator.sandbox_scope != "session":
            raise ValueError("Explicit preparation requires one sandbox per session")
        if self.orchestrator.worker_mode != "persistent":
            raise ValueError("Explicit preparation requires persistent workers")
        agent = self.control.agent(agent_id)
        key = self.key(agent_id, session_id)
        try:
            with self.registry.session("preparation:" + key):
                current = self.get(agent_id, session_id)
                if current["status"] in ACTIVE or (
                    current["status"] != "not_requested" and not retry
                ):
                    return current
                with (
                    self.orchestrator.sandboxes.invocation(key),
                    self.registry.session(key) as session,
                ):
                    session.bind_owner(self.orchestrator.execution_principal())
                    row = session.get()
                    if row["source_hash"] and row["source_hash"] != agent["source_hash"]:
                        raise RevisionConflict("Session is pinned to a different source revision")
                    session.save(
                        agent["source_hash"],
                        row["sandbox_id"],
                        row["status"],
                        source_revision=agent_id,
                    )
                    self._write(
                        key,
                        {
                            "agent_id": agent_id,
                            "session_id": session_id,
                            "status": "queued",
                            "stage": "Waiting to prepare session",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                return self.get(agent_id, session_id)
        except SessionBusy:
            current = self.get(agent_id, session_id)
            if current["status"] in ACTIVE:
                return current
            raise

    def assert_invocable(self, agent_id, session_id):
        if self.get(agent_id, session_id)["status"] in ACTIVE:
            raise SessionBusy(
                "Session preparation is in progress; wait for readiness before invoking"
            )

    @contextmanager
    def stopping_session(self, agent_id, session_id):
        key = self.key(agent_id, session_id)
        with self.registry.session("preparation:" + key):
            current = self.get(agent_id, session_id)
            if current["status"] in ACTIVE:
                self._write(
                    key, {**current, "status": "cancelled", "stage": "Preparation cancelled"}
                )
            yield

    def execute(self, agent_id, session_id):
        key = self.key(agent_id, session_id)
        try:
            with self.registry.session("preparation:" + key):
                current = self.get(agent_id, session_id)
                if current["status"] not in ACTIVE:
                    return
                if current["status"] == "preparing":
                    self._write(
                        key,
                        {
                            **current,
                            "status": "interrupted",
                            "error": "Preparation lost its worker. Retry preparation explicitly.",
                        },
                    )
                    return

                def update(**fields):
                    current.update(fields)
                    self._write(key, current)

                try:
                    update(status="preparing")
                    agent = self.control.agent(agent_id, with_bundle=True)
                    result = self.orchestrator.prepare_session(
                        session_id,
                        bundle=Bundle(
                            bytes(agent["bundle_data"]), agent["source_hash"], agent["file_count"]
                        ),
                        execution_artifact=agent.get("execution_artifact"),
                        entrypoint=agent["entrypoint"],
                        source_revision=agent_id,
                        registry_session_id=scoped_session(agent_id, session_id),
                        progress=lambda stage: update(stage=stage),
                        lifecycle=lambda snapshot: update(lifecycle=snapshot),
                    )
                    update(
                        status="ready",
                        stage="Session prepared",
                        result=result,
                        lifecycle=result["lifecycle"],
                    )
                except SessionBusy:
                    # A turn won the invocation lease before preparation began.
                    update(status="queued", stage="Waiting for current invocation")
                except Exception as exc:
                    logging.getLogger(__name__).exception("Session preparation failed")
                    update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4000])
        except SessionBusy:
            pass  # Another replica owns this preparation.

    def start(self):
        for number in range(2):
            thread = threading.Thread(
                target=self._worker, name=f"session-prepare-{number}", daemon=True
            )
            self.threads.append(thread)
            thread.start()

    def stop(self):
        self.stopping.set()

    def _worker(self):
        while not self.stopping.is_set():
            try:
                for row in self.rows():
                    if self.stopping.is_set():
                        break
                    request = row["preparation"]
                    self.execute(request["agent_id"], request["session_id"])
            except Exception:
                logging.getLogger(__name__).exception("Preparation scheduler unavailable")
            self.stopping.wait(1)
