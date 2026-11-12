"""Bounded worker count; source and invocation records outlive the App process."""

import logging
import queue
import threading

from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.control import scoped_session


class RunManager:
    def __init__(self, control, orchestrator):
        self.control, self.orchestrator = control, orchestrator
        self.pending = queue.Queue()
        self.stopping = threading.Event()
        self.threads = []

    def start(self):
        for row in self.control.recover():
            self.submit(row["id"])
        for number in range(2):
            thread = threading.Thread(target=self._worker, name=f"agent-run-{number}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stopping.set()
        for _ in self.threads:
            self.pending.put(None)

    def submit(self, run_id):
        self.pending.put(run_id)

    def _worker(self):
        while not self.stopping.is_set():
            run_id = self.pending.get()
            if run_id is None:
                return
            try:
                row = self.control.claim(run_id)
                if row is not None:
                    self.execute(row)
            except Exception:
                logging.getLogger(__name__).exception("Background invocation failed")
            finally:
                self.pending.task_done()

    def execute(self, row):
        run_id = row["id"]
        attempt_id = row.get("attempt_id")
        try:
            agent = self.control.agent(row["agent_id"], with_bundle=True)
            bundle = Bundle(bytes(agent["bundle_data"]), agent["source_hash"], agent["file_count"])
            with (
                self.control.track(run_id, attempt_id) as track,
                self.control.stream(run_id, attempt_id=attempt_id) as emit,
            ):
                result = self.orchestrator.invoke(
                    row["session_id"],
                    row["invocation_id"],
                    row["input"],
                    bundle=bundle,
                    execution_artifact=agent.get("execution_artifact"),
                    entrypoint=agent["entrypoint"],
                    source_revision=agent["id"],
                    registry_session_id=scoped_session(agent["id"], row["session_id"]),
                    progress=lambda stage: self.control.progress(
                        run_id, stage, attempt_id=attempt_id
                    ),
                    emit=emit,
                    lifecycle=track,
                )
            result["session_id"] = row["session_id"]
            result["agent_id"] = agent["id"]
            self.control.finish(run_id, result=result, attempt_id=attempt_id)
        except Exception as exc:
            # Isolate arbitrary agent failures at the job boundary and persist the
            # diagnostic so the UI can show a failed run instead of polling forever.
            logging.getLogger(__name__).exception("Agent invocation failed")
            self.control.finish(
                run_id, error=f"{type(exc).__name__}: {exc}"[:4000], attempt_id=attempt_id
            )
