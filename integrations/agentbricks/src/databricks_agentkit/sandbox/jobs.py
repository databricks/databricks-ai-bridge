"""Sandbox execution adapter for the shared AgentKit invocation runtime."""

import asyncio
import time
from datetime import datetime, timezone

from databricks_agentkit.runtime.runtime import Runtime
from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.control import RUNTIME_POLL_SECONDS, scoped_session


class RunManager:
    def __init__(self, control, orchestrator):
        self.control, self.orchestrator = control, orchestrator
        self.runtime = Runtime.from_store(
            self.execute,
            runtime_store=control.runtime_store,
            poll_seconds=RUNTIME_POLL_SECONDS,
        )
        self.capacity = asyncio.Semaphore(2)

    def start(self):
        self.control.call(self.runtime.start())

    def stop(self):
        self.control.call(self.runtime.stop())

    def submit(self, run_id):
        # The shared runtime schedules accepted work and preserves per-session FIFO.
        self.control.call(self.runtime.get_invocation(run_id))

    async def execute(self, request, context):
        if context.is_recovery:
            raise RuntimeError(
                "The previous App worker stopped heartbeating. The sandbox operation may still "
                "be running; inspect the session before submitting a new invocation. "
                "Automatic replay is disabled for arbitrary agent code."
            )
        async with self.capacity:
            return await asyncio.to_thread(self._invoke, request, context)

    def _invoke(self, request, context):
        agent = self.control.agent(request["agent_id"], with_bundle=True)
        bundle = Bundle(bytes(agent["bundle_data"]), agent["source_hash"], agent["file_count"])

        def emit(event):
            return self.control.call(
                context.emit({**event, "at": datetime.now(timezone.utc).isoformat()}), timeout=5
            )

        emit({"type": "sandbox.stage", "stage": "Starting"})
        started_at = time.time()
        result = self.orchestrator.invoke(
            request["session_id"],
            request["invocation_id"],
            request["input"],
            bundle=bundle,
            execution_artifact=agent.get("execution_artifact"),
            entrypoint=agent["entrypoint"],
            source_revision=agent["id"],
            registry_session_id=scoped_session(agent["id"], request["session_id"]),
            progress=lambda stage: emit({"type": "sandbox.stage", "stage": stage}),
            stream_ready=lambda source: emit(
                {
                    "type": "sandbox.stream.ready",
                    "source": {**source, "started_at": started_at},
                }
            ),
            lifecycle=lambda value: emit({"type": "sandbox.lifecycle", "lifecycle": value}),
        )
        result["session_id"] = request["session_id"]
        result["agent_id"] = agent["id"]
        return result
