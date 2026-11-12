"""Sandbox execution adapter for the shared AgentKit invocation runtime."""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone

from databricks_agentkit.runtime.runtime import Runtime
from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.control import STREAM_FLUSH_SECONDS, scoped_session

LOG = logging.getLogger(__name__)


class RunManager:
    def __init__(self, control, orchestrator):
        self.control, self.orchestrator = control, orchestrator
        self.runtime = Runtime.from_store(
            self.execute,
            runtime_store=control.runtime_store,
            poll_seconds=STREAM_FLUSH_SECONDS,
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
        with PartialText(emit) as stream:
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
                emit=stream.emit,
                lifecycle=lambda value: emit({"type": "sandbox.lifecycle", "lifecycle": value}),
            )
        result["session_id"] = request["session_id"]
        result["agent_id"] = agent["id"]
        return result


class PartialText:
    """Coalesce text into bounded, attempt-fenced events without blocking the tail reader."""

    def __init__(self, publish, *, interval=STREAM_FLUSH_SECONDS):
        self.publish, self.interval = publish, interval
        self.started = time.monotonic()
        self.text, self.first_token_ms, self.dirty = "", None, False
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._write, name="sandbox-text-writer", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def emit(self, delta):
        with self.condition:
            if self.closed or self.error or not delta or len(self.text) >= 256 * 1024:
                return
            if self.first_token_ms is None:
                self.first_token_ms = round((time.monotonic() - self.started) * 1000, 2)
            self.text += delta[: 256 * 1024 - len(self.text)]
            self.dirty = True
            self.condition.notify()

    def _write(self):
        flushed, offset = None, 0
        try:
            while True:
                with self.condition:
                    while not self.dirty and not self.closed:
                        self.condition.wait()
                    if not self.dirty:
                        return
                    if not self.closed and flushed is not None:
                        remaining = flushed + self.interval - time.monotonic()
                        if remaining > 0:
                            self.condition.wait(remaining)
                            continue
                    text, first_token_ms = self.text, self.first_token_ms
                    self.dirty = False
                self.publish(
                    {
                        "type": "sandbox.text.delta",
                        "delta": text[offset:],
                        "first_token_ms": first_token_ms,
                    }
                )
                offset = len(text)
                flushed = time.monotonic()
        except Exception as exc:
            with self.condition:
                self.error = exc
            LOG.warning(
                "Live text publication failed; final result will carry the answer", exc_info=True
            )

    def __exit__(self, *exc):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            LOG.warning("Live text publication timed out during shutdown")
