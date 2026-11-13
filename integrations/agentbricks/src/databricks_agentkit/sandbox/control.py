"""Workbench projections over the shared AgentKit Runtime Store."""

import asyncio
import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack

from databricks_agentkit.runtime.store import DEFAULT_RUNTIME_STORE_SCHEMA, RUNTIME_STORE_SCHEMA_ENV
from databricks_agentkit.runtime.types import InvocationStatus
from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog

RUNTIME_POLL_SECONDS = 0.05
STREAM_RESYNC_SECONDS = 1.0


class ControlStore:
    """Adapt synchronous App/Workspace APIs to one shared async Runtime Store loop."""

    def __init__(self, registry, catalog=None, *, runtime_store=None, preview_factory=None):
        self.registry = registry
        self.namespace = registry.namespace
        self.catalog = catalog or WorkspaceCatalog(
            registry.workspace, f"/Shared/agentbricks-runtime/{self.namespace}"
        )
        self.runtime_store = runtime_store
        self.preview_factory = preview_factory or self._preview
        self.loop = None
        self.thread = None

    def initialize(self):
        if self.loop is not None:
            return
        if self.runtime_store is None:
            from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
                LakebaseDurableRuntimeStore,
            )

            self.runtime_store = LakebaseDurableRuntimeStore.from_app_resource(
                endpoint=self.registry.endpoint,
                host=self.registry.host,
                port=5432,
                database=self.registry.database,
                username=self.registry.username,
                workspace_client=self.registry.workspace,
                schema=os.getenv(RUNTIME_STORE_SCHEMA_ENV) or DEFAULT_RUNTIME_STORE_SCHEMA,
                recovery_affinity=f"sandbox:{self.namespace}",
            )
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="sandbox-runtime", daemon=True
        )
        ready = threading.Event()
        self.loop.call_soon(ready.set)
        self.thread.start()
        ready.wait()
        try:
            self.call(self.runtime_store.initialize())
        except BaseException:
            self.close()
            raise

    def call(self, coroutine, *, timeout=None):
        if self.loop is None or not self.loop.is_running():
            coroutine.close()
            raise RuntimeError("Sandbox runtime is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    def close(self):
        if self.loop is None:
            return
        try:
            self.call(self.runtime_store.close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
            self.loop = None

    def create_agent(
        self,
        name,
        kind,
        path,
        entrypoint,
        bundle: Bundle,
        *,
        git_source=None,
        execution_artifact=None,
    ):
        return self.catalog.create_agent(
            name,
            kind,
            path,
            entrypoint,
            bundle,
            git_source=git_source,
            execution_artifact=execution_artifact,
        )

    def agents(self):
        return self.catalog.list("revisions")

    def agent(self, agent_id, *, with_bundle=False):
        return self.catalog.agent(agent_id, with_bundle=with_bundle)

    def _session_id(self, agent_id, session_id):
        return (
            "sandbox:"
            + hashlib.sha256(
                json.dumps([self.namespace, agent_id, session_id], ensure_ascii=True).encode()
            ).hexdigest()
        )

    def create_run(self, agent_id, session_id, invocation_id, value, *, retry=False):
        run_id = (
            "sandbox:"
            + hashlib.sha256(
                json.dumps(
                    [self.namespace, agent_id, session_id, invocation_id], ensure_ascii=True
                ).encode()
            ).hexdigest()
        )
        request = {
            "namespace": self.namespace,
            "agent_id": agent_id,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "input": value,
        }
        if retry:
            state = self.call(self.runtime_store.get(run_id))
            if state is None:
                raise KeyError("Original invocation not found; submit without retry")
            if state.status == InvocationStatus.FAILED:
                raise ValueError("Failed invocations are terminal; retry with a new invocation ID")
        state = self.call(
            self.runtime_store.accept(
                run_id, request, session_id=self._session_id(agent_id, session_id)
            )
        )
        return self._snapshot(state)

    def _state(self, run_id):
        state = self.call(self.runtime_store.get(run_id))
        if (
            state is None
            or not isinstance(state.request, dict)
            or state.request.get("namespace") != self.namespace
        ):
            raise KeyError("Invocation not found")
        return state

    def run(self, run_id):
        return self._snapshot(self._state(run_id))

    def _snapshot(self, state, *, events=None):
        if events is None:
            events = self.call(self.runtime_store.events(state.invocation_id))
        return snapshot(state, events)

    def watch(self, run_id):
        """Relay attempt-file previews, checking durable state at a bounded rate."""
        cursor = None
        row = None
        preview, identity = None, None
        next_check, observed_end = 0, False
        with ExitStack() as readers:
            while True:
                if time.monotonic() >= next_check:
                    state = self._state(run_id)
                    new = self.call(self.runtime_store.events(run_id, after_sequence=cursor))
                    # A claim can commit between reads; consume its events next time.
                    new = [event for event in new if event.attempt <= state.attempt]
                    if new:
                        cursor = new[-1].sequence_number
                    row = snapshot(state, new, previous=row)
                    next_check = time.monotonic() + STREAM_RESYNC_SECONDS
                    source = row.get("stream_source")
                    current = (state.attempt, source) if source else None
                    if current != identity:
                        readers.close()
                        preview, identity, observed_end = None, current, False
                        if source and not state.is_terminal:
                            try:
                                preview = readers.enter_context(
                                    self.preview_factory(source, state.attempt)
                                )
                            except Exception:
                                # A preview failure must not fail or replay the agent.
                                row["stream_error"] = (
                                    "Live preview unavailable; waiting for result."
                                )
                    if state.is_terminal:
                        # Deliver completion before reader cleanup; no extra tail drain
                        # or database flush is required to make the answer visible.
                        yield row
                        return
                if preview:
                    row = {**row, "stream": preview.snapshot()}
                    if preview.follower.error:
                        row["stream_error"] = (
                            "Live preview disconnected; reconnect or wait for result."
                        )
                yield row
                if preview:
                    preview.wait(
                        row["stream"]["sequence"], min(0.25, max(0, next_check - time.monotonic()))
                    )
                    if preview.follower.decoder.ended.is_set() and not observed_end:
                        observed_end = True
                        next_check = 0
                else:
                    time.sleep(max(0, next_check - time.monotonic()))

    def _preview(self, source, attempt):
        from databricks_agentkit.sandbox.lakebox import HttpsRemote
        from databricks_agentkit.sandbox.streaming import LivePreview

        # Observe the stored sandbox directly. Never allocate, resume, or hold
        # session locks in a subscriber, even when the sandbox has disappeared.
        return LivePreview(
            HttpsRemote(self.registry.workspace, source["sandbox_id"]), source, attempt
        )

    def history(self, agent_id, session_id):
        states = self.call(
            self.runtime_store.list_invocations(
                session_id=self._session_id(agent_id, session_id),
                limit=100,
                request_filter={"namespace": self.namespace, "agent_id": agent_id},
            )
        )
        events = self.call(
            self.runtime_store.events(session_id=self._session_id(agent_id, session_id))
        )
        by_run = {}
        for event in events:
            by_run.setdefault(event.invocation_id, []).append(event)
        return [
            self._snapshot(state, events=by_run.get(state.invocation_id, []))
            for state in reversed(states)
        ]

    def sessions(self, agent_id):
        states = self.call(
            self.runtime_store.list_invocations(
                request_filter={"namespace": self.namespace, "agent_id": agent_id},
                limit=1000,
            )
        )
        sessions = {}
        for state in states:
            row = self._snapshot(state, events=[])
            current = sessions.get(row["session_id"])
            if (
                current is None
                or row["session_sequence_number"] > current["session_sequence_number"]
            ):
                sessions[row["session_id"]] = {
                    "session_id": row["session_id"],
                    "run_id": row["id"],
                    "status": row["status"],
                    "stage": row["stage"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "session_sequence_number": row["session_sequence_number"],
                    "sandbox_id": (row["result"] or {}).get("sandbox_id"),
                }
        return list(sessions.values())[:100]


def scoped_session(agent_id: str, session_id: str) -> str:
    return f"agent:{agent_id}\0{session_id}"


def snapshot(state, events, *, previous=None):
    """Project persisted events into the existing chat API without a second state table."""
    status = {
        InvocationStatus.QUEUED: "queued",
        InvocationStatus.ACTIVE: "running",
        InvocationStatus.COMPLETED: "succeeded",
        InvocationStatus.FAILED: "failed",
    }[state.status]
    row = {
        **state.request,
        "id": state.invocation_id,
        "status": status,
        "attempt_id": state.attempt,
        "session_sequence_number": state.session_sequence_number,
        "stage": {
            "queued": "Queued",
            "running": "Starting",
            "succeeded": "Complete",
            "failed": "Failed",
        }[status],
        "result": state.response if status == "succeeded" else None,
        "error": None,
        "lifecycle": None,
        "stream": None,
        "stream_source": None,
        "created_at": None,
        "updated_at": None,
    }
    if previous and previous["attempt_id"] == state.attempt:
        for key in ("created_at", "updated_at", "lifecycle", "stream", "stream_source"):
            row[key] = previous[key]
        if not state.is_terminal:
            row["stage"] = previous["stage"]
    for record in events:
        event = record.event
        at = event.get("at")
        if at:
            row["created_at"] = row["created_at"] or at
            row["updated_at"] = at
        if record.attempt != state.attempt:
            continue
        kind = event.get("type")
        if kind == "sandbox.stage" and not state.is_terminal:
            row["stage"] = event["stage"]
        elif kind == "sandbox.lifecycle":
            row["lifecycle"] = event["lifecycle"]
        elif kind == "sandbox.stream.ready":
            row["stream_source"] = event["source"]
        elif kind == "sandbox.text.delta":
            stream = (
                dict(row["stream"])
                if row["stream"]
                else {
                    "attempt": state.attempt,
                    "text": "",
                    "sequence": 0,
                    "first_token_ms": event.get("first_token_ms"),
                }
            )
            stream["text"] += event["delta"]
            stream["sequence"] = record.sequence_number
            row["stream"] = stream
    if status == "failed":
        error = (state.response or {}).get("error")
        row["error"] = (
            error.get("message") if isinstance(error, dict) else error or "Invocation failed"
        )
    return row
