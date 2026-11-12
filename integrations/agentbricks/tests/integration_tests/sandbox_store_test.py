"""Opt-in real PostgreSQL checks; no Databricks resources are contacted.

AGENTKIT_TEST_POSTGRES_URL=postgresql://... pytest tests/integration_tests/sandbox_store_test.py
Each test owns and removes a unique schema. Sandbox and Workspace APIs are fakes.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from databricks_agentkit.sandbox.control import ControlStore
from databricks_agentkit.sandbox.session_client import RevisionConflict, SessionSandboxClient
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, Registry, SessionBusy


@pytest.fixture
def registry():
    url = os.getenv("AGENTKIT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set AGENTKIT_TEST_POSTGRES_URL to a disposable PostgreSQL database")
    url = url.replace("postgresql+psycopg://", "postgresql://")
    schema = "sandbox_test_" + uuid4().hex

    class LocalRegistry(Registry):
        def connect(self):
            return psycopg.connect(
                url, autocommit=True, row_factory=dict_row, options=f"-csearch_path={schema}"
            )

    value = LocalRegistry(
        MagicMock(), "deployment", endpoint="unused", host="unused", schema=schema
    )
    value.initialize()
    try:
        yield value
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def runtime_store(registry):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
        LakebaseDurableRuntimeStore,
    )

    url = os.environ["AGENTKIT_TEST_POSTGRES_URL"].replace("postgresql://", "postgresql+psycopg://")

    class LocalLakebase:
        engine = create_async_engine(url)

        async def create_schema(self):
            async with self.engine.begin() as connection:
                await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {registry.schema}"))

    return LakebaseDurableRuntimeStore(
        lakebase=LocalLakebase(), schema=registry.schema, recovery_affinity="sandbox-test"
    )


@pytest.fixture
def control(registry):
    value = ControlStore(registry, MagicMock(), runtime_store=runtime_store(registry))
    value.initialize()
    try:
        yield value
    finally:
        value.close()


def client(registry):
    lakebox = MagicMock()
    lakebox.workspace.current_user.me.return_value.user_name = "app"
    lakebox.find_named.return_value = None
    lakebox.create.return_value = {"sandboxId": "sandbox-1", "status": "Running"}
    lakebox.get.return_value = lakebox.create.return_value
    return SessionSandboxClient(registry, lakebox, registry.namespace), lakebox


def test_fresh_schema_uses_standard_runtime_tables_and_separate_sandbox_sessions(registry, control):
    with registry.connect() as db:
        tables = db.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=current_schema()"
        ).fetchall()
    assert {row["tablename"] for row in tables} == {"sessions", "invocations", "invocation_events"}


def test_app_role_can_initialize_and_allocate_after_publication(registry, monkeypatch):
    owner_connect = registry.connect
    principal = "sandbox_app_" + uuid4().hex
    with owner_connect() as db:
        try:
            db.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(principal)))
        except psycopg.errors.InsufficientPrivilege:
            pytest.skip("Role-boundary check requires CREATEROLE on the disposable test database")

    try:
        # Match the App resource's CAN_CONNECT_AND_CREATE database permission.
        with owner_connect() as db:
            database = db.execute("SELECT current_database() AS name").fetchone()["name"]
            db.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(database), sql.Identifier(principal)
                )
            )
        registry.grant_app_access(principal)

        def app_connect():
            db = owner_connect()
            db.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(principal)))
            return db

        monkeypatch.setattr(registry, "connect", app_connect)
        registry.initialize()
        with registry.connect() as db:
            assert db.execute("SELECT current_user AS name").fetchone()["name"] == principal
        sandboxes, _ = client(registry)
        assert sandboxes.get_or_create("session", source_hash="digest").id == "sandbox-1"
    finally:
        with owner_connect() as db:
            db.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(principal)))
            db.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(principal)))


def test_concurrent_first_get_or_create_returns_one_sandbox(registry):
    first, lakebox = client(registry)
    second = SessionSandboxClient(registry, lakebox, registry.namespace)
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()

    def create(label):
        entered.set()
        assert release.wait(5)
        return {"sandboxId": "sandbox-1", "status": "Running"}

    def other():
        second_entered.set()
        return second.get_or_create("session", source_hash="digest", source_revision="revision")

    lakebox.create.side_effect = create
    with ThreadPoolExecutor(max_workers=2) as workers:
        one = workers.submit(
            first.get_or_create, "session", source_hash="digest", source_revision="revision"
        )
        try:
            assert entered.wait(5)
            two = workers.submit(other)
            assert second_entered.wait(5)
        finally:
            release.set()
        assert one.result(5).id == two.result(5).id == "sandbox-1"
    lakebox.create.assert_called_once()
    row = registry.runtime("session")["sandbox"]
    assert row["source_revision"] == "revision"
    assert row["execution_principal"] == "app"
    assert second.get_or_create("session", source_hash="digest").id == "sandbox-1"


def test_allocation_is_independent_of_turn_serialization(registry):
    sandboxes, lakebox = client(registry)
    sandboxes.get_or_create("session", source_hash="digest")
    with sandboxes.invocation("session"):
        assert sandboxes.get_or_create("session", source_hash="digest").id == "sandbox-1"
        with pytest.raises(SessionBusy), sandboxes.invocation("session"):
            pass
    lakebox.create.assert_called_once()


def test_unknown_create_outcome_and_identity_mismatch_never_create_twice(registry):
    sandboxes, lakebox = client(registry)
    lakebox.create.side_effect = TimeoutError("lost reply")
    with pytest.raises(TimeoutError):
        sandboxes.get_or_create("session", source_hash="digest")
    with pytest.raises(RevisionConflict, match="ambiguous"):
        sandboxes.get_or_create("session", source_hash="digest")
    sandboxes._principal = "other-app"
    with pytest.raises(ExecutionIdentityConflict):
        sandboxes.get_or_create("session", source_hash="digest")
    lakebox.create.assert_called_once()
    assert registry.runtime("session")["execution_principal"] == "app"


def test_stop_resume_retains_mapping_and_pinned_revision(registry):
    sandboxes, lakebox = client(registry)
    sandboxes.get_or_create("session", source_hash="digest", source_revision="revision")
    sandboxes.record("session", "digest", "sandbox-1", "ready", {"timings_ms": {"install": 4}})
    lakebox.api.return_value = {"status": "Stopped"}
    sandboxes.stop("session")
    row = registry.runtime("session")["sandbox"]
    assert row["status"] == "stopped" and row["source_revision"] == "revision"
    assert row["bootstrap_metrics"]["timings_ms"]["install"] == 4
    lakebox.get.return_value = {"sandboxId": "sandbox-1", "status": "Stopped"}
    lakebox.start.return_value = {"sandboxId": "sandbox-1", "status": "Running"}
    assert sandboxes.get_or_create("session", source_hash="digest").id == "sandbox-1"
    lakebox.create.assert_called_once()
    with pytest.raises(RevisionConflict):
        sandboxes.get_or_create("session", source_hash="different")


def test_invocations_use_shared_schema_identity_events_and_terminal_semantics(control):
    run = control.create_run("revision", "session", "invoke", {"message": "hi"})
    store = control.runtime_store
    assert control.create_run("revision", "session", "invoke", {"message": "hi"})["id"] == run["id"]
    with pytest.raises(ValueError):
        control.create_run("revision", "session", "invoke", {"message": "changed"})
    claimed = control.call(store.claim(run["id"]))
    assert control.call(store.claim(run["id"])) is None
    control.call(
        store.append_event(
            run["id"],
            claimed.attempt,
            {"type": "sandbox.text.delta", "delta": "Hello", "first_token_ms": 1},
        )
    )
    control.call(store.complete(run["id"], claimed.attempt, {"result": {"reply": "Hello"}}))
    assert control.run(run["id"])["stream"]["text"] == "Hello"
    assert control.history("revision", "session")[0]["status"] == "succeeded"
    assert control.sessions("revision")[0]["session_id"] == "session"
    assert (
        control.call(
            store.append_event(
                run["id"], claimed.attempt, {"type": "sandbox.text.delta", "delta": "late"}
            )
        )
        is None
    )
    with control.registry.connect() as db:
        columns = {
            row["column_name"]
            for row in db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='invocations'"
            ).fetchall()
        }
    assert {"worker_id", "heartbeat_at", "request", "response", "attempt"} <= columns
    assert {"namespace", "stream", "lifecycle", "stage", "attempt_id"}.isdisjoint(columns)


@pytest.fixture
def preparation(control, registry):
    from databricks_agentkit.sandbox.orchestrator import Orchestrator
    from databricks_agentkit.sandbox.preparation import SessionPreparation

    sandboxes, lakebox = client(registry)
    orchestrator = Orchestrator(registry, lakebox, None, registry.namespace)
    orchestrator.sandboxes = sandboxes
    control.catalog.agent.return_value = {
        "id": "revision",
        "source_hash": "digest",
        "bundle_data": b"source",
        "file_count": 1,
        "entrypoint": "agent:invoke",
    }

    def prepare(session_id, **kwargs):
        key = kwargs["registry_session_id"]
        with sandboxes.invocation(key):
            allocation = sandboxes.get_or_create(
                key, source_hash="digest", source_revision="revision"
            )
            kwargs["progress"]("Installing")
            sandboxes.record(key, "digest", allocation.id, "ready", {"timings_ms": {"install": 4}})
            kwargs["lifecycle"]({"steps": {"handler_prepare": {"status": "completed"}}})
        return {
            "sandbox_id": allocation.id,
            "lifecycle": {"preparation_ms": 10},
            "worker": {"id": "resident"},
        }

    orchestrator.prepare_session = MagicMock(side_effect=prepare)
    return SessionPreparation(control, orchestrator)


def test_prepare_before_invoke_is_durable_and_duplicate_creation_reuses_work(
    preparation, registry, control
):
    service = preparation
    assert service.create("revision", "new")["status"] == "queued"
    assert service.create("revision", "new")["status"] == "queued"
    assert len(service.rows()) == 1
    assert control.history("revision", "new") == []
    service.execute("revision", "new")
    assert service.get("revision", "new")["status"] == "ready"
    assert service.get("revision", "new")["sandbox_id"] == "sandbox-1"
    assert service.create("revision", "new")["status"] == "ready"
    service.execute("revision", "new")
    service.orchestrator.prepare_session.assert_called_once()
    assert registry.runtime(service.key("revision", "new"))["sandbox"]["bootstrap_metrics"][
        "timings_ms"
    ] == {"install": 4}
    assert len(service.rows(agent_id="revision")) == 1


def test_preparation_replicas_and_stop_cannot_overlap(preparation):
    from databricks_agentkit.sandbox.preparation import SessionPreparation

    service = preparation
    other = SessionPreparation(service.control, service.orchestrator)
    service.create("revision", "new")
    entered, release = threading.Event(), threading.Event()
    original = service.orchestrator.prepare_session.side_effect

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    service.orchestrator.prepare_session.side_effect = gated
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.execute, "revision", "new")
        try:
            assert entered.wait(5)
            other.execute("revision", "new")
            assert other.create("revision", "new")["status"] == "preparing"
            with pytest.raises(SessionBusy):
                other.assert_invocable("revision", "new")
            with pytest.raises(SessionBusy), other.stopping_session("revision", "new"):
                pass
        finally:
            release.set()
        future.result(5)
    service.orchestrator.prepare_session.assert_called_once()


def test_restart_reloads_queued_requests_but_does_not_replay_unknown_hooks(preparation):
    from databricks_agentkit.sandbox.preparation import SessionPreparation

    service = preparation
    service.create("revision", "queued")
    service.create("revision", "interrupted")
    value = service.get("revision", "interrupted")
    service._write(service.key("revision", "interrupted"), {**value, "status": "preparing"})
    restarted = SessionPreparation(service.control, service.orchestrator)
    for row in restarted.rows():
        restarted.execute(row["preparation"]["agent_id"], row["preparation"]["session_id"])
    assert restarted.get("revision", "queued")["status"] == "ready"
    assert restarted.get("revision", "interrupted")["status"] == "interrupted"
    service.orchestrator.prepare_session.assert_called_once()
    assert restarted.create("revision", "interrupted")["status"] == "interrupted"
    assert restarted.create("revision", "interrupted", retry=True)["status"] == "queued"


def test_cancel_queued_preparation_never_starts_compute(preparation):
    service = preparation
    service.create("revision", "cancel")
    with service.stopping_session("revision", "cancel"):
        pass
    service.execute("revision", "cancel")
    service.orchestrator.prepare_session.assert_not_called()
    assert service.get("revision", "cancel")["status"] == "cancelled"


def test_failed_preparation_exposes_error_and_can_be_explicitly_retried(preparation):
    service = preparation
    service.orchestrator.prepare_session.side_effect = RuntimeError("bad dependency")
    service.create("revision", "failed")
    service.execute("revision", "failed")
    failed = service.get("revision", "failed")
    assert failed["status"] == "failed" and "bad dependency" in failed["error"]
    assert service.create("revision", "failed")["status"] == "failed"
    assert service.create("revision", "failed", retry=True)["status"] == "queued"


def test_event_cursor_replay_and_fencing_across_process_stores(control, registry):
    from databricks_agentkit.runtime.types import InvocationStatus

    reader = ControlStore(registry, control.catalog, runtime_store=runtime_store(registry))
    reader.initialize()
    try:
        run = control.create_run("a", "s", "i", {})
        old = control.call(control.runtime_store.claim(run["id"]))
        control.call(
            control.runtime_store.append_event(
                run["id"], old.attempt, {"type": "sandbox.text.delta", "delta": "before"}
            )
        )
        watch = reader.watch(run["id"])
        assert next(watch)["stream"]["text"] == "before"
        control.call(
            control.runtime_store.append_event(
                run["id"], old.attempt, {"type": "sandbox.text.delta", "delta": " after"}
            )
        )
        assert next(watch)["stream"]["text"] == "before after"
        watch.close()
        with registry.connect() as db:
            db.execute(
                "UPDATE invocations SET heartbeat_at=now()-interval '1 hour' "
                "WHERE invocation_id=%s",
                (run["id"],),
            )
        current = reader.call(reader.runtime_store.claim_recoverable(run["id"], 10))
        assert current.attempt == old.attempt + 1
        assert (
            control.call(
                control.runtime_store.append_event(
                    run["id"], old.attempt, {"type": "sandbox.text.delta", "delta": "stale"}
                )
            )
            is None
        )
        assert control.call(control.runtime_store.complete(run["id"], old.attempt, {})) is False
        reader.call(
            reader.runtime_store.append_event(
                run["id"], current.attempt, {"type": "sandbox.text.delta", "delta": "new attempt"}
            )
        )
        reader.call(reader.runtime_store.complete(run["id"], current.attempt, {"reply": "done"}))
        assert reader.call(reader.runtime_store.get(run["id"])).status == InvocationStatus.COMPLETED
        final = list(control.watch(run["id"]))
        assert final[0]["stream"]["text"] == "new attempt"
        assert final[0]["status"] == "succeeded"
    finally:
        reader.close()


def test_durable_executor_recovers_stale_work_without_replaying_the_sandbox(control, registry):
    from databricks_agentkit.runtime.runtime import Runtime
    from databricks_agentkit.sandbox.jobs import RunManager

    run = control.create_run("a", "s", "i", {})
    control.call(control.runtime_store.claim(run["id"]))
    with registry.connect() as db:
        db.execute(
            "UPDATE invocations SET heartbeat_at=now()-interval '1 hour' WHERE invocation_id=%s",
            (run["id"],),
        )
    replacement = ControlStore(registry, control.catalog, runtime_store=runtime_store(registry))
    replacement.initialize()
    orchestrator = MagicMock()
    manager = RunManager(replacement, orchestrator)
    manager.runtime = Runtime.from_store(
        manager.execute,
        runtime_store=replacement.runtime_store,
        heartbeat_seconds=0.02,
        stale_seconds=0.1,
        scan_seconds=0.02,
    )
    manager.start()
    try:
        deadline = time.monotonic() + 3
        while control.run(run["id"])["status"] != "failed":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        orchestrator.invoke.assert_not_called()
        assert control.run(run["id"])["attempt_id"] == 2
    finally:
        manager.stop()
        replacement.close()
