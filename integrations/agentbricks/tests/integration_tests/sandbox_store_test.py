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


@pytest.fixture
def control(registry):
    value = ControlStore(registry, MagicMock())
    value.initialize()
    return value


def client(registry):
    lakebox = MagicMock()
    lakebox.workspace.current_user.me.return_value.user_name = "app"
    lakebox.find_named.return_value = None
    lakebox.create.return_value = {"sandboxId": "sandbox-1", "status": "Running"}
    lakebox.get.return_value = lakebox.create.return_value
    return SessionSandboxClient(registry, lakebox, registry.namespace), lakebox


def test_fresh_schema_has_only_sessions_and_invocations(registry, control):
    with registry.connect() as db:
        tables = db.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=current_schema()"
        ).fetchall()
    assert {row["tablename"] for row in tables} == {"sessions", "invocations"}


def test_app_role_can_initialize_and_invoke_after_publication(registry, control, monkeypatch):
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
        control.initialize()
        with registry.connect() as db:
            assert db.execute("SELECT current_user AS name").fetchone()["name"] == principal
        run = control.create_run("revision", "session", "invoke", {"message": "hi"})
        claimed = control.claim(run["id"])
        assert claimed["status"] == "running"
        with control.stream(run["id"], attempt_id=claimed["attempt_id"]) as emit:
            watch = control.watch(run["id"])
            try:
                assert next(watch)["stream"]["text"] == ""
                with ThreadPoolExecutor(max_workers=1) as pool:
                    waiting = pool.submit(next, watch)
                    emit("App can publish and subscribe")
                    assert waiting.result(2)["stream"]["text"] == "App can publish and subscribe"
            finally:
                watch.close()
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


def test_invocation_identity_stream_history_and_fenced_retry(control):
    run = control.create_run("revision", "session", "invoke", {"message": "hi"})
    assert control.create_run("revision", "session", "invoke", {"message": "hi"})["id"] == run["id"]
    with pytest.raises(ValueError, match="different input"):
        control.create_run("revision", "session", "invoke", {"message": "changed"})
    with pytest.raises(KeyError):
        control.create_run("revision", "session", "missing", {}, retry=True)
    old = control.claim(run["id"])
    assert control.claim(run["id"]) is None
    with control.stream(run["id"], attempt_id=old["attempt_id"]) as emit:
        emit("Hello")
        deadline = time.monotonic() + 2
        while control.run(run["id"])["stream"]["text"] != "Hello":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        control.finish(run["id"], error="worker exited", attempt_id=old["attempt_id"])
        retried = control.create_run("revision", "session", "invoke", {"message": "hi"}, retry=True)
        assert retried["stream"] is None
        current = control.claim(run["id"])
        with control.stream(run["id"], attempt_id=current["attempt_id"]) as new_emit:
            new_emit("New reply")
        with control.stream(run["id"], attempt_id=old["attempt_id"]) as stale_emit:
            stale_emit("late old attempt")
        emit(" stale")
    control.finish(run["id"], result={"wrong": True}, attempt_id=old["attempt_id"])
    assert control.run(run["id"])["status"] == "running"
    control.finish(
        run["id"], result={"result": {"reply": "New reply"}}, attempt_id=current["attempt_id"]
    )
    history = control.history("revision", "session")
    assert len(history) == 1 and history[0]["stream"]["text"] == "New reply"
    assert control.sessions("revision")[0]["session_id"] == "session"
    assert list(control.watch(run["id"]))[0]["status"] == "succeeded"


def test_restart_marks_inflight_interrupted_and_retains_queued_work(control):
    queued = control.create_run("a", "s", "queued", {})
    running = control.create_run("a", "s", "running", {})
    control.claim(running["id"])
    assert control.recover() == [{"id": queued["id"]}]
    assert control.run(running["id"])["status"] == "interrupted"


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


def test_notifications_deliver_quiet_text_and_terminal_state_across_store_instances(
    control, monkeypatch
):
    from databricks_agentkit.sandbox import control as module

    # A 2s assertion cannot pass by waiting for the 10s heartbeat.
    monkeypatch.setattr(module, "STREAM_RESYNC_SECONDS", 10)
    run = control.create_run("a", "s", "i", {})
    claimed = control.claim(run["id"])
    reader = ControlStore(control.registry, control.catalog)
    with control.stream(run["id"], attempt_id=claimed["attempt_id"]) as emit:
        one, two = reader.watch(run["id"]), reader.watch(run["id"])
        try:
            assert next(one)["stream"]["text"] == next(two)["stream"]["text"] == ""
            with ThreadPoolExecutor(max_workers=2) as pool:
                waiting = [pool.submit(next, watcher) for watcher in (one, two)]
                emit("Hello")
                assert [future.result(2)["stream"]["text"] for future in waiting] == ["Hello"] * 2
                one.close()  # Closing one SSE subscriber does not stop the other or the producer.
                waiting = pool.submit(next, two)
                emit(" world")
                row = waiting.result(2)
                assert row["stream"]["text"] == "Hello world"
                assert row["status"] == "running"
        finally:
            one.close()
            two.close()
    final = reader.watch(run["id"])
    try:
        assert next(final)["status"] == "running"
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(next, final)
            control.finish(
                run["id"], result={"reply": "Hello world"}, attempt_id=claimed["attempt_id"]
            )
            assert waiting.result(2)["status"] == "succeeded"
        with pytest.raises(StopIteration):
            next(final)
    finally:
        final.close()


def test_notification_between_initial_snapshot_and_wait_is_not_lost(control, monkeypatch):
    from databricks_agentkit.sandbox import control as module

    monkeypatch.setattr(module, "STREAM_RESYNC_SECONDS", 10)
    run = control.create_run("a", "s", "i", {})
    claimed = control.claim(run["id"])
    reader = ControlStore(control.registry, control.catalog)
    original = reader.run
    first = True

    def read(run_id, **kwargs):
        nonlocal first
        row = original(run_id, **kwargs)
        if first:
            first = False
            # Update after the first read, before the generator yields it.
            control.progress(run_id, "already changed", attempt_id=claimed["attempt_id"])
        return row

    monkeypatch.setattr(reader, "run", read)
    watch = reader.watch(run["id"])
    try:
        assert next(watch)["stage"] == "Starting"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(next, watch).result(2)["stage"] == "already changed"
    finally:
        watch.close()


def test_heartbeat_resync_and_reconnect_load_durable_snapshot(control, monkeypatch):
    from databricks_agentkit.sandbox import control as module

    monkeypatch.setattr(module, "STREAM_RESYNC_SECONDS", 0.05)
    run = control.create_run("a", "s", "i", {})
    control.claim(run["id"])
    watch = control.watch(run["id"])
    try:
        assert next(watch)["stage"] == "Starting"
        # Simulate a write without a wakeup (e.g. a rolling upgrade from the old App).
        with control.registry.connect() as db:
            db.execute("UPDATE invocations SET stage='missed wakeup' WHERE id=%s", (run["id"],))
        assert next(watch)["stage"] == "missed wakeup"
    finally:
        watch.close()
    with control.stream(run["id"]) as emit:
        emit("durable partial")
    reconnected = control.watch(run["id"])
    try:
        assert next(reconnected)["stream"]["text"] == "durable partial"
    finally:
        reconnected.close()


def test_stream_notifications_do_not_cross_namespace_or_run(control, registry, monkeypatch):
    from databricks_agentkit.sandbox.control import run_channel

    assert run_channel("one", "a") != run_channel("two", "a")
    assert run_channel("one", "a") != run_channel("one", "b")
    assert len(run_channel("x" * 1000, "y" * 1000)) <= 63
    run = control.create_run("a", "s", "i", {})
    old = control.claim(run["id"])
    with registry.connect() as listener:
        listener.execute(
            sql.SQL("LISTEN {}").format(sql.Identifier(run_channel(registry.namespace, run["id"])))
        )
        control.progress(run["id"], "wrong attempt", attempt_id="wrong")
        assert list(listener.notifies(timeout=0.05)) == []
        control.progress(run["id"], "correct attempt", attempt_id=old["attempt_id"])
        assert len(list(listener.notifies(timeout=1, stop_after=1))) == 1
