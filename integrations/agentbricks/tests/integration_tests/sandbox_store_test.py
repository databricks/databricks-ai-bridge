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


# Warm-pool ownership is exercised with real independent database connections.
def pool_store(registry, *, size=2, generation=1, key="revision"):
    from databricks_agentkit.sandbox.warm_pool_store import WarmPoolStore

    value = WarmPoolStore(registry, key, generation, size)
    value.initialize()
    return value


def ready_spare(store, *, owner="app"):
    slot = store.reserve()
    assert slot
    sandbox_id = "sandbox-" + slot["slot_id"]
    with store.registry.session(slot["slot_id"]) as session:
        session.bind_owner(owner)
        session.save(
            "digest",
            sandbox_id,
            "ready",
            {"timings_ms": {"install": 100}},
            source_revision="revision",
        )
    store.mark(slot["slot_id"], "ready", expected="preparing", metadata={"sandbox_id": sandbox_id})
    return slot, sandbox_id


def test_pool_reservations_are_bounded_across_replicas(registry):
    stores = [pool_store(registry) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        reserved = list(workers.map(lambda value: value.reserve(), stores))
    assert len([slot for slot in reserved if slot]) == 2
    assert stores[0].status()["preparing"] == 2


def test_pool_concurrent_claims_move_one_mapping_and_keep_physical_state(registry):
    store = pool_store(registry, size=1)
    slot, sandbox_id = ready_spare(store)

    def claim(number):
        with registry.session(f"real-{number}") as session:
            session.bind_owner("app")
            session.save("digest", None, "pending", {"preparation": {"status": "preparing"}})
            hit = store.claim(session, lambda _: True)
            return hit, session.get()

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(claim, range(8)))
    hits = [row for hit, row in results if hit]
    assert len(hits) == 1
    assert hits[0]["sandbox_id"] == sandbox_id
    assert hits[0]["source_revision"] == "revision"
    metrics = hits[0]["bootstrap_metrics"]
    assert metrics["worker_state_key"] == registry.session_key(slot["slot_id"])
    assert metrics["preparation"] == {"status": "preparing"}
    assert registry.runtime(slot["slot_id"])["sandbox"] is None
    assert store.status()["ready"] == 0
    assert store.reserve()  # Assigned sessions do not consume spare capacity.


def test_pool_new_rollout_fences_old_replenishment_and_claims(registry):
    old = pool_store(registry, size=2)
    ready_spare(old)
    disabled = pool_store(registry, generation=2, size=0)
    old.initialize()  # A delayed old replica must not overwrite the new policy.
    assert not old.current() and disabled.current()
    assert old.reserve() is None and disabled.reserve() is None
    with registry.session("new-session") as session:
        session.bind_owner("app")
        assert not old.claim(session, lambda _: True)
        assert not disabled.claim(session, lambda _: True)
    assert disabled.status()["maximum"] == 0


@pytest.mark.parametrize("problem", ["expired", "unhealthy", "old-revision"])
def test_pool_unusable_spare_is_never_assigned(registry, problem):
    store = pool_store(registry, size=1)
    ready_spare(store)
    if problem == "expired":
        with registry.connect() as db:
            db.execute("UPDATE warm_pool_slots SET expires_at=now()-interval '1 second'")
    if problem == "old-revision":
        store = pool_store(registry, generation=2, size=1, key="new-revision")
    with registry.session("new-session") as session:
        session.bind_owner("app")
        assert not store.claim(session, lambda _: problem != "unhealthy")
        assert session.get()["sandbox_id"] is None
    assert store.reserve() is None  # Still counts until shutdown confirmed.


def test_pool_identity_failure_rolls_back_entire_claim(registry):
    store = pool_store(registry, size=1)
    slot, sandbox_id = ready_spare(store, owner="another-app")
    with registry.session("new-session") as session:
        session.bind_owner("app")
        with pytest.raises(RuntimeError, match="identity mismatch"):
            store.claim(session, lambda _: True)
        assert session.get()["sandbox_id"] is None
    assert registry.runtime(slot["slot_id"])["sandbox"]["sandbox_id"] == sandbox_id
    assert store.status()["ready"] == 1


def test_pool_claim_does_not_wait_for_another_claim_probe(registry):
    store = pool_store(registry, size=2)
    ready_spare(store)
    ready_spare(store)
    entered, release = threading.Event(), threading.Event()

    def slow_probe(slot):
        entered.set()
        assert release.wait(5)
        return True

    def claim(name, probe):
        with registry.session(name) as session:
            session.bind_owner("app")
            assert store.claim(session, probe)
            return session.get()["sandbox_id"]

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(claim, "one", slow_probe)
        try:
            assert entered.wait(5)
            second = workers.submit(claim, "two", lambda _: True)
            second_id = second.result(2)
        finally:
            release.set()
        assert first.result(5) != second_id


def controller(registry, *, size=1, generation=1):
    from types import SimpleNamespace

    from databricks_agentkit.sandbox.warm_pool import WarmPool

    sandboxes, lakebox = client(registry)
    lakebox.idle_timeout = 1800
    lakebox.get.side_effect = lambda sandbox_id: {"sandboxId": sandbox_id, "status": "Running"}
    lakebox.api.return_value = {"status": "stopped"}
    orchestrator = MagicMock(
        registry=registry,
        sandboxes=sandboxes,
        lakebox=lakebox,
        sandbox_scope="session",
        worker_mode="persistent",
    )
    pool = WarmPool(
        orchestrator,
        {"agent_id": "revision", "entrypoint": "agent:run"},
        SimpleNamespace(digest="digest"),
        size=size,
        generation=generation,
    )
    pool.store.initialize()
    pool.healthy = MagicMock(return_value=True)

    def prepare(slot_id, **kwargs):
        sandbox_id = "sandbox-" + slot_id
        with registry.session(slot_id) as session:
            session.bind_owner("app")
            session.save("digest", sandbox_id, "ready", source_revision="revision")
        return {
            "sandbox_id": sandbox_id,
            "worker": {"id": slot_id},
            "pool_worker": {},
            "entrypoint": "agent:run",
            "timings_ms": {},
        }

    orchestrator.prepare_session.side_effect = prepare
    return pool


def test_pool_controller_fills_claims_replenishes_and_downsizes(registry):
    pool = controller(registry, size=2)
    pool.reconcile()
    pool.reconcile()
    pool.reconcile()
    assert pool.orchestrator.prepare_session.call_count == 2
    pool.sandboxes.warm_pool = pool
    allocated = pool.sandboxes.get_or_create(
        "real", source_hash="digest", source_revision="revision"
    )
    assert allocated.pool_hit and allocated.state_key
    assert not allocated.created
    pool.orchestrator.lakebox.create.assert_not_called()
    pool.reconcile()
    assert pool.store.status()["ready"] == 2
    disabled = controller(registry, size=0, generation=2)
    disabled.reconcile()
    assert not disabled.store.slots()
    assert registry.runtime("real")["sandbox"]["sandbox_id"] == allocated.id
    stopped = [
        call.args[2]["sandbox_id"] for call in disabled.orchestrator.lakebox.api.call_args_list
    ]
    assert len(stopped) == 2 and allocated.id not in stopped
    pool.reconcile()
    assert pool.orchestrator.prepare_session.call_count == 3


def test_pool_controller_interrupted_preparation_is_retired_not_replayed(registry):
    pool = controller(registry)
    slot = pool.store.reserve()
    with registry.session(slot["slot_id"]) as session:
        session.bind_owner("app")
        session.save("digest", "interrupted", "bootstrapping")
    pool.reconcile()
    assert registry.runtime(slot["slot_id"])["sandbox"] is None
    assert pool.orchestrator.lakebox.api.call_args.args[2]["sandbox_id"] == "interrupted"
    assert pool.orchestrator.prepare_session.call_count == 1
    assert pool.orchestrator.prepare_session.call_args.args[0] != slot["slot_id"]


def test_pool_ambiguous_create_retains_capacity_without_repeating(registry):
    pool = controller(registry)
    slot = pool.store.reserve()
    with registry.session(slot["slot_id"]) as session:
        session.bind_owner("app")
        session.save("digest", None, "creating")
    pool.reconcile()
    assert pool.store.status()["retiring"] == 1
    assert "unresolved" in pool.store.status()["error"]
    pool.orchestrator.prepare_session.assert_not_called()
    pool.orchestrator.lakebox.create.assert_not_called()


def test_pool_recovery_clears_error_and_refill_backoff(registry):
    pool = controller(registry)
    pool.orchestrator.prepare_session.side_effect = RuntimeError("installation failed")
    pool.reconcile()
    pool.reconcile()
    assert pool.store.status()["error"] == "installation failed"
    assert pool.orchestrator.prepare_session.call_count == 1
    with registry.connect() as db:
        db.execute("UPDATE warm_pool_config SET retry_after=now()-interval '1 second'")
    recovered = controller(registry)
    recovered.reconcile()
    assert recovered.store.status()["ready"] == 1
    assert recovered.store.status()["error"] is None


def test_pool_schema_initialization_is_safe_across_replicas(registry):
    with ThreadPoolExecutor(max_workers=4) as workers:
        stores = list(workers.map(lambda _: pool_store(registry), range(4)))
    assert all(store.current() for store in stores)


def test_pool_controller_cannot_retire_a_concurrently_claimed_sandbox(registry):
    pool = controller(registry)
    slot, sandbox_id = ready_spare(pool.store)
    entered, release = threading.Event(), threading.Event()

    def probe(value):
        if value.get("slot_id") == slot["slot_id"]:
            entered.set()
            assert release.wait(5)
            return False
        return True

    pool.healthy.side_effect = probe
    with ThreadPoolExecutor(max_workers=1) as worker:
        scanning = worker.submit(pool.reconcile)
        try:
            assert entered.wait(5)
            with registry.session("claimed-during-scan") as session:
                session.bind_owner("app")
                assert pool.store.claim(session, lambda _: True)
        finally:
            release.set()
        scanning.result(5)
    assert registry.runtime("claimed-during-scan")["sandbox"]["sandbox_id"] == sandbox_id
    pool.orchestrator.lakebox.api.assert_not_called()


def test_pool_does_not_assign_another_agent_revision(registry):
    pool = controller(registry)
    ready_spare(pool.store)
    with registry.session("other-revision") as session:
        session.bind_owner("app")
        assert not pool.claim(session, "other-digest", "revision")
        assert not pool.claim(session, "digest", "other-revision")
    assert pool.store.status()["ready"] == 1


def test_pool_cannot_replace_an_ambiguous_existing_create(registry):
    pool = controller(registry)
    ready_spare(pool.store)
    pool.sandboxes.warm_pool = pool
    with registry.session("ambiguous-session") as session:
        session.bind_owner("app")
        session.save("digest", None, "creating", source_revision="revision")
    with pytest.raises(RevisionConflict, match="ambiguous"):
        pool.sandboxes.get_or_create(
            "ambiguous-session", source_hash="digest", source_revision="revision"
        )
    assert pool.store.status()["ready"] == 1
    pool.orchestrator.lakebox.create.assert_not_called()
