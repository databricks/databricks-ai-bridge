"""Tests for Runtime Store implementations."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_agentkit.runtime.durability import lakebase_runtime_store as implementation_store
from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
    LakebaseDurableRuntimeStore,
)
from databricks_agentkit.runtime.durability.store import DurableRuntimeStore
from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_LOCAL_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
    InMemoryRuntimeStore,
    runtime_store_from_environment,
    runtime_store_is_persistent_environment,
)
from databricks_agentkit.runtime.types import (
    InvocationConflictError,
    InvocationStatus,
)


def mock_lakebase():
    connection = AsyncMock()
    engine = MagicMock()

    @asynccontextmanager
    async def begin():
        yield connection

    @asynccontextmanager
    async def connect():
        yield connection

    engine.begin = begin
    engine.connect = connect
    engine.dispose = AsyncMock()
    lakebase = MagicMock(engine=engine)
    lakebase.create_schema = AsyncMock()
    return lakebase, connection


def mapping_result(value):
    result = MagicMock()
    result.mappings.return_value.one.return_value = value
    result.mappings.return_value.one_or_none.return_value = value
    result.mappings.return_value.all.return_value = value if isinstance(value, list) else [value]
    return result


def test_environment_store_is_local_without_an_attached_resource(monkeypatch):
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_DATABASE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_USERNAME_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_SCHEMA_ENV, raising=False)

    assert isinstance(runtime_store_from_environment(), InMemoryRuntimeStore)


def test_only_lakebase_store_has_durable_runtime_capabilities():
    lakebase, _ = mock_lakebase()

    assert not isinstance(InMemoryRuntimeStore(), DurableRuntimeStore)
    assert isinstance(LakebaseDurableRuntimeStore(lakebase=lakebase), DurableRuntimeStore)


def test_managed_connection_uses_the_shared_lakebase_connector(monkeypatch):
    from databricks_ai_bridge import lakebase as lakebase_module

    lakebase, _ = mock_lakebase()
    factory = MagicMock(return_value=lakebase)
    monkeypatch.setattr(lakebase_module, "AsyncLakebaseSQLAlchemy", factory)
    client = MagicMock()

    LakebaseDurableRuntimeStore.from_managed_runtime_store(
        branch="projects/project/branches/runtime-branch",
        database="runtime-db",
        username="app-sp",
        workspace_client=client,
    )

    factory.assert_called_once_with(
        autoscaling_endpoint=None,
        project=None,
        branch="projects/project/branches/runtime-branch",
        database="runtime-db",
        username="app-sp",
        workspace_client=client,
        schema="databricks_agentkit_runtime",
        pool_pre_ping=True,
    )


def test_managed_connection_accepts_an_explicit_schema(monkeypatch):
    from databricks_ai_bridge import lakebase as lakebase_module

    lakebase, _ = mock_lakebase()
    factory = MagicMock(return_value=lakebase)
    monkeypatch.setattr(lakebase_module, "AsyncLakebaseSQLAlchemy", factory)

    LakebaseDurableRuntimeStore.from_managed_runtime_store(
        branch="projects/project/branches/runtime-branch",
        database="runtime-db",
        username="app-sp",
        schema="databricks_agentkit_runtime",
    )

    assert factory.call_args.kwargs["schema"] == "databricks_agentkit_runtime"


def test_app_resource_connection_uses_injected_coordinates(monkeypatch):
    endpoint = "projects/project/branches/production/endpoints/primary"
    client = MagicMock()
    client.postgres.generate_database_credential.return_value.token = "test-oauth-token"
    monkeypatch.setenv("PGHOST", "attached.example.com")
    monkeypatch.setenv("PGPORT", "6543")
    monkeypatch.setenv("PGDATABASE", "attached-db")
    monkeypatch.setenv("PGUSER", "attached-user")
    monkeypatch.setenv("PGSSLMODE", "verify-full")
    from databricks_agentkit.runtime.durability import lakebase_runtime_store as module

    create_engine = MagicMock()
    hooks = []
    monkeypatch.setattr(module, "create_async_engine", create_engine)
    monkeypatch.setattr(module.event, "listens_for", lambda *args: lambda fn: hooks.append(fn))

    LakebaseDurableRuntimeStore.from_app_resource(endpoint=endpoint, workspace_client=client)

    url = create_engine.call_args.args[0]
    assert (url.host, url.port, url.database, url.username) == (
        "attached.example.com",
        6543,
        "attached-db",
        "attached-user",
    )
    assert create_engine.call_args.kwargs["connect_args"] == {"sslmode": "verify-full"}
    params = {}
    hooks[0](None, None, None, params)
    assert params["password"] == "test-oauth-token"
    client.postgres.generate_database_credential.assert_called_once_with(endpoint=endpoint)


@pytest.mark.parametrize(
    "missing",
    [
        RUNTIME_STORE_DATABASE_ENV,
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
        RUNTIME_STORE_USERNAME_ENV,
    ],
)
def test_partial_managed_configuration_does_not_fall_back_to_apps_resource(monkeypatch, missing):
    monkeypatch.setenv(RUNTIME_STORE_DATABASE_ENV, "runtime-db-id")
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
        "projects/project/branches/production",
    )
    monkeypatch.setenv(RUNTIME_STORE_USERNAME_ENV, "app-sp")
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
        "projects/other/branches/production/endpoints/primary",
    )
    monkeypatch.setenv(RUNTIME_STORE_SCHEMA_ENV, "legacy_schema")
    monkeypatch.delenv(missing)

    assert not runtime_store_is_persistent_environment()
    with pytest.raises(RuntimeError, match=missing):
        runtime_store_from_environment()


def test_environment_store_uses_managed_api_coordinates(monkeypatch):
    expected = MagicMock()
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_SCHEMA_ENV, raising=False)
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
        "projects/project/branches/runtime-branch",
    )
    monkeypatch.setenv(RUNTIME_STORE_DATABASE_ENV, "runtime-db-id")
    monkeypatch.setenv(RUNTIME_STORE_USERNAME_ENV, "app-sp")
    factory = MagicMock(return_value=expected)
    monkeypatch.setattr(
        implementation_store.LakebaseDurableRuntimeStore,
        "from_managed_runtime_store",
        factory,
    )

    assert runtime_store_from_environment() is expected
    factory.assert_called_once_with(
        branch="projects/project/branches/runtime-branch",
        database="runtime-db-id",
        username="app-sp",
        schema="databricks_agentkit_runtime",
    )
    assert runtime_store_is_persistent_environment()


def test_environment_store_uses_explicit_managed_schema(monkeypatch):
    expected = MagicMock()
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
        "projects/project/branches/runtime-branch",
    )
    monkeypatch.setenv(RUNTIME_STORE_DATABASE_ENV, "runtime-db-id")
    monkeypatch.setenv(RUNTIME_STORE_USERNAME_ENV, "app-sp")
    monkeypatch.setenv(RUNTIME_STORE_SCHEMA_ENV, "databricks_agentkit_runtime")
    factory = MagicMock(return_value=expected)
    monkeypatch.setattr(
        implementation_store.LakebaseDurableRuntimeStore,
        "from_managed_runtime_store",
        factory,
    )

    assert runtime_store_from_environment() is expected
    factory.assert_called_once_with(
        branch="projects/project/branches/runtime-branch",
        database="runtime-db-id",
        username="app-sp",
        schema="databricks_agentkit_runtime",
    )


def test_environment_store_uses_the_attached_lakebase_resource(monkeypatch):
    expected = MagicMock()
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_DATABASE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_USERNAME_ENV, raising=False)
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
        "projects/project/branches/production/endpoints/primary",
    )
    monkeypatch.setenv(RUNTIME_STORE_SCHEMA_ENV, "databricks_agentkit_runtime_app")
    from_app_resource = MagicMock(return_value=expected)
    monkeypatch.setattr(
        implementation_store.LakebaseDurableRuntimeStore,
        "from_app_resource",
        from_app_resource,
    )

    assert runtime_store_from_environment() is expected
    from_app_resource.assert_called_once_with(
        endpoint="projects/project/branches/production/endpoints/primary",
        schema="databricks_agentkit_runtime_app",
    )


def test_environment_store_rejects_an_incomplete_lakebase_resource(monkeypatch):
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_SCHEMA_ENV, raising=False)
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
        "projects/project/branches/production/endpoints/primary",
    )

    with pytest.raises(RuntimeError, match=RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV):
        runtime_store_from_environment()


def test_environment_store_local_marker_overrides_attached_resource(monkeypatch):
    monkeypatch.setenv(RUNTIME_STORE_LOCAL_ENV, "true")
    monkeypatch.setenv(
        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
        "projects/project/branches/production/endpoints/primary",
    )
    monkeypatch.setenv(RUNTIME_STORE_SCHEMA_ENV, "databricks_agentkit_runtime_app")

    assert isinstance(runtime_store_from_environment(), InMemoryRuntimeStore)


def invocation_row(**overrides):
    row = {
        "invocation_id": "session-1",
        "status": "QUEUED",
        "attempt": 0,
        "request_json": '{"input": "hello"}',
        "response_json": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_initialize_creates_invocation_and_event_tables():
    lakebase, connection = mock_lakebase()
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    await store.initialize()

    sql = " ".join(str(call.args[0]) for call in connection.execute.await_args_list)
    assert "databricks_agentkit_runtime.invocations" in sql
    assert "invocation_id TEXT PRIMARY KEY" in sql
    assert "request JSONB NOT NULL" in sql
    assert "response JSONB" in sql
    assert "jsonb_typeof(request)" not in sql
    assert "jsonb_typeof(response)" not in sql
    assert "databricks_agentkit_runtime.invocation_events" in sql
    assert "sequence_number BIGSERIAL PRIMARY KEY" in sql
    lakebase.create_schema.assert_awaited_once()


@pytest.mark.asyncio
async def test_accept_returns_existing_request_when_it_matches():
    lakebase, connection = mock_lakebase()
    connection.execute.side_effect = [MagicMock(), mapping_result(invocation_row())]
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    state = await store.accept("session-1", {"input": "hello"})

    assert state.invocation_id == "session-1"
    assert state.status == InvocationStatus.QUEUED
    assert state.request == {"input": "hello"}


@pytest.mark.asyncio
async def test_accept_rejects_same_id_with_different_request():
    lakebase, connection = mock_lakebase()
    connection.execute.side_effect = [MagicMock(), mapping_result(invocation_row())]
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    with pytest.raises(InvocationConflictError):
        await store.accept("session-1", {"input": "different"})


@pytest.mark.asyncio
async def test_claim_returns_request_and_incremented_attempt():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = mapping_result(invocation_row(status="ACTIVE", attempt=2))
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    state = await store.claim_recoverable("session-1", 10)

    assert state is not None
    assert state.attempt == 2
    assert state.request == {"input": "hello"}
    claim_parameters = connection.execute.await_args_list[0].args[1]
    assert claim_parameters == {"invocation_id": "session-1", "stale": 10}
    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.started"}'


@pytest.mark.asyncio
async def test_queued_invocation_query_only_selects_queued_work():
    lakebase, connection = mock_lakebase()
    result = MagicMock()
    result.scalars.return_value.all.return_value = ["session-1"]
    connection.execute.return_value = result
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    assert await store.queued_invocation_ids() == ["session-1"]

    query = str(connection.execute.await_args.args[0])
    assert "status='QUEUED'" in query
    assert "heartbeat_at" not in query


@pytest.mark.asyncio
async def test_stale_invocation_query_uses_durable_lease():
    lakebase, connection = mock_lakebase()
    result = MagicMock()
    result.scalars.return_value.all.return_value = ["session-1"]
    connection.execute.return_value = result
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    assert await store.stale_invocation_ids(10) == ["session-1"]

    query = str(connection.execute.await_args.args[0])
    assert "SELECT invocation_id" in query
    assert "heartbeat_at" in query
    assert "status='QUEUED'" not in query
    assert connection.execute.await_args.args[1] == {"stale": 10}


@pytest.mark.asyncio
async def test_heartbeat_is_fenced_by_invocation_and_attempt():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    assert await store.heartbeat("session-1", 2) is True

    query = str(connection.execute.await_args.args[0])
    assert "WHERE invocation_id=:invocation_id" in query
    assert connection.execute.await_args.args[1] == {"invocation_id": "session-1", "attempt": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("response_json", ['["done"]', '"done"', "null"])
async def test_get_decodes_any_cached_json_response(response_json):
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = mapping_result(
        invocation_row(
            status="COMPLETED",
            attempt=1,
            response_json=response_json,
        )
    )
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    state = await store.get("session-1")

    assert state is not None
    assert state.status == InvocationStatus.COMPLETED
    assert state.response == json.loads(response_json)


@pytest.mark.asyncio
async def test_complete_persists_response_and_lifecycle_event_atomically():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    assert await store.complete("session-1", 2, ["done"]) is True

    parameters = connection.execute.await_args_list[0].args[1]
    assert parameters["invocation_id"] == "session-1"
    assert parameters["attempt"] == 2
    assert parameters["response"] == '["done"]'
    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.completed"}'


@pytest.mark.asyncio
async def test_fail_persists_lifecycle_event_atomically():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    assert await store.fail("session-1", 2) is True

    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.failed"}'


@pytest.mark.asyncio
async def test_append_event_returns_replay_cursor_for_owned_attempt():
    lakebase, connection = mock_lakebase()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 7
    connection.execute.return_value = result
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    sequence_number = await store.append_event(
        "session-1",
        2,
        {"type": "progress", "step": 1},
    )

    assert sequence_number == 7
    parameters = connection.execute.await_args.args[1]
    assert parameters == {
        "invocation_id": "session-1",
        "attempt": 2,
        "event": '{"type": "progress", "step": 1}',
    }


@pytest.mark.asyncio
async def test_events_returns_ordered_replay_data():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = mapping_result(
        [
            {
                "sequence_number": 8,
                "invocation_id": "session-1",
                "attempt": 2,
                "event_json": '{"type": "progress", "step": 2}',
            }
        ]
    )
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)

    events = await store.events("session-1", after_sequence=7)

    assert len(events) == 1
    assert events[0].sequence_number == 8
    assert events[0].attempt == 2
    assert events[0].event == {"type": "progress", "step": 2}


def test_schema_name_is_validated():
    lakebase, _ = mock_lakebase()
    with pytest.raises(ValueError, match="invalid Runtime Store schema"):
        LakebaseDurableRuntimeStore(lakebase=lakebase, schema="bad-schema;drop")


@pytest.mark.asyncio
async def test_store_rejects_empty_invocation_id():
    lakebase, _ = mock_lakebase()
    store = LakebaseDurableRuntimeStore(lakebase=lakebase)
    with pytest.raises(ValueError, match="must not be empty"):
        await store.accept("", {})
