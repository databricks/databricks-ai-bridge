"""Tests for the Lakebase durability store."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_mason.runtime.durability.store import (
    RUNTIME_DATABASE_ENV,
    RUNTIME_ENDPOINT_ENV,
    RUNTIME_LOCAL_ENV,
    RUNTIME_SCHEMA_ENV,
    RUNTIME_USERNAME_ENV,
    InMemoryDurabilityStore,
    LakebaseDurabilityStore,
    default_durability_store,
)
from databricks_mason.runtime.durability.types import (
    DurableExecutionStatus,
    DurableRequestConflictError,
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


def test_default_store_is_local_without_an_attached_resource(monkeypatch):
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.delenv(RUNTIME_ENDPOINT_ENV, raising=False)

    assert isinstance(default_durability_store(), InMemoryDurabilityStore)


def test_default_store_uses_the_provisioned_runtime_store(monkeypatch):
    expected = MagicMock()
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "mason-app")
    monkeypatch.setenv(
        RUNTIME_ENDPOINT_ENV, "projects/project/branches/production/endpoints/primary"
    )
    monkeypatch.setenv(RUNTIME_DATABASE_ENV, "mason-app-store")
    monkeypatch.setenv(RUNTIME_USERNAME_ENV, "00000000-0000-0000-0000-000000000001")
    monkeypatch.setenv(RUNTIME_SCHEMA_ENV, "databricks_mason_runtime_app")
    from_runtime_store = MagicMock(return_value=expected)
    monkeypatch.setattr(LakebaseDurabilityStore, "from_runtime_store", from_runtime_store)

    assert default_durability_store() is expected
    from_runtime_store.assert_called_once_with(
        endpoint="projects/project/branches/production/endpoints/primary",
        database="mason-app-store",
        username="00000000-0000-0000-0000-000000000001",
        schema="databricks_mason_runtime_app",
    )


def test_app_runtime_store_resolves_endpoint_host(monkeypatch):
    endpoint = "projects/project/branches/production/endpoints/primary"
    workspace_client = MagicMock()
    workspace_client.postgres.get_endpoint.return_value.status.hosts.host = "lakebase.example.com"
    engine = MagicMock()
    create_async_engine = MagicMock(return_value=engine)
    monkeypatch.setattr(
        "databricks_mason.runtime.durability.store.create_async_engine", create_async_engine
    )
    monkeypatch.setattr(
        "databricks_mason.runtime.durability.store.event.listens_for",
        lambda *args, **kwargs: lambda function: function,
    )

    store = LakebaseDurabilityStore.from_runtime_store(
        endpoint=endpoint,
        database="mason-app-store",
        username="00000000-0000-0000-0000-000000000001",
        workspace_client=workspace_client,
    )

    workspace_client.postgres.get_endpoint.assert_called_once_with(name=endpoint)
    url = create_async_engine.call_args.args[0]
    assert url.host == "lakebase.example.com"
    assert url.port == 5432
    assert url.database == "mason-app-store"
    assert url.username == "00000000-0000-0000-0000-000000000001"
    assert store._lakebase.engine is engine


def test_app_runtime_store_rejects_endpoint_without_host():
    workspace_client = MagicMock()
    workspace_client.postgres.get_endpoint.return_value.status.hosts.host = None

    with pytest.raises(RuntimeError, match="did not return a host"):
        LakebaseDurabilityStore.from_runtime_store(
            endpoint="projects/project/branches/production/endpoints/primary",
            database="mason-app-store",
            username="00000000-0000-0000-0000-000000000001",
            workspace_client=workspace_client,
        )


def test_default_store_ignores_deploy_env_outside_apps(monkeypatch):
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.setenv(
        RUNTIME_ENDPOINT_ENV, "projects/project/branches/production/endpoints/primary"
    )

    assert isinstance(default_durability_store(), InMemoryDurabilityStore)


def test_default_store_rejects_missing_resource_inside_apps(monkeypatch):
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "mason-app")
    monkeypatch.delenv(RUNTIME_ENDPOINT_ENV, raising=False)

    with pytest.raises(RuntimeError, match=RUNTIME_ENDPOINT_ENV):
        default_durability_store()


def test_default_store_uses_memory_when_apps_run_local_sets_an_app_name(monkeypatch):
    monkeypatch.setenv(RUNTIME_LOCAL_ENV, "true")
    monkeypatch.setenv("DATABRICKS_APP_NAME", "local-durability-app")
    monkeypatch.setenv(
        RUNTIME_ENDPOINT_ENV, "projects/project/branches/production/endpoints/primary"
    )

    assert isinstance(default_durability_store(), InMemoryDurabilityStore)


def execution_row(**overrides):
    row = {
        "execution_id": "session-1",
        "status": "QUEUED",
        "attempt": 0,
        "heartbeat_at": None,
        "request_json": '{"input": "hello"}',
        "response_json": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_initialize_creates_execution_and_event_tables():
    lakebase, connection = mock_lakebase()
    store = LakebaseDurabilityStore(lakebase=lakebase)

    await store.initialize()

    sql = " ".join(str(call.args[0]) for call in connection.execute.await_args_list)
    assert "databricks_mason_runtime.executions" in sql
    assert "execution_id TEXT PRIMARY KEY" in sql
    assert "request JSONB NOT NULL" in sql
    assert "response JSONB" in sql
    assert "jsonb_typeof(request)" not in sql
    assert "jsonb_typeof(response)" not in sql
    assert "databricks_mason_runtime.execution_events" in sql
    assert "sequence_number BIGSERIAL PRIMARY KEY" in sql
    lakebase.create_schema.assert_awaited_once()


@pytest.mark.asyncio
async def test_accept_returns_existing_request_when_it_matches():
    lakebase, connection = mock_lakebase()
    connection.execute.side_effect = [MagicMock(), mapping_result(execution_row())]
    store = LakebaseDurabilityStore(lakebase=lakebase)

    state = await store.accept("session-1", {"input": "hello"})

    assert state.execution_id == "session-1"
    assert state.status == DurableExecutionStatus.QUEUED
    assert state.request == {"input": "hello"}


@pytest.mark.asyncio
async def test_accept_rejects_same_id_with_different_request():
    lakebase, connection = mock_lakebase()
    connection.execute.side_effect = [MagicMock(), mapping_result(execution_row())]
    store = LakebaseDurabilityStore(lakebase=lakebase)

    with pytest.raises(DurableRequestConflictError):
        await store.accept("session-1", {"input": "different"})


@pytest.mark.asyncio
async def test_claim_returns_request_and_incremented_attempt():
    lakebase, connection = mock_lakebase()
    heartbeat = datetime.now(timezone.utc)
    connection.execute.return_value = mapping_result(
        execution_row(status="ACTIVE", attempt=2, heartbeat_at=heartbeat)
    )
    store = LakebaseDurabilityStore(lakebase=lakebase)

    state = await store.claim("session-1", 10)

    assert state is not None
    assert state.attempt == 2
    assert state.heartbeat_at == heartbeat
    assert state.request == {"input": "hello"}
    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.started"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("response_json", ['["done"]', '"done"', "null"])
async def test_get_decodes_any_cached_json_response(response_json):
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = mapping_result(
        execution_row(
            status="COMPLETED",
            attempt=1,
            response_json=response_json,
        )
    )
    store = LakebaseDurabilityStore(lakebase=lakebase)

    state = await store.get("session-1")

    assert state is not None
    assert state.status == DurableExecutionStatus.COMPLETED
    assert state.response == json.loads(response_json)


@pytest.mark.asyncio
async def test_complete_persists_response_and_lifecycle_event_atomically():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurabilityStore(lakebase=lakebase)

    assert await store.complete("session-1", 2, ["done"]) is True

    parameters = connection.execute.await_args_list[0].args[1]
    assert parameters["execution_id"] == "session-1"
    assert parameters["attempt"] == 2
    assert parameters["response"] == '["done"]'
    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.completed"}'


@pytest.mark.asyncio
async def test_fail_persists_lifecycle_event_atomically():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurabilityStore(lakebase=lakebase)

    assert await store.fail("session-1", 2) is True

    event_parameters = connection.execute.await_args_list[1].args[1]
    assert event_parameters["event"] == '{"type": "run.failed"}'


@pytest.mark.asyncio
async def test_append_event_returns_replay_cursor_for_owned_attempt():
    lakebase, connection = mock_lakebase()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 7
    connection.execute.return_value = result
    store = LakebaseDurabilityStore(lakebase=lakebase)

    sequence_number = await store.append_event(
        "session-1",
        2,
        {"type": "progress", "step": 1},
    )

    assert sequence_number == 7
    parameters = connection.execute.await_args.args[1]
    assert parameters == {
        "execution_id": "session-1",
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
                "execution_id": "session-1",
                "attempt": 2,
                "event_json": '{"type": "progress", "step": 2}',
            }
        ]
    )
    store = LakebaseDurabilityStore(lakebase=lakebase)

    events = await store.events("session-1", after_sequence=7)

    assert len(events) == 1
    assert events[0].sequence_number == 8
    assert events[0].attempt == 2
    assert events[0].event == {"type": "progress", "step": 2}


def test_schema_name_is_validated():
    lakebase, _ = mock_lakebase()
    with pytest.raises(ValueError, match="invalid durability schema"):
        LakebaseDurabilityStore(lakebase=lakebase, schema="bad-schema;drop")


@pytest.mark.asyncio
async def test_store_rejects_empty_execution_id():
    lakebase, _ = mock_lakebase()
    store = LakebaseDurabilityStore(lakebase=lakebase)
    with pytest.raises(ValueError, match="must not be empty"):
        await store.accept("", {})
