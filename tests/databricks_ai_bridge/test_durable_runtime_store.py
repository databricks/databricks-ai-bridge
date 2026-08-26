"""Tests for the Lakebase durability store."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from databricks_ai_bridge.durable_runtime import (
    DurableExecutionStatus,
    DurableRequestConflictError,
    LakebaseDurabilityStore,
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
    assert "databricks_durable_runtime.executions" in sql
    assert "execution_id TEXT PRIMARY KEY" in sql
    assert "request JSONB NOT NULL" in sql
    assert "response JSONB" in sql
    assert "databricks_durable_runtime.execution_events" in sql
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


@pytest.mark.asyncio
async def test_get_decodes_cached_response():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = mapping_result(
        execution_row(
            status="COMPLETED",
            attempt=1,
            response_json='{"output": "done"}',
        )
    )
    store = LakebaseDurabilityStore(lakebase=lakebase)

    state = await store.get("session-1")

    assert state is not None
    assert state.status == DurableExecutionStatus.COMPLETED
    assert state.response == {"output": "done"}


@pytest.mark.asyncio
async def test_complete_persists_response_for_owned_attempt():
    lakebase, connection = mock_lakebase()
    connection.execute.return_value = MagicMock(rowcount=1)
    store = LakebaseDurabilityStore(lakebase=lakebase)

    assert await store.complete("session-1", 2, {"output": "done"}) is True

    parameters = connection.execute.await_args.args[1]
    assert parameters["execution_id"] == "session-1"
    assert parameters["attempt"] == 2
    assert parameters["response"] == '{"output": "done"}'


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
