"""Tests for the long_running repository functions and db lifecycle."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("sqlalchemy")

import databricks_ai_bridge.long_running.db as db_mod
from databricks_ai_bridge.long_running.db import dispose_db, init_db, session_scope
from databricks_ai_bridge.long_running.models import Message, Response
from databricks_ai_bridge.long_running.repository import (
    append_message,
    create_response,
    get_messages,
    get_response,
    update_response_status,
    update_response_trace_id,
)


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture(autouse=True)
def _patch_get_async_session(mock_session):
    from contextlib import asynccontextmanager

    def _make_session():
        @asynccontextmanager
        async def _cm():
            yield mock_session

        return _cm()

    with patch(
        "databricks_ai_bridge.long_running.repository.session_scope",
        side_effect=_make_session,
    ):
        yield


@pytest.mark.asyncio
async def test_create_response(mock_session):
    await create_response("resp_abc123", "in_progress")
    mock_session.add.assert_called_once()
    added = mock_session.add.call_args[0][0]
    assert isinstance(added, Response)
    assert added.response_id == "resp_abc123"
    assert added.status == "in_progress"
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_response_status(mock_session):
    result_mock = MagicMock()
    result_mock.rowcount = 1
    mock_session.execute.return_value = result_mock

    updated = await update_response_status("resp_abc123", "completed")
    assert updated is True
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_response_status_not_found(mock_session):
    result_mock = MagicMock()
    result_mock.rowcount = 0
    mock_session.execute.return_value = result_mock

    updated = await update_response_status("resp_missing", "completed")
    assert updated is False
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_response_status_conditional(mock_session):
    result_mock = MagicMock()
    result_mock.rowcount = 1
    mock_session.execute.return_value = result_mock

    updated = await update_response_status(
        "resp_abc123", "failed", expected_current_status="in_progress"
    )
    assert updated is True
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_response_status_conditional_mismatch(mock_session):
    result_mock = MagicMock()
    result_mock.rowcount = 0
    mock_session.execute.return_value = result_mock

    updated = await update_response_status(
        "resp_abc123", "failed", expected_current_status="in_progress"
    )
    assert updated is False


@pytest.mark.asyncio
async def test_update_response_trace_id(mock_session):
    result_mock = MagicMock()
    result_mock.rowcount = 1
    mock_session.execute.return_value = result_mock

    await update_response_trace_id("resp_abc123", "trace_xyz")
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_append_message(mock_session):
    evt = {"type": "response.output_item.done", "item": {"text": "hello"}}
    await append_message("resp_abc123", 0, item='{"text": "hello"}', stream_event=evt)
    mock_session.add.assert_called_once()
    added = mock_session.add.call_args[0][0]
    assert isinstance(added, Message)
    assert added.response_id == "resp_abc123"
    assert added.sequence_number == 0
    assert added.item == '{"text": "hello"}'
    assert json.loads(added.stream_event) == evt
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_append_message_none_event(mock_session):
    await append_message("resp_abc123", 1, item=None, stream_event=None)
    added = mock_session.add.call_args[0][0]
    assert added.stream_event is None
    assert added.item is None


@pytest.mark.asyncio
async def test_get_messages(mock_session):
    msg1 = MagicMock()
    msg1.sequence_number = 0
    msg1.item = '{"text": "hello"}'
    msg1.stream_event = json.dumps({"type": "response.output_item.done"})

    msg2 = MagicMock()
    msg2.sequence_number = 1
    msg2.item = None
    msg2.stream_event = None

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [msg1, msg2]
    mock_session.execute.return_value = result_mock

    messages = await get_messages("resp_abc123", after_sequence=None)
    assert len(messages) == 2
    assert messages[0] == (0, '{"text": "hello"}', {"type": "response.output_item.done"})
    assert messages[1] == (1, None, None)


@pytest.mark.asyncio
async def test_get_response(mock_session):
    row = MagicMock()
    row.response_id = "resp_abc123"
    row.status = "completed"
    from datetime import datetime, timezone

    row.created_at = datetime(2009, 2, 13, 23, 31, 30, tzinfo=timezone.utc)
    row.trace_id = "trace_xyz"
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = row
    mock_session.execute.return_value = result_mock

    result = await get_response("resp_abc123")
    assert result == (
        "resp_abc123",
        "completed",
        datetime(2009, 2, 13, 23, 31, 30, tzinfo=timezone.utc),
        "trace_xyz",
    )


@pytest.mark.asyncio
async def test_get_response_not_found(mock_session):
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result_mock

    result = await get_response("resp_missing")
    assert result is None


# ---------------------------------------------------------------------------
# init_db / dispose_db / session_scope tests
# ---------------------------------------------------------------------------

DB_MODULE = "databricks_ai_bridge.long_running.db"


@pytest.fixture
def reset_db_globals():
    """Reset db.py module globals after tests that call init_db."""
    yield
    db_mod._session_factory = None
    db_mod._engine = None
    db_mod._lakebase = None
    db_mod._initialized = False


def _mock_lakebase_engine():
    """Return (mock_lakebase_cls, mock_conn) with engine wired up for init_db."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.run_sync = AsyncMock()

    mock_engine = MagicMock()
    mock_engine.sync_engine = MagicMock()

    @asynccontextmanager
    async def fake_begin():
        yield mock_conn

    mock_engine.begin = fake_begin

    mock_lakebase = MagicMock()
    mock_lakebase.engine = mock_engine

    mock_cls = MagicMock(return_value=mock_lakebase)
    return mock_cls, mock_conn


class TestInitDb:
    @pytest.mark.asyncio
    async def test_with_autoscaling_endpoint(self, reset_db_globals):
        mock_cls, _ = _mock_lakebase_engine()
        with patch(f"{DB_MODULE}.AsyncLakebaseSQLAlchemy", mock_cls), patch(f"{DB_MODULE}.event"):
            await init_db(autoscaling_endpoint="ep-abc")

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["autoscaling_endpoint"] == "ep-abc"
        assert kwargs["pool_size"] == 10
        assert kwargs["max_overflow"] == 0
        assert kwargs["pool_pre_ping"] is True
        assert "instance_name" not in kwargs
        assert "project" not in kwargs
        assert "branch" not in kwargs

    @pytest.mark.asyncio
    async def test_with_instance_name(self, reset_db_globals):
        mock_cls, _ = _mock_lakebase_engine()
        with patch(f"{DB_MODULE}.AsyncLakebaseSQLAlchemy", mock_cls), patch(f"{DB_MODULE}.event"):
            await init_db(instance_name="my-instance")

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["instance_name"] == "my-instance"
        assert "autoscaling_endpoint" not in kwargs
        assert "project" not in kwargs
        assert "branch" not in kwargs

    @pytest.mark.asyncio
    async def test_with_project_and_branch(self, reset_db_globals):
        mock_cls, _ = _mock_lakebase_engine()
        with patch(f"{DB_MODULE}.AsyncLakebaseSQLAlchemy", mock_cls), patch(f"{DB_MODULE}.event"):
            await init_db(project="proj", branch="br")

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["project"] == "proj"
        assert kwargs["branch"] == "br"
        assert "instance_name" not in kwargs
        assert "autoscaling_endpoint" not in kwargs

    @pytest.mark.asyncio
    async def test_creates_schema_and_tables(self, reset_db_globals):
        mock_cls, mock_conn = _mock_lakebase_engine()
        with patch(f"{DB_MODULE}.AsyncLakebaseSQLAlchemy", mock_cls), patch(f"{DB_MODULE}.event"):
            await init_db(autoscaling_endpoint="ep")

        mock_conn.execute.assert_awaited_once()
        sql_arg = str(mock_conn.execute.call_args[0][0])
        assert "CREATE SCHEMA IF NOT EXISTS" in sql_arg
        mock_conn.run_sync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_idempotent_skips_second_call(self, reset_db_globals):
        mock_cls, _ = _mock_lakebase_engine()
        with patch(f"{DB_MODULE}.AsyncLakebaseSQLAlchemy", mock_cls), patch(f"{DB_MODULE}.event"):
            await init_db(autoscaling_endpoint="first")
            await init_db(instance_name="second")

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["autoscaling_endpoint"] == "first"


class TestDisposeDb:
    @pytest.mark.asyncio
    async def test_disposes_engine_and_resets_globals(self, monkeypatch):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        monkeypatch.setattr(db_mod, "_engine", mock_engine)
        monkeypatch.setattr(db_mod, "_session_factory", MagicMock())
        monkeypatch.setattr(db_mod, "_lakebase", MagicMock())
        monkeypatch.setattr(db_mod, "_initialized", True)

        await dispose_db()

        mock_engine.dispose.assert_awaited_once()
        assert db_mod._session_factory is None
        assert db_mod._engine is None
        assert db_mod._lakebase is None
        assert db_mod._initialized is False

    @pytest.mark.asyncio
    async def test_noop_when_not_initialized(self, monkeypatch):
        monkeypatch.setattr(db_mod, "_engine", None)
        monkeypatch.setattr(db_mod, "_session_factory", None)
        monkeypatch.setattr(db_mod, "_lakebase", None)
        monkeypatch.setattr(db_mod, "_initialized", False)

        await dispose_db()  # should not raise


class TestSessionScope:
    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, monkeypatch):
        monkeypatch.setattr(db_mod, "_session_factory", None)
        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with session_scope():
                pass

    @pytest.mark.asyncio
    async def test_yields_session_when_initialized(self, monkeypatch):
        mock_session = AsyncMock()

        @asynccontextmanager
        async def fake_factory():
            yield mock_session

        monkeypatch.setattr(db_mod, "_session_factory", fake_factory)
        async with session_scope() as session:
            assert session is mock_session
