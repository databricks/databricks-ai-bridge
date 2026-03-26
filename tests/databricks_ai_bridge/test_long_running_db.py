"""Tests for the long_running repository functions."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("sqlalchemy")

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
