"""Async repository for responses and messages."""

import json
from typing import Any, NamedTuple

try:
    from sqlalchemy import select, update
except ImportError as e:
    raise ImportError(
        "Long-running server requires databricks-ai-bridge[server]. "
        "Please install with: pip install databricks-ai-bridge[server]"
    ) from e

from databricks_ai_bridge.long_running.db import get_async_session
from databricks_ai_bridge.long_running.models import Message, Response


async def create_response(response_id: str, status: str) -> None:
    """Insert a new response."""
    async with get_async_session() as session:
        session.add(Response(response_id=response_id, status=status))
        await session.commit()


async def update_response_status(
    response_id: str, status: str, *, expected_current_status: str | None = None
) -> bool:
    """Update response status. Returns True if a row was updated.

    If *expected_current_status* is given the update only takes effect when the
    row's current status matches, avoiding concurrent-update races.
    """
    async with get_async_session() as session:
        stmt = (
            update(Response)
            .where(Response.response_id == response_id)
        )
        if expected_current_status is not None:
            stmt = stmt.where(Response.status == expected_current_status)
        stmt = stmt.values(status=status)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def update_response_trace_id(response_id: str, trace_id: str) -> None:
    """Update response with trace_id (MLflow trace for observability)."""
    async with get_async_session() as session:
        result = await session.execute(select(Response).where(Response.response_id == response_id))
        row = result.scalar_one_or_none()
        if row:
            row.trace_id = trace_id
            await session.commit()


async def append_message(
    response_id: str,
    sequence_number: int,
    item: str | None = None,
    stream_event: dict[str, Any] | None = None,
) -> None:
    """Append a message (stream event) for a response."""
    async with get_async_session() as session:
        session.add(
            Message(
                response_id=response_id,
                sequence_number=sequence_number,
                item=item,
                stream_event=json.dumps(stream_event) if stream_event is not None else None,
            )
        )
        await session.commit()


async def get_messages(
    response_id: str,
    after_sequence: int | None = None,
) -> list[tuple[int, str | None, dict[str, Any] | None]]:
    """Fetch messages for a response, optionally after a sequence number.

    Returns list of (sequence_number, item, stream_event_dict).
    """
    async with get_async_session() as session:
        stmt = select(Message).where(Message.response_id == response_id)
        if after_sequence is not None:
            stmt = stmt.where(Message.sequence_number > after_sequence)
        stmt = stmt.order_by(Message.sequence_number)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        out = []
        for r in rows:
            evt = json.loads(r.stream_event) if r.stream_event else None
            out.append((r.sequence_number, r.item, evt))
        return out


class ResponseInfo(NamedTuple):
    response_id: str
    status: str
    created_at: float
    trace_id: str | None


async def get_response(response_id: str) -> ResponseInfo | None:
    """Fetch response metadata, or None if not found."""
    async with get_async_session() as session:
        result = await session.execute(select(Response).where(Response.response_id == response_id))
        row = result.scalar_one_or_none()
        if row:
            return ResponseInfo(row.response_id, row.status, row.created_at, row.trace_id)
        return None
