"""Async repository for responses and messages."""

import json
from datetime import datetime
from typing import Any, NamedTuple

from sqlalchemy import select, update
from sqlalchemy.sql import bindparam, text

from databricks_ai_bridge.long_running.db import session_scope
from databricks_ai_bridge.long_running.models import AGENT_DB_SCHEMA, Message, Response


async def create_response(
    response_id: str,
    status: str,
    *,
    owner_pod_id: str | None = None,
    original_request: dict[str, Any] | None = None,
) -> None:
    """Insert a new response row.

    ``owner_pod_id`` and ``original_request`` are optional so that non-durable
    callers (tests, legacy flows) can still create rows without durability
    metadata. When present, they enable heartbeat + crash-resume semantics.
    """
    async with session_scope() as session:
        session.add(
            Response(
                response_id=response_id,
                status=status,
                owner_pod_id=owner_pod_id,
                heartbeat_at=datetime.now().astimezone() if owner_pod_id else None,
                original_request=(
                    json.dumps(original_request) if original_request is not None else None
                ),
            )
        )
        await session.commit()


async def update_response_status(
    response_id: str, status: str, *, expected_current_status: str | None = None
) -> bool:
    """Update response status. Returns True if a row was updated.

    If *expected_current_status* is given the update only takes effect when the
    row's current status matches, avoiding concurrent-update races.
    """
    async with session_scope() as session:
        stmt = update(Response).where(Response.response_id == response_id)
        if expected_current_status is not None:
            stmt = stmt.where(Response.status == expected_current_status)
        stmt = stmt.values(status=status)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def update_response_trace_id(response_id: str, trace_id: str) -> None:
    """Update response with trace_id (MLflow trace for observability)."""
    async with session_scope() as session:
        stmt = update(Response).where(Response.response_id == response_id).values(trace_id=trace_id)
        await session.execute(stmt)
        await session.commit()


async def heartbeat_response(response_id: str, pod_id: str) -> bool:
    """Update heartbeat_at for a response IFF this pod owns it.

    Returns True on success. A False result means the claim has been lost
    (another pod took over, or the run finished and heartbeat should stop).
    """
    async with session_scope() as session:
        stmt = (
            update(Response)
            .where(Response.response_id == response_id, Response.owner_pod_id == pod_id)
            .values(heartbeat_at=datetime.now().astimezone())
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def claim_stale_response(
    response_id: str,
    new_owner_pod_id: str,
    stale_threshold_seconds: float,
) -> int | None:
    """Atomically claim an in-progress response whose heartbeat has gone stale.

    Uses a single conditional UPDATE so exactly one caller wins on contention:
    claim only succeeds if status is ``in_progress`` AND
    (``owner_pod_id IS NULL`` OR ``heartbeat_at`` is older than the threshold).

    Returns the new ``attempt_number`` on success, or ``None`` if the row did
    not satisfy the claim conditions (already completed, already claimed by a
    live pod, or nonexistent).
    """
    # Raw SQL because SQLAlchemy's ORM-level update doesn't expose RETURNING for
    # the incremented column as ergonomically. Using a single statement keeps the
    # claim atomic without an explicit transaction-level lock.
    stmt = text(
        f"""
        UPDATE {AGENT_DB_SCHEMA}.responses
           SET owner_pod_id = :pod,
               heartbeat_at = now(),
               attempt_number = attempt_number + 1
         WHERE response_id = :rid
           AND status = 'in_progress'
           AND (owner_pod_id IS NULL
                OR heartbeat_at IS NULL
                OR heartbeat_at < now() - make_interval(secs => :threshold))
     RETURNING attempt_number
        """
    ).bindparams(
        bindparam("pod", type_=None),
        bindparam("rid", type_=None),
        bindparam("threshold", type_=None),
    )
    async with session_scope() as session:
        result = await session.execute(
            stmt,
            {"pod": new_owner_pod_id, "rid": response_id, "threshold": stale_threshold_seconds},
        )
        row = result.first()
        await session.commit()
        return int(row[0]) if row else None


async def append_message(
    response_id: str,
    sequence_number: int,
    item: str | None = None,
    stream_event: dict[str, Any] | None = None,
    *,
    attempt_number: int = 1,
) -> None:
    """Append a message (stream event) for a response, tagged with attempt_number."""
    async with session_scope() as session:
        session.add(
            Message(
                response_id=response_id,
                sequence_number=sequence_number,
                attempt_number=attempt_number,
                item=item,
                stream_event=json.dumps(stream_event) if stream_event is not None else None,
            )
        )
        await session.commit()


async def get_messages(
    response_id: str,
    after_sequence: int | None = None,
    *,
    attempt_number: int | None = None,
) -> list[tuple[int, str | None, dict[str, Any] | None, int]]:
    """Fetch messages for a response, optionally filtering by sequence / attempt.

    Returns list of ``(sequence_number, item, stream_event_dict, attempt_number)``.
    """
    async with session_scope() as session:
        stmt = select(Message).where(Message.response_id == response_id)
        if after_sequence is not None:
            stmt = stmt.where(Message.sequence_number > after_sequence)
        if attempt_number is not None:
            stmt = stmt.where(Message.attempt_number == attempt_number)
        stmt = stmt.order_by(Message.sequence_number)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        out = []
        for r in rows:
            evt = json.loads(r.stream_event) if r.stream_event else None
            out.append((r.sequence_number, r.item, evt, r.attempt_number))
        return out


class ResponseInfo(NamedTuple):
    response_id: str
    status: str
    created_at: datetime
    trace_id: str | None
    owner_pod_id: str | None
    heartbeat_at: datetime | None
    attempt_number: int
    original_request: dict[str, Any] | None


async def get_response(response_id: str) -> ResponseInfo | None:
    """Fetch response metadata, or None if not found."""
    async with session_scope() as session:
        result = await session.execute(select(Response).where(Response.response_id == response_id))
        row = result.scalar_one_or_none()
        if row:
            return ResponseInfo(
                row.response_id,
                row.status,
                row.created_at,
                row.trace_id,
                row.owner_pod_id,
                row.heartbeat_at,
                row.attempt_number,
                json.loads(row.original_request) if row.original_request else None,
            )
        return None
