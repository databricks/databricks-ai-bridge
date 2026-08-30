"""Async repository for responses and messages."""

import json
from datetime import datetime
from typing import Any, NamedTuple

from sqlalchemy import select, update
from sqlalchemy.sql import bindparam, text

from databricks_ai_bridge.long_running.db import session_scope
from databricks_ai_bridge.long_running.models import (
    AGENT_DB_SCHEMA,
    ConversationAlias,
    Message,
    Response,
)


async def create_response(
    response_id: str,
    status: str,
    *,
    durable: bool = False,
    original_request: dict[str, Any] | None = None,
) -> None:
    """Insert a new response row.

    When ``durable=True``, ``heartbeat_at`` is initialized to ``now()`` so
    the row doesn't immediately look stale. Non-durable callers (tests,
    legacy flows) skip the heartbeat init.
    """
    async with session_scope() as session:
        session.add(
            Response(
                response_id=response_id,
                status=status,
                heartbeat_at=datetime.now().astimezone() if durable else None,
                original_request=(
                    json.dumps(original_request) if original_request is not None else None
                ),
            )
        )
        await session.commit()


async def update_response_status(
    response_id: str,
    status: str,
    *,
    expected_current_status: str | None = None,
    expected_attempt_number: int | None = None,
) -> bool:
    """Update response status. Returns True if a row was updated.

    If *expected_current_status* is given the update only takes effect when the
    row's current status matches, avoiding concurrent-update races.

    If *expected_attempt_number* is given the update only takes effect when the
    row's current ``attempt_number`` matches, ensuring only the pod that owns
    the current attempt can transition the row to a terminal state. This
    prevents a stale background task (e.g. a deferred-fail timer that fired
    after another pod claimed the row for resume) from clobbering the new
    owner's in-progress state.
    """
    async with session_scope() as session:
        stmt = update(Response).where(Response.response_id == response_id)
        if expected_current_status is not None:
            stmt = stmt.where(Response.status == expected_current_status)
        if expected_attempt_number is not None:
            stmt = stmt.where(Response.attempt_number == expected_attempt_number)
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


async def heartbeat_response(response_id: str, expected_attempt_number: int) -> bool:
    """Update heartbeat_at for a response IFF the attempt is still ours.

    Returns True on success. A False result means the claim has been lost —
    another pod CAS-bumped ``attempt_number``, so this pod is no longer the
    owner and the heartbeat task should stop. Implicit-ownership model:
    whichever pod last successfully heartbeats at the current
    ``attempt_number`` is the de facto owner.
    """
    async with session_scope() as session:
        stmt = (
            update(Response)
            .where(
                Response.response_id == response_id,
                Response.attempt_number == expected_attempt_number,
            )
            .values(heartbeat_at=datetime.now().astimezone())
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def claim_stale_response(
    response_id: str,
    stale_threshold_seconds: float,
) -> int | None:
    """Atomically claim an in-progress response whose heartbeat has gone stale.

    Uses a single conditional UPDATE so exactly one caller wins on contention:
    claim only succeeds if status is ``in_progress`` AND
    (``heartbeat_at IS NULL`` OR ``heartbeat_at`` is older than the threshold).
    The new attempt_number is the previous + 1; the prior attempt's heartbeat
    task will detect this on its next heartbeat (rowcount=0) and stop.

    Returns the new ``attempt_number`` on success, or ``None`` if the row did
    not satisfy the claim conditions (already completed, heartbeat still fresh,
    or nonexistent).
    """
    # Raw SQL because SQLAlchemy's ORM-level update doesn't expose RETURNING for
    # the incremented column as ergonomically. Using a single statement keeps the
    # claim atomic without an explicit transaction-level lock.
    stmt = text(
        f"""
        UPDATE {AGENT_DB_SCHEMA}.responses
           SET heartbeat_at = now(),
               attempt_number = attempt_number + 1
         WHERE response_id = :rid
           AND status = 'in_progress'
           AND (heartbeat_at IS NULL
                OR heartbeat_at < now() - make_interval(secs => :threshold))
     RETURNING attempt_number
        """
    ).bindparams(
        bindparam("rid", type_=None),
        bindparam("threshold", type_=None),
    )
    async with session_scope() as session:
        result = await session.execute(
            stmt,
            {"rid": response_id, "threshold": stale_threshold_seconds},
        )
        row = result.first()
        await session.commit()
        return int(row[0]) if row else None


async def find_stale_response_ids(
    stale_threshold_seconds: float,
    limit: int = 50,
) -> list[str]:
    """Return ids of in_progress responses whose heartbeat is older than the
    threshold. Used by the proactive scanner to find candidates for resume
    without waiting for a client GET.

    Limited to ``limit`` rows per scan to bound DB load. Ordered by
    ``heartbeat_at`` ascending so the oldest staleness is handled first.
    """
    stmt = text(
        f"""
        SELECT response_id FROM {AGENT_DB_SCHEMA}.responses
        WHERE status = 'in_progress'
          AND heartbeat_at IS NOT NULL
          AND heartbeat_at < now() - make_interval(secs => :threshold)
        ORDER BY heartbeat_at ASC
        LIMIT :limit
        """
    ).bindparams(
        bindparam("threshold", type_=None),
        bindparam("limit", type_=None),
    )
    async with session_scope() as session:
        result = await session.execute(
            stmt,
            {"threshold": stale_threshold_seconds, "limit": limit},
        )
        return [row[0] for row in result.all()]


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
                row.heartbeat_at,
                row.attempt_number,
                json.loads(row.original_request) if row.original_request else None,
            )
        return None


async def resolve_conversation_alias(base_conversation_id: str) -> str:
    """Return the current rotated form of a conversation id, or the input unchanged.

    Called on the request hot path, so it must be cheap. Misses (no row) are
    expected on first contact for a new conversation; those return the input.
    """
    async with session_scope() as session:
        result = await session.execute(
            select(ConversationAlias.current_conversation_id).where(
                ConversationAlias.base_conversation_id == base_conversation_id
            )
        )
        current = result.scalar_one_or_none()
        return current if current is not None else base_conversation_id


async def upsert_conversation_alias(
    base_conversation_id: str, current_conversation_id: str
) -> None:
    """Record (or update) the mapping ``base -> current``.

    Postgres upsert keeps the table size at one row per logical conversation
    regardless of how many times that conversation has been rotated.
    """
    stmt = text(
        f"""
        INSERT INTO {AGENT_DB_SCHEMA}.conversation_aliases
            (base_conversation_id, current_conversation_id, updated_at)
        VALUES (:base, :current, now())
        ON CONFLICT (base_conversation_id) DO UPDATE
            SET current_conversation_id = EXCLUDED.current_conversation_id,
                updated_at = EXCLUDED.updated_at
        """
    )
    async with session_scope() as session:
        await session.execute(
            stmt, {"base": base_conversation_id, "current": current_conversation_id}
        )
        await session.commit()
