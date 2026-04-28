"""SQLAlchemy models for long-running agent persistence."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dedicated schema for agent tables (responses, messages)
AGENT_DB_SCHEMA = "agent_server"


class Base(DeclarativeBase):
    pass


class Response(Base):
    """Response status tracking for background agent tasks.

    Durability columns (``owner_pod_id``, ``heartbeat_at``, ``attempt_number``,
    ``original_request``) support crash-resume: another pod can atomically
    claim a stale in-progress row and replay the agent loop.
    """

    __tablename__ = "responses"
    __table_args__ = {"schema": AGENT_DB_SCHEMA}

    response_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_pod_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_number: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1", default=1
    )
    original_request: Mapped[str | None] = mapped_column(Text, nullable=True)

    messages = relationship("Message", back_populates="response", cascade="all, delete-orphan")


class Message(Base):
    """Stream events and output items for a response.

    ``attempt_number`` tags events by which run attempt emitted them so that
    resumed runs append to the same event log without overwriting earlier
    (abandoned) attempts, and retrieve can filter to the latest attempt only.
    """

    __tablename__ = "messages"
    __table_args__ = {"schema": AGENT_DB_SCHEMA}

    response_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey(f"{AGENT_DB_SCHEMA}.responses.response_id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer, primary_key=True, nullable=False, default=0
    )
    attempt_number: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1", default=1
    )
    item: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_event: Mapped[str | None] = mapped_column(Text, nullable=True)

    response = relationship("Response", back_populates="messages")
