"""Inbound request handling: pull the session id (and any HITL resume) from the request.

The request body is a plain dict — ``input`` is a list of LangChain message dicts (e.g.
``{"role": "user", "content": "..."}``) passed straight to the agent, plus an optional top-level
``session_id`` for multi-turn. The session id is used as the LangGraph ``thread_id``; with it, the
checkpointer supplies prior history, so send only the new turn's message in ``input``.

A request may instead carry ``resume`` — the client's decisions for a session that paused awaiting
human approval. It's LangGraph's native HITL shape (``{"decisions": [{"type": "approve"}, ...]}``),
passed straight to ``Command(resume=...)`` with the same ``session_id`` to continue that thread.
"""

from uuid_utils import uuid7


def get_session_id(request: dict) -> str:
    """Return the request's ``session_id`` (for multi-turn), or a fresh UUID for a new conversation."""
    return str(request.get("session_id") or uuid7())


def get_resume(request: dict) -> dict | None:
    """Return the native HITL resume payload if this request resumes a paused session, else None."""
    return request.get("resume")
