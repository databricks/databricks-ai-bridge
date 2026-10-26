"""The actor rule shared by the LangGraph memory store and the session and memory helpers."""

from __future__ import annotations

from typing import TypeVar

# The LangGraph memory store keeps each user's entries under ``actor + STORE_ACTOR_SUFFIX``. Memory
# pipelines find the actors to process through their sessions, and no session should use that id,
# so automatic pipeline runs leave the store's entries alone.
STORE_ACTOR_SUFFIX = ":langgraph"

_A = TypeVar("_A")


def check_actor(actor: _A) -> _A:
    """Return ``actor``, or raise if it ends with the suffix the LangGraph memory store reserves.

    A session or memory-tools actor with that ending would reach another user's store entries, so
    the session and memory helpers refuse it. Values other than strings pass through unchanged.
    """
    if isinstance(actor, str) and actor.casefold().endswith(STORE_ACTOR_SUFFIX):
        raise ValueError(
            f"Actor ids can't end with {STORE_ACTOR_SUFFIX!r}: the LangGraph memory store keeps "
            f"each user's entries under '<actor>{STORE_ACTOR_SUFFIX}'."
        )
    return actor
