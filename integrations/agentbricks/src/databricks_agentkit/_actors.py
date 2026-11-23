"""The actor id the LangGraph memory store reserves, shared with the session and memory helpers.

Why the store doesn't keep entries under the plain actor: a memory pipeline distills the
conversations in a session store into a memory store, and it can rewrite, merge, split or delete
the entries of any actor it processes (see the managed memory docs). The LangGraph store's entries
are values the app reads back by key, so a pipeline rewriting them would break ``get`` and
``search``. Pipelines find the actors they process through sessions, so the store keeps each
user's entries under ``<actor>:langgraph``, an actor that never has a session, and pipelines leave
them alone.

``check_actor`` keeps that id reserved. A session under ``alice:langgraph`` would let pipelines
process Alice's store entries, and a memory-tools call with that actor would read and write them,
so the session and memory helpers refuse actors that end with the suffix.
"""

from __future__ import annotations

from typing import TypeVar

STORE_ACTOR_SUFFIX = ":langgraph"

_A = TypeVar("_A")


def check_actor(actor: _A) -> _A:
    """Return ``actor``, or raise ``ValueError`` if it ends with ``:langgraph`` (any case).

    That id belongs to the LangGraph memory store (see the module docstring). Values other than
    strings pass through unchanged.
    """
    if isinstance(actor, str) and actor.casefold().endswith(STORE_ACTOR_SUFFIX):
        raise ValueError(
            f"Actor ids can't end with {STORE_ACTOR_SUFFIX!r}: the LangGraph memory store keeps "
            f"each user's entries under '<actor>{STORE_ACTOR_SUFFIX}'."
        )
    return actor
