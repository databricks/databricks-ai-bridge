"""Per-user agent memory on Databricks managed memory (``/api/2.0/agents/memory-stores``, Beta).

Each Databricks identity gets its own memory store. A store is the only access boundary (``actor_id`` only organizes
entries), so stores are never shared between users. The worker serves four custom tools inside the sandbox
(``memory_search``, ``memory_list``, ``memory_save``, ``memory_delete``) with the sandbox's own credential, so the
store is created and owned by the sandbox's identity: the signed-in user in the App's user mode, or the CLI profile
for the local dispatcher. The store id and ``actor_id`` are fixed by the launcher; the model can't choose them.

Declare the tools on your agent with :func:`memory_custom_tools`.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MEMORY_API",
    "MEMORY_STORE_PREFIX",
    "MemoryConfig",
    "actor_id_for_user",
    "memory_custom_tools",
    "memory_store_id",
]

MEMORY_API = "/api/2.0/agents/memory-stores"  # must equal worker.MEMORY_API
MEMORY_STORE_PREFIX = "cma-memory-"
# Store ids: 3-56 characters, lowercase letters, digits and hyphens, starting with a letter.
_STORE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,54}[a-z0-9]$")
MAX_CONTENT_CHARS = 16_000
MAX_SEARCH_RESULTS = 20


def memory_store_id(key: str | bytes, user_id: str, *, prefix: str = MEMORY_STORE_PREFIX) -> str:
    """The store id for one Databricks identity: an HMAC of its immutable id under the deployment's key.

    Store ids are visible to every workspace user, so the HMAC keeps user ids out of the list and makes the id
    unpredictable (nobody can create a user's store first). The worker still checks the store's owner before use.
    """
    if not key:
        raise ValueError("memory_store_id needs the deployment's HMAC key")
    if not user_id:
        raise ValueError("memory_store_id needs the user's immutable id")
    raw = key.encode() if isinstance(key, str) else key
    digest = hmac.new(raw, b"memory:" + user_id.encode(), hashlib.sha256).hexdigest()
    store_id = prefix + digest[:24]
    if not _STORE_ID_RE.match(store_id):
        raise ValueError(f"invalid memory store prefix {prefix!r}")
    return store_id


def actor_id_for_user(user_id: str) -> str:
    """``actor_id`` for a Databricks identity: its immutable id, which survives an email change."""
    if not user_id:
        raise ValueError("actor_id_for_user needs the user's immutable id")
    return f"databricks-user:{user_id}"


@dataclass(frozen=True)
class MemoryConfig:
    """The memory store the worker serves, and whose it is. Not secret: passed to the worker as environment.

    ``owner_user_id`` is the Databricks id of the sandbox's identity: the worker refuses a store with another owner.
    """

    store_id: str
    actor_id: str
    owner_user_id: str

    def __post_init__(self) -> None:
        if not _STORE_ID_RE.match(self.store_id):
            raise ValueError(f"invalid memory store id {self.store_id!r}")
        if not self.actor_id or not self.owner_user_id:
            raise ValueError("MemoryConfig needs actor_id and owner_user_id")

    @classmethod
    def for_user(cls, key: str | bytes, user_id: str) -> MemoryConfig:
        return cls(memory_store_id(key, user_id), actor_id_for_user(user_id), user_id)

    def to_env(self) -> dict[str, str]:
        return {
            "CMA_MEMORY_STORE": self.store_id,
            "CMA_MEMORY_ACTOR": self.actor_id,
            "CMA_MEMORY_OWNER": self.owner_user_id,
        }


_PATH = {
    "type": "string",
    "description": "A file-like path for the memory, such as /preferences/response-style.md",
}

MEMORY_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "type": "custom",
        "name": "memory_search",
        "description": (
            "Search your long-term memory about this user (kept across sessions) by meaning and keywords. "
            "Use it at the start of a task to recall preferences, facts and earlier work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum results (1-{MAX_SEARCH_RESULTS}, default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "custom",
        "name": "memory_list",
        "description": "List the memories saved for this user under a path prefix (paths and descriptions only).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Only paths starting with this (default /).",
                },
            },
        },
    },
    {
        "type": "custom",
        "name": "memory_save",
        "description": (
            "Save or replace a memory about this user, kept across sessions. Save durable preferences and facts, "
            "not one-off answers. Saving to an existing path replaces its content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "content": {"type": "string", "description": "The memory itself."},
                "description": {
                    "type": "string",
                    "description": "One line on what it is (used for search).",
                },
            },
            "required": ["path", "content", "description"],
        },
    },
    {
        "type": "custom",
        "name": "memory_delete",
        "description": "Delete the memory saved at a path.",
        "input_schema": {"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
    },
)


def memory_custom_tools() -> list[dict[str, Any]]:
    """Custom tool declarations for the agent, matching the tools the worker serves when memory is on."""
    return [dict(d) for d in MEMORY_TOOL_DECLARATIONS]
