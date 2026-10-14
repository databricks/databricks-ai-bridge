"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Unlike the session store (short-term transcript for one conversation), long-term memory is exposed
to the model as four tools over the Databricks managed memory store. ``remember`` appends each fact
to one memory per topic (``/{topic}.md``); ``recall`` searches the actor's memories, including any
written under other paths; ``forget`` removes a fact; ``list_memories`` lists what is stored. Facts
persist across conversations.

``forget`` and ``list_memories`` cover the actor's cross-session memories (those without a
``session_id``), which is where ``remember`` writes. Memories saved with a ``session_id`` are never
listed or edited by these tools.

``memory_tools(actor)`` returns the tools when ``AGENT_MEMORY_STORE`` is set, else an empty list, so
the model never sees them when memory is unconfigured. ``create_agent_graph`` composes them into its
tool list. This is a stand-in for a future ``databricks-langchain`` memory helper — when the SDK
provides one, swap the import in ``agent.py`` and delete this file. Keep it in sync with
``databricks_agentkit.openai.memory``.

Memory entries are per-actor. ``actor`` is the identity whose memory these tools read and write — the
caller (the agent server) supplies it, typically the signed-in user, so each user gets their own
long-term memory. It is **closed over**, not a tool argument: the model only ever sees the tools' own
parameters and cannot set or spoof the actor. Pass a fixed value for one shared memory (e.g. a team
knowledge base).
"""

import re
import threading
import unicodedata
from typing import TYPE_CHECKING, Optional

from langchain_core.tools import BaseTool, tool

from databricks_agentkit.client import AgentKitClient
from databricks_agentkit.runtime.workspace import workspace_client

if TYPE_CHECKING:
    from databricks_agentkit.memory_store import Memory, MemorySearchResult, MemoryStore

_RECALL_RESULTS = 5
_LIST_LIMIT = 100
_LIST_PAGE_SIZE = 100
_MAX_TOPIC_LENGTH = 64
_NON_SLUG = re.compile(r"[^a-z0-9_-]+")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _topic_path(topic: str) -> str:
    """The memory path for a model-supplied topic, always valid as an entry path."""
    # Transliterate first so "Café" files under "cafe" rather than "caf".
    ascii_topic = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode()
    slug = _NON_SLUG.sub("-", ascii_topic.lower()).strip("-_")
    slug = slug[:_MAX_TOPIC_LENGTH].rstrip("-_")
    return f"/{slug or 'general'}.md"


def _forget_path(topic: str) -> Optional[str]:
    """The path ``forget`` edits, or None for a path the server would reject."""
    topic = topic.strip()
    # Paths shown by list_memories or recall are used as-is, so facts in memories written under other
    # paths (older adapters, memory pipelines) can be forgotten too.
    if not topic.startswith("/"):
        return _topic_path(topic)
    # The server's entry-path rules: absolute, with no empty, "." or ".." segment.
    if any(segment in ("", ".", "..") for segment in topic[1:].split("/")):
        return None
    return topic


def _fact_count(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip())


def _format_recall(results: "list[MemorySearchResult]") -> str:
    lines: list[str] = []
    for result in results:
        memory = result.memory
        lines.append(f"{memory.path}:")
        text = memory.content or memory.description or ""
        lines.extend(f"  {line}" for line in text.splitlines() if line.strip())
    return "\n".join(lines) or "No relevant memories."


def _format_listing(memories: "list[Memory]") -> str:
    if not memories:
        return "No memories yet."
    lines: list[str] = []
    for memory in memories[:_LIST_LIMIT]:
        count = _fact_count(memory.content or "")
        line = f"{memory.path} ({count} fact{'' if count == 1 else 's'})"
        lines.append(f"{line}: {memory.description}" if memory.description else line)
    if len(memories) > _LIST_LIMIT:
        lines.append(f"…and {len(memories) - _LIST_LIMIT} more.")
    return "\n".join(lines)


def memory_tools(actor: str, store: str | None = None) -> list[BaseTool]:
    """The long-term-memory tools for ``actor`` when a memory store is configured, else none.

    The store resolves ``store`` arg → ``AGENT_MEMORY_STORE`` env → the ``[memory_store]`` binding in
    agent.toml (`agentbricks memory bind`) → none (no tools). ``actor`` partitions the store; it is captured
    in the tools' closures (not exposed to the model). Call this per request with the identity whose
    memory to use (e.g. the signed-in user).
    """
    from databricks_agentkit.runtime.tool_manifest import resolve_memory_store

    store_id = resolve_memory_store(store)
    if not store_id:
        return []
    handle: Optional["MemoryStore"] = None
    lock = threading.Lock()

    def memory_store() -> "MemoryStore":
        # Built on first use and reused: the client resolves workspace auth, and the id-only handle
        # avoids fetching the store before every write.
        nonlocal handle
        with lock:
            if handle is None:
                handle = AgentKitClient(workspace_client()).memory_stores._handle(store_id)
            return handle

    def shared_memories(path: Optional[str] = None) -> "list[Memory]":
        # The exact-path filter spans sessions, so filter by session and recheck the path here.
        return [
            memory
            for memory in memory_store().list(actor_id=actor, path=path, page_size=_LIST_PAGE_SIZE)
            if memory.session_id is None and (path is None or memory.path == path)
        ]

    @tool
    def remember(fact: str, topic: str) -> str:
        """Persist a durable fact about the user in long-term memory.

        Args:
            fact: One self-contained fact, such as "Prefers email over phone".
            topic: A short category to file the fact under, such as "preferences" or "projects".
        """
        # One whitespace-normalized line per fact keeps topic files readable and makes a repeated
        # fact match its earlier line, so appending it again is a no-op.
        fact = _normalized(fact)
        if not fact:
            return "Nothing to remember: the fact is empty."
        path = _topic_path(topic)
        memory_store().put(actor_id=actor, path=path, content=fact, append=True)
        return f"Remembered in {path}."

    @tool
    def recall(query: str) -> str:
        """Search the user's long-term memory for facts relevant to the query.

        Args:
            query: What to look for, in natural language.
        """
        return _format_recall(
            memory_store().search(actor_id=actor, query=query, limit=_RECALL_RESULTS)
        )

    @tool
    def forget(fact: str, topic: str) -> str:
        """Remove a fact from long-term memory.

        Use it when the user asks to forget something or a remembered fact is no longer true. To
        correct a fact, forget the old one and remember the new one.

        Args:
            fact: The fact to remove, as stored.
            topic: The fact's topic, or a path shown by list_memories or recall.
        """
        fact = _normalized(fact)
        if not fact:
            return "Nothing to forget: the fact is empty."
        path = _forget_path(topic)
        if path is None:
            return (
                f"Invalid memory path {topic.strip()!r}: use a topic or a path shown by "
                "list_memories."
            )
        matches = shared_memories(path)
        if not matches:
            return f"No memories in {path}."
        memory = matches[0]
        lines = (memory.content or "").splitlines()
        target = fact.casefold()
        kept = [line for line in lines if _normalized(line).casefold() != target]
        removed = len(lines) - len(kept)
        if not removed:
            shown = "".join(f"\n  {line}" for line in lines if line.strip())
            return f"Not found in {path}.{shown}"
        # A read-modify-write, like remember's append: not atomic, so a concurrent write to this
        # memory can be lost.
        if any(line.strip() for line in kept):
            memory.update(content="\n".join(kept))
        else:
            memory.delete()
        return f"Forgot {removed} fact{'' if removed == 1 else 's'} in {path}."

    @tool
    def list_memories() -> str:
        """List what is stored in long-term memory: each memory's path and how many facts it holds.

        Check it to reuse an existing topic instead of inventing a near-duplicate, and to find a
        fact's path before forgetting it.
        """
        return _format_listing(sorted(shared_memories(), key=lambda memory: memory.path))

    return [remember, recall, forget, list_memories]
