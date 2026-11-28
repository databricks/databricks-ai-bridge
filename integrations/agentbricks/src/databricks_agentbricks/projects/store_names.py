"""Memory and session store naming rules, and the default store names `agentbricks init` declares.

The rules mirror what the managed store service enforces when it creates a store, so a bad name
fails before `agentbricks deploy` reaches the API. A memory store's ID must start with a letter; a
session store's name may also start with a digit but can't start with ``databricks``. Memory stores
created before IDs were required can have a display name that starts with a digit, so binding an
existing store also accepts that legacy shape.

Kept free of heavy imports so CLI option callbacks can use it without slowing startup.
"""

from __future__ import annotations

import re

_STORE_NAME_MIN_LEN = 3
_MEMORY_STORE_ID_MAX_LEN = 56
_SESSION_STORE_NAME_MAX_LEN = 55
_MEMORY_STORE_ID_RE = re.compile(r"[a-z](?:[a-z0-9-]*[a-z0-9])?")
_LEGACY_MEMORY_STORE_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
_SESSION_STORE_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_RESERVED_MEMORY_STORE_IDS = frozenset(
    {"postgres", "template0", "template1", "databricks", "databricks-postgres"}
)
_RESERVED_SESSION_STORE_NAMES = frozenset({"postgres", "template0", "template1"})
_RESERVED_SESSION_STORE_PREFIX = "databricks"


def memory_store_name_problem(name: str, *, allow_legacy: bool = False) -> str | None:
    """Why ``name`` can't name a memory store, or None if it can.

    ``allow_legacy`` also accepts an existing store's legacy display name, which may start with a
    digit. Creating a store needs the stricter ID rule.
    """
    pattern = _LEGACY_MEMORY_STORE_NAME_RE if allow_legacy else _MEMORY_STORE_ID_RE
    first = "a letter or digit" if allow_legacy else "a letter"
    if not (
        _STORE_NAME_MIN_LEN <= len(name) <= _MEMORY_STORE_ID_MAX_LEN and pattern.fullmatch(name)
    ):
        return (
            f"Memory store name {name!r} must be {_STORE_NAME_MIN_LEN}-{_MEMORY_STORE_ID_MAX_LEN} "
            f"characters of lowercase letters, digits, and hyphens, start with {first}, and end "
            "with a letter or digit."
        )
    if name in _RESERVED_MEMORY_STORE_IDS or _UUID_RE.fullmatch(name):
        return f"Memory store name {name!r} is reserved."
    return None


def session_store_name_problem(name: str) -> str | None:
    """Why ``name`` can't name a session store, or None if it can."""
    if not (
        _STORE_NAME_MIN_LEN <= len(name) <= _SESSION_STORE_NAME_MAX_LEN
        and _SESSION_STORE_NAME_RE.fullmatch(name)
    ):
        return (
            f"Session store name {name!r} must be {_STORE_NAME_MIN_LEN}-"
            f"{_SESSION_STORE_NAME_MAX_LEN} characters of lowercase letters, digits, and hyphens, "
            "and start and end with a letter or digit."
        )
    if name.startswith(_RESERVED_SESSION_STORE_PREFIX):
        return (
            f"Session store name {name!r} can't start with {_RESERVED_SESSION_STORE_PREFIX!r}: "
            "that prefix is reserved."
        )
    if name in _RESERVED_SESSION_STORE_NAMES:
        return f"Session store name {name!r} is reserved."
    return None


def default_store_name(project_name: str, suffix: str, token: str | None = None) -> str:
    """A store name derived from the project directory, e.g. ``my-agent`` -> ``my-agent-memory``.

    Sanitized to the store-name charset (lower-case alphanumerics and hyphens); a name that reduces
    to nothing (e.g. only punctuation) falls back to ``agent``. An optional per-scaffold ``token`` is
    inserted before the suffix (``my-agent-<token>-memory``) so fresh scaffolds get distinct stores
    while the name still ends with the store kind.

    The result is a valid name for both store kinds, and the memory and session names share one
    prefix: a slug that doesn't start with a letter, or starts with the reserved ``databricks``, gets
    an ``agent-`` prefix, and a long slug is shortened to fit the tighter length limit.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", project_name.lower()).strip("-") or "agent"
    if not slug[0].isalpha() or slug.startswith(_RESERVED_SESSION_STORE_PREFIX):
        slug = f"agent-{slug}"
    middle = f"{token}-" if token else ""
    room = min(
        _MEMORY_STORE_ID_MAX_LEN - len(f"-{middle}memory"),
        _SESSION_STORE_NAME_MAX_LEN - len(f"-{middle}sessions"),
    )
    slug = slug[:room].rstrip("-")
    return f"{slug}-{middle}{suffix}"
