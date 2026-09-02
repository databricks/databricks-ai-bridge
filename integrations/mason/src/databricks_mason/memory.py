"""`mason memory` — manage workspace-scoped managed memory stores and entries."""

from __future__ import annotations

import pathlib
from typing import Any

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError
from databricks_mason.render import field

_BREADCRUMB = "Agent Memory"

# Friendly aliases for the memory-entry source type, mapped to the API enum values.
_SOURCE_TYPES = {
    "agent": "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
    "unspecified": "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_UNSPECIFIED",
}


def _normalize_source_type(value):
    """Accept a friendly alias ('agent'/'unspecified') or the full enum; None passes through."""
    if value is None:
        return None
    key = value.strip().lower()
    if key in _SOURCE_TYPES:
        return _SOURCE_TYPES[key]
    if value in _SOURCE_TYPES.values():
        return value
    raise AgentCliError(f"Invalid --source-type {value!r}. Choose one of: agent, unspecified.")


def _require_entry_store(store, entry) -> None:
    """A store is needed unless the entry is a full `memory-stores/.../entries/...` name."""
    if not store and not str(entry).strip().startswith("memory-stores/"):
        raise AgentCliError(
            "Provide --store, or pass the full entry resource name "
            "(memory-stores/<store>/entries/<id>)."
        )


def _store_id(store: dict) -> str:
    name = field(store, "name") or ""
    return name.split("/")[-1] if name else "—"


def _truncate(value: Any, length: int = 60) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


# --- group ------------------------------------------------------------------


@click.group()
def memory() -> None:
    """Manage agent memory stores and entries (/api/agents/v1/memory-stores)."""


@memory.group()
def stores() -> None:
    """Workspace-scoped managed memory stores."""


@memory.group()
def entries() -> None:
    """Memory entries within a store, partitioned by actor."""


# --- stores -----------------------------------------------------------------


def _store_starter_code(obj, store: dict) -> list[tuple[str, str, str]]:
    store_id = _store_id(store)
    name = field(store, "name") or f"memory-stores/{store_id}"
    return [
        (
            "curl",
            "bash",
            f"""
curl -X POST "{obj.client().host}/api/agents/v1/{name}/entries" \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \\
  -d '{{"actor_id": "alice", "path": "/preferences/style.md", "content": "Terse, code first."}}'
""",
        ),
        (
            "mason",
            "bash",
            f"""
mason memory entries create --store {store_id} \\
  --actor-id alice --path /preferences/style.md --content "Terse, code first."
mason memory entries search --store {store_id} --actor-id alice --query "style"
""",
        ),
    ]


def _store_created(store: dict):
    # The API returns RFC 3339 `create_time`; older responses used epoch-millis
    # `created_at`. Read the current field, falling back to the legacy one.
    return field(store, "create_time") or field(store, "created_at")


def _store_updated(store: dict):
    return field(store, "update_time") or field(store, "updated_at")


def _render_store_detail(obj, store: dict) -> None:
    render.detail(
        _BREADCRUMB,
        field(store, "display_name") or _store_id(store),
        {
            "Name": field(store, "name"),
            "Store ID": _store_id(store),
            "Workspace": field(store, "workspace_id"),
            "Owner": field(store, "owner_user_id"),
            "Storage": render.field(field(store, "storage_backend") or {}, "backend_id"),
            "Description": field(store, "description"),
            "Created": timefmt.absolute(_store_created(store)),
            "Updated": timefmt.absolute(_store_updated(store)),
        },
        status="ACTIVE",
        snippets=_store_starter_code(obj, store),
    )


@stores.command("create")
@click.option(
    "--display-name",
    "--name",
    "display_name",
    required=True,
    help="Workspace-unique display name (--name is accepted as an alias).",
)
@click.option("--description", default=None, help="Optional human-readable description.")
@click.pass_obj
def stores_create(obj, display_name, description) -> None:
    """Create a memory store."""
    data = obj.client().create_memory_store(display_name, description)
    if obj.output == "json":
        render.emit_json(data)
        return
    store_id = _store_id(data)
    render.success(
        f"Created memory store '{display_name}'",
        fields={"Store ID": store_id, "Name": field(data, "name")},
        next_steps=[
            f"mason memory entries create --store {store_id} --actor-id <id> --path </p>",
            f"mason memory stores get {store_id}",
        ],
    )


@stores.command("list")
@click.option("--page-size", type=int, default=None)
@click.option("--page-token", default=None)
@click.pass_obj
def stores_list(obj, page_size, page_token) -> None:
    """List memory stores in the workspace."""
    data = obj.client().list_memory_stores(page_size, page_token)
    if obj.output == "json":
        render.emit_json(data)
        return
    items = field(data, "managed_memory_stores") or []
    rows = [
        [
            field(s, "display_name"),
            _store_id(s),
            timefmt.relative(_store_created(s)),
            timefmt.relative(_store_updated(s)),
            _truncate(field(s, "description"), 40),
        ]
        for s in items
    ]
    render.resource_table(
        "Managed Memory Stores",
        [
            ("Name", "left"),
            ("Store ID", "left"),
            ("Created", "left"),
            ("Updated", "left"),
            ("Description", "left"),
        ],
        rows,
        subtitle=_page_note(data),
    )


@stores.command("get")
@click.argument("name")
@click.pass_obj
def stores_get(obj, name) -> None:
    """Get a memory store by id or resource name."""
    data = obj.client().get_memory_store(name)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_store_detail(obj, data)


@stores.command("update")
@click.argument("name")
@click.option("--display-name", default=None)
@click.option("--description", default=None)
@click.pass_obj
def stores_update(obj, name, display_name, description) -> None:
    """Update a store's display name and/or description."""
    data = obj.client().update_memory_store(name, display_name, description)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_store_detail(obj, data)


@stores.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def stores_delete(obj, name, yes) -> None:
    """Delete (soft-delete) a memory store."""
    render.confirm_destroy(f"memory store '{name}'", assume_yes=yes)
    obj.client().delete_memory_store(name)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted memory store '{name}'")


# --- entries ----------------------------------------------------------------


def _render_entry_detail(entry: dict) -> None:
    render.detail(
        f"{_BREADCRUMB} Entry",
        field(entry, "path") or "—",
        {
            "Name": field(entry, "name"),
            "Actor": field(entry, "actor_id"),
            "Session": field(entry, "session_id"),
            "Path": field(entry, "path"),
            "Source": field(entry, "source_type"),
            "Description": field(entry, "description"),
            "Content": field(entry, "content"),
            "Created": timefmt.absolute(field(entry, "create_time")),
            "Updated": timefmt.absolute(field(entry, "update_time")),
        },
        status="ACTIVE",
    )


@entries.command("create")
@click.option("--store", required=True, help="Store id or resource name.")
@click.option("--actor-id", required=True, help="Actor (partition) this entry belongs to.")
@click.option("--path", required=True, help="Absolute path, e.g. /preferences/style.md.")
@click.option(
    "--content", default=None, help="Entry content (inline). Use --content-file for large content."
)
@click.option(
    "--content-file",
    "content_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read entry content from a file (avoids shell arg-length limits on large content).",
)
@click.option("--description", default=None, help="Optional human-readable description.")
@click.option("--session-id", default=None, help="Optional session id to associate the entry with.")
@click.option(
    "--source-type",
    default=None,
    help="Origin of the entry: 'agent' or 'unspecified'.",
)
@click.pass_obj
def entries_create(
    obj, store, actor_id, path, content, content_file, description, session_id, source_type
) -> None:
    """Create a memory entry."""
    if content is not None and content_file is not None:
        raise AgentCliError("Pass either --content or --content-file, not both.")
    if content_file is not None:
        content = pathlib.Path(content_file).read_text()
    data = obj.client().create_memory_entry(
        store, actor_id, path, content, description, session_id, _normalize_source_type(source_type)
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    render.success(f"Created memory entry '{path}'", fields={"Name": field(data, "name")})


@entries.command("get")
@click.option(
    "--store", default=None, help="Store id/name (optional if ENTRY is a full resource name)."
)
@click.argument("entry")
@click.pass_obj
def entries_get(obj, store, entry) -> None:
    """Get an entry by id or resource name (includes content)."""
    _require_entry_store(store, entry)
    data = obj.client().get_memory_entry(store, entry)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_entry_detail(data)


@entries.command("list")
@click.option("--store", required=True)
@click.option("--actor-id", required=True, help="Required partition key.")
@click.option("--path-prefix", default=None)
@click.option("--session-id", default=None)
@click.option("--page-size", type=int, default=None)
@click.option("--page-token", default=None)
@click.pass_obj
def entries_list(obj, store, actor_id, path_prefix, session_id, page_size, page_token) -> None:
    """List entries for an actor. The text view omits content; `-o json` includes it."""
    data = obj.client().list_memory_entries(
        store, actor_id, path_prefix, session_id, page_size, page_token
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    items = field(data, "managed_memory_entries") or []
    rows = [
        [
            field(e, "path"),
            field(e, "actor_id"),
            field(e, "session_id"),
            _truncate(field(e, "description"), 40),
            timefmt.relative(field(e, "update_time")),
        ]
        for e in items
    ]
    render.resource_table(
        f"Memory Entries · actor {actor_id}",
        [
            ("Path", "left"),
            ("Actor", "left"),
            ("Session", "left"),
            ("Description", "left"),
            ("Updated", "left"),
        ],
        rows,
        subtitle=_page_note(data),
    )


@entries.command("search")
@click.option("--store", required=True)
@click.option("--actor-id", required=True)
@click.option("--query", required=True)
@click.option("--limit", type=int, default=None)
@click.pass_obj
def entries_search(obj, store, actor_id, query, limit) -> None:
    """Full-text search an actor's entries, ranked (includes content)."""
    data = obj.client().search_memory_entries(store, actor_id, query, limit)
    if obj.output == "json":
        render.emit_json(data)
        return
    items = field(data, "managed_memory_entries") or []
    rows = [
        [
            field(e, "path"),
            field(e, "actor_id"),
            _truncate(field(e, "content"), 50),
            timefmt.relative(field(e, "update_time")),
        ]
        for e in items
    ]
    render.resource_table(
        f"Memory Search · '{query}'",
        [("Path", "left"), ("Actor", "left"), ("Content", "left"), ("Updated", "left")],
        rows,
    )


@entries.command("update")
@click.option(
    "--store", default=None, help="Store id/name (optional if ENTRY is a full resource name)."
)
@click.argument("entry")
@click.option("--content", default=None, help="New entry content.")
@click.option("--description", default=None, help="New description.")
@click.pass_obj
def entries_update(obj, store, entry, content, description) -> None:
    """Update an entry's content and/or description."""
    _require_entry_store(store, entry)
    data = obj.client().update_memory_entry(store, entry, content, description)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_entry_detail(data)


@entries.command("delete")
@click.option(
    "--store", default=None, help="Store id/name (optional if ENTRY is a full resource name)."
)
@click.argument("entry")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def entries_delete(obj, store, entry, yes) -> None:
    """Delete a memory entry."""
    _require_entry_store(store, entry)
    render.confirm_destroy(f"memory entry '{entry}'", assume_yes=yes)
    obj.client().delete_memory_entry(store, entry)
    if obj.output == "json":
        render.emit_json({"deleted": entry})
        return
    render.success(f"Deleted memory entry '{entry}'")


def _page_note(data: dict) -> str | None:
    token = field(data, "next_page_token")
    return f"More results available — pass --page-token {token}" if token else None
