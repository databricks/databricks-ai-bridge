"""`mason memory` — manage workspace-scoped managed memory stores and entries."""

from __future__ import annotations

from typing import Any

import click

from databricks_mason import render, timefmt
from databricks_mason.render import field

_BREADCRUMB = "Agent Memory"


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
            "Created": timefmt.absolute(field(store, "created_at")),
            "Updated": timefmt.absolute(field(store, "updated_at")),
        },
        status="ACTIVE",
        snippets=_store_starter_code(obj, store),
    )


@stores.command("create")
@click.option("--display-name", required=True, help="Workspace-unique display name.")
@click.option("--description", default=None)
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
            timefmt.relative(field(s, "created_at")),
            timefmt.relative(field(s, "updated_at")),
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
@click.pass_obj
def stores_delete(obj, name) -> None:
    """Delete (soft-delete) a memory store."""
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
@click.option("--actor-id", required=True)
@click.option("--path", required=True, help="Absolute path, e.g. /preferences/style.md.")
@click.option("--content", default=None)
@click.option("--description", default=None)
@click.option("--session-id", default=None)
@click.option(
    "--source-type",
    default=None,
    type=click.Choice(
        ["MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT", "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_UNSPECIFIED"]
    ),
)
@click.pass_obj
def entries_create(
    obj, store, actor_id, path, content, description, session_id, source_type
) -> None:
    """Create a memory entry."""
    data = obj.client().create_memory_entry(
        store, actor_id, path, content, description, session_id, source_type
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    render.success(f"Created memory entry '{path}'", fields={"Name": field(data, "name")})


@entries.command("get")
@click.option("--store", required=True)
@click.argument("entry")
@click.pass_obj
def entries_get(obj, store, entry) -> None:
    """Get an entry by id or resource name (includes content)."""
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
    """List entries for an actor (content omitted)."""
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
@click.option("--store", required=True)
@click.argument("entry")
@click.option("--content", default=None)
@click.option("--description", default=None)
@click.pass_obj
def entries_update(obj, store, entry, content, description) -> None:
    """Update an entry's content and/or description."""
    data = obj.client().update_memory_entry(store, entry, content, description)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_entry_detail(data)


@entries.command("delete")
@click.option("--store", required=True)
@click.argument("entry")
@click.pass_obj
def entries_delete(obj, store, entry) -> None:
    """Delete a memory entry."""
    obj.client().delete_memory_entry(store, entry)
    if obj.output == "json":
        render.emit_json({"deleted": entry})
        return
    render.success(f"Deleted memory entry '{entry}'")


def _page_note(data: dict) -> str | None:
    token = field(data, "next_page_token")
    return f"More results available — pass --page-token {token}" if token else None
