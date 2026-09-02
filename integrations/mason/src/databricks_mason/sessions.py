"""`mason sessions` — manage session stores, sessions, and session items."""

from __future__ import annotations

import json
from typing import Any, Optional

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError
from databricks_mason.render import field

_BREADCRUMB = "Agent Session"


def _truncate(value: Any, length: int = 60) -> str:
    text = "" if value is None else (value if isinstance(value, str) else json.dumps(value))
    return text if len(text) <= length else text[: length - 1] + "…"


# --- group ------------------------------------------------------------------


@click.group()
def sessions() -> None:
    """Manage agent session stores, sessions, and items (/api/agents/v1/session-stores)."""


@sessions.group()
def stores() -> None:
    """Workspace-scoped session stores."""


@sessions.group()
def items() -> None:
    """Transcript items within a session."""


# --- session stores ---------------------------------------------------------


def _render_store_detail(store: dict) -> None:
    render.detail(
        f"{_BREADCRUMB} Store",
        field(store, "session_store_name") or "—",
        {
            "Name": field(store, "session_store_name"),
            "Store ID": field(store, "session_store_id"),
            "Creator": field(store, "creator_user_id"),
            "Description": field(store, "description"),
            "Created": timefmt.absolute(field(store, "create_time")),
            "Updated": timefmt.absolute(field(store, "update_time")),
        },
        status="ACTIVE",
    )


@stores.command("create")
@click.option(
    "--name",
    "--display-name",
    "name",
    required=True,
    help="Workspace-unique store name, 3-63 chars (--display-name is accepted as an alias).",
)
@click.option("--description", default=None)
@click.option("--metadata", default=None, help="JSON object of string labels.")
@click.pass_obj
def stores_create(obj, name, description, metadata) -> None:
    """Create a session store."""
    data = obj.client().create_session_store(name, description, _parse_metadata(metadata))
    if obj.output == "json":
        render.emit_json(data)
        return
    render.success(
        f"Created session store '{name}'",
        fields={"Store ID": field(data, "session_store_id")},
        next_steps=[
            f"mason sessions create --store {name} --actor-id <id>",
            f"mason sessions stores get {name}",
        ],
    )


@stores.command("list")
@click.option("--page-size", type=int, default=None)
@click.option("--page-token", default=None)
@click.pass_obj
def stores_list(obj, page_size, page_token) -> None:
    """List session stores in the workspace."""
    data = obj.client().list_session_stores(page_size, page_token)
    if obj.output == "json":
        render.emit_json(data)
        return
    items_ = field(data, "session_stores") or []
    rows = [
        [
            field(s, "session_store_name"),
            field(s, "creator_user_id"),
            timefmt.relative(field(s, "create_time")),
            timefmt.relative(field(s, "update_time")),
            _truncate(field(s, "description"), 40),
        ]
        for s in items_
    ]
    render.resource_table(
        "Session Stores",
        [
            ("Name", "left"),
            ("Creator", "left"),
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
    """Get a session store by name."""
    data = obj.client().get_session_store(name)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_store_detail(data)


@stores.command("update")
@click.argument("name")
@click.option("--description", default=None)
@click.option("--metadata", default=None, help="JSON object of string labels.")
@click.pass_obj
def stores_update(obj, name, description, metadata) -> None:
    """Update a store's description and/or metadata."""
    data = obj.client().update_session_store(name, description, _parse_metadata(metadata))
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_store_detail(data)


@stores.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def stores_delete(obj, name, yes) -> None:
    """Delete a session store."""
    render.confirm_destroy(f"session store '{name}'", assume_yes=yes)
    obj.client().delete_session_store(name)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted session store '{name}'")


# --- sessions ---------------------------------------------------------------


def _session_starter_code(obj, store: str, session_id: str) -> list[tuple[str, str, str]]:
    return [
        (
            "curl",
            "bash",
            f"""
curl -X POST "{obj.client().host}/api/agents/v1/session-stores/{store}/sessions/{session_id}/items:append" \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \\
  -d '{{"items": [{{"data": {{"role": "user", "content": "Hello"}}}}]}}'
""",
        ),
        (
            "mason",
            "bash",
            f"""
mason sessions items append --store {store} --session-id {session_id} \\
  --data '{{"role": "user", "content": "Hello"}}'
mason sessions items list --store {store} --session-id {session_id}
""",
        ),
    ]


def _render_session_detail(obj, session: dict, store: Optional[str]) -> None:
    store = store or field(session, "session_store_name")
    session_id = field(session, "session_id")
    render.detail(
        _BREADCRUMB,
        session_id or "—",
        {
            "Session ID": session_id,
            "Store": field(session, "session_store_name") or store,
            "Actor": field(session, "actor_id"),
            "Parent": field(session, "parent_session_id"),
            "Root": field(session, "root_session_id"),
            "Created": timefmt.absolute(field(session, "create_time")),
            "Last activity": timefmt.absolute(field(session, "last_activity_time")),
        },
        status="ACTIVE",
        snippets=_session_starter_code(obj, store, session_id) if store and session_id else None,
    )


@sessions.command("create")
@click.option("--store", required=True)
@click.option("--actor-id", required=True, help="Application actor id (child must match parent).")
@click.option("--session-id", default=None, help="Optional caller-chosen id.")
@click.option("--parent-session-id", default=None)
@click.option("--metadata", default=None, help="JSON object of string labels.")
@click.pass_obj
def sessions_create(obj, store, actor_id, session_id, parent_session_id, metadata) -> None:
    """Create a session in a store."""
    data = obj.client().create_session(
        store, actor_id, session_id, parent_session_id, _parse_metadata(metadata)
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    render.success(
        f"Created session '{field(data, 'session_id')}'",
        fields={"Actor": actor_id, "Store": store},
        next_steps=[
            f"mason sessions items append --store {store} "
            f"--session-id {field(data, 'session_id')} --data '{{...}}'",
        ],
    )


@sessions.command("list")
@click.option("--store", required=True)
@click.option("--filter", "filter_", default=None, help='e.g. actor_id = "support-123".')
@click.option("--order-by", default=None, help="e.g. 'last_activity_time desc'.")
@click.option("--page-size", type=int, default=None)
@click.option("--page-token", default=None)
@click.pass_obj
def sessions_list(obj, store, filter_, order_by, page_size, page_token) -> None:
    """List sessions in a store."""
    data = obj.client().list_sessions(store, filter_, order_by, page_size, page_token)
    if obj.output == "json":
        render.emit_json(data)
        return
    items_ = field(data, "sessions") or []
    rows = [
        [
            field(s, "session_id"),
            field(s, "actor_id"),
            field(s, "root_session_id"),
            timefmt.relative(field(s, "create_time")),
            timefmt.relative(field(s, "last_activity_time")),
        ]
        for s in items_
    ]
    render.resource_table(
        f"Sessions · {store}",
        [
            ("Session ID", "left"),
            ("Actor", "left"),
            ("Root", "left"),
            ("Created", "left"),
            ("Last Activity", "left"),
        ],
        rows,
        subtitle=_page_note(data),
    )


@sessions.command("get")
@click.argument("session_id")
@click.option("--store", default=None, help="Session store name (required in this preview).")
@click.pass_obj
def sessions_get(obj, session_id, store) -> None:
    """Get a session by id."""
    if not store:
        raise AgentCliError(
            "Provide --store. Resolving a session by id alone is not supported in this preview."
        )
    data = obj.client().get_session(session_id, store)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_session_detail(obj, data, store)


@sessions.command("update")
@click.argument("session_id")
@click.option("--store", required=True)
@click.option(
    "--metadata", required=True, help="JSON object of string labels (only mutable field)."
)
@click.pass_obj
def sessions_update(obj, session_id, store, metadata) -> None:
    """Update a session's metadata."""
    data = obj.client().update_session(store, session_id, _parse_metadata(metadata) or {})
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_session_detail(obj, data, store)


@sessions.command("delete")
@click.argument("session_id")
@click.option("--store", required=True)
@click.option("--force", is_flag=True, help="Cascade-delete descendant sessions.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def sessions_delete(obj, session_id, store, force, yes) -> None:
    """Delete a session."""
    render.confirm_destroy(f"session '{session_id}'", assume_yes=yes)
    obj.client().delete_session(store, session_id, force)
    if obj.output == "json":
        render.emit_json({"deleted": session_id})
        return
    render.success(f"Deleted session '{session_id}'")


@sessions.command("fork")
@click.argument("source_session_id_arg", required=False, metavar="[SOURCE_SESSION_ID]")
@click.option("--store", required=True)
@click.option(
    "--source-session-id",
    "source_session_id_opt",
    default=None,
    help="Source session to fork (or pass it as the positional argument).",
)
@click.option("--actor-id", required=True)
@click.option("--up-to-item-id", default=None, help="Copy through this item id inclusively.")
@click.option("--session-id", default=None, help="Optional id for the fork.")
@click.option("--metadata", default=None)
@click.pass_obj
def sessions_fork(
    obj,
    source_session_id_arg,
    store,
    source_session_id_opt,
    actor_id,
    up_to_item_id,
    session_id,
    metadata,
) -> None:
    """Fork a session into a new independent top-level session."""
    source_session_id = source_session_id_arg or source_session_id_opt
    if not source_session_id:
        raise AgentCliError(
            "Provide the source session id (as the positional argument or --source-session-id)."
        )
    if source_session_id_arg and source_session_id_opt:
        raise AgentCliError(
            "Pass the source session id once — either positionally or via --source-session-id."
        )
    data = obj.client().fork_session(
        store, source_session_id, actor_id, up_to_item_id, session_id, _parse_metadata(metadata)
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_session_detail(obj, field(data, "session") or {}, store)


# --- session items ----------------------------------------------------------


@items.command("list")
@click.option("--store", required=True)
@click.option("--session-id", required=True)
@click.option("--order-by", default=None, help="'create_time asc' or 'create_time desc'.")
@click.option("--page-size", type=int, default=None)
@click.option("--page-token", default=None)
@click.pass_obj
def items_list(obj, store, session_id, order_by, page_size, page_token) -> None:
    """List transcript items in a session."""
    data = obj.client().list_session_items(store, session_id, order_by, page_size, page_token)
    if obj.output == "json":
        render.emit_json(data)
        return
    items_ = field(data, "session_items") or []
    rows = [
        [
            field(it, "item_id"),
            timefmt.relative(field(it, "create_time")),
            _truncate(field(it, "data"), 70),
        ]
        for it in items_
    ]
    render.resource_table(
        f"Session Items · {session_id}",
        [("Item ID", "left"), ("Created", "left"), ("Data", "left")],
        rows,
        subtitle=_page_note(data),
    )


@items.command("append")
@click.option("--store", required=True)
@click.option("--session-id", required=True)
@click.option("--data", "data_", multiple=True, help="One item's JSON data (repeatable).")
@click.option(
    "--file", "file_", type=click.File("r"), default=None, help="JSON array of item data values."
)
@click.pass_obj
def items_append(obj, store, session_id, data_, file_) -> None:
    """Append one or more items to a session (atomic, in order)."""
    payload = _load_items(data_, file_)
    result = obj.client().append_session_items(store, session_id, payload)
    if obj.output == "json":
        render.emit_json(result)
        return
    appended = field(result, "session_items") or []
    render.success(f"Appended {len(appended)} item(s) to session '{session_id}'")


@items.command("pop")
@click.option("--store", required=True)
@click.option("--session-id", required=True)
@click.pass_obj
def items_pop(obj, store, session_id) -> None:
    """Remove and return the most recent item."""
    data = obj.client().pop_session_item(store, session_id)
    if obj.output == "json":
        render.emit_json(data)
        return
    item = field(data, "item")
    if not item:
        render.success("Session was already empty")
        return
    render.success(
        "Popped last item",
        fields={
            "Item ID": field(item, "item_id"),
            "Data": _truncate(field(item, "data"), 70),
        },
    )


@items.command("clear")
@click.option("--store", required=True)
@click.option("--session-id", required=True)
@click.pass_obj
def items_clear(obj, store, session_id) -> None:
    """Remove all items from a session."""
    obj.client().clear_session_items(store, session_id)
    if obj.output == "json":
        render.emit_json({"cleared": session_id})
        return
    render.success(f"Cleared items from session '{session_id}'")


# --- shared helpers ---------------------------------------------------------


def _parse_metadata(value: Optional[str]) -> Optional[dict]:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgentCliError(f"--metadata must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AgentCliError("--metadata must be a JSON object of string labels.")
    return parsed


def _load_items(data_: tuple[str, ...], file_) -> list[Any]:
    payload: list[Any] = []
    if file_ is not None:
        try:
            loaded = json.load(file_)
        except json.JSONDecodeError as exc:
            raise AgentCliError(f"--file must contain a JSON array: {exc}") from exc
        if not isinstance(loaded, list):
            raise AgentCliError("--file must contain a JSON array of item data values.")
        payload.extend(loaded)
    for raw in data_:
        try:
            payload.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise AgentCliError(f"--data must be valid JSON: {exc}") from exc
    if not payload:
        raise AgentCliError("Provide at least one --data item or a --file.")
    return payload


def _page_note(data: dict) -> str | None:
    token = field(data, "next_page_token")
    return f"More results available — pass --page-token {token}" if token else None
