"""Read immutable memory versions and their provenance."""

from __future__ import annotations

import sys

import click
from rich.text import Text

from databricks_agentbricks.presentation import render
from databricks_agentkit import timefmt

_SOURCE_PREFIX = "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_"


def _source_type(ctx, param, value):
    if value is None:
        return None
    source = value.strip().upper().removeprefix(_SOURCE_PREFIX)
    if source not in {"AGENT", "DREAMER", "UNSPECIFIED"}:
        raise click.BadParameter("Choose agent, dreamer, unspecified, or a full source enum.")
    return _SOURCE_PREFIX + source


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _reference(change) -> str:
    # Session ids are caller-chosen, run and request ids are generated; naming the kind keeps a
    # column of mixed ids readable.
    if change.get("session_id"):
        return f"session:{change['session_id']}"
    if change.get("pipeline_run"):
        return f"run:{change['pipeline_run'].rsplit('/', 1)[-1]}"
    if change.get("request_id"):
        return f"request:{change['request_id']}"
    return "—"


def _actors(client, store, change) -> str:
    # List rows carry no actor; it is only on the entry versions of the change itself.
    change_id = change.get("name", "").rsplit("/", 1)[-1]
    entries = client.get_memory_change(store, change_id).get("entries", [])
    return ", ".join(sorted({e["actor_id"] for e in entries if e.get("actor_id")})) or "—"


def _counts(change):
    return (
        f"+{change.get('entries_added', 0)} "
        f"~{change.get('entries_updated', 0)} "
        f"-{change.get('entries_deleted', 0)}"
    )


@click.group()
def changes() -> None:
    """Inspect memory write history.

    List returns change metadata a page at a time; get returns one whole change, including its
    immutable entry versions. JSON preserves the API response, including list's continuation
    token.
    """


@changes.command("list")
@click.option("--store", required=True, help="Store id or resource name.")
@click.option("--actor-id", default=None, help="Only changes touching this actor's entries.")
@click.option(
    "--source-type",
    callback=_source_type,
    default=None,
    help="Origin: agent, dreamer, unspecified, or a full source enum. Omit for all sources.",
)
@click.option(
    "--path",
    "--path-prefix",
    "path_prefix",
    default=None,
    help="Literal entry-path prefix (not an exact path or filesystem path).",
)
@click.option(
    "--memory-entry-id",
    default=None,
    help="Only changes touching this entry id: one entry's version history.",
)
@click.option(
    "--page-size",
    type=click.IntRange(1, 100),
    default=10,
    show_default=True,
    help="Maximum changes per page.",
)
@click.option(
    "--page-token",
    default=None,
    help="Continue from a next_page_token returned by -o json, with the same store and filters.",
)
@click.pass_obj
def changes_list(
    obj, store, actor_id, source_type, path_prefix, memory_entry_id, page_size, page_token
) -> None:
    """List change metadata, newest first, without entry contents.

    On a terminal, text output offers the next page; otherwise it prints one page. The actor
    column costs one extra request per listed change. Actor, path, and entry filters must match the same touched entry. Counts describe the whole
    change. For one entry's history, including after it was deleted, pass --memory-entry-id with
    the final segment of its memory_entry name.
    """
    client = obj.client()
    interactive = obj.output != "json" and _interactive()
    while True:
        data = client.list_memory_changes(
            store,
            actor_id=actor_id,
            source_type=source_type,
            path_prefix=path_prefix,
            memory_entry_id=memory_entry_id,
            page_size=page_size,
            page_token=page_token,
        )
        if obj.output == "json":
            render.emit_json(data)
            return
        rows = [
            [
                Text(change.get("name", "").rsplit("/", 1)[-1]),
                Text(change.get("source_type", "").removeprefix(_SOURCE_PREFIX)),
                Text(_actors(client, store, change)),
                Text(_reference(change)),
                _counts(change),
                timefmt.absolute(change.get("create_time")),
            ]
            for change in data.get("memory_changes", [])
        ]
        render.resource_table(
            "Memory Changes",
            [
                ("Change ID", "left"),
                ("Source", "left"),
                ("Actor", "left"),
                ("Reference", "left"),
                ("Entries", "left"),
                ("Created", "left"),
            ],
            rows,
            no_wrap=[0],
        )
        page_token = data.get("next_page_token")
        if not page_token:
            return
        if not interactive:
            click.echo("More changes available. Run in a terminal to page through them.")
            return
        if not click.confirm("Show next page?", default=False):
            return


@changes.command("get")
@click.argument("change_id")
@click.option("--store", required=True, help="Store id or resource name.")
@click.pass_obj
def changes_get(obj, change_id, store) -> None:
    """Get one change: its header and the entry versions it wrote, including content.

    To compare versions, get the change named by an entry's previous_change_id. A missing
    predecessor marks the start of recorded history. Restoring an older version requires a new
    entry update; these commands only read history.
    """
    data = obj.client().get_memory_change(store, change_id)
    if obj.output == "json":
        render.emit_json(data)
        return
    fields = {
        "Name": data.get("name"),
        "Source": data.get("source_type"),
        "Session": data.get("session_id"),
        "Pipeline run": data.get("pipeline_run"),
        "Request": data.get("request_id"),
        "Principal": data.get("principal"),
        "Session store": data.get("session_store"),
        "Sessions read": data.get("sessions_read"),
        "Entries": _counts(data),
        "Created": timefmt.absolute(data.get("create_time")),
        "Updated": timefmt.absolute(data.get("update_time")),
    }
    render.detail("Memory Change", change_id, _literal_fields(fields))
    entries = data.get("entries", [])
    for entry in entries:
        fields = {
            "Memory entry": entry.get("memory_entry"),
            "Actor": entry.get("actor_id"),
            "Path": entry.get("path"),
            "Type": entry.get("change_type"),
            "Session": entry.get("session_id"),
            "Previous change ID": entry.get("previous_change_id"),
            "Description": entry.get("description"),
            "Content": entry.get("content"),
            "Summary": entry.get("summary"),
            "Created": timefmt.absolute(entry.get("create_time")),
        }
        render.detail("Entry Version", entry.get("path") or "—", _literal_fields(fields))
    click.echo(f"{len(entries)} entry versions.")


def _literal_fields(fields):
    # Memory text may contain brackets that Rich would otherwise interpret as markup.
    return {key: Text(value) if isinstance(value, str) else value for key, value in fields.items()}
