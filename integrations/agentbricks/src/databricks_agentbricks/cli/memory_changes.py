"""Read immutable memory versions and their provenance."""

from __future__ import annotations

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


def _page_options(function):
    function = click.option(
        "--page-token", default=None, help="Continue a page with the same store and filters."
    )(function)
    return click.option(
        "--page-size",
        type=click.IntRange(1, 100),
        default=10,
        show_default=True,
        help="Maximum results per page.",
    )(function)


def _continuation(data):
    token = data.get("next_page_token")
    if token:
        click.echo(f"More available. Repeat with the same filters and --page-token {token}")


def _counts(change):
    return (
        f"+{change.get('entries_added', 0)} "
        f"~{change.get('entries_updated', 0)} "
        f"-{change.get('entries_deleted', 0)}"
    )


@click.group()
def changes() -> None:
    """Inspect memory write history and immutable entry versions.

    Requires memory history to be enabled for the workspace and store read access.
    Each command returns one page; JSON preserves the API response and continuation token.
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
@_page_options
@click.pass_obj
def changes_list(obj, store, actor_id, source_type, path_prefix, page_size, page_token) -> None:
    """List one page of change metadata, newest first, without entry contents.

    Actor and path filters must match the same touched entry. Counts describe the whole change.
    """
    data = obj.client().list_memory_changes(
        store,
        actor_id=actor_id,
        source_type=source_type,
        path_prefix=path_prefix,
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
            Text(
                change.get("session_id")
                or change.get("pipeline_run")
                or change.get("request_id")
                or "—"
            ),
            _counts(change),
            timefmt.absolute(change.get("create_time")),
        ]
        for change in data.get("changes", [])
    ]
    render.resource_table(
        "Memory Changes",
        [
            ("Change ID", "left"),
            ("Source", "left"),
            ("Reference", "left"),
            ("Entries", "left"),
            ("Created", "left"),
        ],
        rows,
        no_wrap=[0],
    )
    _continuation(data)


@changes.command("get")
@click.argument("change_id")
@click.option("--store", required=True, help="Store id or resource name.")
@click.option("--memory-entry-id", default=None, help="Exact entry id within the change.")
@click.option("--path", default=None, help="Exact entry path within the change.")
@_page_options
@click.pass_obj
def changes_get(obj, change_id, store, memory_entry_id, path, page_size, page_token) -> None:
    """Get a change header and one page of entry versions, including content.

    To compare versions, get an entry's previous_change_id with --memory-entry-id set to that
    entry's id. A missing predecessor marks the start of recorded history. Restoring an older
    version requires a new entry update; these commands only read history.
    """
    data = obj.client().get_memory_changes(
        store,
        change_id,
        memory_entry_id=memory_entry_id,
        path=path,
        page_size=page_size,
        page_token=page_token,
    )
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
    click.echo(f"{len(entries)} entry versions on this page.")
    _continuation(data)


def _literal_fields(fields):
    # Memory text may contain brackets that Rich would otherwise interpret as markup.
    return {key: Text(value) if isinstance(value, str) else value for key, value in fields.items()}
