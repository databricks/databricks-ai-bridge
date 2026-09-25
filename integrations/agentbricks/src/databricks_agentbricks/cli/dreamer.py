"""``ab dreamer`` — manage cross-session memory pipelines."""

from __future__ import annotations

from typing import Any

import click

from databricks_agentbricks import render
from databricks_agentbricks.render import field
from databricks_agentkit import timefmt


def _truncate(value: Any, length: int = 48) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


def _render_detail(pipeline: dict) -> None:
    policy = field(pipeline, "dreamer_policy") or {}
    render.detail(
        "Dreamer",
        field(pipeline, "display_name") or field(pipeline, "name") or "—",
        {
            "Resource name": field(pipeline, "name"),
            "Session store": field(pipeline, "session_store"),
            "Memory store": field(pipeline, "memory_store"),
            "Model": field(pipeline, "model"),
            "Instructions": field(pipeline, "instructions"),
            "Enabled": field(policy, "enabled"),
            "Trigger": field(policy, "trigger"),
            "ETag": field(pipeline, "etag"),
            "Created": timefmt.absolute(field(pipeline, "create_time")),
            "Updated": timefmt.absolute(field(pipeline, "update_time")),
        },
    )


def _render_run(run: dict) -> None:
    stats = field(run, "stats") or {}
    render.detail(
        "Dreamer Run",
        field(run, "name") or "—",
        {
            "Resource name": field(run, "name"),
            "State": field(run, "state"),
            "Stats": stats or None,
            "Error": field(run, "error"),
            "Created": timefmt.absolute(field(run, "create_time")),
            "Finished": timefmt.absolute(field(run, "end_time")),
        },
    )


@click.group()
def dreamer() -> None:
    """Manage pipelines that distill session history into long-term memory."""


@dreamer.command("create")
@click.option("--memory-store", required=True, help="Memory store name or resource name.")
@click.option("--session-store", required=True, help="Session store name or resource name.")
@click.option("--model", default=None, help="Model service used for Dreamer distillation.")
@click.option("--display-name", default=None, help="Optional human-readable pipeline name.")
@click.option("--instructions", default=None, help="Instructions steering distillation.")
@click.pass_obj
def create(obj, memory_store, session_store, model, display_name, instructions) -> None:
    """Create a Dreamer memory pipeline."""
    data = obj.client().create_memory_pipeline(
        memory_store=memory_store,
        session_store=session_store,
        model=model,
        display_name=display_name,
        instructions=instructions,
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    render.success(
        "Created Dreamer memory pipeline",
        fields={"Resource name": field(data, "name")},
        next_steps=[
            (f"ab dreamer get {field(data, 'name')}", "View the pipeline"),
        ],
    )


@dreamer.command("list")
@click.option("--page-size", type=int, default=25, show_default=True)
@click.option("--page-token", default=None)
@click.pass_obj
def list_(obj, page_size, page_token) -> None:
    """List Dreamer memory pipelines in the workspace."""
    data = obj.client().list_memory_pipelines(page_size, page_token)
    if obj.output == "json":
        render.emit_json(data)
        return
    pipelines = field(data, "memory_pipelines") or []
    render.resource_table(
        "Dreamer Memory Pipelines",
        [
            ("Resource name", "left"),
            ("Session store", "left"),
            ("Memory store", "left"),
            ("Model", "left"),
        ],
        [
            [
                field(item, "name"),
                field(item, "session_store"),
                field(item, "memory_store"),
                _truncate(field(item, "model")),
            ]
            for item in pipelines
        ],
        subtitle=(
            f"Next page: --page-token {field(data, 'next_page_token')}"
            if field(data, "next_page_token")
            else None
        ),
        no_wrap=[0],
    )


@dreamer.command("get")
@click.argument("name")
@click.pass_obj
def get(obj, name) -> None:
    """Get a Dreamer memory pipeline by id or resource name."""
    data = obj.client().get_memory_pipeline(name)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_detail(data)


@dreamer.command("update")
@click.argument("name")
@click.option("--display-name", default=None)
@click.option("--instructions", default=None)
@click.option("--enable", "enabled", flag_value=True, default=None)
@click.option("--disable", "enabled", flag_value=False)
@click.pass_obj
def update(obj, name, display_name, instructions, enabled) -> None:
    """Update a pipeline's display name, instructions, or enabled state."""
    data = obj.client().update_memory_pipeline(
        name,
        display_name=display_name,
        instructions=instructions,
        enabled=enabled,
    )
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_detail(data)


@dreamer.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def delete(obj, name, yes) -> None:
    """Delete a Dreamer memory pipeline and its backing job."""
    render.confirm_destroy(f"Dreamer memory pipeline '{name}'", assume_yes=yes)
    obj.client().delete_memory_pipeline(name)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted Dreamer memory pipeline '{name}'")


@dreamer.command("run")
@click.argument("name")
@click.pass_obj
def run(obj, name) -> None:
    """Manually run a Dreamer memory pipeline."""
    data = obj.client().run_memory_pipeline(name)
    if obj.output == "json":
        render.emit_json(data)
        return
    _render_run(data)
