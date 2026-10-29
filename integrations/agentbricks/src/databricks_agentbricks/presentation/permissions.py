"""Rendering shared by session and memory store permission commands."""

from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.render import field


def permission_table(items: list[dict], *, title: str, footer: str | None = None) -> None:
    rows = []
    for item in items:
        principal = field(item, "principal") or {}
        permission = field(item, "permission") or "PERMISSION_LEVEL_UNSPECIFIED"
        rows.append(
            (
                field(principal, "principal_type"),
                field(principal, "name"),
                "No access" if permission == "PERMISSION_LEVEL_UNSPECIFIED" else permission,
            )
        )
    render.resource_table(
        title,
        [("Principal type", "left"), ("Principal", "left"), ("Permission", "left")],
        rows,
        footer=footer,
        no_wrap=[1],
    )
