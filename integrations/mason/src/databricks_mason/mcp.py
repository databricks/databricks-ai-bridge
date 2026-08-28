"""Discover Unity Catalog MCP Services that can be added to an agent."""

from __future__ import annotations

from typing import Any

import click

from databricks_mason import render
from databricks_mason.errors import AgentCliError

_RESOURCE_PREFIX = "mcp-services/"


def _validate_schema(schema: str) -> str:
    parts = schema.strip().split(".")
    if (
        len(parts) != 2
        or any(not part for part in parts)
        or any(character.isspace() for character in schema)
    ):
        raise AgentCliError(
            f"Invalid schema {schema!r}.",
            hint="Use a two-part Unity Catalog schema name: catalog.schema.",
        )
    return schema.strip()


def _service_record(service: Any) -> dict[str, str] | None:
    if not isinstance(service, dict):
        return None
    raw_name = service.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        return None
    name = raw_name.removeprefix(_RESOURCE_PREFIX)
    record = {"name": name}
    for field in ("id", "comment"):
        value = service.get(field)
        if isinstance(value, str) and value:
            record[field] = value
    return record


def _list_services(client: Any, schema: str) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    page_token = None
    while True:
        response = client.list_mcp_services(schema, page_token=page_token)
        if not isinstance(response, dict):
            raise AgentCliError("The MCP Services API returned an invalid response.")
        services = response.get("mcp_services", [])
        if not isinstance(services, list):
            raise AgentCliError("The MCP Services API returned an invalid response.")
        for service in services:
            record = _service_record(service)
            if record is not None and record["name"] not in by_name:
                by_name[record["name"]] = record
        page_token = response.get("next_page_token")
        if not isinstance(page_token, str) or not page_token:
            break
    return [by_name[name] for name in sorted(by_name)]


@click.group()
def mcp() -> None:
    """Discover managed MCP Services available through Unity Catalog."""


@mcp.command("list")
@click.option(
    "--schema",
    default="system.ai",
    show_default=True,
    help="Two-part Unity Catalog schema containing MCP Services.",
)
@click.pass_obj
def list_mcp(obj: Any, schema: str) -> None:
    """List MCP Services that can be added with ``mason tools add mcp``."""
    schema = _validate_schema(schema)
    services = _list_services(obj.client(), schema)
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "mcp_services": services})
        return
    render.resource_table(
        "MCP Services",
        [("Service", "left"), ("Add command", "left")],
        [(service["name"], f"mason tools add mcp {service['name']}") for service in services],
        subtitle=f"Available in {schema}",
    )
