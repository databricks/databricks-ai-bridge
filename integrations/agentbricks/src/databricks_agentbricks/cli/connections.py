"""Bind governed external connections in ``agent.toml``."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import click
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ConnectionInfo, ConnectionType, CredentialType

from databricks_agentbricks import render
from databricks_agentbricks.agent_project import AgentProject, ConnectionSpec
from databricks_agentbricks.errors import AgentCliError


def _workspace_client(profile: str | None) -> Any:
    from databricks_agentkit._api_client import _workspace_client as build_client

    return build_client(profile)


def _record(spec: ConnectionSpec) -> dict[str, str]:
    return {
        "name": spec.name,
        "uc_connection": spec.uc_connection,
        "transport": spec.transport,
        "principal": spec.principal,
    }


def _validate_connection(info: ConnectionInfo, spec: ConnectionSpec) -> None:
    if info.connection_type != ConnectionType.HTTP:
        raise AgentCliError(f"{spec.uc_connection!r} is not a UC HTTP Connection.")
    if spec.principal == "user":
        raise AgentCliError(
            "Request-user UC Connections are not currently supported by Agent Bricks.",
            hint="Bind an existing BEARER_TOKEN Connection with --principal app.",
        )
    credential = info.credential_type
    if credential is not None and credential.value.startswith("OAUTH_"):
        raise AgentCliError(
            "OAuth UC Connections are not currently supported by Agent Bricks.",
            hint="Bind an existing BEARER_TOKEN Connection with --principal app.",
        )
    if credential != CredentialType.BEARER_TOKEN:
        rendered = credential.value if credential is not None else "unknown"
        raise AgentCliError(
            f"UC Connection {spec.uc_connection!r} uses unsupported credential type {rendered!r}.",
            hint="Only existing BEARER_TOKEN Connections are currently supported.",
        )


def _existing_connection(workspace_client: Any, spec: ConnectionSpec) -> ConnectionInfo:
    try:
        info = workspace_client.connections.get(spec.uc_connection)
    except NotFound as exc:
        raise AgentCliError(f"UC Connection {spec.uc_connection!r} does not exist.") from exc
    except Exception as exc:
        raise AgentCliError(f"Could not read UC Connection {spec.uc_connection!r}.") from exc
    _validate_connection(info, spec)
    return info


def _emit(obj: Any, action: str, project: AgentProject, spec: ConnectionSpec) -> None:
    if obj.output == "json":
        render.emit_json(
            {
                "action": action,
                "manifest": str(project.path),
                "connection": _record(spec),
            }
        )
        return
    render.success(
        f"{action.capitalize()} connection {spec.name!r}",
        fields={
            "UC Connection": spec.uc_connection,
            "Transport": spec.transport,
            "Principal": spec.principal,
            "Manifest": str(project.path),
        },
    )


@click.group("auth")
def auth() -> None:
    """Configure authentication and governed external connections."""


@auth.group()
def connections() -> None:
    """Bind existing UC Connections used by agent code."""


def _source_option(command):
    return click.option(
        "--source",
        default=".",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        help="Project directory containing agent.toml. Defaults to the current directory.",
    )(command)


@connections.command("bind")
@click.argument("name")
@click.option("--uc-connection", required=True, help="Existing three-part UC Connection name.")
@click.option("--transport", required=True, type=click.Choice(["mcp", "http"]))
@click.option(
    "--principal",
    required=True,
    type=click.Choice(["user", "app"]),
    help="Invocation identity. Only app is currently supported; user fails closed.",
)
@_source_option
@click.pass_obj
def bind_connection(
    obj: Any,
    name: str,
    uc_connection: str,
    transport: Literal["mcp", "http"],
    principal: Literal["user", "app"],
    source: pathlib.Path,
) -> None:
    """Bind NAME to an existing UC HTTP Connection."""

    project = AgentProject.load(source)
    spec = ConnectionSpec(name, uc_connection, transport, principal)
    changed = project.add_connection(spec)
    _existing_connection(_workspace_client(obj.profile), spec)
    if changed:
        project.write()
    _emit(obj, "bound", project, spec)
