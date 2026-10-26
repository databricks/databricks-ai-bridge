"""Permission commands shared by the session and memory store groups."""

from __future__ import annotations

import sys
from typing import Callable

import click

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.permissions import permission_table
from databricks_agentbricks.presentation.render import field
from databricks_agentkit._permissions import MAX_PERMISSION_PAGE_SIZE


def _principal(user: str | None, service_principal: str | None) -> tuple[str, str]:
    if (user is None) == (service_principal is None):
        raise click.UsageError("Specify exactly one of --user or --service-principal.")
    name = user if user is not None else service_principal
    if not name or not name.strip():
        raise click.UsageError("The principal must be a non-empty email or application ID.")
    return name, "USER" if user is not None else "SERVICE_PRINCIPAL"


def _principal_options(function):
    function = click.option(
        "--user", default=None, help="User email (not with --service-principal)."
    )(function)
    return click.option(
        "--service-principal",
        default=None,
        help="Service principal application/client ID, not a numeric SCIM ID (not with --user).",
    )(function)


def store_permissions(kind: str, *, resolve_get: Callable | None = None) -> click.Group:
    """Build identical permission commands using the store group's existing identifiers."""
    key = f"{kind}_store_permissions"

    @click.group("permissions")
    def permissions() -> None:
        """Inspect and manage store access.

        Get returns your effective permission. List shows direct grants, excluding implicit
        ownership and inherited access. Grant, List, and Revoke require the creator or a
        workspace admin. STORE uses the same identifiers as the store commands.
        """

    @permissions.command("get")
    @click.argument("store")
    @click.pass_obj
    def get_permission(obj, store) -> None:
        """Show the authenticated caller's effective permission on STORE.

        Session stores accept a store name or resource name. Memory stores accept an ID,
        resource name, or display name; ambiguous display names fail. No access is reported as
        PERMISSION_LEVEL_UNSPECIFIED in JSON.
        """
        get = getattr(obj.client(), f"get_{kind}_store_permission")
        try:
            data = get(store)
        except AgentCliError as exc:
            # Metadata GET may deny a caller with no access, so try the permission endpoint first.
            if (
                resolve_get is None
                or store.strip().startswith(f"{kind}-stores/")
                or exc.error_code
                not in {
                    "NOT_FOUND",
                    "RESOURCE_DOES_NOT_EXIST",
                    "INVALID_PARAMETER_VALUE",
                    "BAD_REQUEST",
                }
            ):
                raise
            data = get(field(resolve_get(obj, store), "name"))
        if obj.output == "json":
            render.emit_json(data)
            return
        permission_table([data], title="Your effective permission")

    @permissions.command("list")
    @click.argument("store")
    @click.option(
        "--page-size",
        type=click.IntRange(1, MAX_PERMISSION_PAGE_SIZE),
        default=25,
        show_default=True,
        help="Grants per page.",
    )
    @click.option(
        "--page-token", default=None, help="Resume from a previous page (not with --all)."
    )
    @click.option("--all", "all_", is_flag=True, help="Fetch every page of direct grants.")
    @click.pass_obj
    def list_permissions(obj, store, page_size, page_token, all_) -> None:
        """List explicit grants on STORE as its creator or a workspace admin.

        Accepts a session store name or memory store ID, or a resource name. Implicit ownership
        and inherited access are excluded. Text output offers subsequent pages on a terminal;
        JSON and piped output return one page unless --all is supplied.
        """
        if all_ and page_token is not None:
            raise click.UsageError("--page-token can't be combined with --all.")
        list_page = getattr(obj.client(), f"list_{kind}_store_permissions")
        interactive = obj.output != "json" and sys.stdin.isatty() and sys.stdout.isatty()
        found = []
        while True:
            data = list_page(store, page_size=page_size, page_token=page_token)
            items = field(data, key) or []
            token = field(data, "next_page_token")
            if all_:
                found.extend(items)
                if token:
                    page_token = token
                    continue
                data = {key: found}
                items = found
            if obj.output == "json":
                render.emit_json(data)
                return
            permission_table(
                items,
                title="Direct store grants",
                footer=None if interactive or all_ else render.more_pages_footer(len(items), token),
            )
            if (
                all_
                or not token
                or not interactive
                or not click.confirm("Show next page?", default=False)
            ):
                return
            page_token = token

    @permissions.command("grant")
    @click.argument("store")
    @_principal_options
    @click.option(
        "--permission",
        required=True,
        type=click.Choice(["READ", "WRITE"], case_sensitive=False),
        help="Access to grant. WRITE includes READ; READ preserves an existing WRITE grant.",
    )
    @click.pass_obj
    def grant_permission(obj, store, user, service_principal, permission) -> None:
        """Grant access on STORE as its creator or a workspace admin.

        Specify exactly one of --user or --service-principal.
        Accepts a session store name or memory store ID, or a resource name.
        """
        name, principal_type = _principal(user, service_principal)
        grant = getattr(obj.client(), f"grant_{kind}_store_permission")
        data = grant(store, name, permission=permission, principal_type=principal_type)
        if obj.output == "json":
            render.emit_json(data)
            return
        render.success(f"Granted {permission} access to '{name}' on '{store}'")

    @permissions.command("revoke")
    @click.argument("store")
    @_principal_options
    @click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
    @click.pass_obj
    def revoke_permission(obj, store, user, service_principal, yes) -> None:
        """Revoke both READ and WRITE on STORE as its creator or a workspace admin.

        Specify exactly one of --user or --service-principal.
        Accepts a session store name or memory store ID, or a resource name. Repeating a revoke
        succeeds. Ownership cannot be revoked; inherited access is not removed.
        """
        name, principal_type = _principal(user, service_principal)
        if not yes:
            click.confirm(
                f"Revoke READ and WRITE for '{name}' on '{store}'?", default=False, abort=True
            )
        revoke = getattr(obj.client(), f"revoke_{kind}_store_permission")
        data = revoke(store, name, principal_type=principal_type)
        if obj.output == "json":
            render.emit_json(data)
            return
        render.success(f"Revoked READ and WRITE grants for '{name}' on '{store}'")

    return permissions
