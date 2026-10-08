"""Databricks SDK helpers for the matrix tests and fixtures: the workspace, which is not under test.

Everything here goes through the SDK except ``app_logs``, since the SDK has no Apps log API.
"""

from __future__ import annotations

import datetime as dt
import io
import pathlib
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any, cast

from common import TOOL_RESOURCE_PREFIX, MatrixError, log, now
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import SecurableType
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, State, StatementState

GrantTuple = tuple[str, str, str, str]


class Workspace:
    def __init__(
        self,
        profile: str | None,
        *,
        app_auth_profile: str | None = None,
        warehouse_id: str | None = None,
        preprovisioned_app_catalog_access: bool = False,
    ):
        self.profile = profile
        self.app_auth_profile = app_auth_profile or profile
        self.warehouse_id = warehouse_id
        self.preprovisioned_app_catalog_access = preprovisioned_app_catalog_access
        self.client = WorkspaceClient(profile=profile)
        self._app_auth_client: WorkspaceClient | None = None
        self._app_auth_checked = False
        self._warehouse_started = False

    # App auth: Databricks Apps /api routes need OAuth, so a PAT is rejected.

    def _app_auth_label(self) -> str:
        return (
            f"App auth profile {self.app_auth_profile!r}"
            if self.app_auth_profile
            else "ambient environment credentials"
        )

    def _app_client(self) -> WorkspaceClient:
        if self._app_auth_client is None:
            self._app_auth_client = WorkspaceClient(profile=self.app_auth_profile)
        return self._app_auth_client

    def check_app_auth(self) -> None:
        if self._app_auth_checked:
            return
        if self._app_client().config.auth_type == "pat":
            raise MatrixError(
                f"{self._app_auth_label()} uses a PAT. "
                "Databricks Apps /api routes require OAuth; run `databricks auth login` "
                "for a profile on the same workspace."
            )
        self._app_auth_checked = True

    @property
    def app_headers(self) -> dict[str, str]:
        # Resolved per call so OAuth tokens refresh during a long run.
        authorization = self._app_client().config.authenticate().get("Authorization")
        if not authorization:
            raise MatrixError(f"Could not resolve credentials from {self._app_auth_label()}.")
        return {"Authorization": authorization}

    # Existing resources the suite depends on

    def require_catalog(self, name: str) -> None:
        try:
            self.client.catalogs.get(name)
        except DatabricksError as exc:
            raise MatrixError(f"Catalog {name!r} is not accessible: {exc}") from exc

    def require_genie_space(self, space_id: str) -> None:
        try:
            self.client.genie.get_space(space_id)
        except DatabricksError as exc:
            raise MatrixError(f"Genie space {space_id!r} is not accessible: {exc}") from exc

    # SQL, warehouse, files

    def start_warehouse(self, override: str | None = None) -> str:
        if override:
            self.warehouse_id = override
        else:
            warehouses = list(self.client.warehouses.list())
            if not warehouses:
                raise MatrixError("The workspace has no SQL warehouse available for UC setup.")
            running = next(
                (item for item in warehouses if item.state == State.RUNNING), warehouses[0]
            )
            self.warehouse_id = str(running.id)
        log(f"# start warehouse {self.warehouse_id}")
        self.client.warehouses.start_and_wait(self.warehouse_id, timeout=dt.timedelta(minutes=20))
        return self.warehouse_id

    def _ensure_warehouse(self) -> str:
        # Each xdist worker has its own Workspace, so the warehouse is started on first use.
        if not self._warehouse_started:
            self.start_warehouse(self.warehouse_id)
            self._warehouse_started = True
        assert self.warehouse_id is not None
        return self.warehouse_id

    def sql(self, statement: str, *, timeout: float = 600) -> None:
        warehouse_id = self._ensure_warehouse()
        log(f"$ sql: {statement}")
        response = self.client.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=warehouse_id,
            wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
        )
        statement_id = response.statement_id
        while response.status and response.status.state in {
            StatementState.PENDING,
            StatementState.RUNNING,
        }:
            if not statement_id:
                raise MatrixError(f"SQL response has no statement_id: {response}")
            if timeout <= 0:
                raise MatrixError(f"SQL statement timed out: {statement_id}")
            time.sleep(10)
            timeout -= 10
            response = self.client.statement_execution.get_statement(statement_id)
        if not response.status or response.status.state != StatementState.SUCCEEDED:
            raise MatrixError(f"SQL failed: {response.as_dict()}")

    def drop(self, kind: str, full_name: str) -> None:
        catalog, schema, name = full_name.split(".")
        self.sql(f"DROP {kind} IF EXISTS `{catalog}`.`{schema}`.`{name}`")

    def upload(self, path: str, data: bytes) -> None:
        self.client.files.upload(path, io.BytesIO(data), overwrite=True)
        log(f"# uploaded {path}")

    def delete_file(self, path: str) -> None:
        try:
            self.client.files.delete(path)
        except NotFound:
            pass

    # Apps

    def app(self, name: str) -> dict[str, Any]:
        return self.client.apps.get(name).as_dict()

    def assert_app_absent(self, name: str) -> None:
        try:
            self.client.apps.get(name)
        except NotFound:
            return
        except DatabricksError as exc:
            raise MatrixError(f"Could not verify App {name} is absent: {exc}") from exc
        raise MatrixError(f"App {name} already exists; refusing to deploy over it.")

    def wait_for_app(self, name: str) -> dict[str, Any]:
        """The App once it is ACTIVE with a URL."""
        started = time.monotonic()
        next_tick = 0.0
        while time.monotonic() - started < 1200:
            app = self.app(name)
            compute = app.get("compute_status", {})
            state = compute.get("state") if isinstance(compute, dict) else None
            if state == "ACTIVE" and app.get("url"):
                return app
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                log(f"tick {now():%H:%M} | app-{name} | {state or 'UNKNOWN'}")
                next_tick += 60
            time.sleep(15)
        raise MatrixError(f"App {name} did not become ACTIVE.")

    def app_logs(self, name: str, log_path: pathlib.Path) -> pathlib.Path | None:
        """Write the App's recent logs to ``log_path``; the SDK has no Apps log API."""
        argv = ["databricks", "apps", "logs", name, "--tail-lines", "200"]
        if self.profile:
            argv += ["--profile", self.profile]
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=120, check=False)
            content = result.stdout if result.returncode == 0 else result.stderr or result.stdout
        except Exception as exc:
            content = f"Could not retrieve App logs: {exc}\n"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            log(f"App runtime log capture warning for {name}: {exc}")
            return None
        log(f"App runtime logs captured: {log_path}")
        return log_path

    # Grants

    def granted(self, app_name: str, grant: GrantTuple) -> bool:
        """Whether Agent Bricks granted the App ``grant``, as (kind, name, type, permission)."""
        return grant in app_resource_tuples(tool_resources(self.app(app_name)))

    def granted_tuples(self, app_name: str) -> set[GrantTuple]:
        return app_resource_tuples(tool_resources(self.app(app_name)))

    def experiment_id(self, name: str) -> str:
        experiment = self.client.experiments.get_by_name(name).experiment
        if experiment is None or not experiment.experiment_id:
            raise MatrixError(f"Experiment {name!r} does not exist.")
        return str(experiment.experiment_id)

    def non_tool_resources(self, app_name: str) -> list[dict[str, Any]]:
        return non_tool_resources(self.app(app_name))

    def transitive_state(self, app_name: str, function: str) -> dict[str, Any]:
        """The App principal's privileges on a nested function before any manual grant."""
        app = self.app(app_name)
        principal = self._principal(app)
        direct = direct_privileges(self.client, SecurableType.FUNCTION, function, principal)
        if "EXECUTE" in direct:
            raise MatrixError(
                f"Agent Bricks granted the transitive function directly: {function} -> {direct}"
            )
        effective = effective_privileges(self.client, SecurableType.FUNCTION, function, principal)
        if "EXECUTE" in effective:
            raise MatrixError(
                "The transitive function was already effective before the manual grant: "
                f"{function} -> {effective}"
            )
        if any(
            resource.get("uc_securable", {}).get("securable_full_name") == function
            for resource in tool_resources(app)
        ):
            raise MatrixError(
                "The transitive function appeared in Agent Bricks-owned Apps resources."
            )
        return {
            "service_principal_client_id": principal,
            "direct_privileges": direct,
            "effective_privileges": effective,
        }

    def grant_transitive(self, app_name: str, function: str) -> dict[str, Any]:
        """Grant the App principal EXECUTE on a nested function and confirm it is direct and effective."""
        principal = self._principal(self.app(app_name))
        catalog, schema, function_name = function.split(".")
        quoted_principal = f"`{principal.replace('`', '``')}`"
        statements = []
        if not self.preprovisioned_app_catalog_access:
            statements.append(f"GRANT USE CATALOG ON CATALOG `{catalog}` TO {quoted_principal}")
        statements.extend(
            (
                f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema}` TO {quoted_principal}",
                f"GRANT EXECUTE ON FUNCTION `{catalog}`.`{schema}`.`{function_name}` "
                f"TO {quoted_principal}",
            )
        )
        for statement in statements:
            self.sql(statement)
        direct = direct_privileges(self.client, SecurableType.FUNCTION, function, principal)
        effective = effective_privileges(self.client, SecurableType.FUNCTION, function, principal)
        if "EXECUTE" not in effective:
            raise MatrixError(
                "Manual EXECUTE grant did not become effective for the transitive function: "
                f"{function} -> {effective}"
            )
        if "EXECUTE" not in direct:
            raise MatrixError(
                "Manual EXECUTE grant was not persisted directly for the transitive function: "
                f"{function} -> {direct}"
            )
        return {
            "granted_at": now().isoformat(),
            "direct_privileges": direct,
            "effective_privileges": effective,
        }

    @staticmethod
    def _principal(app: dict[str, Any]) -> str:
        principal = app.get("service_principal_client_id")
        if not principal:
            raise MatrixError(f"App response has no service_principal_client_id: {app}")
        return str(principal)

    # Cleanup

    def delete_app(self, name: str) -> None:
        self.client.apps.delete(name)
        self._wait_for_app_deleted(name)

    def _wait_for_app_deleted(self, name: str, timeout: float = 1200) -> None:
        started = time.monotonic()
        next_tick = 0.0
        while True:
            try:
                self.client.apps.get(name)
            except NotFound:
                log(f"tick {now():%H:%M} | delete-{name} | absent")
                return
            except DatabricksError as exc:
                raise MatrixError(f"Could not verify deletion of App {name!r}: {exc}") from exc
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                raise MatrixError(
                    f"App {name!r} still existed {timeout:.0f}s after delete returned."
                )
            if elapsed >= next_tick:
                log(f"tick {now():%H:%M} | delete-{name} | deleting")
                next_tick += 60
            time.sleep(15)

    def runtime_store(self, app_name: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.client.api_client.do("GET", f"/api/2.0/agents/runtime-stores/{app_name}"),
        )

    def delete_runtime_store(self, app_name: str) -> None:
        """Delete the deploy-created Runtime Store; a missing store counts as deleted."""
        try:
            self.client.api_client.do("DELETE", f"/api/2.0/agents/runtime-stores/{app_name}")
        except NotFound:
            pass

    def app_role_target(self, app_name: str) -> str | None:
        """The Lakebase role of the App's principal, only if ownership is verified end to end."""
        try:
            principal = self.app(app_name).get("service_principal_client_id")
            store = self.runtime_store(app_name)
            owner = store.get("owner", {}).get("app", {})
            branch = store.get("storage_backend", {}).get("lakebase", {}).get("branch")
            if (
                not principal
                or store.get("name") != f"runtime-stores/{app_name}"
                or owner.get("name") != app_name
                or owner.get("service_principal_id") != principal
                or not isinstance(branch, str)
                or not re.fullmatch(r"projects/[^/]+/branches/[^/]+", branch)
            ):
                log(f"cleanup warning | Lakebase role for {app_name} | ownership not verified")
                return None
            for role in self.client.postgres.list_roles(parent=branch):
                data = role.as_dict()
                status = data.get("status", {})
                name = data.get("name")
                if (
                    status.get("postgres_role") == principal
                    and status.get("identity_type") == "SERVICE_PRINCIPAL"
                    and isinstance(name, str)
                    and name.startswith(f"{branch}/roles/")
                ):
                    return name
        except Exception as exc:
            log(f"cleanup warning | Lakebase role lookup for {app_name} | {exc}")
        return None

    def delete_role(self, role_name: str) -> None:
        operation = self.client.postgres.delete_role(name=role_name)
        wait = getattr(operation, "wait", None)
        if callable(wait):
            wait()


def cleanup_app(
    workspace: Workspace,
    name: str,
    *,
    delete_store: Callable[[str, str], subprocess.CompletedProcess[str]] | None = None,
    memory_store: str | None = None,
    session_store: str | None = None,
    has_app: bool = True,
    runtime_store: bool = False,
) -> list[str]:
    """Delete one project's stores and, if it has an App, its runtime store, App and Lakebase role.

    ``delete_store(kind, store)`` is the agentbricks CLI's store delete, which has no SDK surface.
    Returns a description of each failure; an empty list means everything was deleted.
    """
    failures: list[str] = []
    # Looked up before anything is deleted, since it needs the App and its runtime store.
    role_target = workspace.app_role_target(name) if has_app and runtime_store else None
    for kind, store in (("memory", memory_store), ("sessions", session_store)):
        if not (store and delete_store):
            continue
        try:
            result = delete_store(kind, store)
            if result.returncode != 0:
                failures.append(f"{kind} store {store}: {(result.stderr or result.stdout).strip()}")
        except Exception as exc:
            failures.append(f"{kind} store {store}: {exc}")
    if not has_app:
        return failures

    runtime_store_deleted = False
    if runtime_store:
        # The Runtime Store owns a dedicated Lakebase database; it must be deleted before the
        # role, or the role delete fails on database ownership.
        try:
            workspace.delete_runtime_store(name)
            runtime_store_deleted = True
        except Exception as exc:
            failures.append(f"runtime store {name}: {exc}")
    try:
        workspace.delete_app(name)
    except Exception as exc:
        failures.append(f"app {name}: {exc}")
        return failures
    if role_target and not failures and runtime_store_deleted:
        try:
            workspace.delete_role(role_target)
        except Exception as exc:
            failures.append(f"Lakebase role {role_target}: {exc}")
    return failures


# Grant helpers


def _split_resources(app: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resources = app.get("resources") or []
    if not isinstance(resources, list):
        raise MatrixError(f"App resources are not a list: {resources}")

    def name(resource: dict[str, Any]) -> str:
        return str(resource.get("name", ""))

    owned = [r for r in resources if name(r).startswith(TOOL_RESOURCE_PREFIX)]
    other = [r for r in resources if not name(r).startswith(TOOL_RESOURCE_PREFIX)]
    return sorted(owned, key=name), sorted(other, key=name)


def tool_resources(app: dict[str, Any]) -> list[dict[str, Any]]:
    """The App resources Agent Bricks owns, i.e. those it names with ``TOOL_RESOURCE_PREFIX``."""
    return _split_resources(app)[0]


def non_tool_resources(app: dict[str, Any]) -> list[dict[str, Any]]:
    """The App resources Agent Bricks does not own, e.g. the tracing experiment."""
    return _split_resources(app)[1]


def app_resource_tuples(resources: Sequence[dict[str, Any]]) -> set[GrantTuple]:
    actual = set()
    for resource in resources:
        if "uc_securable" in resource:
            value = resource["uc_securable"]
            actual.add(
                (
                    "uc_securable",
                    value.get("securable_full_name"),
                    value.get("securable_type"),
                    value.get("permission"),
                )
            )
        elif "genie_space" in resource:
            value = resource["genie_space"]
            actual.add(
                ("genie_space", value.get("space_id"), "GENIE_SPACE", value.get("permission"))
            )
    return actual


def effective_privileges(
    client: WorkspaceClient, securable_type: SecurableType, full_name: str, principal: str
) -> list[str]:
    privileges: set[str] = set()
    page_token: str | None = None
    while True:
        response = client.grants.get_effective(
            securable_type.value,
            full_name,
            max_results=0,
            principal=principal,
            **({"page_token": page_token} if page_token else {}),
        )
        for assignment in response.privilege_assignments or ():
            if assignment.principal != principal:
                continue
            privileges.update(
                privilege.privilege.value
                for privilege in assignment.privileges or ()
                if privilege.privilege is not None
            )
        page_token = response.next_page_token
        if not page_token:
            return sorted(privileges)


def direct_privileges(
    client: WorkspaceClient, securable_type: SecurableType, full_name: str, principal: str
) -> list[str]:
    privileges: set[str] = set()
    page_token: str | None = None
    while True:
        response = client.grants.get(
            securable_type.value,
            full_name,
            max_results=0,
            principal=principal,
            **({"page_token": page_token} if page_token else {}),
        )
        for assignment in response.privilege_assignments or ():
            if assignment.principal == principal:
                privileges.update(privilege.value for privilege in assignment.privileges or ())
        page_token = response.next_page_token
        if not page_token:
            return sorted(privileges)
