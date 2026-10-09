"""Databricks SDK helpers for the matrix tests and fixtures: the workspace, which is not under test.

Everything here goes through the SDK except ``app_logs``, since the SDK has no Apps log API.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import pathlib
import re
import subprocess
import time
from collections.abc import Callable, Collection, Sequence
from typing import Any, cast

from common import TOOL_RESOURCE_PREFIX, MatrixError, log, now
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import SecurableType, VolumeType
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, State, StatementState

GrantTuple = tuple[str, str, str, str]

_BRANCH = re.compile(r"projects/[^/]+/branches/[^/]+")
_RUNTIME_DATABASE_PREFIX = "runtime-"
_APPS_RECORD = "apps.jsonl"


@dataclasses.dataclass(frozen=True)
class AppIdentity:
    """What cleanup needs to find an App's Lakebase leftovers once the App and its Runtime Store are gone."""

    service_principal: str | None = None
    branch: str | None = None
    database_id: str | None = None

    def merged(self, other: AppIdentity) -> AppIdentity:
        """This identity with any missing field taken from ``other``."""
        return AppIdentity(
            self.service_principal or other.service_principal,
            self.branch or other.branch,
            self.database_id or other.database_id,
        )


def record_app_identity(output: pathlib.Path, app: str, identity: AppIdentity) -> None:
    """Append one line for the controller's end-of-run sweep, which cannot see worker memory."""
    line = json.dumps({"app": app, **dataclasses.asdict(identity)}) + "\n"
    # One write per line: appends this small do not interleave across processes on POSIX.
    with (output / _APPS_RECORD).open("a", encoding="utf-8") as record:
        record.write(line)


def recorded_app_identities(output: pathlib.Path) -> dict[str, AppIdentity]:
    """Every App identity the run's workers recorded, by App name."""
    path = output / _APPS_RECORD
    identities: dict[str, AppIdentity] = {}
    if not path.exists():
        return identities
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
            identity = AppIdentity(
                data.get("service_principal"), data.get("branch"), data.get("database_id")
            )
            identities[data["app"]] = identities.get(data["app"], AppIdentity()).merged(identity)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return identities


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
        self._user_name: str | None = None

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

    def upload(self, path: str, data: bytes) -> None:
        self.client.files.upload(path, io.BytesIO(data), overwrite=True)
        log(f"# uploaded {path}")

    # Temporary UC objects

    def create_schema(self, catalog: str, name: str, *, remove_after: dt.datetime) -> str:
        """Create a schema and return its full name; ``RemoveAfter`` lets a sweeper delete it later."""
        info = self.client.schemas.create(
            name,
            catalog,
            comment="Temporary Agent Bricks E2E schema; safe to delete",
            properties={"RemoveAfter": remove_after.strftime("%Y%m%d%H")},
        )
        log(f"# created schema {info.full_name}")
        return f"{catalog}.{name}"

    def delete_schema(self, full_name: str) -> None:
        """Delete a schema with everything in it, grants included; a missing schema is fine."""
        try:
            self.client.schemas.delete(full_name, force=True)
        except NotFound:
            return
        log(f"# deleted schema {full_name}")

    def schemas_with_prefix(self, catalog: str, prefix: str) -> list[str]:
        return [
            info.full_name
            for info in self.client.schemas.list(catalog)
            if info.full_name and info.name and info.name.startswith(prefix)
        ]

    def create_volume(self, catalog: str, schema: str, name: str) -> None:
        self.client.volumes.create(catalog, schema, name, VolumeType.MANAGED)
        log(f"# created volume {catalog}.{schema}.{name}")

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

    def app_identity(self, app_name: str) -> AppIdentity:
        """The App's principal and Lakebase location as far as they still exist; never raises."""
        principal: str | None = None
        branch: str | None = None
        database_id: str | None = None
        try:
            principal = self.app(app_name).get("service_principal_client_id")
        except NotFound:
            pass
        except Exception as exc:
            log(f"cleanup warning | principal lookup for {app_name} | {exc}")
        try:
            lakebase = self.runtime_store(app_name).get("storage_backend", {}).get("lakebase", {})
            candidate = lakebase.get("branch")
            if isinstance(candidate, str) and _BRANCH.fullmatch(candidate):
                branch = candidate
                database_id = lakebase.get("database_id")
        except NotFound:
            pass
        except Exception as exc:
            log(f"cleanup warning | Runtime Store lookup for {app_name} | {exc}")
        return AppIdentity(principal, branch, database_id)

    def runtime_databases(
        self, branch: str, principals: Collection[str], database_ids: Collection[str]
    ) -> list[str]:
        """Resource names of the branch's ``runtime-`` databases owned by these principals.

        Database ids embed a truncated App name plus a UUID, so ownership is read from the owner
        role instead, which is named after the principal. ``database_ids`` are matched as given.
        """
        wanted = [principal for principal in principals if principal]
        names = []
        for database in self.client.postgres.list_databases(parent=branch):
            database_id = (database.name or "").rsplit("/", 1)[-1]
            owner = (
                (database.status.role if database.status else None)
                or (database.spec.role if database.spec else None)
                or ""
            )
            if (
                database.name
                and database_id.startswith(_RUNTIME_DATABASE_PREFIX)
                and (
                    database_id in database_ids
                    or any(owner.rsplit("/", 1)[-1].endswith(item) for item in wanted)
                )
            ):
                names.append(database.name)
        return names

    def roles_of(self, branch: str, principals: Collection[str]) -> list[str]:
        """Resource names of the branch's roles for these principals.

        Role ids are the principal's UUID, sometimes behind a prefix such as ``agents-``.
        """
        wanted = [principal for principal in principals if principal]
        return [
            role.name
            for role in self.client.postgres.list_roles(parent=branch)
            if role.name and any(role.name.rsplit("/", 1)[-1].endswith(item) for item in wanted)
        ]

    def delete_database(self, name: str) -> None:
        try:
            _wait(self.client.postgres.delete_database(name=name))
        except NotFound:
            pass

    def delete_role(self, name: str) -> None:
        try:
            _wait(self.client.postgres.delete_role(name=name))
        except NotFound:
            pass

    # Deployment sources: `agentbricks deploy` syncs them into the user's workspace home and
    # `apps delete` leaves them behind.

    @property
    def user_name(self) -> str:
        if self._user_name is None:
            name = self.client.current_user.me().user_name
            if not name:
                raise MatrixError("The current user has no user_name.")
            self._user_name = name
        return self._user_name

    def _deployments_dir(self) -> str:
        return f"/Workspace/Users/{self.user_name}/agentbricks_deployments"

    def delete_deployment_source(self, app_name: str) -> None:
        """Delete the App's synced source folder; a missing folder counts as deleted."""
        self.delete_workspace_path(f"{self._deployments_dir()}/{app_name}")

    def deployment_sources_with_prefix(self, prefix: str) -> list[str]:
        try:
            entries = list(self.client.workspace.list(self._deployments_dir()))
        except NotFound:
            return []
        return [
            entry.path
            for entry in entries
            if entry.path and entry.path.rsplit("/", 1)[-1].startswith(prefix)
        ]

    def delete_workspace_path(self, path: str) -> None:
        try:
            self.client.workspace.delete(path, recursive=True)
        except NotFound:
            pass


def _wait(operation: Any) -> None:
    wait = getattr(operation, "wait", None)
    if callable(wait):
        wait()


def _delete_each(
    kind: str, find: Callable[[], list[str]], delete: Callable[[str], None]
) -> list[str]:
    """Delete everything ``find`` returns, continuing past failures; returns one line per failure."""
    try:
        names = find()
    except Exception as exc:
        return [f"{kind} lookup: {exc}"]
    failures = []
    for name in names:
        try:
            delete(name)
        except Exception as exc:
            failures.append(f"{kind} {name}: {exc}")
    return failures


def delete_lakebase_leftovers(
    workspace: Workspace,
    branch: str,
    *,
    principals: Collection[str],
    database_ids: Collection[str] = (),
) -> list[str]:
    """Delete runtime databases and then the principals' roles on one branch.

    A role that owns a database cannot be deleted, so databases go first. Only ``runtime-``
    databases are touched, and both kinds are matched by principal UUID, never by name pattern.
    """
    return _delete_each(
        "runtime database",
        lambda: workspace.runtime_databases(branch, principals, database_ids),
        workspace.delete_database,
    ) + _delete_each(
        "Lakebase role",
        lambda: workspace.roles_of(branch, principals),
        workspace.delete_role,
    )


def cleanup_app(
    workspace: Workspace,
    name: str,
    *,
    delete_store: Callable[[str, str], subprocess.CompletedProcess[str]] | None = None,
    memory_store: str | None = None,
    session_store: str | None = None,
    has_app: bool = True,
    runtime_store: bool = False,
    identity: AppIdentity | None = None,
) -> list[str]:
    """Delete one project's stores and, if it has an App, everything the App's deploy created.

    ``delete_store(kind, store)`` is the agentbricks CLI's store delete, which has no SDK surface.
    ``identity`` is what was recorded at deploy time; it fills in whatever the live lookup misses.
    Every step runs regardless of earlier failures except those that need the App gone. Returns a
    description of each failure; an empty list means everything was deleted.
    """
    failures: list[str] = []

    def attempt(label: str, action: Callable[..., object], *args: Any) -> bool:
        try:
            action(*args)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            return False
        return True

    def delete_cli_store(kind: str, store: str) -> None:
        assert delete_store is not None
        result = delete_store(kind, store)
        if result.returncode != 0:
            raise MatrixError((result.stderr or result.stdout).strip())

    # Read before any delete: the principal and branch come from the App and its Runtime Store.
    known = (identity or AppIdentity()).merged(
        workspace.app_identity(name) if has_app else AppIdentity()
    )
    for kind, store in (("memory", memory_store), ("sessions", session_store)):
        if store and delete_store:
            attempt(f"{kind} store {store}", delete_cli_store, kind, store)
    if not has_app:
        return failures

    if runtime_store:
        attempt(f"runtime store {name}", workspace.delete_runtime_store, name)
    app_deleted = attempt(f"app {name}", workspace.delete_app, name)
    if app_deleted:
        attempt(f"deployment source {name}", workspace.delete_deployment_source, name)
    if known.branch:
        failures += delete_lakebase_leftovers(
            workspace,
            known.branch,
            principals=[known.service_principal or ""],
            database_ids=[known.database_id or ""],
        )
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
