"""Databricks SDK helpers for what is not under test: the prerequisites the suites need, and cleanup.

Tests drive and inspect the agent through the ``agentbricks`` CLI. This module is for the rest:
UC objects, the warehouse and Genie space a tool needs, and, under "Workarounds", the cleanup no
public command does yet. ``Workspace`` holds one operation per method; deciding what to delete and
in which order is ``harness.cleanup``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import pathlib
import re
import subprocess
import time
from collections.abc import Collection
from typing import Any, cast

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, State, StatementState

from harness.logs import log
from harness.run import E2EError

_BRANCH = re.compile(r"projects/[^/]+/branches/[^/]+")


@dataclasses.dataclass(frozen=True)
class AppIdentity:
    """What cleanup needs to find the App's Lakebase role once the App and its Runtime Store are gone."""

    service_principal: str | None = None
    branch: str | None = None

    def merged(self, other: AppIdentity) -> AppIdentity:
        """This identity with any missing field taken from ``other``."""
        return AppIdentity(
            self.service_principal or other.service_principal,
            self.branch or other.branch,
        )


class Workspace:
    def __init__(
        self,
        profile: str | None,
        *,
        warehouse_id: str | None = None,
    ):
        self.profile = profile
        self.warehouse_id = warehouse_id
        self.client = WorkspaceClient(profile=profile)
        self._warehouse_started = False
        self._user_name: str | None = None

    # Existing resources the suite depends on

    def require_catalog(self, name: str) -> None:
        try:
            self.client.catalogs.get(name)
        except DatabricksError as exc:
            raise E2EError(f"Catalog {name!r} is not accessible: {exc}") from exc

    def require_genie_space(self, space_id: str) -> None:
        try:
            self.client.genie.get_space(space_id)
        except DatabricksError as exc:
            raise E2EError(f"Genie space {space_id!r} is not accessible: {exc}") from exc

    # SQL, warehouse, files

    def start_warehouse(self, override: str | None = None) -> str:
        if override:
            self.warehouse_id = override
        else:
            warehouses = list(self.client.warehouses.list())
            if not warehouses:
                raise E2EError("The workspace has no SQL warehouse available for UC setup.")
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
                raise E2EError(f"SQL response has no statement_id: {response}")
            if timeout <= 0:
                raise E2EError(f"SQL statement timed out: {statement_id}")
            time.sleep(10)
            timeout -= 10
            response = self.client.statement_execution.get_statement(statement_id)
        if not response.status or response.status.state != StatementState.SUCCEEDED:
            raise E2EError(f"SQL failed: {response.as_dict()}")

    def upload(self, path: str, data: bytes) -> None:
        self.client.files.upload(path, io.BytesIO(data), overwrite=True)
        log(f"# uploaded {path}")

    # Temporary UC objects

    def create_schema(self, catalog: str, name: str, *, remove_after: dt.datetime) -> str:
        """Create a schema and return its full name; ``RemoveAfter`` lets a workspace janitor delete it later."""
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

    # Workarounds for the end-of-run cleanup: what `agentbricks deployments delete` leaves behind.

    def runtime_store(self, app_name: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.client.api_client.do("GET", f"/api/2.0/agents/runtime-stores/{app_name}"),
        )

    def lakebase_branch(self, app_name: str) -> str | None:
        """The Lakebase branch of the App's Runtime Store, if it still exists; never raises."""
        try:
            lakebase = self.runtime_store(app_name).get("storage_backend", {}).get("lakebase", {})
        except NotFound:
            return None
        except Exception as exc:
            log(f"cleanup warning | Runtime Store lookup for {app_name} | {exc}")
            return None
        branch = lakebase.get("branch")
        return branch if isinstance(branch, str) and _BRANCH.fullmatch(branch) else None

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
                raise E2EError("The current user has no user_name.")
            self._user_name = name
        return self._user_name

    def _deployments_dir(self) -> str:
        return f"/Workspace/Users/{self.user_name}/agentbricks_deployments"

    def experiments_with_prefix(self, prefix: str) -> list[str]:
        """Ids of the experiments whose full path starts with ``prefix``."""
        return [
            str(experiment.experiment_id)
            for experiment in self.client.experiments.search_experiments(
                filter=f"name LIKE '{prefix}%'"
            )
            if experiment.experiment_id
        ]

    def delete_experiment(self, experiment_id: str) -> None:
        try:
            self.client.experiments.delete_experiment(experiment_id)
        except NotFound:
            pass

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
