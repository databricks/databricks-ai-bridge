#!/usr/bin/env python3
"""Run the deployed LangGraph/OpenAI declarative user-auth scope matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from typing import Any, cast

from databricks.sdk import WorkspaceClient

FRAMEWORKS = ("langgraph", "openai")
SCOPE_SOURCES = ("explicit", "combined")
IDENTITY_DEFAULT_SCOPES = frozenset({"iam.access-control:read", "iam.current-user:read"})
USER_SQL_MARKER = "AGENTBRICKS_USER_SQL_OK"
APP_SQL_DENIED_MARKER = "AGENTBRICKS_APP_SQL_DENIED"


class MatrixError(RuntimeError):
    """A reproducible setup or execution failure."""


class InvocationHTTPError(MatrixError):
    """An HTTP failure returned by a deployed App invocation."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} from /api/invocations: {detail}")


@dataclasses.dataclass(frozen=True)
class ProjectCase:
    framework: str
    scope_source: str
    path: pathlib.Path
    app_name: str

    @property
    def required_scopes(self) -> list[str]:
        if self.scope_source == "combined":
            return ["ai-gateway", "sql"]
        return ["sql"]


@dataclasses.dataclass
class EvidenceRow:
    framework: str
    scope_source: str
    status: str
    required_scopes: list[str]
    configured_scopes: list[str]
    effective_scopes: list[str]
    user_sql_marker: str
    app_sql_denied: bool
    web_search_verified: bool
    invocation_status: str
    freshness_marker: str
    duration_seconds: float
    request_path: str = "/api/invocations"
    http_status: int | None = None
    artifact_paths: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None


class Transcript:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, value: str) -> None:
        line = value.rstrip() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def command(self, argv: Sequence[str], cwd: pathlib.Path | None = None) -> None:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        self.write(f"$ {prefix}{shlex.join(list(argv))}")

    def file_step(self, path: pathlib.Path, description: str) -> None:
        self.write(f"# write {path}: {description}")


class Runner:
    def __init__(
        self,
        *,
        profile: str,
        app_auth_profile: str,
        wheel: pathlib.Path,
        output: pathlib.Path,
        source_repo: str,
        source_ref: str,
    ) -> None:
        self.profile = profile
        self.app_auth_profile = app_auth_profile
        self.wheel = wheel.resolve()
        self.output = output.resolve()
        self.source_repo = source_repo
        self.source_ref = source_ref
        self.freshness_marker = f"AUTH_SCOPE_E2E_{source_ref[:12]}"
        self.transcript = Transcript(self.output / "commands.log")
        self.runner_venv = self.output / "runner-venv"
        self.agentbricks = self.runner_venv / "bin" / "agentbricks"
        self.rows: list[EvidenceRow] = []
        self.apps: list[str] = []
        self.warehouse_id: str | None = None
        self.table_name: str | None = None
        self.catalog: str | None = None
        self.schema: str | None = None
        self.host: str | None = None
        self.headers: dict[str, str] = {}
        self.cleanup_state = {"apps_deleted": False, "sql_asset_deleted": False}

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 300,
        check: bool = True,
        log_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.transcript.command(argv, cwd)
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log_output and result.stdout.strip():
            self.transcript.write(result.stdout)
        if log_output and result.stderr.strip():
            self.transcript.write(result.stderr)
        if check and result.returncode != 0:
            detail = result.stderr or result.stdout
            raise MatrixError(
                f"Command failed ({result.returncode}): {shlex.join(list(argv))}\n{detail}"
            )
        return result

    def run_long(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 2400,
    ) -> pathlib.Path:
        self.transcript.command(argv, cwd)
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s; log: {log_path}")
                if elapsed >= next_tick:
                    last = _last_nonempty_line(log_path)
                    error = _last_error_line(log_path)
                    suffix = f" | {error}" if error else ""
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | "
                        f"running | {last}{suffix}"
                    )
                    next_tick += 60.0
                time.sleep(2)
        if process.returncode != 0:
            raise MatrixError(
                f"{label} failed ({process.returncode}); log: {log_path}\n"
                f"{_last_lines(log_path, 40)}"
            )
        self.transcript.write(f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | success")
        return log_path

    def databricks(self, args: Sequence[str], *, timeout: float = 300, check: bool = True) -> Any:
        result = self.run(
            ["databricks", *args, "--profile", self.profile, "--output", "json"],
            timeout=timeout,
            check=check,
            log_output=False,
        )
        if not check and result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MatrixError("Databricks CLI returned invalid JSON.") from exc

    def bootstrap(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_ref):
            raise MatrixError("--source-ref must be a full lowercase 40-character git SHA.")
        self.output.mkdir(parents=True, exist_ok=True)
        self.run(["uv", "venv", str(self.runner_venv)], timeout=300)
        self.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.runner_venv / "bin" / "python"),
                str(self.wheel),
            ],
            timeout=600,
        )
        workspace = WorkspaceClient(profile=self.profile)
        app_auth = WorkspaceClient(profile=self.app_auth_profile)
        if not workspace.config.host or not app_auth.config.host:
            raise MatrixError("Both workspace profiles must resolve a Databricks host.")
        self.host = workspace.config.host.rstrip("/")
        if app_auth.config.host.rstrip("/") != self.host:
            raise MatrixError("--app-auth-profile must target the same workspace as --profile.")
        if app_auth.config.auth_type == "pat":
            raise MatrixError("Deployed App API calls require an OAuth profile, not PAT auth.")
        authorization = app_auth.config.authenticate().get("Authorization")
        if not authorization:
            raise MatrixError("Could not resolve OAuth credentials for deployed App calls.")
        self.headers = {"Authorization": authorization}

    def select_warehouse(self, override: str | None) -> str:
        if override:
            self.warehouse_id = override
        else:
            warehouses = self.databricks(["warehouses", "list"])
            if not isinstance(warehouses, list) or not warehouses:
                raise MatrixError("No SQL warehouse is available for the auth-scope matrix.")
            selected = next(
                (item for item in warehouses if item.get("state") == "RUNNING"), warehouses[0]
            )
            self.warehouse_id = str(selected["id"])
        self.run_long(
            "warehouse-start",
            [
                "databricks",
                "warehouses",
                "start",
                self.warehouse_id,
                "--profile",
                self.profile,
                "--timeout",
                "20m",
            ],
            timeout=1250,
        )
        return self.warehouse_id

    def sql(self, statement: str, *, allow_failure: bool = False) -> dict[str, Any]:
        if self.warehouse_id is None:
            raise MatrixError("SQL warehouse was not selected.")
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": statement,
            "wait_timeout": "30s",
            "on_wait_timeout": "CONTINUE",
        }
        response = self.databricks(
            ["api", "post", "/api/2.0/sql/statements", "--json", json.dumps(payload)],
            timeout=60,
        )
        statement_id = response.get("statement_id")
        remaining = 600
        while response.get("status", {}).get("state") in {"PENDING", "RUNNING"}:
            if not statement_id or remaining <= 0:
                raise MatrixError("SQL statement did not reach a terminal state.")
            time.sleep(10)
            remaining -= 10
            response = self.databricks(
                ["api", "get", f"/api/2.0/sql/statements/{statement_id}"], timeout=60
            )
        if response.get("status", {}).get("state") != "SUCCEEDED" and not allow_failure:
            raise MatrixError(f"SQL failed with state {response.get('status', {}).get('state')}.")
        return response

    def create_sql_asset(self, uc_schema: str) -> str:
        catalog, separator, schema = uc_schema.partition(".")
        if not separator or not catalog or not schema or "." in schema:
            raise MatrixError("--uc-schema must be a two-part catalog.schema name.")
        self.catalog = catalog
        self.schema = schema
        table = f"agentbricks_auth_scope_{uuid.uuid4().hex[:8]}"
        self.table_name = table
        self.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
        self.sql(
            f"CREATE TABLE `{catalog}`.`{schema}`.`{table}` USING DELTA AS "
            f"SELECT '{USER_SQL_MARKER}' AS marker, current_user() AS owner"
        )
        return f"{catalog}.{schema}.{table}"

    def create_projects(self) -> list[ProjectCase]:
        if not all((self.catalog, self.schema, self.table_name, self.warehouse_id)):
            raise MatrixError("SQL asset and warehouse must exist before project creation.")
        projects_root = self.output / "projects"
        projects_root.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex[:6]
        cases: list[ProjectCase] = []
        for framework in FRAMEWORKS:
            for scope_source in SCOPE_SOURCES:
                project = projects_root / f"{framework}-{scope_source}"
                self.run_long(
                    f"init-{framework}-{scope_source}",
                    [
                        str(self.agentbricks),
                        "--profile",
                        self.profile,
                        "init",
                        "--framework",
                        framework,
                        "--profile",
                        self.profile,
                        str(project),
                    ],
                    timeout=600,
                )
                self._append_user_auth(project)
                if scope_source == "combined":
                    self.run(
                        [
                            str(self.agentbricks),
                            "--profile",
                            self.profile,
                            "tools",
                            "add",
                            "mcp",
                            "system.ai.web_search",
                            "--auth",
                            "user",
                            "--source",
                            str(project),
                        ],
                        timeout=180,
                    )
                self._pin_runtime_source(project, framework)
                self._write_auth_scope_tool(project, framework)
                self._patch_agent(project, framework)
                manifest = (project / "agent.toml").read_text(encoding="utf-8")
                expected_binding = scope_source == "combined"
                if ("system.ai.web_search" in manifest) != expected_binding:
                    raise MatrixError(f"Managed scope source mismatch for {project}.")
                app_name = f"agent-bricks-as-{framework[:2]}-{scope_source[:2]}-{suffix}"
                cases.append(ProjectCase(framework, scope_source, project, app_name))
        return cases

    def _append_user_auth(self, project: pathlib.Path) -> None:
        manifest = project / "agent.toml"
        self.transcript.file_step(manifest, "declare request-user auth with explicit SQL scope")
        with manifest.open("a", encoding="utf-8") as output:
            output.write('\n[auth.user]\nrequired = true\nadditional_api_scopes = ["sql"]\n')

    def _pin_runtime_source(self, project: pathlib.Path, framework: str) -> None:
        pyproject = project / "pyproject.toml"
        self.transcript.file_step(pyproject, f"pin runtime to pushed SHA {self.source_ref}")
        framework_package, framework_subdirectory = {
            "langgraph": ("databricks-langchain", "integrations/langchain"),
            "openai": ("databricks-openai", "integrations/openai"),
        }[framework]
        source = pyproject.read_text(encoding="utf-8")
        dependencies_needle = "dependencies = [\n"
        if dependencies_needle not in source:
            raise MatrixError(f"Could not find dependency list in {pyproject}.")
        pyproject.write_text(
            source.replace(
                dependencies_needle,
                dependencies_needle + f'    "{framework_package}",\n',
                1,
            ),
            encoding="utf-8",
        )
        with pyproject.open("a", encoding="utf-8") as output:
            output.write(
                "\n[tool.uv.sources]\n"
                "databricks-agentbricks = { "
                f'git = "{self.source_repo}", rev = "{self.source_ref}", '
                'subdirectory = "integrations/agentbricks" }\n'
                f'{framework_package} = {{ git = "{self.source_repo}", '
                f'rev = "{self.source_ref}", subdirectory = "{framework_subdirectory}" }}\n'
            )

    def _write_auth_scope_tool(self, project: pathlib.Path, framework: str) -> None:
        decorator_import = (
            "from langchain_core.tools import tool"
            if framework == "langgraph"
            else "from agents import function_tool"
        )
        decorator = "tool" if framework == "langgraph" else "function_tool"
        query = (
            f"SELECT marker FROM `{self.catalog}`.`{self.schema}`.`{self.table_name}` "
            "WHERE owner = current_user()"
        )
        source = f'''from collections.abc import Callable

from databricks.sdk import WorkspaceClient
{decorator_import}

FRESHNESS_MARKER = {self.freshness_marker!r}
SQL_QUERY = {query!r}
WAREHOUSE_ID = {self.warehouse_id!r}


def _read_marker(client: WorkspaceClient) -> str | None:
    response = client.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=SQL_QUERY,
        wait_timeout="30s",
    )
    state = str(getattr(getattr(response, "status", None), "state", "")).rsplit(".", 1)[-1]
    if state != "SUCCEEDED":
        raise RuntimeError(f"SQL statement failed with state {{state or 'UNKNOWN'}}")
    data = getattr(getattr(response, "result", None), "data_array", None)
    if not data or not data[0]:
        return None
    return str(data[0][0])


def auth_scope_tools(
    workspace_client_for: Callable[[str], WorkspaceClient] | None,
):
    @{decorator}
    def verify_sql_auth_scope() -> str:
        """Prove user SQL access succeeds while the App principal is denied."""
        if workspace_client_for is None:
            raise RuntimeError("request-user client resolver is missing")
        print(FRESHNESS_MARKER, flush=True)
        if _read_marker(workspace_client_for("app")) is not None:
            raise RuntimeError("App principal unexpectedly read the user-only SQL asset")
        marker = _read_marker(workspace_client_for("user"))
        if marker != {USER_SQL_MARKER!r}:
            raise RuntimeError("User SQL marker did not match")
        return f"{{FRESHNESS_MARKER}}|{{marker}}|{APP_SQL_DENIED_MARKER}:empty-result"

    return [verify_sql_auth_scope]
'''
        target = project / "agent" / "auth_scope_tools.py"
        self.transcript.file_step(target, "request-bound custom SQL tool with identity control")
        target.write_text(source, encoding="utf-8")

    def _patch_agent(self, project: pathlib.Path, framework: str) -> None:
        target = project / "agent" / "agent.py"
        source = target.read_text(encoding="utf-8")
        import_needle = "from agent.tools import all_tools\n"
        if import_needle not in source:
            raise MatrixError(f"Could not find tool import seam in {target}.")
        source = source.replace(
            import_needle,
            import_needle + "from agent.auth_scope_tools import auth_scope_tools\n",
            1,
        )
        tools_needle = "        *all_tools(),\n"
        if tools_needle not in source:
            raise MatrixError(f"Could not find framework tool list seam in {target}.")
        source = source.replace(
            tools_needle,
            "        *auth_scope_tools(workspace_client_for),\n" + tools_needle,
            1,
        )
        if framework == "langgraph":
            source = source.replace(
                'REQUIRE_APPROVAL = {"send_message": True}', "REQUIRE_APPROVAL = {}", 1
            )
        else:
            source = source.replace(
                'REQUIRE_APPROVAL = {"send_message"}', "REQUIRE_APPROVAL = set()", 1
            )
        self.transcript.file_step(target, "register request-bound auth scope tool")
        target.write_text(source, encoding="utf-8")

    def deploy_case(self, case: ProjectCase) -> None:
        started = time.monotonic()
        configured: list[str] = []
        effective: list[str] = []
        invocation_status = ""
        user_marker = ""
        app_denied = False
        web_verified = False
        freshness = ""
        http_status: int | None = None
        artifacts: list[str] = []
        error: str | None = None
        self.apps.append(case.app_name)
        try:
            deploy_log = self.run_long(
                f"deploy-{case.framework}-{case.scope_source}",
                [
                    str(self.agentbricks),
                    "--profile",
                    self.profile,
                    "deploy",
                    case.app_name,
                    "--source",
                    str(case.path),
                    "--allow-user-scope-update",
                ],
            )
            artifacts.append(str(deploy_log))
            app = self._wait_for_app(case.app_name)
            configured = sorted(set(app.get("user_api_scopes") or []))
            effective = sorted(set(app.get("effective_user_api_scopes") or []))
            required = set(case.required_scopes)
            if set(configured) != required:
                raise MatrixError(
                    f"Configured scopes mismatch: expected {sorted(required)}, got {configured}."
                )
            if set(effective) - IDENTITY_DEFAULT_SCOPES != required:
                raise MatrixError(
                    "Effective scopes did not equal required scopes plus identity defaults."
                )
            base_url = str(app.get("url") or "").rstrip("/")
            if not base_url:
                raise MatrixError("Deployed App did not expose a URL.")
            sql_response, http_status = self._invoke(
                f"invoke-sql-{case.framework}-{case.scope_source}",
                base_url,
                "Call verify_sql_auth_scope exactly once. Return its exact tool result and nothing else.",
            )
            serialized = json.dumps(sql_response, sort_keys=True, default=str)
            invocation_status = str(sql_response.get("status") or "")
            user_marker = USER_SQL_MARKER if USER_SQL_MARKER in serialized else ""
            app_denied = APP_SQL_DENIED_MARKER in serialized
            freshness = self.freshness_marker if self.freshness_marker in serialized else ""
            if (
                invocation_status != "completed"
                or not user_marker
                or not app_denied
                or not freshness
            ):
                raise MatrixError(f"Custom SQL tool evidence was incomplete: {serialized[:2000]}")
            freshness_log = self._verify_freshness_log(case.app_name)
            artifacts.append(str(freshness_log))
            if case.scope_source == "combined":
                web_response, _ = self._invoke(
                    f"invoke-web-{case.framework}",
                    base_url,
                    "Use the configured web search tool to find one official Databricks documentation page about OAuth scopes. Return its HTTPS URL.",
                )
                web_serialized = json.dumps(web_response, sort_keys=True, default=str).lower()
                web_verified = (
                    web_response.get("status") == "completed"
                    and _has_completed_search_tool_call(web_response)
                    and "https" in web_serialized
                )
                if not web_verified:
                    raise MatrixError(
                        f"Managed web-search evidence was incomplete: {web_serialized[:2000]}"
                    )
            else:
                web_verified = False
            status = "pass"
        except Exception as exc:
            status = "fail"
            error = self._redact(str(exc))
        self.rows.append(
            EvidenceRow(
                framework=case.framework,
                scope_source=case.scope_source,
                status=status,
                required_scopes=case.required_scopes,
                configured_scopes=configured,
                effective_scopes=effective,
                user_sql_marker=user_marker,
                app_sql_denied=app_denied,
                web_search_verified=web_verified,
                invocation_status=invocation_status,
                freshness_marker=freshness,
                duration_seconds=round(time.monotonic() - started, 3),
                http_status=http_status,
                artifact_paths=artifacts,
                error=error,
            )
        )
        self.write_evidence()

    def _wait_for_app(self, name: str) -> dict[str, Any]:
        started = time.monotonic()
        next_tick = 0.0
        while time.monotonic() - started < 1200:
            app = self.databricks(["apps", "get", name])
            compute = app.get("compute_status", {})
            state = compute.get("state") if isinstance(compute, dict) else None
            if state == "ACTIVE" and app.get("url"):
                return app
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | app-{name} | "
                    f"{state or 'UNKNOWN'}"
                )
                next_tick += 60.0
            time.sleep(15)
        raise MatrixError(f"App {name} did not become ACTIVE.")

    def _invoke(self, label: str, base_url: str, prompt: str) -> tuple[dict[str, Any], int]:
        invocation_id = str(uuid.uuid4())
        body = {
            "id": invocation_id,
            "input": {
                "session_id": invocation_id,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        url = f"{base_url}/api/invocations"
        response, status = _invoke_with_readiness_retry(
            label,
            lambda: _http_json(url, body, self.headers),
            self.transcript,
            timeout=360,
            retry_interval=15,
        )
        return response, status

    def _verify_freshness_log(self, app_name: str) -> pathlib.Path:
        label = f"freshness-{app_name}"
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = ["databricks", "apps", "logs", app_name, "--profile", self.profile]
        self.transcript.command(argv)
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        started = time.monotonic()
        next_tick = 60.0
        try:
            while time.monotonic() - started < 180:
                if self.freshness_marker in log_path.read_text(encoding="utf-8", errors="replace"):
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | success"
                    )
                    return log_path
                if process.poll() is not None:
                    raise MatrixError(
                        f"App log stream ended before freshness proof: {_last_lines(log_path, 30)}"
                    )
                elapsed = time.monotonic() - started
                if elapsed >= next_tick:
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | "
                        f"running | {_last_nonempty_line(log_path)}"
                    )
                    next_tick += 60.0
                time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        raise MatrixError("Freshness marker did not appear in deployed App logs.")

    def _redact(self, value: str) -> str:
        replacements = {
            self.host or "": "<workspace-host>",
            self.warehouse_id or "": "<warehouse-id>",
            self.table_name or "": "<sql-asset>",
        }
        result = value
        for secret, replacement in replacements.items():
            if secret:
                result = result.replace(secret, replacement)
        return result

    def cleanup(self) -> None:
        deleted_apps: list[bool] = []
        for app in self.apps:
            self.run(
                ["databricks", "apps", "delete", app, "--profile", self.profile],
                timeout=600,
                check=False,
                log_output=False,
            )
            deleted_apps.append(self._wait_for_app_absence(app))
        self.cleanup_state["apps_deleted"] = bool(deleted_apps) and all(deleted_apps)
        if self.catalog and self.schema and self.table_name:
            self.sql(f"DROP TABLE IF EXISTS `{self.catalog}`.`{self.schema}`.`{self.table_name}`")
            check = self.sql(
                f"SELECT count(*) FROM `{self.catalog}`.information_schema.tables "
                f"WHERE table_schema = '{self.schema}' AND table_name = '{self.table_name}'"
            )
            data = check.get("result", {}).get("data_array") or []
            self.cleanup_state["sql_asset_deleted"] = bool(data and str(data[0][0]) == "0")

    def _wait_for_app_absence(self, name: str) -> bool:
        started = time.monotonic()
        next_tick = 60.0
        while time.monotonic() - started < 600:
            result = self.run(
                ["databricks", "apps", "get", name, "--profile", self.profile],
                timeout=60,
                check=False,
                log_output=False,
            )
            if result.returncode != 0:
                return True
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | cleanup-{name} | deleting"
                )
                next_tick += 60.0
            time.sleep(10)
        return False

    def write_evidence(self) -> None:
        payload = {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_sha": self.source_ref,
            "wheel_sha256": _sha256(self.wheel),
            "rows": [dataclasses.asdict(row) for row in self.rows],
            "cleanup": self.cleanup_state,
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)


def _last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _last_error_line(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 40).splitlines()):
        if re.search(r"ERROR|FAIL|Traceback|panic|exit code [1-9]", line, re.IGNORECASE):
            return line.strip()[:300]
    return ""


def _has_completed_search_tool_call(response: dict[str, Any]) -> bool:
    called_at: dict[str, int] = {}
    output = response.get("output")
    if isinstance(output, dict):
        output = output.get("output")
    if not isinstance(output, list):
        return False
    for index, raw_message in enumerate(output):
        if not isinstance(raw_message, dict):
            continue
        message = cast(dict[str, Any], raw_message)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                name = tool_call.get("name")
                if isinstance(name, str) and "search" in name.lower():
                    called_at.setdefault(name, index)
        if not called_at:
            continue
        content = json.dumps(message.get("content"), default=str).lower()
        if "https://" not in content:
            continue
        message_type = message.get("role") or message.get("type")
        name = message.get("name")
        if (
            message_type == "tool"
            and isinstance(name, str)
            and name in called_at
            and index > called_at[name]
        ):
            return True
        if message_type in {"assistant", "ai"} and any(
            index > call_index for call_index in called_at.values()
        ):
            return True
    return False


def _monitored(
    label: str,
    operation: Callable[[], tuple[dict[str, Any], int]],
    transcript: Transcript,
    *,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started = time.monotonic()
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise MatrixError(f"{label} timed out after {timeout:.0f}s")
            try:
                return future.result(timeout=min(60, remaining))
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | "
                    f"running | {elapsed:.0f}s"
                )


def _http_json(
    url: str, body: dict[str, Any], headers: dict[str, str]
) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=340) as response:
            payload = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise InvocationHTTPError(exc.code, detail) from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MatrixError(f"Invalid JSON from /api/invocations: {payload[:2000]}") from exc
    if not isinstance(value, dict):
        raise MatrixError("Expected an object response from /api/invocations.")
    return value, status


def _invoke_with_readiness_retry(
    label: str,
    operation: Callable[[], tuple[dict[str, Any], int]],
    transcript: Transcript,
    *,
    timeout: float,
    retry_interval: float,
) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    next_tick = 60.0
    attempt = 0
    while True:
        elapsed = time.monotonic() - started
        remaining = timeout - elapsed
        if remaining <= 0:
            raise MatrixError(f"{label} did not become ready within {timeout:.0f}s")
        attempt += 1
        transcript.write(f"attempt {attempt} | {label} | request")
        try:
            return _monitored(label, operation, transcript, timeout=remaining)
        except InvocationHTTPError as exc:
            if exc.status_code not in {502, 503}:
                raise
            elapsed = time.monotonic() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                raise MatrixError(f"{label} did not become ready within {timeout:.0f}s") from exc
            transcript.write(f"attempt {attempt} | HTTP {exc.status_code} | retrying")
            if elapsed >= next_tick:
                transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | "
                    f"waiting for App route | {elapsed:.0f}s"
                )
                while elapsed >= next_tick:
                    next_tick += 60.0
            time.sleep(min(retry_interval, remaining))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(path: pathlib.Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    source_sha = str(document.get("source_sha") or "")
    wheel_sha = str(document.get("wheel_sha256") or "")
    rows = document.get("rows", [])
    expected = {(framework, source) for framework in FRAMEWORKS for source in SCOPE_SOURCES}
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = (str(row.get("framework")), str(row.get("scope_source")))
        if key in actual:
            duplicate_count += 1
        actual[key] = row
    failures: list[str] = []
    passed = 0
    for key in sorted(expected & set(actual)):
        row = actual[key]
        required = {"sql"} if key[1] == "explicit" else {"ai-gateway", "sql"}
        configured = set(row.get("configured_scopes") or [])
        effective = set(row.get("effective_scopes") or [])
        expected_freshness = f"AUTH_SCOPE_E2E_{source_sha[:12]}"
        valid = all(
            (
                row.get("status") == "pass",
                set(row.get("required_scopes") or []) == required,
                configured == required,
                effective - IDENTITY_DEFAULT_SCOPES == required,
                row.get("user_sql_marker") == USER_SQL_MARKER,
                row.get("app_sql_denied") is True,
                row.get("web_search_verified") is (key[1] == "combined"),
                row.get("invocation_status") == "completed",
                row.get("freshness_marker") == expected_freshness,
            )
        )
        if valid:
            passed += 1
        else:
            failures.append(f"invalid cell: {key}")
    skipped = len(expected - set(actual))
    failed = len(failures) + len(set(actual) - expected) + duplicate_count
    cleanup = document.get("cleanup", {})
    global_errors = []
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        global_errors.append("source_sha is not a full git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", wheel_sha):
        global_errors.append("wheel_sha256 is invalid")
    if cleanup != {"apps_deleted": True, "sql_asset_deleted": True}:
        global_errors.append("cleanup verification is incomplete")
    sys.stdout.write(f"{passed} passed, {failed} failed, {skipped} skipped\n")
    for error in [*failures, *global_errors]:
        sys.stdout.write(error + "\n")
    return 0 if passed == len(expected) and failed == skipped == 0 and not global_errors else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="df1")
    parser.add_argument("--app-auth-profile")
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--warehouse-id")
    parser.add_argument("--uc-schema", default="main.agentbricks_auth_scope_e2e")
    parser.add_argument("--source-repo")
    parser.add_argument("--source-ref")
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args()
    required = (args.wheel, args.output, args.source_repo, args.source_ref)
    if args.verify_evidence is None and not all(required):
        parser.error(
            "--wheel, --output, --source-repo, and --source-ref are required unless "
            "--verify-evidence is used"
        )
    if args.verify_evidence is None and not args.app_auth_profile:
        parser.error("--app-auth-profile is required for deployed App OAuth calls")
    return args


def main() -> int:
    args = parse_args()
    if args.verify_evidence:
        return verify_evidence(args.verify_evidence)
    runner = Runner(
        profile=args.profile,
        app_auth_profile=args.app_auth_profile,
        wheel=args.wheel,
        output=args.output,
        source_repo=args.source_repo,
        source_ref=args.source_ref,
    )
    setup_error: str | None = None
    try:
        runner.bootstrap()
        runner.select_warehouse(args.warehouse_id)
        runner.create_sql_asset(args.uc_schema)
        cases = runner.create_projects()
        for case in cases:
            runner.deploy_case(case)
    except Exception as exc:
        setup_error = runner._redact(str(exc))
        runner.transcript.write(f"matrix setup failure | {setup_error}")
    finally:
        if not args.keep_resources:
            try:
                runner.cleanup()
            except Exception as exc:
                runner.transcript.write(f"cleanup failure | {runner._redact(str(exc))}")
        runner.write_evidence()
    result = verify_evidence(runner.output / "evidence.json")
    if setup_error:
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
