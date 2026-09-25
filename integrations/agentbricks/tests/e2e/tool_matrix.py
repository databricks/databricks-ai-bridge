#!/usr/bin/env python3
"""Run the LangGraph × CLI/direct × dev/deploy × tool and access-grant E2E matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Sequence
from typing import Any

import tomli
import tomlkit
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import SecurableType

FRAMEWORKS = ("langgraph",)
AUTHORING_PATHS = ("cli", "direct")
RUNTIMES = ("dev", "deploy")
TOOL_KINDS = ("sandbox", "mcp", "python", "uc_function", "genie")
E2E_MODEL = "system.ai.gpt-5-2"

PROMPTS = {
    "sandbox": (
        "You must call the sandbox tool and run Python code "
        "print('AGENTBRICKS_SANDBOX_OK'). Return the exact stdout marker."
    ),
    "mcp": (
        "You must use a tool from the configured system.ai.web_search MCP server. "
        "Search official Databricks documentation for Model Context Protocol, then return the "
        "title and https URL of one result. Do not answer from memory."
    ),
    "python": (
        "You must call the matrix_marker Python tool with value 'matrix'. Return its exact result."
    ),
    "uc_function": "",
    "genie": (
        "You must call the genie_ask tool and ask what data is available in the configured "
        "Genie space. Return a one-sentence summary based only on the tool response."
    ),
}

EXPECTED = {
    "sandbox": "AGENTBRICKS_SANDBOX_OK",
    "python": "AGENTBRICKS_PYTHON_OK",
    "uc_function": "AGENTBRICKS_UC_OK:matrix",
    "mcp": "a web-search tool call and a non-empty https result",
    "genie": "a genie_ask tool call and a non-empty Genie response",
}

_WHEEL_SOURCE_FILES = (
    "databricks_agentkit/_api_client.py",
    "databricks_agentbricks/tool_access.py",
    "databricks_agentbricks/app_resources.py",
    "databricks_agentbricks/cli/deploy.py",
)


class MatrixError(RuntimeError):
    """A reproducible setup or execution failure."""


@dataclasses.dataclass
class EvidenceRow:
    framework: str
    authoring: str
    runtime: str
    tool_kind: str
    status: str
    command: str
    expected: str
    actual: str
    duration_seconds: float
    artifact_paths: list[str]
    app_name: str | None = None
    app_url: str | None = None
    error: str | None = None


@dataclasses.dataclass
class ProjectCase:
    framework: str
    authoring: str
    path: pathlib.Path
    app_name: str


class Transcript:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        line = text.rstrip() + "\n"
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
        profile: str,
        output: pathlib.Path,
        wheel: pathlib.Path,
        template_repo: str | None = None,
        template_ref: str | None = None,
        app_auth_profile: str | None = None,
        genie_space_id: str | None = None,
        commit_sha: str | None = None,
        source_root: pathlib.Path | None = None,
        cleanup_required: bool = True,
    ):
        self.profile = profile
        self.output = output
        self.wheel = wheel.resolve()
        self.template_repo = template_repo
        self.template_ref = template_ref
        self.app_auth_profile = app_auth_profile or profile
        self.genie_space_id = genie_space_id
        self.commit_sha = commit_sha.lower() if commit_sha else None
        self.source_root = source_root.resolve() if source_root is not None else None
        self.source_provenance: dict[str, Any] = {}
        self.cleanup_required = cleanup_required
        self.cleanup_complete = False
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.ended_at: str | None = None
        self.versions: dict[str, str] = {}
        self.transcript = Transcript(output / "commands.log")
        self.runner_venv = output / "runner-venv"
        self.agentbricks = self.runner_venv / "bin" / "ab"
        self.rows: list[EvidenceRow] = []
        self.apps: list[str] = []
        self.uc_function: str | None = None
        self.transitive_uc_function: str | None = None
        self.uc_table: str | None = None
        self.uc_volume: str | None = None
        self.warehouse_id: str | None = None
        self.host: str | None = None
        self.headers: dict[str, str] = {}
        self.grant_checks: list[dict[str, Any]] = []
        self.cleanup_results: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 300,
        log: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if log:
            self.transcript.command(argv, cwd)
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log and result.stdout.strip():
            self.transcript.write(result.stdout)
        if log and result.stderr.strip():
            self.transcript.write(result.stderr)
        if check and result.returncode != 0:
            raise MatrixError(
                f"Command failed ({result.returncode}): {shlex.join(list(argv))}\n"
                f"{result.stderr or result.stdout}"
            )
        return result

    def run_long(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 1800,
    ) -> str:
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
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | running | {last}"
                    )
                    next_tick += 60.0
                time.sleep(2)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        self.transcript.write(output)
        if process.returncode != 0:
            raise MatrixError(f"{label} failed ({process.returncode}); log: {log_path}")
        self.transcript.write(f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | success")
        return output

    def databricks(self, args: Sequence[str], *, timeout: float = 300) -> dict[str, Any]:
        result = self.run(
            ["databricks", *args, "--profile", self.profile, "--output", "json"],
            timeout=timeout,
        )
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MatrixError(f"Databricks CLI returned invalid JSON: {result.stdout}") from exc

    def bootstrap(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.source_root is None or self.commit_sha is None:
            raise MatrixError("Live runs require a source root and commit SHA.")
        self.source_provenance = _source_provenance(self.source_root, self.commit_sha, self.wheel)
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
        self.run([str(self.agentbricks), "tools", "--help"])
        version_commands = {
            "agentbricks": [str(self.agentbricks), "--version"],
            "databricks": ["databricks", "version"],
            "uv": ["uv", "--version"],
            "python": [str(self.runner_venv / "bin" / "python"), "--version"],
        }
        for name, command in version_commands.items():
            result = self.run(command, log=False, check=False)
            version = (result.stdout or result.stderr).strip()
            if result.returncode != 0 or not version:
                raise MatrixError(f"Could not record {name} version: {version or 'no output'}")
            self.versions[name] = version
        workspace_client = WorkspaceClient(profile=self.profile)
        app_auth_client = WorkspaceClient(profile=self.app_auth_profile)
        if not workspace_client.config.host:
            raise MatrixError(f"Could not resolve a host from profile {self.profile!r}.")
        if not app_auth_client.config.host:
            raise MatrixError(
                f"Could not resolve a host from App auth profile {self.app_auth_profile!r}."
            )
        self.host = workspace_client.config.host.rstrip("/")
        app_auth_host = app_auth_client.config.host.rstrip("/")
        if app_auth_host != self.host:
            raise MatrixError(
                f"App auth profile {self.app_auth_profile!r} targets {app_auth_host}, "
                f"not {self.host}."
            )
        if app_auth_client.config.auth_type == "pat":
            raise MatrixError(
                f"App auth profile {self.app_auth_profile!r} uses a PAT. "
                "Databricks Apps /api routes require OAuth; run `databricks auth login` "
                "for a profile on the same workspace."
            )
        authorization = app_auth_client.config.authenticate().get("Authorization")
        if not authorization:
            raise MatrixError(
                f"Could not resolve credentials from App auth profile {self.app_auth_profile!r}."
            )
        self.headers = {"Authorization": authorization}

    def select_warehouse(self, override: str | None) -> str:
        if override:
            self.warehouse_id = override
        else:
            warehouses = self.databricks(["warehouses", "list"])
            if not isinstance(warehouses, list) or not warehouses:
                raise MatrixError("df1 has no SQL warehouse available for UC function setup.")
            running = next(
                (item for item in warehouses if item.get("state") == "RUNNING"), warehouses[0]
            )
            self.warehouse_id = str(running["id"])
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

    def sql(self, statement: str, *, timeout: float = 600) -> dict[str, Any]:
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
        while response.get("status", {}).get("state") in {"PENDING", "RUNNING"}:
            if not statement_id:
                raise MatrixError(f"SQL response has no statement_id: {response}")
            if timeout <= 0:
                raise MatrixError(f"SQL statement timed out: {statement_id}")
            time.sleep(10)
            timeout -= 10
            response = self.databricks(
                ["api", "get", f"/api/2.0/sql/statements/{statement_id}"], timeout=60
            )
        if response.get("status", {}).get("state") != "SUCCEEDED":
            raise MatrixError(f"SQL failed: {json.dumps(response, indent=2)}")
        return response

    def create_uc_function(self, schema: str) -> str:
        catalog, separator, schema_name = schema.partition(".")
        if not separator or not catalog or not schema_name or "." in schema_name:
            raise MatrixError("--uc-schema must be a two-part catalog.schema name.")
        self.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema_name}`")
        suffix = uuid.uuid4().hex[:8]
        nested_function_name = f"agentbricks_nested_{suffix}"
        self.transitive_uc_function = f"{catalog}.{schema_name}.{nested_function_name}"
        self.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{nested_function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Transitive Agent Bricks E2E marker; never declared in agent.toml' "
            "RETURN concat('AGENTBRICKS_UC_OK:', value)"
        )
        function_name = f"agentbricks_uc_{suffix}"
        self.uc_function = f"{catalog}.{schema_name}.{function_name}"
        table_name = f"agentbricks_table_{suffix}"
        self.uc_table = f"{catalog}.{schema_name}.{table_name}"
        self.sql(f"CREATE TABLE `{catalog}`.`{schema_name}`.`{table_name}` AS SELECT 1 AS marker")
        volume_name = f"agentbricks_volume_{suffix}"
        self.uc_volume = f"{catalog}.{schema_name}.{volume_name}"
        self.sql(f"CREATE VOLUME `{catalog}`.`{schema_name}`.`{volume_name}`")
        exposed_tool_name = self.uc_function.replace(".", "__")
        if len(exposed_tool_name) > 64:
            raise MatrixError(
                "The UC function's MCP tool name would exceed 64 characters: "
                f"{exposed_tool_name!r}. Use a shorter --uc-schema."
            )
        self.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Deterministic Agent Bricks E2E marker tool' "
            f"RETURN `{catalog}`.`{schema_name}`.`{nested_function_name}`(value)"
        )
        return self.uc_function

    def create_projects(self) -> list[ProjectCase]:
        if self.uc_function is None:
            raise MatrixError("UC function was not created.")
        projects_root = self.output / "projects"
        projects_root.mkdir(parents=True, exist_ok=True)
        run_suffix = uuid.uuid4().hex[:6]
        cases: list[ProjectCase] = []
        for framework in FRAMEWORKS:
            for authoring in AUTHORING_PATHS:
                project = projects_root / f"{framework}-{authoring}"
                init_args = [
                    str(self.agentbricks),
                    "--profile",
                    self.profile,
                    "init",
                    "--framework",
                    framework,
                    "--profile",
                    self.profile,
                ]
                if self.template_repo:
                    init_args.extend(["--repo", self.template_repo])
                if self.template_ref:
                    init_args.extend(["--ref", self.template_ref])
                init_args.append(str(project))
                self.run_long(
                    f"init-{framework}-{authoring}",
                    init_args,
                    timeout=600,
                )
                self._pin_project_wheel(project)
                if authoring == "cli":
                    self._author_cli(project)
                else:
                    self._author_direct(project, framework)
                self._write_python_marker(project)
                app_name = f"agent-bricks-t-{framework[:2]}-{authoring[:2]}-{run_suffix}"
                cases.append(ProjectCase(framework, authoring, project, app_name))
        return cases

    def _pin_project_wheel(self, project: pathlib.Path) -> None:
        vendor_dir = project / "agentbricks_e2e_wheels"
        vendor_dir.mkdir()
        vendored_wheel = vendor_dir / self.wheel.name
        shutil.copy2(self.wheel, vendored_wheel)
        pyproject = project / "pyproject.toml"
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
        tool = document.setdefault("tool", {})
        uv = tool.setdefault("uv", {})
        sources = uv.setdefault("sources", {})
        sources["databricks-agentbricks"] = {"path": vendored_wheel.relative_to(project).as_posix()}
        self.transcript.file_step(
            pyproject,
            f"pin databricks-agentbricks runtime to {vendored_wheel.relative_to(project)}",
        )
        pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")

    def _author_cli(self, project: pathlib.Path) -> None:
        if self.uc_table is None or self.uc_volume is None:
            raise MatrixError("Sandbox table and volume were not created.")
        manifest = project / "agent.toml"
        before = manifest.read_bytes()
        rejected = self.run(
            [
                str(self.agentbricks),
                "--profile",
                self.profile,
                "--output",
                "json",
                "tools",
                "add",
                "mcp",
                "system.ai.missing_service",
                "--name",
                "broken_mcp",
                "--source",
                str(project),
            ],
            check=False,
        )
        if rejected.returncode == 0 or manifest.read_bytes() != before:
            raise MatrixError(
                "ab tools add accepted an unavailable MCP service or changed agent.toml"
            )
        if json.loads(rejected.stderr).get("error", {}).get("code") not in {
            "NOT_FOUND",
            "RESOURCE_DOES_NOT_EXIST",
        }:
            raise MatrixError(f"Unexpected MCP validation error: {rejected.stderr}")
        self.run(
            [
                str(self.agentbricks),
                "tools",
                "remove",
                "mcp",
                "system.ai.missing_service",
                "--source",
                str(project),
            ]
        )
        manifest = tomli.loads((project / "agent.toml").read_text())
        if any(tool["id"] == "broken_mcp" for tool in manifest.get("tools", [])):
            raise MatrixError("ab tools remove left the broken MCP binding in agent.toml")
        commands = [
            [
                "tools",
                "add",
                "sandbox",
                "--scope",
                f"table:{self.uc_table}",
                "--scope",
                f"volume:{self.uc_volume}",
                "--auth",
                "app",
            ],
            ["tools", "add", "mcp", "system.ai.web_search", "--auth", "app"],
            [
                "tools",
                "add",
                "uc-function",
                self.uc_function or "",
                "--name",
                "agentbricks_uc_marker",
            ],
            [
                "tools",
                "add",
                "genie-agent",
                self.genie_space_id or "",
                "--name",
                "genie",
                "--auth",
                "app",
            ],
        ]
        for args in commands:
            self.run(
                [
                    str(self.agentbricks),
                    "--profile",
                    self.profile,
                    *args,
                    "--source",
                    str(project),
                ]
            )
        manifest = tomli.loads((project / "agent.toml").read_text())
        tool_ids = {tool["id"] for tool in manifest.get("tools", [])}
        expected = {"sandbox", "web_search", "agentbricks_uc_marker", "genie"}
        if tool_ids != expected:
            raise MatrixError(
                f"managed bindings mismatch: expected {sorted(expected)}, got {sorted(tool_ids)}"
            )
        tools_by_id = {tool["id"]: tool for tool in manifest.get("tools", [])}
        for tool_id in ("sandbox", "web_search", "genie"):
            if tools_by_id[tool_id].get("auth") != "app":
                raise MatrixError(f"CLI-authored {tool_id} binding is not App-auth.")

    def _author_direct(self, project: pathlib.Path, framework: str) -> None:
        if self.uc_table is None or self.uc_volume is None:
            raise MatrixError("Sandbox table and volume were not created.")
        fixture = pathlib.Path(__file__).parent / "fixtures" / "direct_agent.toml"
        manifest = (
            fixture.read_text(encoding="utf-8")
            .replace("__FRAMEWORK__", framework)
            .replace("__UC_FUNCTION__", self.uc_function or "")
            .replace("__GENIE_SPACE_ID__", self.genie_space_id or "")
            .replace("__UC_TABLE__", self.uc_table)
            .replace("__UC_VOLUME__", self.uc_volume)
        )
        target = project / "agent.toml"
        self.transcript.file_step(target, "direct authoring; no ab tools command")
        target.write_text(manifest, encoding="utf-8")

    def _write_python_marker(self, project: pathlib.Path) -> None:
        body = (
            "from langchain_core.tools import tool\n\n\n"
            "@tool\n"
            "def matrix_marker(value: str) -> str:\n"
            '    """Return the deterministic AgentBricks E2E marker."""\n'
            "    return 'AGENTBRICKS_PYTHON_OK'\n"
        )
        target = project / "agent" / "tools" / "matrix_marker.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.transcript.file_step(
            target, "user-owned deterministic AGENTBRICKS_PYTHON_OK implementation"
        )
        target.write_text(body, encoding="utf-8")

    def run_dev(self, case: ProjectCase, port: int) -> None:
        label = f"dev-{case.framework}-{case.authoring}"
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.agentbricks),
            "--profile",
            self.profile,
            "dev",
            "--source",
            str(case.path),
            "--app-port",
            str(port),
            "--prepare-environment",
        ]
        self.transcript.command(argv)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            self._wait_for_local(process, port, label, log_path)
            self._exercise(case, "dev", f"http://127.0.0.1:{port}", {}, log_path)
        except Exception as exc:
            self._record_runtime_failure(case, "dev", exc, log_path)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            self.transcript.write(
                f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | stopped"
            )

    def _wait_for_local(
        self,
        process: subprocess.Popen[str],
        port: int,
        label: str,
        log_path: pathlib.Path,
    ) -> None:
        started = time.monotonic()
        next_tick = 60.0
        while True:
            if process.poll() is not None:
                raise MatrixError(
                    f"{label} exited {process.returncode}: {_last_lines(log_path, 30)}"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - started
            if elapsed > 1200:
                raise MatrixError(f"{label} did not become reachable: {_last_lines(log_path, 30)}")
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | starting | "
                    f"{_last_nonempty_line(log_path)}"
                )
                next_tick += 60
            time.sleep(5)

    def deploy(self, case: ProjectCase) -> None:
        label = f"deploy-{case.framework}-{case.authoring}"
        log_path = self.output / "logs" / f"{label}.log"
        try:
            self.run_long(
                label,
                [
                    str(self.agentbricks),
                    "--profile",
                    self.profile,
                    "deploy",
                    case.app_name,
                    "--source",
                    str(case.path),
                ],
                timeout=2400,
            )
            self.apps.append(case.app_name)
            app = self._wait_for_app(case.app_name)
            initial_grants = self._grant_snapshot(app)
            repeat_grants = None
            idempotent = None
            if case.authoring == "cli":
                self.run_long(
                    f"{label}-repeat",
                    [
                        str(self.agentbricks),
                        "--profile",
                        self.profile,
                        "deploy",
                        case.app_name,
                        "--source",
                        str(case.path),
                    ],
                    timeout=2400,
                )
                app = self._wait_for_app(case.app_name)
                repeat_grants = self._grant_snapshot(app)
                idempotent = (
                    initial_grants["tool_resources"] == repeat_grants["tool_resources"]
                    and initial_grants["unrelated_resources"]
                    == repeat_grants["unrelated_resources"]
                    and initial_grants["uc_effective"] == repeat_grants["uc_effective"]
                    and initial_grants["transitive_direct_privileges"]
                    == repeat_grants["transitive_direct_privileges"]
                    and initial_grants["transitive_effective_privileges"]
                    == repeat_grants["transitive_effective_privileges"]
                )
                if not idempotent:
                    raise MatrixError(
                        "Repeat deploy changed grant state:\n"
                        f"initial={json.dumps(initial_grants, indent=2)}\n"
                        f"repeat={json.dumps(repeat_grants, indent=2)}"
                    )
            post_manual_grant = self._grant_transitive_function(app)
            self.grant_checks.append(
                {
                    "app_name": case.app_name,
                    "authoring": case.authoring,
                    "service_principal_client_id": app.get("service_principal_client_id"),
                    "initial": initial_grants,
                    "repeat": repeat_grants,
                    "repeat_deploy_idempotent": idempotent,
                    "post_manual_transitive_grant": post_manual_grant,
                    "manual_transitive_grant_applied": True,
                }
            )
            self._write_evidence()
            url = str(app.get("url") or "").rstrip("/")
            if not url:
                raise MatrixError(f"App {case.app_name} has no URL: {app}")
            self._exercise(case, "deploy", url, self.headers, log_path, app_name=case.app_name)
        except Exception as exc:
            self._record_runtime_failure(case, "deploy", exc, log_path, case.app_name)

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
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | app-{name} | {state or 'UNKNOWN'}"
                )
                next_tick += 60
            time.sleep(15)
        raise MatrixError(f"App {name} did not become ACTIVE.")

    def _grant_snapshot(self, app: dict[str, Any]) -> dict[str, Any]:
        principal = app.get("service_principal_client_id")
        if not principal or self.uc_function is None or self.transitive_uc_function is None:
            raise MatrixError(f"App response has no service_principal_client_id: {app}")
        resources = app.get("resources") or []
        if not isinstance(resources, list):
            raise MatrixError(f"App resources are not a list: {resources}")
        tool_resources = sorted(
            (
                resource
                for resource in resources
                if str(resource.get("name", "")).startswith("agentbricks-tool-")
            ),
            key=lambda resource: str(resource.get("name", "")),
        )
        unrelated_resources = sorted(
            (
                resource
                for resource in resources
                if not str(resource.get("name", "")).startswith("agentbricks-tool-")
            ),
            key=lambda resource: str(resource.get("name", "")),
        )
        if not unrelated_resources:
            raise MatrixError("Expected a non-tool App resource to prove preservation on redeploy.")
        expected_app_resources = {
            ("uc_securable", self.uc_function, "FUNCTION", "EXECUTE"),
            ("uc_securable", self.uc_table, "TABLE", "SELECT"),
            ("uc_securable", self.uc_volume, "VOLUME", "READ_VOLUME"),
            ("genie_space", self.genie_space_id or "", "GENIE_SPACE", "CAN_RUN"),
        }
        actual_app_resources = set()
        for resource in tool_resources:
            if "uc_securable" in resource:
                value = resource["uc_securable"]
                actual_app_resources.add(
                    (
                        "uc_securable",
                        value.get("securable_full_name"),
                        value.get("securable_type"),
                        value.get("permission"),
                    )
                )
            elif "genie_space" in resource:
                value = resource["genie_space"]
                actual_app_resources.add(
                    (
                        "genie_space",
                        value.get("space_id"),
                        "GENIE_SPACE",
                        value.get("permission"),
                    )
                )
        if actual_app_resources != expected_app_resources:
            raise MatrixError(
                "Unexpected automatic Apps resources: "
                f"expected={sorted(expected_app_resources)}, actual={sorted(actual_app_resources)}"
            )

        client = WorkspaceClient(profile=self.profile)
        transitive_direct = _direct_privileges(
            client,
            SecurableType.FUNCTION,
            self.transitive_uc_function,
            str(principal),
        )
        if "EXECUTE" in transitive_direct:
            raise MatrixError(
                "Agent Bricks granted the transitive function directly: "
                f"{self.transitive_uc_function} -> {transitive_direct}"
            )
        transitive_effective = _effective_privileges(
            client,
            SecurableType.FUNCTION,
            self.transitive_uc_function,
            str(principal),
        )
        if "EXECUTE" in transitive_effective:
            raise MatrixError(
                "The transitive function was already effective before the manual grant: "
                f"{self.transitive_uc_function} -> {transitive_effective}"
            )
        if any(
            resource.get("uc_securable", {}).get("securable_full_name")
            == self.transitive_uc_function
            for resource in tool_resources
        ):
            raise MatrixError(
                "The transitive function appeared in Agent Bricks-owned Apps resources."
            )
        return {
            "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tool_resources": tool_resources,
            "unrelated_resources": unrelated_resources,
            "uc_effective": {},
            "transitive_resource": self.transitive_uc_function,
            "transitive_direct_privileges": transitive_direct,
            "transitive_effective_privileges": transitive_effective,
        }

    def _grant_transitive_function(self, app: dict[str, Any]) -> dict[str, Any]:
        principal = app.get("service_principal_client_id")
        if not principal or self.transitive_uc_function is None:
            raise MatrixError("Cannot grant the transitive control without an App principal.")
        catalog, schema, function_name = self.transitive_uc_function.split(".")
        quoted_principal = f"`{str(principal).replace('`', '``')}`"
        self.sql(
            f"GRANT EXECUTE ON FUNCTION `{catalog}`.`{schema}`.`{function_name}` "
            f"TO {quoted_principal}"
        )
        client = WorkspaceClient(profile=self.profile)
        direct = _direct_privileges(
            client,
            SecurableType.FUNCTION,
            self.transitive_uc_function,
            str(principal),
        )
        effective = _effective_privileges(
            client,
            SecurableType.FUNCTION,
            self.transitive_uc_function,
            str(principal),
        )
        if "EXECUTE" not in effective:
            raise MatrixError(
                "Manual EXECUTE grant did not become effective for the transitive function: "
                f"{self.transitive_uc_function} -> {effective}"
            )
        if "EXECUTE" not in direct:
            raise MatrixError(
                "Manual EXECUTE grant was not persisted directly for the transitive function: "
                f"{self.transitive_uc_function} -> {direct}"
            )
        return {
            "granted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "direct_privileges": direct,
            "effective_privileges": effective,
        }

    def _exercise(
        self,
        case: ProjectCase,
        runtime: str,
        base_url: str,
        headers: dict[str, str],
        log_path: pathlib.Path,
        app_name: str | None = None,
    ) -> None:
        invocation_url = f"{base_url}/api/invocations"
        for tool_kind in TOOL_KINDS:
            started = time.monotonic()
            prompt = PROMPTS[tool_kind]
            if tool_kind == "uc_function":
                if self.uc_function is None:
                    raise MatrixError("UC function was not created.")
                exposed_tool_name = self.uc_function.replace(".", "__")
                prompt = (
                    f"You must call the tool named {exposed_tool_name} with value 'matrix'. "
                    "Do not call matrix_marker. Return the called tool's exact result."
                )
            command = _curl_command(invocation_url, prompt, bool(headers))
            try:
                response = self._invoke_with_retry(
                    f"{runtime}-{case.framework}-{case.authoring}-{tool_kind}",
                    invocation_url,
                    prompt,
                    headers,
                )
                serialized = json.dumps(response, sort_keys=True, default=str)
                _assert_semantics(tool_kind, serialized)
                status, error = "pass", None
            except Exception as exc:
                serialized = ""
                status, error = "fail", str(exc)
            self.rows.append(
                EvidenceRow(
                    framework=case.framework,
                    authoring=case.authoring,
                    runtime=runtime,
                    tool_kind=tool_kind,
                    status=status,
                    command=command,
                    expected=EXPECTED[tool_kind],
                    actual=serialized[:6000],
                    duration_seconds=round(time.monotonic() - started, 3),
                    artifact_paths=[str(log_path)],
                    app_name=app_name,
                    app_url=base_url if runtime == "deploy" else None,
                    error=error,
                )
            )
            self._write_evidence()

    def _invoke_with_retry(
        self, label: str, url: str, prompt: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        last: Exception | None = None
        invocation_id = str(uuid.uuid4())
        body = {
            "id": invocation_id,
            "input": {
                "session_id": invocation_id,
                "model": E2E_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for attempt in range(1, 4):
            try:
                return _monitored(
                    label,
                    lambda: _http_json(url, body, headers),
                    self.transcript,
                    timeout=360,
                )
            except Exception as exc:
                last = exc
                self.transcript.write(f"attempt {attempt}/3 | {label} | {exc}")
                if attempt < 3:
                    time.sleep(15)
        raise MatrixError(f"{label} failed after 3 attempts: {last}")

    def _record_runtime_failure(
        self,
        case: ProjectCase,
        runtime: str,
        exc: Exception,
        log_path: pathlib.Path,
        app_name: str | None = None,
    ) -> None:
        existing = {
            row.tool_kind
            for row in self.rows
            if row.framework == case.framework
            and row.authoring == case.authoring
            and row.runtime == runtime
        }
        for tool_kind in TOOL_KINDS:
            if tool_kind in existing:
                continue
            self.rows.append(
                EvidenceRow(
                    framework=case.framework,
                    authoring=case.authoring,
                    runtime=runtime,
                    tool_kind=tool_kind,
                    status="fail",
                    command="runtime setup",
                    expected=EXPECTED[tool_kind],
                    actual="",
                    duration_seconds=0.0,
                    artifact_paths=[str(log_path)],
                    app_name=app_name,
                    error=str(exc),
                )
            )
        self._write_evidence()

    def _write_evidence(self) -> None:
        payload = {
            "schema_version": 1,
            "commit_sha": self.commit_sha,
            "source_provenance": self.source_provenance,
            "template_repo": self.template_repo or "wheel://databricks-agentbricks",
            "template_ref": self.template_ref or _sha256(self.wheel),
            "workspace_host": self.host,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "versions": self.versions,
            "profile": self.profile,
            "app_auth_profile": self.app_auth_profile,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wheel": str(self.wheel),
            "wheel_sha256": _sha256(self.wheel),
            "uc_function": self.uc_function,
            "transitive_uc_function": self.transitive_uc_function,
            "uc_table": self.uc_table,
            "uc_volume": self.uc_volume,
            "genie_space_id": self.genie_space_id,
            "warehouse_id": self.warehouse_id,
            "grant_checks": self.grant_checks,
            "cleanup_required": self.cleanup_required,
            "cleanup_complete": self.cleanup_complete,
            "cleanup": self.cleanup_results,
            "rows": [dataclasses.asdict(row) for row in self.rows],
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def cleanup(self) -> None:
        for app in self.apps:
            result = self.run(
                ["databricks", "apps", "delete", app, "--profile", self.profile],
                timeout=600,
                check=False,
            )
            if result.returncode != 0:
                self.cleanup_results.append(
                    {
                        "resource": f"app:{app}",
                        "status": "failed",
                        "detail": (result.stderr or result.stdout).strip(),
                    }
                )
                continue
            try:
                self._wait_for_app_deleted(app)
            except MatrixError as exc:
                self.cleanup_results.append(
                    {"resource": f"app:{app}", "status": "failed", "detail": str(exc)}
                )
            else:
                self.cleanup_results.append(
                    {
                        "resource": f"app:{app}",
                        "status": "deleted",
                        "confirmed_absent_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
        if self.uc_function:
            catalog, schema, function_name = self.uc_function.split(".")
            try:
                self.sql(f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`")
                self.cleanup_results.append(
                    {"resource": f"function:{self.uc_function}", "status": "deleted"}
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC function | {exc}")
                self.cleanup_results.append(
                    {
                        "resource": f"function:{self.uc_function}",
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        if self.transitive_uc_function:
            catalog, schema, function_name = self.transitive_uc_function.split(".")
            try:
                self.sql(f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`")
                self.cleanup_results.append(
                    {
                        "resource": f"function:{self.transitive_uc_function}",
                        "status": "deleted",
                    }
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | transitive UC function | {exc}")
                self.cleanup_results.append(
                    {
                        "resource": f"function:{self.transitive_uc_function}",
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        if self.uc_table:
            catalog, schema, table_name = self.uc_table.split(".")
            try:
                self.sql(f"DROP TABLE IF EXISTS `{catalog}`.`{schema}`.`{table_name}`")
                self.cleanup_results.append(
                    {"resource": f"table:{self.uc_table}", "status": "deleted"}
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC table | {exc}")
                self.cleanup_results.append(
                    {
                        "resource": f"table:{self.uc_table}",
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        if self.uc_volume:
            catalog, schema, volume_name = self.uc_volume.split(".")
            try:
                self.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{schema}`.`{volume_name}`")
                self.cleanup_results.append(
                    {"resource": f"volume:{self.uc_volume}", "status": "deleted"}
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC volume | {exc}")
                self.cleanup_results.append(
                    {
                        "resource": f"volume:{self.uc_volume}",
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        self.cleanup_complete = not any(
            result.get("status") == "failed" for result in self.cleanup_results
        )
        self._write_evidence()

    def _wait_for_app_deleted(self, name: str, timeout: float = 1200) -> None:
        client = WorkspaceClient(profile=self.profile)
        started = time.monotonic()
        next_tick = 0.0
        while True:
            try:
                client.apps.get(name)
            except NotFound:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | delete-{name} | absent"
                )
                return
            except DatabricksError as exc:
                raise MatrixError(f"Could not verify deletion of App {name!r}: {exc}") from exc
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                raise MatrixError(
                    f"App {name!r} still existed {timeout:.0f}s after delete returned."
                )
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | delete-{name} | deleting"
                )
                next_tick += 60
            time.sleep(15)


def _last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _monitored(
    label: str,
    operation: Callable[[], dict[str, Any]],
    transcript: Transcript,
    *,
    timeout: float,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started = time.monotonic()
        while True:
            try:
                return future.result(
                    timeout=min(60, max(1, timeout - (time.monotonic() - started)))
                )
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | running | {elapsed:.0f}s"
                )
                if elapsed >= timeout:
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s") from None


def _http_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=340) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise MatrixError(f"HTTP {exc.code} from {url}: {detail}") from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MatrixError(f"Invalid JSON from {url}: {payload[:2000]}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"Expected object response from {url}, got {type(value).__name__}")
    return value


def _assert_semantics(tool_kind: str, serialized: str) -> None:
    lowered = serialized.lower()
    if tool_kind in {"sandbox", "python", "uc_function"}:
        marker = EXPECTED[tool_kind]
        if marker not in serialized:
            raise MatrixError(f"Missing semantic marker {marker!r}: {serialized[:2000]}")
        return
    if tool_kind == "mcp":
        tool_evidence = any(value in lowered for value in ("web_search", "web search", "search"))
        if not tool_evidence or "https" not in lowered or len(serialized) < 80:
            raise MatrixError(f"Missing web-search execution/result evidence: {serialized[:2000]}")
        return
    if tool_kind == "genie":
        if (
            not any(value in lowered for value in ("genie_ask", "genie", "conversation_id"))
            or len(serialized) < 80
        ):
            raise MatrixError(f"Missing Genie execution/result evidence: {serialized[:2000]}")
        return
    raise MatrixError(f"No semantic assertion is defined for tool kind {tool_kind!r}.")


def _curl_command(invocation_url: str, prompt: str, authenticated: bool) -> str:
    auth = " -H 'Authorization: Bearer <redacted>'" if authenticated else ""
    body = json.dumps(
        {
            "id": "<client-generated-uuid>",
            "input": {
                "session_id": "<stable-session-id>",
                "model": E2E_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
    )
    return (
        f"curl -sS -X POST {shlex.quote(invocation_url)}"
        f" -H 'Content-Type: application/json'{auth} --data {shlex.quote(body)}"
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(
    source_root: pathlib.Path, commit_sha: str, wheel: pathlib.Path
) -> dict[str, Any]:
    def git(*args: str, text: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=source_root,
            capture_output=True,
            text=text,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr if text else result.stderr.decode(errors="replace")
            raise MatrixError(f"Could not inspect source provenance: {detail.strip()}")
        return result

    head = git("rev-parse", "HEAD").stdout.strip().lower()
    if head != commit_sha.lower():
        raise MatrixError(
            f"Claimed commit {commit_sha} does not match source checkout HEAD {head}."
        )
    status = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    source_hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(wheel) as archive:
            for member in _WHEEL_SOURCE_FILES:
                source_path = source_root / "integrations" / "agentbricks" / "src" / member
                source_bytes = source_path.read_bytes()
                try:
                    wheel_bytes = archive.read(member)
                except KeyError as exc:
                    raise MatrixError(f"Built wheel is missing source module {member}.") from exc
                if wheel_bytes != source_bytes:
                    raise MatrixError(
                        f"Built wheel module {member} does not match source checkout."
                    )
                source_hashes[member] = hashlib.sha256(wheel_bytes).hexdigest()
    except zipfile.BadZipFile as exc:
        raise MatrixError(f"Built wheel is not a readable zip archive: {wheel}") from exc
    diff = git("diff", "--binary", "HEAD", text=False).stdout
    return {
        "source_head_sha": head,
        "source_dirty": bool(status),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "wheel_source_matches": True,
        "wheel_source_sha256": source_hashes,
    }


def _effective_privileges(
    client: WorkspaceClient,
    securable_type: SecurableType,
    full_name: str,
    principal: str,
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


def _direct_privileges(
    client: WorkspaceClient,
    securable_type: SecurableType,
    full_name: str,
    principal: str,
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


def verify_evidence(path: pathlib.Path, *, require_cleanup: bool = True) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    provenance_fields = ("commit_sha", "workspace_host", "started_at", "ended_at")
    missing_provenance = [field for field in provenance_fields if not document.get(field)]
    template_repo = document.get("template_repo")
    template_ref = document.get("template_ref")
    if (
        not isinstance(template_repo, str)
        or not template_repo
        or not isinstance(template_ref, str)
        or not template_ref
    ):
        missing_provenance.append("template_repo/template_ref")
    elif template_repo.startswith("wheel://") and template_ref != document.get("wheel_sha256"):
        missing_provenance.append("wheel_template_ref")
    source_provenance = document.get("source_provenance")
    if (
        not isinstance(source_provenance, dict)
        or source_provenance.get("source_head_sha") != document.get("commit_sha")
        or source_provenance.get("wheel_source_matches") is not True
        or not isinstance(source_provenance.get("wheel_source_sha256"), dict)
        or not source_provenance.get("wheel_source_sha256")
    ):
        missing_provenance.append("source_provenance")
    versions = document.get("versions")
    if not isinstance(versions, dict) or any(
        not versions.get(name) for name in ("agentbricks", "databricks", "uv", "python")
    ):
        missing_provenance.append("versions")
    if missing_provenance:
        sys.stdout.write(f"evidence provenance missing: {sorted(missing_provenance)}\n")
        return 1
    rows = document.get("rows", [])
    grant_checks = document.get("grant_checks", [])
    expected = {
        (framework, authoring, runtime, tool)
        for framework in FRAMEWORKS
        for authoring in AUTHORING_PATHS
        for runtime in RUNTIMES
        for tool in TOOL_KINDS
    }
    actual = {
        (row["framework"], row["authoring"], row["runtime"], row["tool_kind"]) for row in rows
    }
    duplicates = len(rows) - len(actual)
    passed = sum(row.get("status") == "pass" for row in rows)
    failed = sum(row.get("status") == "fail" for row in rows)
    skipped = len(expected - actual)
    sys.stdout.write(f"{passed} passed, {failed} failed, {skipped} skipped\n")
    if actual != expected or duplicates or passed != len(expected):
        if expected - actual:
            sys.stdout.write(f"missing cells: {sorted(expected - actual)}\n")
        if duplicates:
            sys.stdout.write(f"duplicate rows: {duplicates}\n")
        return 1
    expected_deployments = len(FRAMEWORKS) * len(AUTHORING_PATHS)
    if len(grant_checks) != expected_deployments:
        sys.stdout.write(
            f"grant evidence: expected {expected_deployments} deployments, got {len(grant_checks)}\n"
        )
        return 1
    repeated = [check for check in grant_checks if check.get("repeat") is not None]
    if len(repeated) != len(FRAMEWORKS) or any(
        check.get("repeat_deploy_idempotent") is not True for check in repeated
    ):
        sys.stdout.write("grant evidence: repeat-deploy idempotency proof is missing or failed\n")
        return 1
    if any(
        "EXECUTE" in check.get("initial", {}).get("transitive_direct_privileges", [])
        or "EXECUTE" in check.get("initial", {}).get("transitive_effective_privileges", [])
        or check.get("manual_transitive_grant_applied") is not True
        or "EXECUTE"
        not in check.get("post_manual_transitive_grant", {}).get("direct_privileges", [])
        or "EXECUTE"
        not in check.get("post_manual_transitive_grant", {}).get("effective_privileges", [])
        for check in grant_checks
    ):
        sys.stdout.write("grant evidence: transitive exclusion/manual-grant proof failed\n")
        return 1
    cleanup_required = document.get("cleanup_required")
    if not isinstance(cleanup_required, bool):
        sys.stdout.write("cleanup evidence: cleanup_required was not recorded\n")
        return 1
    cleanup = document.get("cleanup", [])
    if (
        require_cleanup
        and cleanup_required
        and (
            document.get("cleanup_complete") is not True
            or not isinstance(cleanup, list)
            or any(result.get("status") == "failed" for result in cleanup)
            or any(
                result.get("resource", "").startswith("app:")
                and result.get("status") == "deleted"
                and not result.get("confirmed_absent_at")
                for result in cleanup
            )
        )
    ):
        sys.stdout.write("cleanup evidence: required cleanup is incomplete or failed\n")
        return 1
    sys.stdout.write(
        f"{len(grant_checks)} grant snapshots passed; {len(repeated)} repeat deploy idempotent\n"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="df1")
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--warehouse-id")
    parser.add_argument("--uc-schema", default="supervisor_agent.mason_agent_tools_e2e")
    parser.add_argument("--template-repo")
    parser.add_argument("--template-ref")
    parser.add_argument(
        "--source-root",
        type=pathlib.Path,
        help="Required source checkout whose HEAD and wheel contents are verified for provenance.",
    )
    parser.add_argument(
        "--commit-sha",
        help="Required source commit SHA for provenance in a live matrix run.",
    )
    parser.add_argument(
        "--genie-space-id",
        default=os.environ.get("AGENTBRICKS_E2E_GENIE_SPACE_ID"),
        help="Existing 32-character Genie space ID for App-auth grant/invocation coverage.",
    )
    parser.add_argument(
        "--app-auth-profile",
        help="OAuth profile for deployed App /api calls; defaults to --profile.",
    )
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args()
    if args.verify_evidence is None and (args.wheel is None or args.output is None):
        parser.error("--wheel and --output are required unless --verify-evidence is used")
    if args.verify_evidence is None and not args.commit_sha:
        parser.error("--commit-sha is required unless --verify-evidence is used")
    if args.verify_evidence is None and args.source_root is None:
        parser.error("--source-root is required unless --verify-evidence is used")
    if args.verify_evidence is None and (
        len(args.commit_sha) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in args.commit_sha)
    ):
        parser.error("--commit-sha must be a full 40-character hexadecimal Git SHA")
    if bool(args.template_repo) != bool(args.template_ref):
        parser.error("--template-repo and --template-ref must be provided together")
    if args.verify_evidence is None and (
        not isinstance(args.genie_space_id, str)
        or len(args.genie_space_id) != 32
        or any(character not in "0123456789abcdef" for character in args.genie_space_id)
    ):
        parser.error("--genie-space-id must be a 32-character lowercase hexadecimal ID")
    return args


def main() -> int:
    args = parse_args()
    if args.verify_evidence:
        return verify_evidence(args.verify_evidence)
    runner = Runner(
        args.profile,
        args.output.resolve(),
        args.wheel.resolve(),
        args.template_repo,
        args.template_ref,
        args.app_auth_profile,
        args.genie_space_id,
        args.commit_sha,
        args.source_root,
        cleanup_required=not args.keep_resources,
    )
    precheck_passed = False
    try:
        runner.bootstrap()
        runner.select_warehouse(args.warehouse_id)
        runner.create_uc_function(args.uc_schema)
        cases = runner.create_projects()
        for index, case in enumerate(cases):
            runner.run_dev(case, 8400 + index)
        for case in cases:
            runner.deploy(case)
        runner.ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
        runner._write_evidence()
        precheck_passed = (
            verify_evidence(runner.output / "evidence.json", require_cleanup=False) == 0
        )
        if not precheck_passed:
            return 1
        if runner.cleanup_required:
            runner.cleanup()
        else:
            runner._write_evidence()
        runner.ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
        runner._write_evidence()
        return verify_evidence(runner.output / "evidence.json")
    finally:
        if not precheck_passed:
            runner.ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
            runner._write_evidence()
            runner.transcript.write(
                "Resources retained after failure for diagnosis; rerun cleanup after fixing."
            )


if __name__ == "__main__":
    sys.exit(main())
