#!/usr/bin/env python3
"""Run the LangGraph × CLI/direct × dev/deploy × tool E2E matrix."""

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
from collections.abc import Callable, Sequence
from typing import Any

from databricks.sdk import WorkspaceClient

FRAMEWORKS = ("langgraph",)
AUTHORING_PATHS = ("cli", "direct")
RUNTIMES = ("dev", "deploy")
TOOL_KINDS = ("sandbox", "mcp", "python", "uc_function")

PROMPTS = {
    "sandbox": (
        "You must call the sandbox tool and run Python code "
        "print('MASON_SANDBOX_OK'). Return the exact stdout marker."
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
}

EXPECTED = {
    "sandbox": "MASON_SANDBOX_OK",
    "python": "MASON_PYTHON_OK",
    "uc_function": "MASON_UC_OK:matrix",
    "mcp": "a web-search tool call and a non-empty https result",
}

PYTHON_TOOL_ID = "matrix-marker"
PYTHON_TOOL_ENTRYPOINT = "agent.tools.matrix_marker:matrix_marker"
VALIDATION_CHECKS = (
    "undeclared_warning",
    "broken_entrypoint_rejection",
    "valid_contract_check",
    "direct_custom_tool_run",
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
class ValidationCheck:
    framework: str
    authoring: str
    check: str
    status: str
    command: str
    return_code: int
    stdout: str
    stderr: str
    expected: str
    error: str | None = None


@dataclasses.dataclass
class CleanupCheck:
    resource_kind: str
    resource: str
    status: str
    command: str
    return_code: int
    stdout: str
    stderr: str
    error: str | None = None


@dataclasses.dataclass
class ProjectCase:
    framework: str
    authoring: str
    path: pathlib.Path
    app_name: str


def append_python_activation(
    manifest: pathlib.Path, entrypoint: str = PYTHON_TOOL_ENTRYPOINT
) -> None:
    """Append the literal Python-tool activation used by the code-first CLI lane."""
    existing = manifest.read_text(encoding="utf-8")
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    with manifest.open("a", encoding="utf-8") as output:
        output.write(
            f'{prefix}\n[[tools]]\nid = "{PYTHON_TOOL_ID}"\n'
            f'source = {{ kind = "python", entrypoint = "{entrypoint}" }}\n'
        )


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
    ):
        self.profile = profile
        self.output = output
        self.wheel = wheel.resolve()
        self.template_repo = template_repo
        self.template_ref = template_ref
        self.app_auth_profile = app_auth_profile or profile
        self.transcript = Transcript(output / "commands.log")
        self.runner_venv = output / "runner-venv"
        self.mason = self.runner_venv / "bin" / "mason"
        self.rows: list[EvidenceRow] = []
        self.validation_checks: list[ValidationCheck] = []
        self.cleanup_checks: list[CleanupCheck] = []
        self.apps: list[str] = []
        self.uc_function: str | None = None
        self.warehouse_id: str | None = None
        self.host: str | None = None
        self.headers: dict[str, str] = {}

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
        self.run([str(self.mason), "tools", "--help"])
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
        function_name = f"mason_uc_{uuid.uuid4().hex[:8]}"
        self.uc_function = f"{catalog}.{schema_name}.{function_name}"
        exposed_tool_name = self.uc_function.replace(".", "__")
        if len(exposed_tool_name) > 64:
            raise MatrixError(
                "The UC function's MCP tool name would exceed 64 characters: "
                f"{exposed_tool_name!r}. Use a shorter --uc-schema."
            )
        self.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Deterministic Mason E2E marker tool' "
            "RETURN concat('MASON_UC_OK:', value)"
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
                    str(self.mason),
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
                self._install_sdk_wheel(project)
                if authoring == "cli":
                    self._author_cli(project)
                    self._write_python_marker(project)
                    self._validate_undeclared_warning(project, framework, authoring)
                    self._validate_broken_entrypoint(project, framework, authoring)
                    manifest = project / "agent.toml"
                    self.transcript.file_step(
                        manifest, "activate matrix-marker with a literal Python manifest entry"
                    )
                    append_python_activation(manifest)
                    self._validate_active_python_tool(project, framework, authoring)
                else:
                    self._write_python_marker(project)
                    self._author_direct(project, framework)
                app_name = f"mason-tools-{framework[:2]}-{authoring[:2]}-{run_suffix}"
                cases.append(ProjectCase(framework, authoring, project, app_name))
        return cases

    def _install_sdk_wheel(self, project: pathlib.Path) -> None:
        """Pin the checkout-under-test SDK into generated dev and deploy environments."""
        wheel_dir = project / ".mason" / "sdk"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        target = wheel_dir / self.wheel.name
        shutil.copy2(self.wheel, target)
        relative = target.relative_to(project).as_posix()
        pyproject = project / "pyproject.toml"
        existing = pyproject.read_text(encoding="utf-8")
        prefix = "" if existing.endswith("\n") else "\n"
        self.transcript.file_step(
            target,
            "bundle the Mason SDK wheel under test for both local and deployed resolution",
        )
        with pyproject.open("a", encoding="utf-8") as output:
            output.write(
                f'{prefix}\n[tool.uv.sources]\ndatabricks-mason = {{ path = "{relative}" }}\n'
            )

    def _author_cli(self, project: pathlib.Path) -> None:
        commands = [
            ["tools", "add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
            ["tools", "add", "mcp", "system.ai.web_search"],
            [
                "tools",
                "add",
                "uc-function",
                self.uc_function or "",
                "--name",
                "mason_uc_marker",
            ],
        ]
        for args in commands:
            self.run([str(self.mason), *args, "--source", str(project)])

    def _author_direct(self, project: pathlib.Path, framework: str) -> None:
        fixture = pathlib.Path(__file__).parent / "fixtures" / "direct_agent.toml"
        manifest = (
            fixture.read_text(encoding="utf-8")
            .replace("__FRAMEWORK__", framework)
            .replace("__UC_FUNCTION__", self.uc_function or "")
        )
        target = project / "agent.toml"
        self.transcript.file_step(target, "direct authoring; no mason tools command")
        target.write_text(manifest, encoding="utf-8")

    def _write_python_marker(self, project: pathlib.Path) -> None:
        body = (
            "from langchain_core.tools import tool\n\n\n"
            "@tool\n"
            "def matrix_marker(value: str) -> str:\n"
            '    """Return the deterministic Mason E2E marker."""\n'
            "    return 'MASON_PYTHON_OK'\n"
        )
        target = project / "agent" / "tools" / "matrix_marker.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.transcript.file_step(target, "user-owned deterministic MASON_PYTHON_OK implementation")
        target.write_text(body, encoding="utf-8")

    def _run_validation(
        self,
        framework: str,
        authoring: str,
        check_name: str,
        argv: Sequence[str],
        expected: str,
        validate: Callable[[subprocess.CompletedProcess[str]], None],
    ) -> subprocess.CompletedProcess[str]:
        result: subprocess.CompletedProcess[str] | None = None
        error: str | None = None
        try:
            result = self.run(argv, check=False)
            validate(result)
        except Exception as exc:
            error = str(exc)
        self.validation_checks.append(
            ValidationCheck(
                framework=framework,
                authoring=authoring,
                check=check_name,
                status="pass" if error is None else "fail",
                command=shlex.join(list(argv)),
                return_code=result.returncode if result is not None else -1,
                stdout=result.stdout if result is not None else "",
                stderr=result.stderr if result is not None else "",
                expected=expected,
                error=error,
            )
        )
        self._write_evidence()
        if error is not None:
            raise MatrixError(f"Validation {check_name} failed: {error}")
        if result is None:
            raise MatrixError(f"Validation {check_name} did not produce a command result.")
        return result

    def _validate_undeclared_warning(
        self, project: pathlib.Path, framework: str, authoring: str
    ) -> None:
        argv = [
            str(self.mason),
            "--output",
            "json",
            "tools",
            "check",
            "--source",
            str(project),
        ]

        def validate(result: subprocess.CompletedProcess[str]) -> None:
            payload = _json_command_output(result)
            warnings = payload.get("warnings")
            found = isinstance(warnings, list) and any(
                isinstance(warning, dict)
                and warning.get("code") == "MASON001"
                and warning.get("entrypoint") == PYTHON_TOOL_ENTRYPOINT
                for warning in warnings
            )
            if result.returncode != 0 or payload.get("ok") is not True or not found:
                raise MatrixError(
                    f"Expected successful MASON001 warning for {PYTHON_TOOL_ENTRYPOINT}: {payload}"
                )

        self._run_validation(
            framework,
            authoring,
            "undeclared_warning",
            argv,
            f"exit 0 with MASON001 for {PYTHON_TOOL_ENTRYPOINT}",
            validate,
        )

    def _validate_broken_entrypoint(
        self, project: pathlib.Path, framework: str, authoring: str
    ) -> None:
        manifest = project / "agent.toml"
        original = manifest.read_text(encoding="utf-8")
        broken_entrypoint = "agent.tools.missing_matrix_marker:matrix_marker"
        self.transcript.file_step(manifest, "temporarily activate a missing Python module")
        append_python_activation(manifest, broken_entrypoint)
        argv = [
            str(self.mason),
            "--output",
            "json",
            "tools",
            "check",
            PYTHON_TOOL_ID,
            "--source",
            str(project),
        ]

        def validate(result: subprocess.CompletedProcess[str]) -> None:
            payload = _json_command_output(result)
            error = payload.get("error")
            if (
                result.returncode == 0
                or payload.get("ok") is not False
                or not isinstance(error, str)
                or "agent.tools.missing_matrix_marker" not in error
            ):
                raise MatrixError(f"Expected a hard missing-entrypoint failure: {payload}")

        try:
            self._run_validation(
                framework,
                authoring,
                "broken_entrypoint_rejection",
                argv,
                "non-zero exit and structured error for the missing module",
                validate,
            )
        finally:
            self.transcript.file_step(manifest, "restore the manifest after the failure probe")
            manifest.write_text(original, encoding="utf-8")

    def _validate_active_python_tool(
        self, project: pathlib.Path, framework: str, authoring: str
    ) -> None:
        check_argv = [
            str(self.mason),
            "--output",
            "json",
            "tools",
            "check",
            PYTHON_TOOL_ID,
            "--source",
            str(project),
        ]

        def validate_check(result: subprocess.CompletedProcess[str]) -> None:
            payload = _json_command_output(result)
            tools = payload.get("tools")
            found = isinstance(tools, list) and any(
                isinstance(tool, dict)
                and tool.get("id") == PYTHON_TOOL_ID
                and tool.get("entrypoint") == PYTHON_TOOL_ENTRYPOINT
                for tool in tools
            )
            if result.returncode != 0 or payload.get("ok") is not True or not found:
                raise MatrixError(f"Expected a valid matrix-marker contract: {payload}")

        self._run_validation(
            framework,
            authoring,
            "valid_contract_check",
            check_argv,
            f"exit 0 with the contract for {PYTHON_TOOL_ENTRYPOINT}",
            validate_check,
        )

        run_argv = [
            str(self.mason),
            "--output",
            "json",
            "tools",
            "run",
            PYTHON_TOOL_ID,
            "--input",
            '{"value":"matrix"}',
            "--source",
            str(project),
        ]

        def validate_run(result: subprocess.CompletedProcess[str]) -> None:
            payload = _json_command_output(result)
            if (
                result.returncode != 0
                or payload.get("ok") is not True
                or payload.get("tool") != PYTHON_TOOL_ID
                or payload.get("result") != EXPECTED["python"]
            ):
                raise MatrixError(f"Expected direct MASON_PYTHON_OK invocation: {payload}")

        self._run_validation(
            framework,
            authoring,
            "direct_custom_tool_run",
            run_argv,
            f"exit 0 with result {EXPECTED['python']}",
            validate_run,
        )

    def run_dev(self, case: ProjectCase, port: int) -> None:
        label = f"dev-{case.framework}-{case.authoring}"
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.mason),
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
        if case.app_name not in self.apps:
            self.apps.append(case.app_name)
        try:
            self.run_long(
                label,
                [
                    str(self.mason),
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
            self._grant_function(app)
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

    def _grant_function(self, app: dict[str, Any]) -> None:
        principal = app.get("service_principal_client_id")
        if not principal or self.uc_function is None:
            raise MatrixError(f"App response has no service_principal_client_id: {app}")
        catalog, schema, function_name = self.uc_function.split(".")
        quoted_principal = f"`{str(principal).replace('`', '``')}`"
        for statement in (
            f"GRANT USE CATALOG ON CATALOG `{catalog}` TO {quoted_principal}",
            f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema}` TO {quoted_principal}",
            f"GRANT EXECUTE ON FUNCTION `{catalog}`.`{schema}`.`{function_name}` TO {quoted_principal}",
        ):
            self.sql(statement)

    def _exercise(
        self,
        case: ProjectCase,
        runtime: str,
        base_url: str,
        headers: dict[str, str],
        log_path: pathlib.Path,
        app_name: str | None = None,
    ) -> None:
        invocation_url = f"{base_url}{'/api' if runtime == 'deploy' else ''}/invocations"
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
        for attempt in range(1, 4):
            try:
                return _monitored(
                    label,
                    lambda: _http_json(
                        url, {"input": [{"role": "user", "content": prompt}]}, headers
                    ),
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
            "profile": self.profile,
            "app_auth_profile": self.app_auth_profile,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wheel": str(self.wheel),
            "wheel_sha256": _sha256(self.wheel),
            "uc_function": self.uc_function,
            "warehouse_id": self.warehouse_id,
            "validation_checks": [dataclasses.asdict(check) for check in self.validation_checks],
            "rows": [dataclasses.asdict(row) for row in self.rows],
            "cleanup": [dataclasses.asdict(check) for check in self.cleanup_checks],
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def cleanup(self) -> bool:
        for app in self.apps:
            argv = ["databricks", "apps", "delete", app, "--profile", self.profile]
            result: subprocess.CompletedProcess[str] | None = None
            error: str | None = None
            try:
                result = self.run(argv, timeout=600, check=False)
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "no command output").strip()
                    error = f"App delete exited {result.returncode}: {detail}"
            except Exception as exc:
                error = str(exc)
            self.cleanup_checks.append(
                CleanupCheck(
                    resource_kind="app",
                    resource=app,
                    status="pass" if error is None else "fail",
                    command=shlex.join(argv),
                    return_code=result.returncode if result is not None else -1,
                    stdout=result.stdout if result is not None else "",
                    stderr=result.stderr if result is not None else "",
                    error=error,
                )
            )
            self._write_evidence()
        if self.uc_function:
            catalog, schema, function_name = self.uc_function.split(".")
            statement = f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`"
            error = None
            try:
                self.sql(statement)
            except Exception as exc:
                error = str(exc)
            self.cleanup_checks.append(
                CleanupCheck(
                    resource_kind="uc_function",
                    resource=self.uc_function,
                    status="pass" if error is None else "fail",
                    command=statement,
                    return_code=0 if error is None else 1,
                    stdout="",
                    stderr="",
                    error=error,
                )
            )
            self._write_evidence()
        return all(check.status == "pass" for check in self.cleanup_checks)


def _last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _json_command_output(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MatrixError(f"Command returned invalid JSON: {result.stdout[:2000]}") from exc
    if not isinstance(payload, dict):
        raise MatrixError(f"Command returned {type(payload).__name__}, expected a JSON object.")
    return payload


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
    tool_evidence = any(value in lowered for value in ("web_search", "web search", "search"))
    if not tool_evidence or "https" not in lowered or len(serialized) < 80:
        raise MatrixError(f"Missing web-search execution/result evidence: {serialized[:2000]}")


def _curl_command(invocation_url: str, prompt: str, authenticated: bool) -> str:
    auth = " -H 'Authorization: Bearer <redacted>'" if authenticated else ""
    body = json.dumps({"input": [{"role": "user", "content": prompt}]})
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


def _execution_evidence_succeeded(document: dict[str, Any]) -> bool:
    rows = document.get("rows", [])
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
    validation_checks = document.get("validation_checks", [])
    expected_validations = {
        (framework, "cli", check) for framework in FRAMEWORKS for check in VALIDATION_CHECKS
    }
    actual_validations = {
        (check["framework"], check["authoring"], check["check"]) for check in validation_checks
    }
    return (
        actual == expected
        and len(rows) == len(actual)
        and all(row.get("status") == "pass" for row in rows)
        and actual_validations == expected_validations
        and len(validation_checks) == len(actual_validations)
        and all(check.get("status") == "pass" for check in validation_checks)
    )


def verify_evidence(path: pathlib.Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows", [])
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
    validation_checks = document.get("validation_checks", [])
    expected_validations = {
        (framework, "cli", check) for framework in FRAMEWORKS for check in VALIDATION_CHECKS
    }
    actual_validations = {
        (check["framework"], check["authoring"], check["check"]) for check in validation_checks
    }
    validation_duplicates = len(validation_checks) - len(actual_validations)
    validation_passed = sum(check.get("status") == "pass" for check in validation_checks)
    validation_failed = sum(check.get("status") == "fail" for check in validation_checks)
    validation_skipped = len(expected_validations - actual_validations)
    sys.stdout.write(
        "validation: "
        f"{validation_passed} passed, {validation_failed} failed, "
        f"{validation_skipped} skipped\n"
    )
    rows_ok = actual == expected and not duplicates and passed == len(expected)
    validations_ok = (
        actual_validations == expected_validations
        and not validation_duplicates
        and validation_passed == len(expected_validations)
    )
    cleanup = document.get("cleanup", [])
    expected_apps = {
        row["app_name"]
        for row in rows
        if row.get("runtime") == "deploy" and isinstance(row.get("app_name"), str)
    }
    app_identities_ok = len(expected_apps) == len(FRAMEWORKS) * len(AUTHORING_PATHS)
    expected_cleanup = {("app", app) for app in expected_apps}
    uc_function = document.get("uc_function")
    uc_identity_ok = isinstance(uc_function, str) and bool(uc_function)
    if uc_identity_ok:
        expected_cleanup.add(("uc_function", uc_function))
    actual_cleanup = {(check["resource_kind"], check["resource"]) for check in cleanup}
    cleanup_duplicates = len(cleanup) - len(actual_cleanup)
    cleanup_passed = sum(
        check.get("status") == "pass" and check.get("return_code") == 0 for check in cleanup
    )
    cleanup_failed = len(cleanup) - cleanup_passed
    cleanup_skipped = len(expected_cleanup - actual_cleanup)
    sys.stdout.write(
        f"cleanup: {cleanup_passed} passed, {cleanup_failed} failed, {cleanup_skipped} skipped\n"
    )
    cleanup_ok = (
        app_identities_ok
        and uc_identity_ok
        and actual_cleanup == expected_cleanup
        and not cleanup_duplicates
        and cleanup_passed == len(expected_cleanup)
    )
    if not rows_ok or not validations_ok or not cleanup_ok:
        if expected - actual:
            sys.stdout.write(f"missing cells: {sorted(expected - actual)}\n")
        if duplicates:
            sys.stdout.write(f"duplicate rows: {duplicates}\n")
        if expected_validations - actual_validations:
            sys.stdout.write(
                f"missing validation checks: {sorted(expected_validations - actual_validations)}\n"
            )
        if validation_duplicates:
            sys.stdout.write(f"duplicate validation checks: {validation_duplicates}\n")
        if expected_cleanup - actual_cleanup:
            sys.stdout.write(
                f"missing cleanup checks: {sorted(expected_cleanup - actual_cleanup)}\n"
            )
        if cleanup_duplicates:
            sys.stdout.write(f"duplicate cleanup checks: {cleanup_duplicates}\n")
        if not app_identities_ok:
            sys.stdout.write("cleanup evidence must identify one App per deploy authoring lane\n")
        if not uc_identity_ok:
            sys.stdout.write("cleanup evidence must identify the created UC function\n")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="df1")
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--warehouse-id")
    parser.add_argument("--uc-schema", default="main.mason_agent_tools_e2e")
    parser.add_argument("--template-repo")
    parser.add_argument("--template-ref")
    parser.add_argument(
        "--app-auth-profile",
        help="OAuth profile for deployed App /api calls; defaults to --profile.",
    )
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args()
    if args.verify_evidence is None and (args.wheel is None or args.output is None):
        parser.error("--wheel and --output are required unless --verify-evidence is used")
    if bool(args.template_repo) != bool(args.template_ref):
        parser.error("--template-repo and --template-ref must be provided together")
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
    )
    try:
        runner.bootstrap()
        runner.select_warehouse(args.warehouse_id)
        runner.create_uc_function(args.uc_schema)
        cases = runner.create_projects()
        for index, case in enumerate(cases):
            runner.run_dev(case, 8400 + index)
        for case in cases:
            runner.deploy(case)
        runner._write_evidence()
    except Exception:
        runner.transcript.write(
            "cleanup retained | run failed before complete evidence; preserving Apps and UC function"
        )
        runner._write_evidence()
        raise

    evidence_path = runner.output / "evidence.json"
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not _execution_evidence_succeeded(document):
        runner.transcript.write(
            "cleanup retained | semantic or validation checks failed; preserving Apps and UC function"
        )
        verify_evidence(evidence_path)
        return 1
    if args.keep_resources:
        runner.transcript.write(
            "cleanup retained | --keep-resources requested; cleanup proof is intentionally absent"
        )
        verify_evidence(evidence_path)
        return 1

    runner.cleanup()
    runner._write_evidence()
    return verify_evidence(evidence_path)


if __name__ == "__main__":
    sys.exit(main())
