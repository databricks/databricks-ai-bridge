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
                if authoring == "cli":
                    self._author_cli(project)
                else:
                    self._author_direct(project)
                self._write_python_marker(project)
                app_name = f"mason-tools-{framework[:2]}-{authoring[:2]}-{run_suffix}"
                cases.append(ProjectCase(framework, authoring, project, app_name))
        return cases

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
            ["tools", "add", "python", "matrix-marker"],
        ]
        for args in commands:
            self.run([str(self.mason), *args, "--source", str(project)])

    def _author_direct(self, project: pathlib.Path) -> None:
        fixture = pathlib.Path(__file__).parent / "fixtures" / "direct_databricks_tools.py"
        registry = fixture.read_text(encoding="utf-8").replace(
            "__UC_FUNCTION__", self.uc_function or ""
        )
        target = project / "agent" / "databricks_tools.py"
        self.transcript.file_step(target, "direct authoring; no mason tools command")
        target.write_text(registry, encoding="utf-8")

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
            self.apps.append(case.app_name)
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
            "rows": [dataclasses.asdict(row) for row in self.rows],
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def cleanup(self) -> None:
        for app in self.apps:
            self.run(
                ["databricks", "apps", "delete", app, "--profile", self.profile],
                timeout=600,
                check=False,
            )
        if self.uc_function:
            catalog, schema, function_name = self.uc_function.split(".")
            try:
                self.sql(f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`")
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC function | {exc}")


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
    if actual != expected or duplicates or passed != len(expected):
        if expected - actual:
            sys.stdout.write(f"missing cells: {sorted(expected - actual)}\n")
        if duplicates:
            sys.stdout.write(f"duplicate rows: {duplicates}\n")
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
    succeeded = False
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
        succeeded = verify_evidence(runner.output / "evidence.json") == 0
        return 0 if succeeded else 1
    finally:
        if not args.keep_resources and succeeded:
            runner.cleanup()
        elif not succeeded:
            runner.transcript.write(
                "Resources retained after failure for diagnosis; rerun cleanup after fixing."
            )


if __name__ == "__main__":
    sys.exit(main())
