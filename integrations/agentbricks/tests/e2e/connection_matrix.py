#!/usr/bin/env python3
"""Run the governed UC Connection setup × transport × framework × execution matrix."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urljoin

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ConnectionType

Setup = Literal["new", "existing"]
Transport = Literal["mcp", "http"]
Framework = Literal["langgraph", "openai"]
Execution = Literal["foreground", "background"]

SETUPS: tuple[Setup, ...] = ("new", "existing")
TRANSPORTS: tuple[Transport, ...] = ("mcp", "http")
FRAMEWORKS: tuple[Framework, ...] = ("langgraph", "openai")
EXECUTIONS: tuple[Execution, ...] = ("foreground", "background")

_SENSITIVE = (
    re.compile(r"SENTINEL-[A-Z0-9_-]+", re.IGNORECASE),
    re.compile(r"authorization\s*:\s*bearer\s+\S+", re.IGNORECASE),
    re.compile(r"cookie\s*:\s*\S+", re.IGNORECASE),
    re.compile(r'"(?:access_token|refresh_token|client_secret)"\s*:', re.IGNORECASE),
)


class MatrixError(RuntimeError):
    """A reproducible harness, environment, or matrix failure."""


@dataclasses.dataclass(frozen=True)
class MatrixCase:
    setup: Setup
    transport: Transport
    framework: Framework
    execution: Execution

    @property
    def id(self) -> str:
        return f"{self.setup}-{self.transport}-{self.framework}-{self.execution}"


@dataclasses.dataclass(frozen=True)
class DeploymentCase:
    setup: Setup
    framework: Framework


@dataclasses.dataclass(frozen=True)
class Response:
    status_code: int
    body: dict[str, Any]


def matrix_cases() -> tuple[MatrixCase, ...]:
    """Return the exact primary 16-cell matrix in stable order."""
    return tuple(
        MatrixCase(*values)
        for values in itertools.product(SETUPS, TRANSPORTS, FRAMEWORKS, EXECUTIONS)
    )


def deployment_cases() -> tuple[DeploymentCase, ...]:
    """Group the 16 cells into four setup/framework App deployments."""
    return tuple(DeploymentCase(setup, framework) for setup in SETUPS for framework in FRAMEWORKS)


def render_probe_files(
    framework: Framework,
    *,
    aliases: Mapping[str, str],
    http_path: str,
    mcp_tool: str,
    freshness_marker: str = "agentbricks-connection-e2e",
) -> dict[str, str]:
    """Render a deterministic native framework tool and its Agent Bricks adapter."""
    alias_values = {}
    for transport in TRANSPORTS:
        user_alias = aliases.get(f"user:{transport}", aliases[transport])
        alias_values[f"user:{transport}"] = user_alias
        alias_values[f"app:{transport}"] = aliases.get(f"app:{transport}", user_alias)
    alias_literal = repr(alias_values)
    shared_tool = f"""from typing import Any

from databricks_agentkit.auth import context

ALIASES = {alias_literal}


async def _connection_probe(
    transport: str,
    request_id: str,
    principal: str = "user",
    control: str | None = None,
) -> dict[str, Any]:
    print({freshness_marker!r}, transport, principal, flush=True)
    if control == "unknown-alias":
        context.connections.client("agentbricks-e2e-unknown-alias")
        raise AssertionError("unknown alias unexpectedly resolved")
    client = context.connections.client(ALIASES[f"{{principal}}:{{transport}}"])
    if control == "forbidden-header":
        method, path = ("POST", "") if transport == "mcp" else ("GET", "/")
        await client.request(method, path, headers={{"Authorization": "not-a-token"}})
        raise AssertionError("forbidden header unexpectedly succeeded")
    if transport == "http":
        response = await client.request(
            "GET",
            {http_path!r},
            params={{"request_id": request_id, "marker": "agentbricks-connection-e2e"}},
        )
    else:
        response = await client.request(
            "POST",
            "",
            json={{
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {{
                    "name": {mcp_tool!r},
                    "arguments": {{
                        "request_id": request_id,
                        "marker": "agentbricks-connection-e2e",
                    }},
                }},
            }},
        )
    response.raise_for_status()
    return {{
        "transport": transport,
        "status_code": response.status_code,
        "provider": response.json(),
        "freshness_marker": {freshness_marker!r},
    }}
"""
    if framework == "langgraph":
        tool_source = (
            "from langchain_core.tools import tool\n\n"
            + shared_tool
            + '''\n\n@tool
async def connection_probe(
    transport: str,
    request_id: str,
    principal: str = "user",
    control: str | None = None,
) -> dict[str, Any]:
    """Call the governed connection and return its deterministic fixture marker."""
    return await _connection_probe(transport, request_id, principal, control)
'''
        )
        adapter = """from typing import Any

from agent.tools.connection_probe import connection_probe
from databricks_agentkit import InvocationContext


def _transport(value: Any) -> str:
    if not isinstance(value, dict) or value.get("transport") not in {"mcp", "http"}:
        raise ValueError("input.transport must be 'mcp' or 'http'")
    return value["transport"]


async def invoke(value: Any, context: InvocationContext) -> dict[str, Any]:
    transport = _transport(value)
    result = await connection_probe.ainvoke(
        {
            "transport": transport,
            "request_id": context.invocation_id,
            "principal": value.get("principal", "user"),
            "control": value.get("control"),
        }
    )
    await context.emit({"type": "connection.probe", "transport": transport})
    return result


async def recover(value: Any, context: InvocationContext) -> dict[str, Any]:
    return await invoke(value, context)
"""
    else:
        tool_source = (
            "from agents import function_tool\n\n"
            + shared_tool
            + '''\n\n@function_tool
async def connection_probe(
    transport: str,
    request_id: str,
    principal: str = "user",
    control: str | None = None,
) -> dict[str, Any]:
    """Call the governed connection and return its deterministic fixture marker."""
    return await _connection_probe(transport, request_id, principal, control)
'''
        )
        adapter = """import json
from typing import Any

from agent.tools.connection_probe import connection_probe
from agents.tool_context import ToolContext
from databricks_agentkit import InvocationContext


def _transport(value: Any) -> str:
    if not isinstance(value, dict) or value.get("transport") not in {"mcp", "http"}:
        raise ValueError("input.transport must be 'mcp' or 'http'")
    return value["transport"]


async def invoke(value: Any, context: InvocationContext) -> dict[str, Any]:
    transport = _transport(value)
    arguments = json.dumps(
        {
            "transport": transport,
            "request_id": context.invocation_id,
            "principal": value.get("principal", "user"),
            "control": value.get("control"),
        }
    )
    tool_context = ToolContext(
        context=None,
        tool_name=connection_probe.name,
        tool_call_id=context.invocation_id,
        tool_arguments=arguments,
    )
    result = await connection_probe.on_invoke_tool(tool_context, arguments)
    if isinstance(result, str):
        result = json.loads(result)
    await context.emit({"type": "connection.probe", "transport": transport})
    return result


async def recover(value: Any, context: InvocationContext) -> dict[str, Any]:
    return await invoke(value, context)
"""
    return {
        "agent/tools/connection_probe.py": tool_source,
        "runtime/adapter.py": adapter,
    }


def connection_command(
    case: MatrixCase,
    *,
    ab: pathlib.Path,
    profile: str,
    project: pathlib.Path,
    parent: str,
    urls: Mapping[str, str],
    aliases: Mapping[str, str],
    existing: Mapping[str, str],
) -> list[str]:
    """Build the installed-Agent Bricks create/bind command for a matrix case."""
    alias_key = f"{case.setup}:{case.transport}"
    try:
        alias = aliases[alias_key]
    except KeyError as exc:
        raise MatrixError(f"Missing connection alias for {alias_key}") from exc
    common = [
        str(ab),
        "--profile",
        profile,
        "auth",
        "connections",
    ]
    if case.setup == "new":
        try:
            url = urls[case.transport]
        except KeyError as exc:
            raise MatrixError(f"Missing {case.transport} provider URL") from exc
        return [
            *common,
            "create",
            alias,
            "--url",
            url,
            "--transport",
            case.transport,
            "--oauth",
            "dcr",
            "--principal",
            "user",
            "--parent",
            parent,
            "--source",
            str(project),
        ]
    try:
        fqn = existing[case.transport]
    except KeyError as exc:
        raise MatrixError(f"Missing existing {case.transport} UC Connection") from exc
    return [
        *common,
        "bind",
        alias,
        "--uc-connection",
        fqn,
        "--transport",
        case.transport,
        "--principal",
        "user",
        "--source",
        str(project),
    ]


def assert_markers(payload: Any, expected_marker: str, expected_user_marker: str) -> None:
    """Require provider and invoking-user proof in a terminal response."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    if expected_marker not in serialized:
        raise MatrixError(f"Missing provider marker {expected_marker!r}")
    if expected_user_marker not in serialized:
        raise MatrixError(f"Missing invoking-user marker {expected_user_marker!r}")


def expect_denial(response: Response, name: str) -> None:
    """Require a negative control to fail closed without leaking credentials."""
    serialized = json.dumps(response.body, sort_keys=True, default=str)
    _scan_sensitive(serialized)
    if response.status_code < 400:
        raise MatrixError(f"{name} expected an HTTP denial, got {response.status_code}")


def execute_invocation(
    execution: Execution,
    invocation_url: str,
    body: dict[str, Any],
    request: Callable[[str, str, dict[str, Any] | None], Response],
    *,
    expected_marker: str,
    expected_user_marker: str,
    poll_interval: float = 5,
    timeout: float = 900,
) -> dict[str, Any]:
    """Execute one foreground or background invocation through a scripted transport."""
    submitted = request("POST", invocation_url, body)
    if execution == "foreground":
        if submitted.status_code != 200:
            raise MatrixError(
                f"Foreground invocation expected HTTP 200, got {submitted.status_code}"
            )
        terminal = submitted.body
    else:
        if submitted.status_code != 202:
            raise MatrixError(
                f"Background invocation expected HTTP 202, got {submitted.status_code}"
            )
        status_url = submitted.body.get("status_url")
        if not isinstance(status_url, str) or not status_url:
            raise MatrixError("Background invocation response has no status_url")
        status_url = urljoin(invocation_url, status_url)
        deadline = time.monotonic() + timeout
        while True:
            polled = request("GET", status_url, None)
            if polled.status_code != 200:
                raise MatrixError(f"Background status expected HTTP 200, got {polled.status_code}")
            state = polled.body.get("status")
            if state == "completed":
                terminal = polled.body
                break
            if state in {"failed", "cancelled"}:
                raise MatrixError(f"Background invocation ended in state {state!r}")
            if state not in {"queued", "running"}:
                raise MatrixError(f"Background invocation returned unknown state {state!r}")
            if time.monotonic() >= deadline:
                raise MatrixError(f"Background invocation timed out after {timeout:.0f}s")
            time.sleep(poll_interval)
    if terminal.get("status") != "completed":
        raise MatrixError(f"Invocation returned non-terminal status {terminal.get('status')!r}")
    assert_markers(terminal, expected_marker, expected_user_marker)
    return terminal


def _scan_sensitive(text: str) -> None:
    for pattern in _SENSITIVE:
        if pattern.search(text):
            raise MatrixError("Evidence contains sensitive data")


def write_evidence(
    output: pathlib.Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    cleanup: Sequence[Mapping[str, Any]],
    scanned_text: str,
    metadata: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Secret-scan and atomically write machine-readable evidence."""
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
        "rows": [dict(row) for row in rows],
        "controls": [dict(control) for control in controls],
        "cleanup": [dict(item) for item in cleanup],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    _scan_sensitive(scanned_text)
    _scan_sensitive(serialized)
    target = output / "evidence.json"
    temporary = output / "evidence.json.tmp"
    temporary.write_text(serialized + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def resource_name(run_id: str, case: MatrixCase, *, include_execution: bool = False) -> str:
    """Return a stable App/project-safe name for one setup/framework deployment."""
    setup = {"new": "new", "existing": "ex"}[case.setup]
    framework = {"langgraph": "lg", "openai": "oa"}[case.framework]
    parts = ["cx", _slug(run_id), setup, case.transport, framework]
    if include_execution:
        parts.append({"foreground": "fg", "background": "bg"}[case.execution])
    return "-".join(parts)[:63].rstrip("-")


def cleanup_plan(*, apps: Sequence[str], connections: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return every created resource exactly once in deterministic cleanup order."""
    return tuple(
        [("app", name) for name in dict.fromkeys(apps)]
        + [("connection", name) for name in dict.fromkeys(connections)]
    )


def verify_evidence(path: pathlib.Path) -> int:
    """Validate exact matrix coverage plus successful controls and cleanup."""
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows", [])
    expected = {case.id for case in matrix_cases()}
    actual = [row.get("case") for row in rows]
    matrix_ok = (
        len(rows) == len(expected)
        and set(actual) == expected
        and all(row.get("status") == "pass" for row in rows)
    )
    controls = document.get("controls", [])
    controls_ok = bool(controls) and all(item.get("status") == "pass" for item in controls)
    cleanup = document.get("cleanup", [])
    cleanup_ok = bool(cleanup) and all(
        item.get("status") in {"deleted", "not_found"} for item in cleanup
    )
    return 0 if matrix_ok and controls_ok and cleanup_ok else 1


class MatrixRunner:
    """Orchestrate the matrix while keeping live boundaries overridable offline."""

    def __init__(self, *, output: pathlib.Path, keep_resources: bool) -> None:
        self.output = output
        self.keep_resources = keep_resources
        self.rows: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.cleanup_rows: list[dict[str, Any]] = []

    def bootstrap(self) -> None:
        raise NotImplementedError

    def prepare_deployment(self, deployment: DeploymentCase) -> str:
        raise NotImplementedError

    def execute_case(self, case: MatrixCase, app_url: str) -> dict[str, Any]:
        raise NotImplementedError

    def execute_controls(self, app_urls: dict[DeploymentCase, str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def cleanup(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def evidence_metadata(self) -> dict[str, Any]:
        return {}

    def evidence_scan_text(self) -> str:
        return ""

    def run(self) -> int:
        self.output.mkdir(parents=True, exist_ok=True)
        app_urls: dict[DeploymentCase, str] = {}
        preparation_errors: dict[DeploymentCase, str] = {}
        try:
            self.bootstrap()
            for deployment in deployment_cases():
                try:
                    app_urls[deployment] = self.prepare_deployment(deployment)
                except Exception as exc:
                    preparation_errors[deployment] = str(exc)

            for case in matrix_cases():
                deployment = DeploymentCase(case.setup, case.framework)
                if deployment in preparation_errors:
                    self.rows.append(
                        {
                            "case": case.id,
                            "status": "fail",
                            "error": preparation_errors[deployment],
                        }
                    )
                    continue
                try:
                    row = self.execute_case(case, app_urls[deployment])
                except Exception as exc:
                    row = {"case": case.id, "status": "fail", "error": str(exc)}
                self.rows.append(row)
            try:
                self.controls = self.execute_controls(app_urls)
            except Exception as exc:
                self.controls = [{"name": "control-runner", "status": "fail", "error": str(exc)}]
        except Exception as exc:
            existing = {row.get("case") for row in self.rows}
            self.rows.extend(
                {"case": case.id, "status": "fail", "error": str(exc)}
                for case in matrix_cases()
                if case.id not in existing
            )
            self.controls = [{"name": "control-runner", "status": "fail", "error": str(exc)}]
        finally:
            if self.keep_resources:
                self.cleanup_rows = [{"kind": "all", "name": "retained", "status": "retained"}]
            else:
                try:
                    self.cleanup_rows = self.cleanup()
                except Exception as exc:
                    self.cleanup_rows = [
                        {"kind": "cleanup", "name": "runner", "status": "failed", "error": str(exc)}
                    ]
            target = write_evidence(
                self.output,
                rows=self.rows,
                controls=self.controls,
                cleanup=self.cleanup_rows,
                scanned_text=self.evidence_scan_text(),
                metadata=self.evidence_metadata(),
            )
        return verify_evidence(target)


class Transcript:
    """Thread-safe transcript that rejects credentials before writing them."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, value: str) -> None:
        text = value.rstrip() + "\n"
        _scan_sensitive(text)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    def command(self, argv: Sequence[str], cwd: pathlib.Path | None = None) -> None:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        self.write(f"$ {prefix}{shlex.join(list(argv))}")


class LiveMatrixRunner(MatrixRunner):
    """Run the matrix using a built wheel and live Databricks/provider endpoints."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(output=args.output.resolve(), keep_resources=args.keep_resources)
        self.args = args
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_token = uuid.uuid4().hex[:6]
        self.transcript = Transcript(self.output / "commands.log")
        self.runner_venv = self.output / "runner-venv"
        self.ab = self.runner_venv / "bin" / "ab"
        self.workspace: WorkspaceClient | None = None
        self.app_auth: WorkspaceClient | None = None
        self.host = ""
        self.created_apps: list[str] = []
        self.created_connections: list[str] = []
        self.app_names: dict[DeploymentCase, str] = {}
        self.aliases: dict[tuple[Setup, Framework, Transport], str] = {}
        self.connection_fqns: dict[tuple[Setup, Framework, Transport], str] = {}
        self.app_aliases: dict[tuple[Setup, Framework, Transport], str] = {}
        self._logs: list[pathlib.Path] = []
        self._freshness_checked: set[str] = set()

    def _run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 600,
        label: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.transcript.command(argv, cwd)
        label = label or pathlib.Path(argv[0]).name
        log_path = self.output / "logs" / f"{_slug(label)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logs.append(log_path)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s")
                if elapsed >= next_tick:
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | running | "
                        f"{_last_nonempty_line(log_path)}"
                    )
                    next_tick += 60
                time.sleep(2)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        _scan_sensitive(output)
        result = subprocess.CompletedProcess(list(argv), process.returncode, output, "")
        state = "success" if result.returncode == 0 else f"failed:{result.returncode}"
        self.transcript.write(f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | {state}")
        if check and result.returncode != 0:
            raise MatrixError(f"{label} failed with exit code {result.returncode}; log: {log_path}")
        return result

    def bootstrap(self) -> None:
        wheel = self.args.wheel.resolve()
        if not wheel.is_file():
            raise MatrixError(f"Agent Bricks wheel does not exist: {wheel}")
        if self.output.exists() and any(self.output.iterdir()):
            allowed = {"commands.log", "logs"}
            unexpected = {path.name for path in self.output.iterdir()} - allowed
            if unexpected:
                raise MatrixError(f"Output directory is not empty: {self.output}")
        self._validate_inputs()
        self._run(["uv", "venv", str(self.runner_venv)], label="runner-venv")
        self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.runner_venv / "bin" / "python"),
                str(wheel),
            ],
            timeout=900,
            label="install-wheel",
        )
        self._run([str(self.ab), "auth", "connections", "--help"], label="ab-help")
        self.workspace = WorkspaceClient(profile=self.args.profile)
        self.app_auth = WorkspaceClient(profile=self.args.app_auth_profile)
        self.host = str(self.workspace.config.host or "").rstrip("/")
        app_host = str(self.app_auth.config.host or "").rstrip("/")
        if not self.host or app_host != self.host:
            raise MatrixError("Workspace and App-auth profiles must resolve to the same host")
        if self.app_auth.config.auth_type == "pat":
            raise MatrixError("--app-auth-profile must use OAuth, not PAT")
        authorization = self.app_auth.config.authenticate().get("Authorization")
        if not authorization:
            raise MatrixError("Could not resolve App invocation OAuth credentials")
        del authorization
        self._validate_existing_connections()

    def _validate_inputs(self) -> None:
        parent_parts = self.args.parent.split(".")
        if len(parent_parts) != 2 or not all(parent_parts):
            raise MatrixError("--parent must be a two-part catalog.schema name")
        for label, value in (("MCP", self.args.mcp_url), ("HTTP", self.args.http_url)):
            if not value.startswith("https://"):
                raise MatrixError(f"{label} fixture URL must use HTTPS")
        for value in (
            self.args.existing_mcp_connection,
            self.args.existing_http_connection,
        ):
            if len(value.split(".")) != 3:
                raise MatrixError("Existing UC Connection names must have three parts")

    def _validate_existing_connections(self) -> None:
        assert self.workspace is not None
        for transport, fqn in (
            ("mcp", self.args.existing_mcp_connection),
            ("http", self.args.existing_http_connection),
        ):
            try:
                info = self.workspace.connections.get(fqn)
            except NotFound as exc:
                raise MatrixError(f"Existing {transport} UC Connection was not found") from exc
            if info.connection_type != ConnectionType.HTTP:
                raise MatrixError(f"Existing {transport} binding is not a UC HTTP Connection")

    def _alias(self, deployment: DeploymentCase, transport: Transport) -> str:
        key = (deployment.setup, deployment.framework, transport)
        if key in self.aliases:
            return self.aliases[key]
        prefix = getattr(self.args, f"{deployment.setup}_{transport}_alias")
        framework = "lg" if deployment.framework == "langgraph" else "oa"
        value = _slug(f"{prefix}-{framework}-{self.run_token}")[:63].rstrip("-")
        self.aliases[key] = value
        return value

    def _deployment_name(self, deployment: DeploymentCase) -> str:
        setup = "new" if deployment.setup == "new" else "ex"
        framework = "lg" if deployment.framework == "langgraph" else "oa"
        return _slug(f"cx-{self.run_token}-{setup}-{framework}")[:45].rstrip("-")

    def prepare_deployment(self, deployment: DeploymentCase) -> str:
        project = self.output / "projects" / f"{deployment.setup}-{deployment.framework}"
        self._run(
            [
                str(self.ab),
                "--profile",
                self.args.profile,
                "init",
                "--framework",
                deployment.framework,
                "--server",
                "agentbricks",
                "--disable-chat-app",
                str(project),
            ],
            timeout=900,
            label=f"init-{deployment.setup}-{deployment.framework}",
        )
        self._install_wheel_source(project)
        self._run([str(self.ab), "memory", "unbind", "--source", str(project)])
        self._run([str(self.ab), "sessions", "unbind", "--source", str(project)])
        rendered_aliases: dict[str, str] = {}
        for transport in TRANSPORTS:
            alias = self._alias(deployment, transport)
            case = MatrixCase(deployment.setup, transport, deployment.framework, "foreground")
            command = connection_command(
                case,
                ab=self.ab,
                profile=self.args.profile,
                project=project,
                parent=self.args.parent,
                urls={"mcp": self.args.mcp_url, "http": self.args.http_url},
                aliases={f"{deployment.setup}:{transport}": alias},
                existing={
                    "mcp": self.args.existing_mcp_connection,
                    "http": self.args.existing_http_connection,
                },
            )
            self._run(
                command,
                timeout=600,
                label=f"connection-{deployment.setup}-{transport}-{deployment.framework}",
            )
            fqn = (
                f"{self.args.parent}.{alias}"
                if deployment.setup == "new"
                else getattr(self.args, f"existing_{transport}_connection")
            )
            self.connection_fqns[(deployment.setup, deployment.framework, transport)] = fqn
            if deployment.setup == "new":
                self.created_connections.append(fqn)
            app_alias = _slug(f"app-{transport}-{deployment.framework}-{self.run_token}")[:63]
            self._run(
                [
                    str(self.ab),
                    "--profile",
                    self.args.profile,
                    "auth",
                    "connections",
                    "bind",
                    app_alias,
                    "--uc-connection",
                    fqn,
                    "--transport",
                    transport,
                    "--principal",
                    "app",
                    "--source",
                    str(project),
                ],
                timeout=600,
                label=f"connection-app-{transport}-{deployment.framework}",
            )
            self.app_aliases[(deployment.setup, deployment.framework, transport)] = app_alias
            rendered_aliases[transport] = alias
            rendered_aliases[f"user:{transport}"] = alias
            rendered_aliases[f"app:{transport}"] = app_alias
        for relative_path, source in render_probe_files(
            deployment.framework,
            aliases=rendered_aliases,
            http_path=self.args.http_path,
            mcp_tool=self.args.mcp_tool,
            freshness_marker=f"agentbricks-connection-e2e-{self.run_token}",
        ).items():
            target = project / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        deployment_name = self._deployment_name(deployment)
        app_name = f"agent-bricks-{deployment_name}"
        self._run(
            [
                str(self.ab),
                "--profile",
                self.args.profile,
                "--output",
                "json",
                "deploy",
                deployment_name,
                "--source",
                str(project),
            ],
            timeout=2700,
            label=f"deploy-{deployment.setup}-{deployment.framework}",
        )
        self.created_apps.append(app_name)
        self.app_names[deployment] = app_name
        return self._wait_for_app(app_name)

    def _install_wheel_source(self, project: pathlib.Path) -> None:
        vendor = project / "vendor"
        vendor.mkdir(parents=True, exist_ok=True)
        target = vendor / self.args.wheel.name
        shutil.copy2(self.args.wheel, target)
        pyproject = project / "pyproject.toml"
        contents = pyproject.read_text(encoding="utf-8")
        contents = re.sub(
            r'"databricks-agentbricks\[([^]]+)\]>=?[^\"]*"',
            r'"databricks-agentbricks[\1]"',
            contents,
        )
        contents += (
            f'\n[tool.uv.sources]\ndatabricks-agentbricks = {{ path = "vendor/{target.name}" }}\n'
        )
        pyproject.write_text(contents, encoding="utf-8")

    def _wait_for_app(self, name: str) -> str:
        assert self.workspace is not None
        deadline = time.monotonic() + 1200
        next_tick = 0.0
        while time.monotonic() < deadline:
            app = self.workspace.apps.get(name)
            compute = getattr(app, "compute_status", None)
            state = str(getattr(compute, "state", "UNKNOWN"))
            url = str(getattr(app, "url", "") or "").rstrip("/")
            if state.endswith("ACTIVE") and url:
                return url
            elapsed = 1200 - max(0, deadline - time.monotonic())
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | app-{name} | {state}"
                )
                next_tick += 60
            time.sleep(15)
        raise MatrixError(f"App {name!r} did not become ACTIVE")

    def _authorization(self) -> str:
        assert self.app_auth is not None
        value = self.app_auth.config.authenticate().get("Authorization")
        if not value:
            raise MatrixError("Could not refresh App invocation OAuth credentials")
        return value

    def _request(self, method: str, url: str, body: dict[str, Any] | None) -> Response:
        return self._request_with_headers(
            method,
            url,
            body,
            {"Authorization": self._authorization()},
        )

    def _request_with_headers(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> Response:
        headers = dict(headers)
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=340) as raw:
                status = raw.status
                payload = raw.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status = exc.code
            payload = exc.read().decode("utf-8", errors="replace")
        try:
            document = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise MatrixError(f"App returned non-JSON HTTP {status}") from exc
        if not isinstance(document, dict):
            raise MatrixError(f"App returned a non-object HTTP {status} response")
        return Response(status, document)

    def execute_case(self, case: MatrixCase, app_url: str) -> dict[str, Any]:
        invocation_id = str(uuid.uuid4())
        body = {
            "id": invocation_id,
            "input": {"transport": case.transport},
            "background": case.execution == "background",
        }
        started = time.monotonic()
        terminal = execute_invocation(
            case.execution,
            f"{app_url}/api/invocations",
            body,
            self._request,
            expected_marker=getattr(self.args, f"{case.transport}_marker"),
            expected_user_marker=self.args.user_marker,
        )
        freshness_marker = f"agentbricks-connection-e2e-{self.run_token}"
        if freshness_marker not in json.dumps(terminal, sort_keys=True, default=str):
            raise MatrixError("Deployed response did not prove the generated harness code is fresh")
        app_name = self.app_names[DeploymentCase(case.setup, case.framework)]
        if app_name not in self._freshness_checked:
            self._wait_for_freshness_log(app_name, freshness_marker)
            self._freshness_checked.add(app_name)
        elapsed = round(time.monotonic() - started, 3)
        self.transcript.write(
            f"cell {case.id} | pass | invocation={invocation_id} | duration={elapsed}s"
        )
        fqn = self.connection_fqns[(case.setup, case.framework, case.transport)]
        marker = getattr(self.args, f"{case.transport}_marker")
        return {
            "case": case.id,
            "status": "pass",
            "setup": case.setup,
            "transport": case.transport,
            "framework": case.framework,
            "execution": case.execution,
            "connection_alias": self.aliases[(case.setup, case.framework, case.transport)],
            "uc_connection": fqn,
            "app": app_name,
            "invocation_id": invocation_id,
            "terminal_status": terminal.get("status"),
            "duration_seconds": elapsed,
            "marker_sha256": hashlib.sha256(marker.encode()).hexdigest(),
            "user_marker_sha256": hashlib.sha256(self.args.user_marker.encode()).hexdigest(),
        }

    def _wait_for_freshness_log(self, app_name: str, marker: str) -> None:
        deadline = time.monotonic() + 240
        while True:
            result = self._run(
                [
                    "databricks",
                    "apps",
                    "logs",
                    app_name,
                    "--profile",
                    self.args.profile,
                    "--tail-lines",
                    "500",
                    "--search",
                    marker,
                ],
                timeout=120,
                label=f"freshness-{app_name}",
                check=False,
            )
            if marker in result.stdout:
                self.transcript.write(f"freshness {app_name} | verified | marker={marker}")
                return
            if time.monotonic() >= deadline:
                raise MatrixError(f"Freshness marker did not appear in logs for {app_name}")
            self.transcript.write(
                f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | freshness-{app_name} | pending"
            )
            time.sleep(60)

    def execute_controls(self, app_urls: dict[DeploymentCase, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        expected = {case.id for case in matrix_cases()}
        actual = {row.get("case") for row in self.rows if row.get("status") == "pass"}
        status = "pass" if actual == expected else "fail"
        rows.append(
            {
                "name": "exact-primary-matrix",
                "status": status,
                "expected_cells": 16,
                "passed_cells": len(actual),
                "deployed_apps": len(app_urls),
            }
        )
        control_deployment = DeploymentCase("new", "langgraph")
        control_url = app_urls.get(control_deployment)
        if control_url is None:
            rows.append(
                {
                    "name": "live-controls",
                    "status": "fail",
                    "error": "control deployment was unavailable",
                }
            )
            return rows
        for transport in TRANSPORTS:
            invocation_id = str(uuid.uuid4())
            try:
                execute_invocation(
                    "foreground",
                    f"{control_url}/api/invocations",
                    {
                        "id": invocation_id,
                        "input": {"transport": transport, "principal": "app"},
                    },
                    self._request,
                    expected_marker=getattr(self.args, f"{transport}_marker"),
                    expected_user_marker=self.args.app_user_marker or self.args.user_marker,
                )
            except Exception as exc:
                rows.append(
                    {
                        "name": f"app-principal-{transport}",
                        "status": "fail",
                        "error": str(exc),
                    }
                )
            else:
                rows.append({"name": f"app-principal-{transport}", "status": "pass"})
        for name in ("unknown-alias", "forbidden-header"):
            response = self._request(
                "POST",
                f"{control_url}/api/invocations",
                {
                    "id": str(uuid.uuid4()),
                    "input": {"transport": "http", "control": name},
                },
            )
            try:
                expect_denial(response, name)
            except Exception as exc:
                rows.append({"name": name, "status": "fail", "error": str(exc)})
            else:
                rows.append({"name": name, "status": "pass", "http_status": response.status_code})
        missing_identity = self._request_with_headers(
            "POST",
            f"{control_url}/api/invocations",
            {
                "id": str(uuid.uuid4()),
                "input": {"transport": "http"},
            },
            {},
        )
        try:
            expect_denial(missing_identity, "missing-user-identity")
        except Exception as exc:
            rows.append({"name": "missing-user-identity", "status": "fail", "error": str(exc)})
        else:
            rows.append(
                {
                    "name": "missing-user-identity",
                    "status": "pass",
                    "http_status": missing_identity.status_code,
                }
            )
        return rows

    def cleanup(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for kind, name in cleanup_plan(
            apps=self.created_apps, connections=self.created_connections
        ):
            if kind == "app":
                result = self._run(
                    [
                        str(self.ab),
                        "--profile",
                        self.args.profile,
                        "deployments",
                        "delete",
                        name,
                        "--yes",
                    ],
                    timeout=1200,
                    label=f"cleanup-app-{name}",
                    check=False,
                )
                status = "deleted" if result.returncode == 0 else "failed"
            else:
                assert self.workspace is not None
                try:
                    self.workspace.connections.delete(name)
                    status = "deleted"
                except NotFound:
                    status = "not_found"
                except Exception:
                    status = "failed"
            rows.append({"kind": kind, "name": name, "status": status})
        return rows

    def evidence_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profile": self.args.profile,
            "app_auth_profile": self.args.app_auth_profile,
            "wheel": self.args.wheel.name,
            "wheel_sha256": _sha256(self.args.wheel),
            "parent": self.args.parent,
            "matrix_axes": {
                "setup": list(SETUPS),
                "transport": list(TRANSPORTS),
                "framework": list(FRAMEWORKS),
                "execution": list(EXECUTIONS),
            },
        }

    def evidence_scan_text(self) -> str:
        parts = []
        for path in [self.transcript.path, *self._logs]:
            if path.exists():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)


def _last_nonempty_line(path: pathlib.Path) -> str:
    if not path.exists():
        return "no output yet"
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile")
    parser.add_argument("--app-auth-profile")
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--parent")
    parser.add_argument("--mcp-url")
    parser.add_argument("--http-url")
    parser.add_argument("--existing-mcp-connection")
    parser.add_argument("--existing-http-connection")
    parser.add_argument("--new-mcp-alias", default="new-mcp")
    parser.add_argument("--new-http-alias", default="new-http")
    parser.add_argument("--existing-mcp-alias", default="existing-mcp")
    parser.add_argument("--existing-http-alias", default="existing-http")
    parser.add_argument("--mcp-marker")
    parser.add_argument("--http-marker")
    parser.add_argument("--user-marker")
    parser.add_argument("--app-user-marker")
    parser.add_argument("--mcp-tool", default="agentbricks_connection_probe")
    parser.add_argument("--http-path", default="/agentbricks-e2e")
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args(argv)
    required = (
        "profile",
        "app_auth_profile",
        "wheel",
        "output",
        "parent",
        "mcp_url",
        "http_url",
        "existing_mcp_connection",
        "existing_http_connection",
        "mcp_marker",
        "http_marker",
        "user_marker",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if args.verify_evidence is None and missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_evidence:
        return verify_evidence(args.verify_evidence)
    return LiveMatrixRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
