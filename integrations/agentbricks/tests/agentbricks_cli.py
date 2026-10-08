"""The ``agentbricks`` CLI and the agents it produces: the system under test.

Every call here is a real ``agentbricks`` subprocess, so a regression in the CLI fails the test
that makes it. The only non-subprocess part is ``Agent.invoke``, which POSTs to a running agent.
Nothing in this module knows about any particular matrix; tests say what to bind.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import tomli
import tomlkit
from common import (
    E2E_MODEL,
    FRAMEWORKS,
    Evidence,
    Inputs,
    MatrixError,
    Transcript,
    last_lines,
    last_nonempty_line,
    now,
)
from workspace_client import Workspace, cleanup_app

_STORE_KINDS = {"memory": ("memory", "stores"), "sessions": ("sessions", "stores")}
_DIRECT_HEADER = (
    'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "agentbricks"\n'
)


class ToolSpec(Protocol):
    """What ``write_manifest`` needs from a tool: its [[tools]] TOML and any user-owned files."""

    toml: str
    files: Mapping[str, str]


@dataclasses.dataclass
class Project:
    authoring: str
    path: pathlib.Path
    app_name: str
    memory_store_config: str | None = None
    session_store_config: str | None = None
    memory_resource: dict[str, Any] | None = None
    session_resource: dict[str, Any] | None = None
    memory_store_name: str | None = None
    session_store_name: str | None = None
    experiment_name: str | None = None
    # Set only once this project registered the App, so cleanup never deletes one it did not create.
    app_registered: bool = False
    environment_prepared: bool = False


class Agent:
    """A running agent: a dev server or a deployed App."""

    def __init__(
        self,
        base_url: str,
        headers: Callable[[], dict[str, str]],
        log_path: pathlib.Path,
        transcript: Transcript,
        *,
        app_name: str | None = None,
        app: dict[str, Any] | None = None,
    ):
        self.base_url = base_url
        self.log_path = log_path
        self.app_name = app_name
        self.app = app
        self._headers = headers
        self._transcript = transcript

    @property
    def runtime(self) -> str:
        return "deploy" if self.app_name else "dev"

    def invoke(self, prompt: str, label: str = "invoke") -> str:
        """POST the prompt to /api/invocations and return the serialized response."""
        url = f"{self.base_url}/api/invocations"
        last: Exception | None = None
        invocation_id = str(uuid.uuid4())
        body = {
            "id": invocation_id,
            # session_id is a top-level invocation field; the adapter rejects it nested in input.
            "session_id": invocation_id,
            "input": {
                "model": E2E_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for attempt in range(1, 4):
            try:
                response = _monitored(
                    label,
                    lambda: _http_json(url, body, self._headers()),
                    self._transcript,
                    timeout=360,
                )
                return json.dumps(response, sort_keys=True, default=str)
            except Exception as exc:
                last = exc
                self._transcript.write(f"attempt {attempt}/3 | {label} | {exc}")
                if attempt < 3:
                    time.sleep(15)
        raise MatrixError(f"{label} failed after 3 attempts: {last}")

    def warm_up(self, deadline_seconds: float = 180.0) -> None:
        """Invoke until the App answers once. A new App's session store can drop the first calls."""
        started = time.monotonic()
        while True:
            try:
                self.invoke("Reply with the single word OK.", label="warm-up")
                return
            except MatrixError:
                if time.monotonic() - started > deadline_seconds:
                    raise

    def curl(self, prompt: str) -> str:
        """A redacted curl equivalent of ``invoke``, recorded as the evidence row's command."""
        auth = " -H 'Authorization: Bearer <redacted>'" if self.app_name else ""
        body = json.dumps(
            {
                "id": "<client-generated-uuid>",
                "session_id": "<stable-session-id>",
                "input": {
                    "model": E2E_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
        return (
            f"curl -sS -X POST {shlex.quote(f'{self.base_url}/api/invocations')}"
            f" -H 'Content-Type: application/json'{auth} --data {shlex.quote(body)}"
        )


class AgentbricksCli:
    """Authors, runs and deploys projects with the CLI, and deletes what it created on ``close``."""

    def __init__(
        self,
        inputs: Inputs,
        workspace: Workspace,
        transcript: Transcript,
        evidence: Evidence,
    ):
        self.inputs = inputs
        self.workspace = workspace
        self.transcript = transcript
        self.evidence = evidence
        self.logs_dir = inputs.output / "logs"
        self._projects: list[Project] = []
        self._sequence = 0

    # Process plumbing

    def argv(self, *args: str) -> list[str]:
        profile = (
            ["--profile", self.inputs.databricks_profile] if self.inputs.databricks_profile else []
        )
        return [str(self.inputs.agentbricks_bin), *profile, *args]

    def cli(
        self,
        *args: str,
        cwd: pathlib.Path | None = None,
        timeout: float = 300,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = self.argv(*args)
        self.transcript.command(argv, cwd)
        result = subprocess.run(
            argv, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False
        )
        if result.stdout.strip():
            self.transcript.write(result.stdout)
        if result.stderr.strip():
            self.transcript.write(result.stderr)
        if check and result.returncode != 0:
            raise MatrixError(
                f"Command failed ({result.returncode}): {shlex.join(argv)}\n"
                f"{result.stderr or result.stdout}"
            )
        return result

    def _run_long(self, label: str, argv: Sequence[str], *, timeout: float) -> None:
        self.transcript.command(argv)
        log_path = self.logs_dir / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                list(argv),
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    terminate_process_group(process, grace=0)
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s; log: {log_path}")
                if elapsed >= next_tick:
                    self.transcript.write(
                        f"tick {now():%H:%M} | {label} | running | {last_nonempty_line(log_path)}"
                    )
                    next_tick += 60.0
                time.sleep(2)
        self.transcript.write(log_path.read_text(encoding="utf-8", errors="replace"))
        if process.returncode != 0:
            raise MatrixError(f"{label} failed ({process.returncode}); log: {log_path}")
        self.transcript.write(f"tick {now():%H:%M} | {label} | success")

    def _next(self) -> int:
        self._sequence += 1
        return self._sequence

    # Authoring

    def new_project(self, authoring: str) -> Project:
        """`agentbricks init` a uniquely named project and pin the runtime under test.

        `init` binds a session store, a memory store and a tracing experiment by default, and CLI
        authoring keeps them. Direct authoring replaces the whole manifest in ``write_manifest``.
        """
        path = self.inputs.output / "projects" / f"t-{uuid.uuid4().hex[:8]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        project = Project(authoring, path, f"agent-bricks-{path.name}")
        self._projects.append(project)
        self._run_long(
            f"init-{path.name}",
            self.argv("init", "--framework", FRAMEWORKS[0], str(path)),
            timeout=600,
        )
        if self.inputs.bridge_sha:
            self._pin_bridge_sources(path)
        else:
            self._pin_project_wheel(path)
        # Registered before any store exists so cleanup knows the app name if creation fails.
        self.evidence.register_project(project.app_name, authoring=authoring)
        if authoring == "cli":
            self._sync_manifest(project)
        return project

    def tools_add(
        self, project: Project, *args: str, check: bool = True, json_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self._tools("add", project, args, check, json_output)

    def tools_remove(
        self, project: Project, *args: str, check: bool = True, json_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self._tools("remove", project, args, check, json_output)

    def _tools(
        self,
        verb: str,
        project: Project,
        args: Sequence[str],
        check: bool,
        json_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        prefix = ("--output", "json") if json_output else ()
        return self.cli(*prefix, "tools", verb, *args, "--source", str(project.path), check=check)

    def manifest(self, project: Project) -> dict[str, Any]:
        return tomli.loads((project.path / "agent.toml").read_text(encoding="utf-8"))

    def write_file(self, project: Project, relative_path: str, content: str) -> None:
        """Write a user-owned file in the project, as a developer would."""
        target = project.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.transcript.file_step(target, "user-owned file")
        target.write_text(content, encoding="utf-8")

    def write_manifest(self, project: Project, tool_specs: Sequence[ToolSpec]) -> None:
        """Direct authoring: write agent.toml with exactly these tools plus a tracing binding."""
        sections: list[str] = [_DIRECT_HEADER.format(framework=FRAMEWORKS[0])]
        sections.extend(spec.toml for spec in tool_specs if spec.toml)
        # Follows default_experiment_name's shape so direct authoring carries the same tracing
        # binding `agentbricks init` gives CLI authoring.
        slug = re.sub(r"[^a-z0-9-]+", "-", project.path.name.lower()).strip("-") or "agent"
        experiment = f"/Shared/agentbricks_traces/{slug}-{self.inputs.run_suffix}"
        sections.append(f'[tracing]\nexperiment_name = "{experiment}"\n')
        target = project.path / "agent.toml"
        self.transcript.file_step(target, "direct authoring; no agentbricks tools command")
        target.write_text("\n".join(sections), encoding="utf-8")
        for spec in tool_specs:
            for relative_path, content in spec.files.items():
                self.write_file(project, relative_path, content)
        self._sync_manifest(project)

    def _sync_manifest(self, project: Project) -> None:
        """Read the authored agent.toml and create the stores it declares."""
        manifest = self.manifest(project)
        project.memory_store_config = manifest.get("memory_store", {}).get("name")
        project.session_store_config = manifest.get("session_store", {}).get("name")
        project.experiment_name = manifest.get("tracing", {}).get("experiment_name")
        if project.memory_store_config and not project.memory_store_name:
            result = self.cli(
                "--output",
                "json",
                "memory",
                "stores",
                "create",
                "--name",
                project.memory_store_config,
            )
            project.memory_resource = json.loads(result.stdout)
            name = project.memory_resource.get("name")
            if isinstance(name, str) and name:
                project.memory_store_name = name
        if project.session_store_config and not project.session_store_name:
            result = self.cli(
                "--output",
                "json",
                "sessions",
                "stores",
                "create",
                "--name",
                project.session_store_config,
            )
            project.session_resource = json.loads(result.stdout)
            project.session_store_name = project.session_store_config
        self.evidence.register_project(
            project.app_name,
            authoring=project.authoring,
            memory_store_name=project.memory_store_name,
            session_store_name=project.session_store_name,
        )

    def _pin_project_wheel(self, project: pathlib.Path) -> None:
        wheel = self.inputs.wheel
        vendor_dir = project / "agentbricks_e2e_wheels"
        vendor_dir.mkdir()
        vendored_wheel = vendor_dir / wheel.name
        shutil.copy2(wheel, vendored_wheel)
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

    def _pin_bridge_sources(self, project: pathlib.Path) -> None:
        bridge_sha = self.inputs.bridge_sha
        pyproject = project / "pyproject.toml"
        document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
        dependencies = document["project"]["dependencies"]
        if not any(
            str(dependency).startswith("databricks-langchain") for dependency in dependencies
        ):
            dependencies.append("databricks-langchain>=0.17.0")
        if "tool" not in document:
            document["tool"] = tomlkit.table()
        if "uv" not in document["tool"]:
            document["tool"]["uv"] = tomlkit.table()
        if "sources" not in document["tool"]["uv"]:
            document["tool"]["uv"]["sources"] = tomlkit.table()
        for package, subdirectory in (
            ("databricks-agentbricks", "integrations/agentbricks"),
            ("databricks-langchain", "integrations/langchain"),
        ):
            source = tomlkit.inline_table()
            source.update(
                {
                    "git": "https://github.com/databricks/databricks-ai-bridge.git",
                    "rev": bridge_sha,
                    "subdirectory": subdirectory,
                }
            )
            document["tool"]["uv"]["sources"][package] = source
        self.transcript.file_step(pyproject, f"pin Agent Bricks and LangChain to {bridge_sha}")
        pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")

    # Running

    @contextlib.contextmanager
    def dev(self, project: Project) -> Iterator[Agent]:
        """`agentbricks dev` the project locally for the duration of the block."""
        label = f"dev-{project.path.name}-{self._next()}"
        log_path = self.logs_dir / f"{label}.log"
        port = _free_port()
        # Only the first start prepares the venv; run-local refuses to recreate an existing one.
        prepare = [] if project.environment_prepared else ["--prepare-environment"]
        argv = self.argv("dev", "--source", str(project.path), "--app-port", str(port), *prepare)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript.command(argv)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                # The template server binds $PORT, so an inherited PORT would override --app-port.
                env={**os.environ, "PORT": str(port)},
            )
        try:
            self._wait_for_local(process, port, label, log_path)
            project.environment_prepared = True
            yield Agent(f"http://127.0.0.1:{port}", lambda: {}, log_path, self.transcript)
        finally:
            if process.poll() is None:
                terminate_process_group(process)
            self.transcript.write(f"{label} | stopped")

    def deploy(self, project: Project, app: str | None = None) -> Agent:
        """`agentbricks deploy` the project; deploying again updates the same App."""
        name = app or project.app_name
        creating = not project.app_registered
        if creating:
            self.workspace.assert_app_absent(name)
            # Deploy can create the App and then fail while waiting for it, so register first.
            self.evidence.register_app(name)
            project.app_registered = True
        label = f"deploy-{name}-{self._next()}"
        self._run_long(
            label,
            self.argv("deploy", name, "--source", str(project.path)),
            timeout=2400,
        )
        deployed = self.workspace.wait_for_app(name)
        url = str(deployed.get("url") or "").rstrip("/")
        if not url:
            raise MatrixError(f"App {name} has no URL: {deployed}")
        agent = Agent(
            url,
            lambda: self.workspace.app_headers,
            self.logs_dir / f"{label}.log",
            self.transcript,
            app_name=name,
            app=deployed,
        )
        if creating:
            agent.warm_up()
        return agent

    def _wait_for_local(
        self, process: subprocess.Popen[str], port: int, label: str, log_path: pathlib.Path
    ) -> None:
        started = time.monotonic()
        next_tick = 60.0
        while True:
            if process.poll() is not None:
                raise MatrixError(
                    f"{label} exited {process.returncode}: {last_lines(log_path, 30)}"
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
                raise MatrixError(f"{label} did not become reachable: {last_lines(log_path, 30)}")
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {now():%H:%M} | {label} | starting | {last_nonempty_line(log_path)}"
                )
                next_tick += 60
            time.sleep(5)

    # Cleanup

    def delete_store(self, kind: str, name: str) -> subprocess.CompletedProcess[str]:
        return delete_store(
            self.inputs.agentbricks_bin, self.inputs.databricks_profile, self.transcript, kind, name
        )

    def close(self) -> None:
        """Delete every App, store and role this CLI created; failures are left for the sweep."""
        for project in self._projects:
            if project.app_registered:
                self._attach_app_logs(project)
            info = {
                "memory_store_name": project.memory_store_name,
                "session_store_name": project.session_store_name,
            }
            try:
                results = cleanup_app(
                    self.workspace,
                    self.delete_store,
                    project.app_name,
                    info,
                    has_app=project.app_registered,
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | {project.app_name} | {exc}")
                continue
            if any(result.get("status") == "failed" for result in results):
                self.transcript.write(
                    f"cleanup incomplete | {project.app_name} | deferred to the runner sweep"
                )
                continue
            self.evidence.mark_cleaned(project.app_name, results)

    def _attach_app_logs(self, project: Project) -> None:
        # The App is deleted next, so a failed cell must capture its logs while it exists.
        failed = any(
            row["status"] == "fail" and row.get("app_name") == project.app_name
            for row in self.evidence.read_rows()
        )
        if not failed:
            return
        log_path = self.workspace.app_logs(
            project.app_name, self.logs_dir / f"deploy-runtime-{project.app_name}.log"
        )
        if log_path is not None:
            self.evidence.attach_artifact_to_failed_rows(project.app_name, log_path)


def delete_store(
    agentbricks: pathlib.Path | str,
    profile: str | None,
    transcript: Transcript,
    kind: str,
    name: str,
) -> subprocess.CompletedProcess[str]:
    """`agentbricks <memory|sessions> stores delete`; also used by the runner's final sweep."""
    command = _STORE_KINDS[kind]
    argv = [
        str(agentbricks),
        *(["--profile", profile] if profile else []),
        *command,
        "delete",
        name,
        "--yes",
    ]
    transcript.command(argv)
    result = subprocess.run(argv, text=True, capture_output=True, timeout=600, check=False)
    for text in (result.stdout, result.stderr):
        if text.strip():
            transcript.write(text)
    return result


def terminate_process_group(process: subprocess.Popen[str], *, grace: float = 20) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if grace <= 0:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
                transcript.write(f"tick {now():%H:%M} | {label} | running | {elapsed:.0f}s")
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
