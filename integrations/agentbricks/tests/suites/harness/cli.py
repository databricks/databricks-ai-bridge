"""The ``agentbricks`` CLI and the agents it produces: the system under test.

Every CLI step is a real ``agentbricks`` subprocess, written as the command a developer would type
(``cli.run("tools add ...")``), so a regression in the CLI fails the test that makes it. The only
non-subprocess part is ``AgentEndpoint.invoke``, which POSTs to a running agent. Nothing in this
module knows about any particular suite. It records the projects it creates in ``projects`` but
deletes nothing: the ``agentbricks_cli`` fixture does.
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
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import tomli
import tomlkit

from harness.logs import last_lines, last_nonempty_line, log, log_command
from harness.run import APP_PREFIX, E2E_MODEL, FRAMEWORKS, E2EError, RunConfig, now, project_prefix
from harness.target import TargetWorkspace
from harness.workspace import Workspace

_DIRECT_HEADER = (
    'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "agentbricks"\n'
)
# The Apps gateway's answers while an App is not up, e.g. restarting after a redeploy.
_GATEWAY_UNAVAILABLE = re.compile(r"HTTP 50[234]\b")


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

    def manifest(self) -> dict[str, Any]:
        return tomli.loads((self.path / "agent.toml").read_text(encoding="utf-8"))

    def write_file(self, relative_path: str, content: str) -> None:
        """Write a user-owned file in the project, as a developer would."""
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"# write {target}: user-owned file")
        target.write_text(content, encoding="utf-8")

    def write_manifest(self, tool_specs: Sequence[ToolSpec]) -> None:
        """Direct authoring: write agent.toml with exactly these tools plus a tracing binding."""
        sections: list[str] = [_DIRECT_HEADER.format(framework=FRAMEWORKS[0])]
        sections.extend(spec.toml for spec in tool_specs if spec.toml)
        # Follows default_experiment_name's shape so direct authoring carries the same tracing
        # binding `agentbricks init` gives CLI authoring.
        slug = re.sub(r"[^a-z0-9-]+", "-", self.path.name.lower()).strip("-") or "agent"
        self.experiment_name = f"/Shared/agentbricks_traces/{slug}"
        sections.append(f'[tracing]\nexperiment_name = "{self.experiment_name}"\n')
        target = self.path / "agent.toml"
        log(f"# write {target}: direct authoring; no agentbricks tools command")
        target.write_text("\n".join(sections), encoding="utf-8")
        for spec in tool_specs:
            for relative_path, content in spec.files.items():
                self.write_file(relative_path, content)


@dataclasses.dataclass
class BackgroundCommand:
    """A CLI command left running by ``AgentbricksCli.start``, e.g. ``agentbricks dev``."""

    label: str
    process: subprocess.Popen[str]
    log_path: pathlib.Path


class AgentEndpoint:
    """Where a running agent answers, a dev server or a deployed App, and how to call it."""

    def __init__(
        self,
        base_url: str,
        headers: Callable[[], dict[str, str]],
        *,
        app_name: str | None = None,
        app: dict[str, Any] | None = None,
    ):
        self.base_url = base_url
        self.app_name = app_name
        self.app = app
        self._headers = headers

    @classmethod
    def local(cls, server: BackgroundCommand, port: int) -> AgentEndpoint:
        """The `agentbricks dev` server on ``port``, once it answers; fails if the server exits."""
        started = time.monotonic()
        next_tick = 60.0
        while True:
            if server.process.poll() is not None:
                raise E2EError(
                    f"{server.label} exited {server.process.returncode}: "
                    f"{last_lines(server.log_path, 30)}"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5):
                    break
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - started
            if elapsed > 1200:
                raise E2EError(
                    f"{server.label} did not become reachable: {last_lines(server.log_path, 30)}"
                )
            if elapsed >= next_tick:
                log(
                    f"tick {now():%H:%M} | {server.label} | starting | "
                    f"{last_nonempty_line(server.log_path)}"
                )
                next_tick += 60
            time.sleep(5)
        return cls(f"http://127.0.0.1:{port}", lambda: {})

    @classmethod
    def deployed(cls, workspace: Workspace, app_name: str) -> AgentEndpoint:
        """The App once it is up and has answered once."""
        workspace.check_app_auth()
        app = workspace.wait_for_app(app_name)
        url = str(app.get("url") or "").rstrip("/")
        if not url:
            raise E2EError(f"App {app_name} has no URL: {app}")
        endpoint = cls(url, lambda: workspace.app_headers, app_name=app_name, app=app)
        endpoint.warm_up()
        return endpoint

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
                    label, lambda: _http_json(url, body, self._headers()), timeout=360
                )
                return json.dumps(response, sort_keys=True, default=str)
            except Exception as exc:
                last = exc
                log(f"attempt {attempt}/3 | {label} | {exc}")
                if attempt < 3:
                    time.sleep(15)
        raise E2EError(f"{label} failed after 3 attempts: {last}")

    def warm_up(self, deadline_seconds: float = 180.0) -> None:
        """Invoke until the agent answers once. A new App's session store can drop the first calls,
        and a redeployed one restarts."""
        started = time.monotonic()
        while True:
            try:
                self.invoke("Reply with the single word OK.", label="warm-up")
                return
            except E2EError:
                if time.monotonic() - started > deadline_seconds:
                    raise


class AgentbricksCli:
    """Runs ``agentbricks`` commands against a target workspace and authors projects with them."""

    def __init__(self, target_workspace: TargetWorkspace, run_config: RunConfig, test_name: str):
        self.target_workspace = target_workspace
        self.run_config = run_config
        # One directory per test: tests run concurrently and each numbers its commands from 1.
        self.logs_dir = (
            run_config.output / "logs" / re.sub(r"[^A-Za-z0-9_.-]+", "-", test_name).strip("-")
        )
        self.projects: list[Project] = []
        # The CLI prints a hint when it detects a coding agent; the suite must behave the same
        # whoever launches it, and the hint would corrupt --output json errors on stderr.
        self._env = {**os.environ, **target_workspace.env, "AGENTBRICKS_DISABLE_AGENT_HINT": "1"}
        self._sequence = 0

    def run(
        self,
        command: str,
        *,
        timeout: float = 300,
        check: bool = True,
        retry_unavailable: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``agentbricks <command>`` to completion.

        ``retry_unavailable`` retries only the Apps gateway's "App not up" answers (a staging App
        can restart under the test); any other failure is the CLI's.
        """
        argv = self._argv(command)
        for attempt in range(1, 4):
            result = self._run_once(argv, self._label(command), timeout)
            output = result.stderr or result.stdout
            if (
                result.returncode == 0
                or not retry_unavailable
                or attempt == 3
                or _GATEWAY_UNAVAILABLE.search(output) is None
            ):
                break
            log(f"attempt {attempt}/3 | {command.split()[0]} | App unavailable; retrying")
            time.sleep(15)
        if check and result.returncode != 0:
            raise E2EError(f"Command failed ({result.returncode}): {shlex.join(argv)}\n{output}")
        return result

    @contextlib.contextmanager
    def start(
        self, command: str, *, env: Mapping[str, str] | None = None
    ) -> Iterator[BackgroundCommand]:
        """Run ``agentbricks <command>`` in the background for the block, then stop it."""
        argv = self._argv(command)
        label = self._label(command)
        log_path = self.logs_dir / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_command(argv)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**self._env, **(env or {})},
            )
        try:
            yield BackgroundCommand(label, process, log_path)
        finally:
            if process.poll() is None:
                terminate_process_group(process)
            log(f"{label} | stopped")

    def _argv(self, command: str) -> list[str]:
        return [_agentbricks_bin(), *self.target_workspace.cli_args, *shlex.split(command)]

    def _label(self, command: str) -> str:
        """A log name from the command's first two words, e.g. ``007-tools-add``."""
        self._sequence += 1
        words = shlex.split(command)
        while words and words[0].startswith("-"):
            # A leading global option such as ``--output json``, and its value.
            words = words[1:] if "=" in words[0] else words[2:]
        # The second word only when it is a subcommand or name, never a flag or a path.
        names = words[:1] + [w for w in words[1:2] if re.fullmatch(r"[a-z][a-z0-9_-]*", w)]
        return "-".join([f"{self._sequence:03d}", *names])

    def _run_once(
        self, argv: list[str], label: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        """Run with stdout and stderr in log files, printing progress while a command is slow."""
        stdout_path = self.logs_dir / f"{label}.out"
        stderr_path = self.logs_dir / f"{label}.err"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        log_command(argv)
        started = time.monotonic()
        with (
            stdout_path.open("w", encoding="utf-8") as out,
            stderr_path.open("w", encoding="utf-8") as err,
        ):
            process = subprocess.Popen(
                argv,
                env=self._env,
                text=True,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    terminate_process_group(process, grace=0)
                    raise E2EError(f"{label} timed out after {timeout:.0f}s; log: {stderr_path}")
                if elapsed >= next_tick:
                    progress = last_nonempty_line(stderr_path) or last_nonempty_line(stdout_path)
                    log(f"tick {now():%H:%M} | {label} | running | {progress}")
                    next_tick += 60.0
                time.sleep(0.2 if elapsed < 5 else 2)
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        for text in (stdout, stderr):
            if text.strip():
                log(text)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    # Authoring

    def new_project(self, authoring: str) -> Project:
        """`agentbricks init` a uniquely named project and pin the runtime under test.

        `init` binds a session store, a memory store and a tracing experiment by default, and CLI
        authoring keeps them, so their stores are created here. Direct authoring replaces the
        whole manifest with ``Project.write_manifest``.
        """
        name = f"{project_prefix(self.run_config.run_id)}{uuid.uuid4().hex[:6]}"
        path = self.run_config.output / "projects" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        project = Project(authoring, path, f"{APP_PREFIX}{name}")
        workspace = self.target_workspace.client
        if workspace is not None:
            # Teardown deletes whatever App has this name, so it must not belong to anyone else.
            workspace.assert_app_absent(project.app_name)
        self.projects.append(project)
        self.run(f"init --framework {FRAMEWORKS[0]} {path}", timeout=600)
        if self.run_config.bridge_sha:
            self._pin_bridge_sources(path)
        else:
            self._pin_project_wheel(path)
        if authoring == "cli":
            self._create_declared_stores(project)
        return project

    def _create_declared_stores(self, project: Project) -> None:
        """Create the stores `init` declared in agent.toml, which deploy expects to exist."""
        manifest = project.manifest()
        project.memory_store_config = manifest.get("memory_store", {}).get("name")
        project.session_store_config = manifest.get("session_store", {}).get("name")
        project.experiment_name = manifest.get("tracing", {}).get("experiment_name")
        if project.memory_store_config:
            result = self.run(
                f"--output json memory stores create --name {project.memory_store_config}"
            )
            project.memory_resource = json.loads(result.stdout)
            name = project.memory_resource.get("name")
            if isinstance(name, str) and name:
                project.memory_store_name = name
        if project.session_store_config:
            result = self.run(
                f"--output json sessions stores create --name {project.session_store_config}"
            )
            project.session_resource = json.loads(result.stdout)
            project.session_store_name = project.session_store_config

    def _pin_project_wheel(self, project: pathlib.Path) -> None:
        wheel = self.run_config.wheel
        assert wheel is not None
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
        log(
            f"# write {pyproject}: "
            f"pin databricks-agentbricks runtime to {vendored_wheel.relative_to(project)}"
        )
        pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")

    def _pin_bridge_sources(self, project: pathlib.Path) -> None:
        bridge_sha = self.run_config.bridge_sha
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
        log(f"# write {pyproject}: pin Agent Bricks and LangChain to {bridge_sha}")
        pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")


def _agentbricks_bin() -> str:
    """The CLI of the interpreter running the tests, i.e. the source checkout's own install."""
    sibling = pathlib.Path(sys.executable).parent / "agentbricks"
    found = str(sibling) if sibling.exists() else shutil.which("agentbricks")
    if found is None:
        raise E2EError("The agentbricks CLI is not installed in the test environment.")
    return found


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


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _monitored(
    label: str, operation: Callable[[], dict[str, Any]], *, timeout: float
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
                log(f"tick {now():%H:%M} | {label} | running | {elapsed:.0f}s")
                if elapsed >= timeout:
                    raise E2EError(f"{label} timed out after {timeout:.0f}s") from None


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
        raise E2EError(f"HTTP {exc.code} from {url}: {detail}") from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise E2EError(f"Invalid JSON from {url}: {payload[:2000]}") from exc
    if not isinstance(value, dict):
        raise E2EError(f"Expected object response from {url}, got {type(value).__name__}")
    return value
