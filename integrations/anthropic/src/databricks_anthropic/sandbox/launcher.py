"""Launch a Claude Managed Agents worker for one claimed work item in a Databricks Sandbox."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Literal

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from databricks_anthropic import _useragent
from databricks_anthropic.sandbox import _ops

if TYPE_CHECKING:
    from anthropic.types.beta.environments import BetaSelfHostedWork

__all__ = [
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DatabricksSandboxLauncher",
    "LaunchResult",
    "Lifecycle",
    "SandboxMcpServer",
    "sandbox_id_for_session",
]

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
HOME_DIR = "/home/sandbox-agent"
TOOLS_DIR = f"{HOME_DIR}/.cma"  # persistent home: survives stop/start
VENV_DIR = f"{TOOLS_DIR}/venv"
VENV_PYTHON = f"{VENV_DIR}/bin/python"
WORKER_PATH = f"{TOOLS_DIR}/worker.py"
WORKER_LOG = f"{TOOLS_DIR}/worker.log"
WORKER_PID = f"{TOOLS_DIR}/worker.pid"
# Anthropic secrets for the next worker: written 0600, read and deleted by the worker at startup, so they
# are not in its process environment, which the agent could read via /proc/<pid>/environ. They do stay in the
# worker's memory and the agent runs as the same Linux user, so treat the sandbox as able to reach the key.
WORKER_SECRETS = f"{TOOLS_DIR}/worker.secrets"
# Present while a previous worker is being replaced: that worker then skips its delete/stop step.
WORKER_REPLACING = f"{TOOLS_DIR}/worker.replacing"
VENV_MARKER = f"{TOOLS_DIR}/venv.spec"
DEFAULT_WORKDIR = f"{HOME_DIR}/agent-workspace"
# EnvironmentWorker downloads attached memory stores to /mnt/memory/<store-name>/. `/` is an ephemeral
# overlay (wiped on stop) owned by root, so the launcher recreates it via the image's passwordless sudo.
MEMORY_DIR = "/mnt/memory"

# Keep in step with pyproject.toml (unit-tested). The worker imports anthropic.lib.*, hence the cap.
WORKER_PACKAGES = ("anthropic>=2.3,<3", "databricks-sdk[sandbox]>=0.157.0")

# Tags written on every sandbox at creation; cleanup uses them to find the owning session.
TAG_MANAGED_BY = "managed_by"
TAG_SESSION = "claude_session_id"
TAG_ENVIRONMENT = "claude_environment_id"
TAG_LIFECYCLE = "lifecycle"
MANAGED_BY = "databricks-anthropic"

Lifecycle = Literal["ephemeral", "resumable"]
# mcp brings the HTTP client its transport is built on (httpx for 1.x, httpx2 for 2.x).
MCP_PACKAGES = ("mcp>=1.24",)


_PREFIX_RE = re.compile(r"^[a-z][a-z0-9-]{0,28}-$")


def sandbox_id_for_session(session_id: str, *, prefix: str = "cma-") -> str:
    """Deterministic sandbox id for a Claude session.

    Sandbox ids must match AIP-122 (``^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$``, at most 63
    characters), while session ids are mixed case with underscores, so lower-casing could
    collide. A SHA-256 prefix is stable across resumes and collision-resistant.
    """
    return prefix + hashlib.sha256(session_id.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class SandboxMcpServer:
    """A Databricks MCP server whose tools the worker serves as custom tools."""

    name: str
    url: str
    exclude: tuple[str, ...] = ()
    # Genie One: also serve genie_ask_and_wait, which polls inside the sandbox.
    wait_tool: bool = False

    def to_env(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "exclude": list(self.exclude),
            "wait_tool": self.wait_tool,
        }


@dataclass(frozen=True)
class LaunchResult:
    sandbox_name: str
    pid: int
    created: bool
    seconds: float


@dataclass(kw_only=True)
class DatabricksSandboxLauncher:
    """Get or create a session's sandbox and start the Anthropic worker inside it.

    The sandbox's built-in Databricks credential authenticates as the identity of
    ``workspace_client``, so the agent's tools (and any MCP servers) act as that user or
    service principal.

    Only the environment key and the work item's per-session secret enter the sandbox. They are
    written to a 0600 file that the worker reads and deletes before it runs any tool, so they are
    not in its process environment; the Anthropic API key never enters the sandbox.

    Lifecycle (``lifecycle``):

    - ``"ephemeral"`` (default): when the session goes idle the worker exports
      ``workdir/export_dir`` to ``export_volume_path`` (if set) and **deletes** its sandbox.
      The next message gets a fresh sandbox.
    - ``"resumable"``: when idle the worker **stops** its sandbox (files kept); the next
      message restarts it. The sandbox is deleted when its Claude session is deleted or
      archived, or after ``retention_days`` unused (see ``DatabricksSandboxDispatcher.cleanup``).
    """

    workspace_client: WorkspaceClient
    environment_id: str
    environment_key: str
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    workdir: str = DEFAULT_WORKDIR
    # Safety net: if the worker dies, the missing heartbeat lets the sandbox auto-stop.
    inactivity_timeout_s: int = 600
    image_uri: str | None = None
    max_idle_s: float = 300
    heartbeat_s: float = 30
    lifecycle: Lifecycle = "ephemeral"
    # resumable: delete stopped sandboxes unused for this long (enforced by dispatcher cleanup)
    retention_days: float = 7.0
    # Optional UC Volume folder, e.g. /Volumes/main/agents/session_outputs; files under
    # workdir/export_dir are copied to <export_volume_path>/<session_id>/ when the worker finishes.
    export_volume_path: str | None = None
    export_dir: str = "outputs"
    mcp_servers: list[SandboxMcpServer] = field(default_factory=list)
    index_url: str | None = None
    extra_packages: tuple[str, ...] = ()
    sandbox_id_prefix: str = "cma-"
    confirm_s: float = 5

    def __post_init__(self) -> None:
        # The prefix scopes cleanup; an empty or odd prefix could match other sandboxes.
        if not _PREFIX_RE.match(self.sandbox_id_prefix):
            raise ValueError(
                "sandbox_id_prefix must be 2-30 characters: a lowercase letter, then lowercase letters, "
                f"digits or '-', ending in '-' (got {self.sandbox_id_prefix!r})"
            )
        if os.path.isabs(self.export_dir) or os.path.normpath(self.export_dir).startswith(".."):
            raise ValueError(
                f"export_dir must be a relative path inside workdir, not {self.export_dir!r}"
            )
        if self.lifecycle not in ("ephemeral", "resumable"):
            raise ValueError(
                f"lifecycle must be 'ephemeral' or 'resumable', not {self.lifecycle!r}"
            )
        _useragent.apply_process_env()
        _useragent.tag(self.workspace_client)

    def sandbox_name_for(self, session_id: str) -> str:
        return _ops.sandbox_name(sandbox_id_for_session(session_id, prefix=self.sandbox_id_prefix))

    def sandbox_tags(self, session_id: str) -> dict[str, str]:
        return {
            TAG_MANAGED_BY: MANAGED_BY,
            TAG_SESSION: session_id,
            TAG_ENVIRONMENT: self.environment_id,
            TAG_LIFECYCLE: self.lifecycle,
        }

    def ensure_sandbox(self, session_id: str) -> tuple[str, bool]:
        """Start the session's sandbox if it exists, else create it (tagged); return (name, created)."""
        w = self.workspace_client
        sandbox_id = sandbox_id_for_session(session_id, prefix=self.sandbox_id_prefix)
        name = _ops.sandbox_name(sandbox_id)
        try:
            _ops.ensure_running(w, name)
            return name, False
        except NotFound:
            pass
        name = _ops.create(
            w,
            sandbox_id,
            inactivity_timeout_s=self.inactivity_timeout_s,
            image_uri=self.image_uri,
            display_name=f"Claude session {session_id}",
            tags=self.sandbox_tags(session_id),
        )
        try:
            _ops.wait_for_state(w, name, _ops.SandboxState.SANDBOX_STATE_RUNNING)
        except BaseException:
            _ops.delete_quietly(w, name)  # never leak a tagged sandbox that never came up
            raise
        return name, True

    def packages(self) -> list[str]:
        pkgs = [*WORKER_PACKAGES, *self.extra_packages]
        if self.mcp_servers:
            pkgs.extend(MCP_PACKAGES)
        return pkgs

    def _worker_secrets(self, work: BetaSelfHostedWork) -> dict[str, str]:
        # The work secret is needed to mount the session's memory stores.
        return {
            "ANTHROPIC_ENVIRONMENT_KEY": self.environment_key,
            "ANTHROPIC_WORK_SECRET": work.secret or "",
        }

    def _worker_env(self, work: BetaSelfHostedWork, sandbox_name: str) -> dict[str, str]:
        """Non-secret worker settings (the CMA_* protocol with worker.py); secrets go via WORKER_SECRETS."""
        env = {
            "ANTHROPIC_ENVIRONMENT_ID": self.environment_id,
            "ANTHROPIC_WORK_ID": work.id,
            "ANTHROPIC_SESSION_ID": work.data.id,
            "CMA_SECRETS_FILE": WORKER_SECRETS,
            "CMA_REPLACING_MARKER": WORKER_REPLACING,
            "CMA_ANTHROPIC_BASE_URL": self.anthropic_base_url,
            "DBX_SANDBOX_NAME": sandbox_name,
            "CMA_WORKDIR": self.workdir,
            "CMA_MAX_IDLE": str(self.max_idle_s),
            "CMA_HEARTBEAT_SECONDS": str(self.heartbeat_s),
            "CMA_ON_EXIT": "delete" if self.lifecycle == "ephemeral" else "stop",
            # Partner attribution for the agent's own Databricks SDK/CLI calls (tool subprocesses inherit it).
            **_useragent.UPSTREAM_ENV,
        }
        if self.export_volume_path:
            env["CMA_EXPORT_VOLUME"] = self.export_volume_path.rstrip("/")
            env["CMA_EXPORT_DIR"] = os.path.join(self.workdir, self.export_dir)
        if self.mcp_servers:
            env["CMA_MCP_SERVERS"] = json.dumps([s.to_env() for s in self.mcp_servers])
        return env

    # ------------------------------------------------------------------ lifecycle

    def launch(self, work: BetaSelfHostedWork) -> LaunchResult:
        """Blocking: provision the sandbox, bootstrap, start the worker, confirm it is alive."""
        t0 = time.monotonic()
        w = self.workspace_client
        session_id = work.data.id
        name, created = self.ensure_sandbox(session_id)
        try:
            self._prepare_memory_dir(name)
            self._bootstrap(name)
            # A resumed session may still have the previous item's worker; replace it.
            _ops.stop_pidfile_process(w, name, WORKER_PID, marker_path=WORKER_REPLACING)
            _ops.upload_text(
                w, name, WORKER_SECRETS, json.dumps(self._worker_secrets(work)), mode="600"
            )
            pid = _ops.start_detached(
                w,
                name,
                f"mkdir -p {shlex.quote(self.workdir)} && cd {shlex.quote(self.workdir)} && "
                f"exec {VENV_PYTHON} {WORKER_PATH}",
                log_path=WORKER_LOG,
                pid_path=WORKER_PID,
                envs=self._worker_env(work, name),
            )
            time.sleep(self.confirm_s)
            if not _ops.is_alive(w, name, pid):
                raise _ops.SandboxCommandError(
                    f"worker exited right after launch in {name}:\n{_ops.tail(w, name, WORKER_LOG)}"
                )
        except BaseException:
            # Never leak a sandbox we just created if bring-up fails.
            if created:
                _ops.delete_quietly(w, name)
            else:  # kept sandbox: don't leave the secrets file behind if the worker never read it
                with contextlib.suppress(Exception):
                    _ops.run(w, name, f"rm -f {shlex.quote(WORKER_SECRETS)}", check=False)
            raise
        seconds = time.monotonic() - t0
        log.info(
            "launched worker session=%s work=%s sandbox=%s pid=%s created=%s in %.1fs",
            session_id,
            work.id,
            name,
            pid,
            created,
            seconds,
        )
        return LaunchResult(sandbox_name=name, pid=pid, created=created, seconds=seconds)

    def _prepare_memory_dir(self, name: str) -> None:
        """Make /mnt/memory writable for the worker (needed for memory stores); runs on every launch."""
        out = (
            _ops.run(
                self.workspace_client,
                name,
                f"if [ -w {MEMORY_DIR} ] || "
                f"{{ mkdir -p {MEMORY_DIR} 2>/dev/null; [ -w {MEMORY_DIR} ]; }}; then echo ok; "
                f"elif sudo -n mkdir -p {MEMORY_DIR} && "
                f'sudo -n chown "$(id -u):$(id -g)" {MEMORY_DIR}; then echo ok; '
                "else echo unavailable; fi",
                check=False,
            ).stdout
            or ""
        )
        if "ok" not in out:
            log.warning(
                "%s is not writable in %s; memory stores will fail to mount", MEMORY_DIR, name
            )

    def _bootstrap(self, name: str) -> None:
        """Idempotently build the worker venv (persists across stop/start) and upload the worker."""
        w = self.workspace_client
        # The index URL may embed credentials: only its hash goes on disk, and it reaches uv via env.
        index_tag = (
            hashlib.sha256(self.index_url.encode()).hexdigest()[:12] if self.index_url else ""
        )
        spec = " ".join(sorted(self.packages())) + (
            f" index-sha256={index_tag}" if index_tag else ""
        )
        current = _ops.run(w, name, f"cat {VENV_MARKER} 2>/dev/null", check=False).stdout or ""
        if current.strip() != spec:
            pkgs = " ".join(shlex.quote(p) for p in self.packages())
            # The default image ships python3 + uv but not pip.
            _ops.run(
                w,
                name,
                f"set -e; mkdir -p {TOOLS_DIR}; "
                f"[ -x {VENV_PYTHON} ] || uv venv --quiet --python python3 {VENV_DIR}; "
                f"uv pip install --quiet --python {VENV_PYTHON} {pkgs}; "
                f'printf %s "$DBX_SPEC" > {VENV_MARKER}',
                envs={
                    "DBX_SPEC": spec,
                    **({"UV_DEFAULT_INDEX": self.index_url} if self.index_url else {}),
                },
                timeout_s=600,
            )
        source = resources.files("databricks_anthropic.sandbox").joinpath("worker.py").read_text()
        _ops.upload_text(w, name, WORKER_PATH, source)

    def stop_session_sandbox(self, session_id: str) -> None:
        """Stop a session's sandbox (home directory is kept for resume)."""
        try:
            self.workspace_client.sandbox.stop_sandbox(self.sandbox_name_for(session_id))
        except NotFound:
            pass

    def delete_session_sandbox(self, session_id: str) -> None:
        """Delete a session's sandbox and its home directory."""
        _ops.delete_quietly(self.workspace_client, self.sandbox_name_for(session_id))
