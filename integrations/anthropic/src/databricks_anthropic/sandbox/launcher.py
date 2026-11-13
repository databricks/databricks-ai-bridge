"""Launch a Claude Managed Agents worker for one claimed work item in a Databricks Sandbox."""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import hmac
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
from databricks.sdk.errors import AlreadyExists, NotFound

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
    "SandboxOwnershipError",
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
# The work item the running worker serves (written by the start command): a re-delivered item whose worker
# is still alive is left alone instead of being replaced (idempotent launch).
WORKER_WORK_ID = f"{TOOLS_DIR}/worker.work_id"
# The per-session secret for the next worker: written 0600, read and deleted by the worker at startup, so it
# is not in its process environment, which the agent could read via /proc/<pid>/environ. It carries only the
# work item's sessions token (scoped to this one session); the environment key never enters a sandbox.
WORKER_SECRETS = f"{TOOLS_DIR}/worker.secrets"
# Present while a previous worker is being replaced: that worker then skips its delete/stop step.
WORKER_REPLACING = f"{TOOLS_DIR}/worker.replacing"
# Written by a worker when it starts finishing (export, then delete or stop): it will not serve another turn,
# so a new turn for the session must replace it rather than reuse it. Cleared when a new worker starts.
WORKER_FINISHING = f"{TOOLS_DIR}/worker.finishing"
# Written by the worker once it is about to serve the session (MCP tools open, credential accepted): the
# launcher waits for it instead of a fixed sleep. Cleared when a new worker starts.
WORKER_READY = f"{TOOLS_DIR}/worker.ready"
# A finishing worker being replaced may be exporting outputs: give it this long before KILL.
FINISHING_GRACE_S = 120
VENV_MARKER = f"{TOOLS_DIR}/venv.spec"
DEFAULT_WORKDIR = f"{HOME_DIR}/agent-workspace"
# EnvironmentWorker downloads attached memory stores to /mnt/memory/<store-name>/. `/` is an ephemeral
# overlay (wiped on stop) owned by root, so the launcher recreates it via the image's passwordless sudo.
MEMORY_DIR = "/mnt/memory"

# Exact top-level pins for the in-sandbox venv. The venv itself is installed from hash-locked requirements
# generated from these pins (worker-requirements*.txt, scripts/lock_worker.sh), so every dependency is pinned
# too. Each pin must satisfy the package's range in pyproject.toml (unit-tested). The worker imports anthropic.lib.*.
WORKER_PACKAGES = ("anthropic==2.3.0", "databricks-sdk[sandbox]==0.159.0")

# Tags written on every sandbox at creation; cleanup uses them to find the owning session.
TAG_MANAGED_BY = "managed_by"
TAG_SESSION = "claude_session_id"
TAG_ENVIRONMENT = "claude_environment_id"
TAG_LIFECYCLE = "lifecycle"
MANAGED_BY = "databricks-anthropic"

Lifecycle = Literal["ephemeral", "resumable"]
# mcp brings the HTTP client its transport is built on (httpx for 1.x, httpx2 for 2.x).
MCP_PACKAGES = ("mcp==2.3.0",)


_PREFIX_RE = re.compile(r"^[a-z][a-z0-9-]{0,28}-$")


def sandbox_id_for_session(
    session_id: str, *, prefix: str = "cma-", key: str | bytes | None = None
) -> str:
    """Deterministic sandbox id for a Claude session.

    Sandbox ids must match AIP-122 (``^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$``, at most 63
    characters), while session ids are mixed case with underscores, so lower-casing could
    collide. A SHA-256 prefix is stable across resumes and collision-resistant.

    With ``key`` (a deployment secret) the id is an HMAC instead, so nobody without the key can
    predict a session's sandbox name and create it first (name squatting).
    """
    if key:
        raw = key.encode() if isinstance(key, str) else key
        digest = hmac.new(raw, session_id.encode(), hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(session_id.encode()).hexdigest()
    return prefix + digest[:24]


@functools.lru_cache(maxsize=1)
def _worker_source() -> tuple[str, str]:
    """``worker.py`` and its SHA-256 (read once per process)."""
    source = resources.files("databricks_anthropic.sandbox").joinpath("worker.py").read_text()
    return source, hashlib.sha256(source.encode()).hexdigest()


class SandboxOwnershipError(RuntimeError):
    """An existing sandbox with the session's name is not one this launcher created for that session."""


def sessions_token_from_secret(secret: str | None) -> str | None:
    """The sessions token inside a work item's secret (URL-safe base64 JSON), or None. Never log it."""
    if not secret:
        return None
    try:
        parsed = json.loads(base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4)))
    except (ValueError, TypeError):
        return None
    token = parsed.get("sessions_token") if isinstance(parsed, dict) else None
    return token if isinstance(token, str) and token else None


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
    # True when a worker for this same work item was already running and was left in place.
    reused: bool = False


@dataclass(kw_only=True)
class DatabricksSandboxLauncher:
    """Get or create a session's sandbox and start the Anthropic worker inside it.

    The sandbox's built-in Databricks credential authenticates as the identity of
    ``workspace_client``, so the agent's tools (and any MCP servers) act as that user or
    service principal.

    Only the work item's sessions token enters the sandbox (it can act on that one session and its
    attached memory stores, nothing else). It is written to a 0600 file that the worker reads and deletes
    before it runs any tool, so it is not in its process environment. The environment key (which can claim
    any session in the environment) and the Anthropic API key never enter a sandbox.

    ``sandbox_id_key`` (recommended for shared deployments) makes sandbox names unpredictable; an existing
    sandbox is only used if its tags show it was created for this session and environment.

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
    # Secret for HMAC sandbox ids (see sandbox_id_for_session). Changing it orphans existing sandboxes.
    sandbox_id_key: str | bytes | None = field(default=None, repr=False)
    # Start-up: poll every confirm_s until the worker reports ready, exits, or start_timeout_s passes.
    confirm_s: float = 2
    start_timeout_s: float = 120

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
        # The worker sends the session's token to this URL: never over plain HTTP.
        if not self.anthropic_base_url.startswith("https://"):
            raise ValueError(f"anthropic_base_url must use https, not {self.anthropic_base_url!r}")
        if self.lifecycle not in ("ephemeral", "resumable"):
            raise ValueError(
                f"lifecycle must be 'ephemeral' or 'resumable', not {self.lifecycle!r}"
            )
        _useragent.apply()
        _useragent.tag(self.workspace_client)

    def sandbox_id_for(self, session_id: str) -> str:
        return sandbox_id_for_session(
            session_id, prefix=self.sandbox_id_prefix, key=self.sandbox_id_key
        )

    def sandbox_name_for(self, session_id: str) -> str:
        return _ops.sandbox_name(self.sandbox_id_for(session_id))

    def sandbox_tags(self, session_id: str) -> dict[str, str]:
        return {
            TAG_MANAGED_BY: MANAGED_BY,
            TAG_SESSION: session_id,
            TAG_ENVIRONMENT: self.environment_id,
            TAG_LIFECYCLE: self.lifecycle,
        }

    def verify_ownership(self, name: str, session_id: str) -> None:
        """Raise SandboxOwnershipError unless ``name`` carries this session's and environment's tags.

        The sandbox API lists only sandboxes the caller can see, but exposes no creator field, so the
        tags (written at creation) plus an unpredictable name (``sandbox_id_key``) are the check.
        """
        sandbox = self.workspace_client.sandbox.get_sandbox(name)
        tags = _ops.tags_of(sandbox)
        expected = {
            TAG_MANAGED_BY: MANAGED_BY,
            TAG_SESSION: session_id,
            TAG_ENVIRONMENT: self.environment_id,
        }
        mismatched = sorted(k for k, v in expected.items() if tags.get(k) != v)
        if mismatched:
            raise SandboxOwnershipError(
                f"{name} exists but was not created for session {session_id} in environment "
                f"{self.environment_id} (tags differ: {', '.join(mismatched)}); refusing to use it"
            )

    def ensure_sandbox(self, session_id: str) -> tuple[str, bool]:
        """Start the session's sandbox if it exists, else create it (tagged); return (name, created).

        An existing sandbox is verified (tags) before it is started or used.
        """
        w = self.workspace_client
        sandbox_id = self.sandbox_id_for(session_id)
        name = _ops.sandbox_name(sandbox_id)
        try:
            self.verify_ownership(name, session_id)
            _ops.ensure_running(w, name)
            return name, False
        except NotFound:
            pass
        try:
            name = _ops.create(
                w,
                sandbox_id,
                inactivity_timeout_s=self.inactivity_timeout_s,
                image_uri=self.image_uri,
                display_name=f"Claude session {session_id}",
                tags=self.sandbox_tags(session_id),
            )
        except AlreadyExists:
            # A concurrent launch (e.g. a re-delivered item) created it first: verify, then share it.
            self.verify_ownership(name, session_id)
            _ops.ensure_running(w, name)
            return name, False
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
        """The worker's only credential: the work secret, re-encoded to carry just the sessions token.

        The sessions token authenticates this item's heartbeat, session events and memory stores; the
        ingress and source tokens in the original payload are dropped. Fails closed: no sessions token
        means no launch (the worker would otherwise need the environment key).
        """
        token = sessions_token_from_secret(getattr(work, "secret", None))
        if token is None:
            raise ValueError(
                f"work item {work.id} carries no sessions token; refusing to launch without one "
                "(the environment key is never sent to a sandbox)"
            )
        reduced = base64.urlsafe_b64encode(json.dumps({"sessions_token": token}).encode()).decode()
        return {"ANTHROPIC_WORK_SECRET": reduced}

    def _check_no_environment_key(self, *payloads: object) -> None:
        """Fail closed if the environment key would reach the sandbox in any payload."""
        key = self.environment_key
        if key and any(key in json.dumps(p) for p in payloads):
            raise RuntimeError("refusing to launch: the environment key would enter the sandbox")

    def _worker_env(self, work: BetaSelfHostedWork, sandbox_name: str) -> dict[str, str]:
        """Non-secret worker settings (the CMA_* protocol with worker.py); secrets go via WORKER_SECRETS."""
        env = {
            "ANTHROPIC_ENVIRONMENT_ID": self.environment_id,
            "ANTHROPIC_WORK_ID": work.id,
            "ANTHROPIC_SESSION_ID": work.data.id,
            "CMA_SECRETS_FILE": WORKER_SECRETS,
            "CMA_REPLACING_MARKER": WORKER_REPLACING,
            "CMA_FINISHING_MARKER": WORKER_FINISHING,
            "CMA_READY_MARKER": WORKER_READY,
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
        secrets = self._worker_secrets(work)  # fails closed before anything is created
        envs = self._worker_env(work, self.sandbox_name_for(session_id))
        self._check_no_environment_key(secrets, envs)
        name, created = self.ensure_sandbox(session_id)
        live: _ops.WorkerState | None = None
        try:
            if not created:
                live = _ops.live_worker(
                    w,
                    name,
                    WORKER_PID,
                    WORKER_WORK_ID,
                    worker_path=WORKER_PATH,
                    finishing_path=WORKER_FINISHING,
                )
                # Work ids are per session, so a new turn arrives with the same id as the last one: a live,
                # non-finishing worker for it is already serving the session. Look twice, a poll interval
                # apart, so a worker that starts finishing right after the first look is not reused.
                if live is not None and not live.finishing and live.work_id == work.id:
                    time.sleep(self.confirm_s)
                    live = _ops.live_worker(
                        w,
                        name,
                        WORKER_PID,
                        WORKER_WORK_ID,
                        worker_path=WORKER_PATH,
                        finishing_path=WORKER_FINISHING,
                    )
                if live is not None and not live.finishing and live.work_id == work.id:
                    # Re-delivered item whose worker is still running: never start a second one.
                    log.info("worker for work=%s already running in %s; leaving it", work.id, name)
                    return LaunchResult(
                        sandbox_name=name,
                        pid=live.pid,
                        created=False,
                        seconds=time.monotonic() - t0,
                        reused=True,
                    )
            # Replace any previous worker *before* touching the sandbox: the replace marker must be set before a
            # finishing worker can delete or stop the sandbox, and the venv must not be reinstalled under a
            # running worker. A finishing worker may be exporting; it keeps the sandbox once it sees the marker.
            grace_s = FINISHING_GRACE_S if live is not None and live.finishing else 10
            _ops.stop_pidfile_process(
                w,
                name,
                WORKER_PID,
                marker_path=WORKER_REPLACING,
                worker_path=WORKER_PATH,
                grace_s=grace_s,
            )
            if not created and not self._still_usable(name):
                # The previous worker deleted the sandbox as it finished, before it saw the replace marker.
                log.warning("%s was deleted by its previous worker; recreating it", name)
                name, created = self.ensure_sandbox(session_id)
            self._prepare_memory_dir(name)
            worker_sha256 = self._bootstrap(name)
            _ops.upload_text(w, name, WORKER_SECRETS, json.dumps(secrets), mode="600")
            pid = _ops.start_detached(
                w,
                name,
                # Verify the worker file in the same exec that starts it, and record the work id it serves.
                f"echo {worker_sha256}'  '{WORKER_PATH} | sha256sum -c --quiet - || exit 97; "
                f'rm -f {WORKER_FINISHING} {WORKER_READY}; printf %s "$ANTHROPIC_WORK_ID" > {WORKER_WORK_ID}; '
                f"mkdir -p {shlex.quote(self.workdir)} && cd {shlex.quote(self.workdir)} && "
                f"exec {VENV_PYTHON} -I {WORKER_PATH}",
                log_path=WORKER_LOG,
                pid_path=WORKER_PID,
                envs=envs,
            )
            self._wait_until_ready(name, pid)
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

    def _wait_until_ready(self, name: str, pid: int) -> None:
        """Wait for the worker's ready marker. Raise with its log if it exits first.

        Liveness, readiness and the log tail come from one call each time, so the reason survives even if
        the sandbox goes away. A worker that is still starting after ``start_timeout_s`` is left running.
        """
        deadline = time.monotonic() + self.start_timeout_s
        while True:
            time.sleep(self.confirm_s)
            try:
                alive, ready, log_tail = _ops.worker_progress(
                    self.workspace_client, name, pid, WORKER_LOG, WORKER_READY
                )
            except NotFound as e:
                raise _ops.SandboxCommandError(
                    f"{name} was deleted while its worker was starting; see the App or dispatcher log"
                ) from e
            if ready:
                return
            if not alive:
                raise _ops.SandboxCommandError(
                    f"worker exited during start-up in {name}:\n{log_tail}"
                )
            if time.monotonic() > deadline:
                log.warning(
                    "worker in %s still starting after %.0fs; leaving it running",
                    name,
                    self.start_timeout_s,
                )
                return

    def _still_usable(self, name: str) -> bool:
        """False if the sandbox is gone; otherwise make sure it is running (a finishing worker may have stopped it)."""
        try:
            state = _ops.state_of(self.workspace_client, name)
            if state != _ops.SandboxState.SANDBOX_STATE_RUNNING:
                _ops.ensure_running(self.workspace_client, name)
        except NotFound:  # gone, or deleted while we waited for it
            return False
        return True

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

    def worker_lock(self) -> tuple[str, str]:
        """(file name, contents) of the hash-locked requirements for this launcher's worker venv."""
        filename = "worker-requirements-mcp.txt" if self.mcp_servers else "worker-requirements.txt"
        return filename, resources.files("databricks_anthropic.sandbox").joinpath(
            filename
        ).read_text()

    def _bootstrap(self, name: str) -> str:
        """Idempotently build the worker venv (persists across stop/start), upload the worker; return its sha256.

        The venv is installed from a hash-locked requirements file (every package, with ``--require-hashes``),
        so a fresh sandbox never resolves an unreviewed release. ``extra_packages`` are installed afterwards
        without hashes (an advanced option; document what you add).
        """
        w = self.workspace_client
        lock_name, lock = self.worker_lock()
        lock_path = f"{TOOLS_DIR}/{lock_name}"
        # The index URL may embed credentials: only its hash goes on disk, and it reaches uv via env.
        index_tag = (
            hashlib.sha256(self.index_url.encode()).hexdigest()[:12] if self.index_url else ""
        )
        spec = " ".join(
            [
                f"lock-sha256={hashlib.sha256(lock.encode()).hexdigest()[:16]}",
                *sorted(self.extra_packages),
                *([f"index-sha256={index_tag}"] if index_tag else []),
            ]
        )
        current = _ops.run(w, name, f"cat {VENV_MARKER} 2>/dev/null", check=False).stdout or ""
        if current.strip() != spec:
            _ops.upload_text(w, name, lock_path, lock)
            extra = " ".join(shlex.quote(p) for p in self.extra_packages)
            # The default image ships python3 + uv but not pip.
            _ops.run(
                w,
                name,
                f"set -e; mkdir -p {TOOLS_DIR}; "
                f"[ -x {VENV_PYTHON} ] || uv venv --quiet --python python3 {VENV_DIR}; "
                f"uv pip install --quiet --python {VENV_PYTHON} --require-hashes -r {lock_path}; "
                + (f"uv pip install --quiet --python {VENV_PYTHON} {extra}; " if extra else "")
                + f'printf %s "$DBX_SPEC" > {VENV_MARKER}',
                envs={
                    "DBX_SPEC": spec,
                    **({"UV_DEFAULT_INDEX": self.index_url} if self.index_url else {}),
                },
                timeout_s=600,
            )
        source, digest = _worker_source()
        _ops.upload_text(w, name, WORKER_PATH, source)
        return digest

    def stop_session_sandbox(self, session_id: str) -> None:
        """Stop a session's sandbox (home directory is kept for resume)."""
        try:
            self.workspace_client.sandbox.stop_sandbox(self.sandbox_name_for(session_id))
        except NotFound:
            pass

    def delete_session_sandbox(self, session_id: str) -> None:
        """Delete a session's sandbox and its home directory."""
        _ops.delete_quietly(self.workspace_client, self.sandbox_name_for(session_id))
