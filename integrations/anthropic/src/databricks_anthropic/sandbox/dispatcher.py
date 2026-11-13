"""Poll a self-hosted environment's work queue and run each session in a Databricks Sandbox."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from anthropic import NOT_GIVEN, AsyncAnthropic

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox.launcher import (
    MANAGED_BY,
    TAG_ENVIRONMENT,
    TAG_LIFECYCLE,
    TAG_MANAGED_BY,
    TAG_SESSION,
    DatabricksSandboxLauncher,
    LaunchResult,
)

# An ephemeral sandbox deletes itself when its worker finishes; one still around (stopped) after
# this long means the worker crashed or was killed, so cleanup removes it.
EPHEMERAL_GRACE_S = 3600
# A worker whose export failed stops (not deletes) its sandbox and appends this to its display name, so the
# files survive: such sandboxes are kept until retention_days, not EPHEMERAL_GRACE_S.
KEPT_MARKER = "[export failed, kept]"
# results / failures keep only the most recent launches (the dispatcher can run for weeks).
MAX_RECORDED = 1000


if TYPE_CHECKING:
    import anthropic
    from anthropic.types.beta.environments import BetaSelfHostedWork

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupAction:
    sandbox_name: str
    action: str  # "delete" | "keep"
    reason: str


@dataclass
class DispatcherStatus:
    """Counters for a health or status endpoint (no secrets, no session content)."""

    started_at: float | None = None
    polling: bool = False
    last_claim_at: float | None = None
    claimed: int = 0
    launched: int = 0
    reused: int = 0
    failed: int = 0
    skipped: int = 0
    retries: int = 0
    inflight: int = 0
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class DatabricksSandboxDispatcher:
    """Claim work from a Claude Managed Agents self-hosted environment.

    Every claimed ``session`` item is handed to :class:`DatabricksSandboxLauncher`, which
    runs the session's worker in its own sandbox. Needs only outbound HTTPS: no webhook.

    ``drain=True`` returns once the queue is empty (scripts, tests); the default polls
    until cancelled. ``accept`` optionally filters claimed session items (e.g. only the
    caller's own session in a test). The poller acknowledges every item it yields, and
    ``reclaim_older_than_ms`` only reclaims *un-acknowledged* work, so a skipped item is not handed
    straight back; it waits unserved until its lease expires. Only skip items another dispatcher serves.
    An item whose launch fails is stopped (``force``), so it is not relaunched in a loop.

    While polling (not in drain mode) it also sweeps every ``sweep_interval_s``: stopped
    resumable sandboxes unused for ``launcher.retention_days`` are deleted, as are leftover
    ephemeral ones. Pass ``sessions_api`` (an ``anthropic.Anthropic`` client with an API key)
    to also delete sandboxes whose Claude session was deleted or archived.
    """

    launcher: DatabricksSandboxLauncher
    _: KW_ONLY
    max_concurrent_launches: int = 8
    drain: bool = False
    reclaim_older_than_ms: int | None = 2000
    accept: Callable[[BetaSelfHostedWork], bool] | None = None
    sweep_interval_s: float | None = 3600
    sessions_api: anthropic.Anthropic | None = None  # with an API key, for session-gone cleanup
    # Back-off before each retry of a launch that failed with a transient error (then the item is stopped).
    retry_delays_s: tuple[float, ...] = (5.0, 15.0)
    results: list[LaunchResult] = field(default_factory=list, init=False)
    failures: list[tuple[str, str]] = field(default_factory=list, init=False)
    status: DispatcherStatus = field(default_factory=DispatcherStatus, init=False)

    def _client(self) -> AsyncAnthropic:
        # Explicit auth_token + base_url: ambient ANTHROPIC_* (e.g. Claude Code routed via a
        # gateway) must not redirect or re-authenticate queue calls.
        return AsyncAnthropic(
            auth_token=self.launcher.environment_key, base_url=self.launcher.anthropic_base_url
        )

    async def run(self) -> list[LaunchResult]:
        semaphore = asyncio.Semaphore(self.max_concurrent_launches)
        inflight: dict[str, asyncio.Task[None]] = {}
        status = self.status
        status.started_at, status.polling = time.time(), True
        # Launches block on sandbox RPCs for seconds each: give them their own threads, sized to the
        # concurrency limit, instead of the small shared default pool (6 threads on 2 vCPUs).
        pool = ThreadPoolExecutor(
            max_workers=self.max_concurrent_launches, thread_name_prefix="cma-launch"
        )
        loop = asyncio.get_running_loop()

        async def launch(work: BetaSelfHostedWork) -> None:
            async with semaphore:
                status.inflight += 1
                try:
                    result = await self._launch_with_retries(loop, pool, work)
                    self.results.append(result)
                    del self.results[:-MAX_RECORDED]  # bounded history in poll mode
                    if getattr(result, "reused", False):
                        status.reused += 1
                    else:
                        status.launched += 1
                except Exception as e:  # noqa: BLE001 - one bad item must not stop the dispatcher
                    detail = f"{type(e).__name__}: {e}"
                    self.failures.append((work.data.id, detail))
                    del self.failures[:-MAX_RECORDED]
                    status.failed += 1
                    status.last_error = type(e).__name__
                    log.error("launch failed session=%s work=%s: %s", work.data.id, work.id, detail)
                    await self._stop_work(work)
                finally:
                    status.inflight -= 1

        sweeper = None
        if not self.drain and self.sweep_interval_s:
            sweeper = asyncio.create_task(self._sweep_forever())
        async with self._client() as client:
            try:
                poller = client.beta.environments.work.poller(
                    environment_id=self.launcher.environment_id,
                    environment_key=self.launcher.environment_key,
                    reclaim_older_than_ms=self.reclaim_older_than_ms,
                    auto_stop=False,  # the sandboxed worker owns the stop call
                    drain=self.drain,
                    # Non-blocking in drain mode (the API rejects block_ms=0).
                    block_ms=None if self.drain else NOT_GIVEN,
                )
                async for work in poller:
                    if work.data.type != "session":
                        log.info("skipping work=%s type=%s", work.id, work.data.type)
                        status.skipped += 1
                        continue
                    session_id = work.data.id
                    if self.accept is not None and not self.accept(work):
                        log.info("not accepted: work=%s session=%s", work.id, session_id)
                        status.skipped += 1
                        continue
                    running = inflight.get(session_id)
                    if running and not running.done():
                        log.info(
                            "session=%s already launching; skipping work=%s", session_id, work.id
                        )
                        status.skipped += 1
                        continue
                    log.info("claimed work=%s session=%s", work.id, session_id)
                    status.claimed += 1
                    status.last_claim_at = time.time()
                    for done in [sid for sid, task in inflight.items() if task.done()]:
                        del inflight[done]  # keep the map bounded in long-running poll mode
                    inflight[session_id] = asyncio.create_task(launch(work))
            except Exception as e:
                status.last_error = type(e).__name__
                raise
            finally:
                status.polling = False
                if sweeper:
                    sweeper.cancel()
                # Wait for launches already claimed, including ones still queued for the semaphore: each is an
                # acknowledged work item, and cancelling it would drop that session's turn. Shutdown can take as
                # long as the slowest launch; the platform's termination grace period bounds it.
                pending = [t for t in inflight.values() if not t.done()]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                pool.shutdown(wait=False)
        return self.results

    async def _launch_with_retries(
        self, loop: asyncio.AbstractEventLoop, pool: ThreadPoolExecutor, work: BetaSelfHostedWork
    ) -> LaunchResult:
        """Launch, retrying clearly transient failures (the launch is idempotent) before giving up.

        A failed launch force-stops the work item, and a stopped item is not re-delivered, so a passing
        503 or timeout would otherwise drop the user's turn. Permanent errors are raised at once.
        """
        for attempt, delay in enumerate((*self.retry_delays_s, None)):
            try:
                return await loop.run_in_executor(pool, self.launcher.launch, work)
            except Exception as e:
                if delay is None or not _is_transient(e):
                    raise
                self.status.retries += 1
                log.warning(
                    "launch attempt %d for session=%s hit a transient error (%s); retrying in %.0fs",
                    attempt + 1,
                    work.data.id,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _stop_work(self, work: BetaSelfHostedWork) -> None:
        """Release a work item we could not serve, so the poller does not relaunch it every few seconds."""
        try:
            async with self._client() as client:
                await client.beta.environments.work.stop(
                    work.id, environment_id=self.launcher.environment_id, force=True
                )
        except Exception as e:  # noqa: BLE001 - best effort
            log.warning("could not stop work=%s: %s: %s", work.id, type(e).__name__, e)

    async def _sweep_forever(self) -> None:
        while True:
            try:
                for action in await asyncio.to_thread(self.cleanup):
                    if action.action == "delete":
                        log.info("cleanup deleted %s (%s)", action.sandbox_name, action.reason)
            except Exception as e:  # noqa: BLE001 - never let cleanup stop the dispatcher
                log.warning("cleanup sweep failed: %s: %s", type(e).__name__, e)
            await asyncio.sleep(self.sweep_interval_s or 3600)

    def list_session_sandboxes(self) -> list[Any]:
        """Sandboxes this launcher created: its id prefix **and** our tags for this environment.

        Sandboxes created by anything else (other tools, other environments, or untagged ones that
        happen to share the prefix) are never returned, so cleanup can't touch them.
        """
        prefix = _ops.sandbox_name(self.launcher.sandbox_id_prefix)
        return [
            sb
            for sb in self.launcher.workspace_client.sandbox.list_sandboxes()
            if sb.name and sb.name.startswith(prefix) and self._owns(sb)
        ]

    def _owns(self, sandbox: Any) -> bool:
        tags = _ops.tags_of(sandbox)
        return (
            tags.get(TAG_MANAGED_BY) == MANAGED_BY
            and tags.get(TAG_ENVIRONMENT) == self.launcher.environment_id
        )

    def plan_cleanup(
        self,
        *,
        delete_all: bool = False,
        force: bool = False,
        retention_days: float | None = None,
        sessions_api: anthropic.Anthropic | None = None,
        now: float | None = None,
    ) -> list[CleanupAction]:
        """Decide what to delete without deleting anything (``cleanup`` applies the plan).

        ``delete_all`` deletes every sandbox this launcher owns, except running ones unless ``force``.
        """
        now = time.time() if now is None else now
        retention_s = (
            self.launcher.retention_days if retention_days is None else retention_days
        ) * 86400
        sessions_api = sessions_api if sessions_api is not None else self.sessions_api
        plan = []
        for sb in self.list_session_sandboxes():
            tags = _ops.tags_of(sb)
            state = sb.status.state if sb.status else None
            stopped = state == _ops.SandboxState.SANDBOX_STATE_STOPPED
            last_used = max(_seconds(sb.update_time), _seconds(sb.create_time))
            idle_s = now - last_used if last_used else 0
            gone = False
            if not delete_all and sessions_api is not None and tags.get(TAG_SESSION):
                try:
                    gone = _session_gone(sessions_api, tags[TAG_SESSION])
                except Exception as e:  # noqa: BLE001 - one failed lookup must not abort the sweep
                    plan.append(
                        CleanupAction(sb.name, "keep", f"session lookup failed: {type(e).__name__}")
                    )
                    continue
            if delete_all and state == _ops.SandboxState.SANDBOX_STATE_RUNNING and not force:
                plan.append(CleanupAction(sb.name, "keep", "running (pass force to delete it)"))
            elif delete_all:
                plan.append(CleanupAction(sb.name, "delete", "delete-all requested"))
            elif gone:
                plan.append(
                    CleanupAction(
                        sb.name, "delete", f"Claude session {tags[TAG_SESSION]} deleted or archived"
                    )
                )
            elif (
                stopped
                and tags.get(TAG_LIFECYCLE) == "ephemeral"
                and KEPT_MARKER in (getattr(sb, "display_name", None) or "")
                and idle_s <= retention_s
            ):
                plan.append(
                    CleanupAction(sb.name, "keep", "kept: export failed (within retention)")
                )
            elif stopped and tags.get(TAG_LIFECYCLE) == "ephemeral" and idle_s > EPHEMERAL_GRACE_S:
                plan.append(
                    CleanupAction(
                        sb.name, "delete", "leftover ephemeral sandbox (worker did not delete it)"
                    )
                )
            elif stopped and idle_s > retention_s:
                plan.append(
                    CleanupAction(
                        sb.name, "delete", f"stopped and unused for {idle_s / 86400:.1f} days"
                    )
                )
            else:
                plan.append(
                    CleanupAction(
                        sb.name,
                        "keep",
                        f"{state.value if state else 'unknown'}",
                    )
                )
        return plan

    def cleanup(
        self,
        *,
        dry_run: bool = False,
        delete_all: bool = False,
        force: bool = False,
        retention_days: float | None = None,
        sessions_api: anthropic.Anthropic | None = None,
    ) -> list[CleanupAction]:
        """Apply :meth:`plan_cleanup`: delete what it marks for deletion (unless ``dry_run``)."""
        plan = self.plan_cleanup(
            delete_all=delete_all,
            force=force,
            retention_days=retention_days,
            sessions_api=sessions_api,
        )
        if not dry_run:
            for action in plan:
                if action.action != "delete":
                    continue
                try:
                    _ops.delete_quietly(self.launcher.workspace_client, action.sandbox_name)
                except Exception as e:  # noqa: BLE001 - keep going; the next sweep retries
                    log.warning(
                        "cleanup could not delete %s: %s: %s",
                        action.sandbox_name,
                        type(e).__name__,
                        e,
                    )
        return plan


def _is_transient(error: BaseException) -> bool:
    """Errors worth retrying: timeouts, dropped connections and the service's own retry-later responses."""
    import requests
    from databricks.sdk import errors

    transient = (
        TimeoutError,
        requests.ConnectionError,
        requests.Timeout,
        errors.TemporarilyUnavailable,
        errors.InternalError,
        errors.DeadlineExceeded,
        errors.ResourceExhausted,
        errors.TooManyRequests,
    )
    return isinstance(error, transient)


def _seconds(timestamp: Any) -> float:
    """protobuf Timestamp (or None) -> epoch seconds."""
    if timestamp is None:
        return 0.0
    return (
        float(getattr(timestamp, "seconds", 0) or 0)
        + float(getattr(timestamp, "nanos", 0) or 0) / 1e9
    )


def _session_gone(sessions_api: Any, session_id: str) -> bool:
    import anthropic

    try:
        session = sessions_api.beta.sessions.retrieve(session_id)
    except anthropic.NotFoundError:
        return True
    return getattr(session, "archived_at", None) is not None
