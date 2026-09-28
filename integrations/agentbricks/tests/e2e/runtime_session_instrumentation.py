"""Opt-in probes for isolated deployed template tests, never production templates.

Call ``install(app, build={...})`` after registering the template's real hooks. The probes keep
the real Runtime, Lakebase store, framework adapter, and model. They add bounded scheduling
delays/faults and persisted markers around those hooks. Do not install in a shared application.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Query

BUILD_FIELDS = {
    "sdk_commit",
    "variant",
    "app_name",
    "lakebase_project",
    "lakebase_branch",
    "lakebase_database",
    "source_sha256",
    "wheel_sha256",
}
TIMINGS = {"heartbeat_seconds": 1.0, "stale_seconds": 8.0, "scan_seconds": 1.0}


def install(app: Any, build: dict[str, Any]) -> None:
    """Install fixed read-only probes and wrap existing hooks in a dedicated E2E deployment."""
    from databricks_agentkit.runtime.types import InvocationFailedError, InvocationNotFoundError

    if getattr(app.state, "runtime_e2e_installed", False):
        raise ValueError("runtime E2E instrumentation is already installed")
    if set(build) - BUILD_FIELDS:
        raise ValueError("build metadata contains fields outside the safe allowlist")
    app.state.runtime_e2e_installed = True
    boot_id = str(uuid.uuid4())
    metadata = copy.deepcopy(build)
    runtime = getattr(app, "_runtime", None)
    source_hashes = {}
    for module_name in (
        "app",
        "runtime",
        "store",
        "types",
        "durability.lakebase_runtime_store",
    ):
        module = importlib.import_module(f"databricks_agentkit.runtime.{module_name}")
        module_path = module.__file__
        if module_path is None:
            raise ValueError(f"cannot verify source for {module_name}")
        source_hashes[module_name] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()

    @app.get("/api/runtime-e2e/build")
    async def build_identity() -> dict[str, Any]:
        return {
            **metadata,
            "pid": os.getpid(),
            "boot_id": boot_id,
            "durable": runtime.is_durable if runtime else False,
            "instrumented": runtime is not None,
            "timings": TIMINGS if runtime and runtime.is_durable else None,
            "installed_version": importlib.metadata.version("databricks-agentbricks"),
            "installed_source_sha256": source_hashes,
        }

    if runtime is None:
        return
    if runtime._started:
        raise ValueError("install probes before the application's lifespan starts")
    if app.auth_policy.requires_user:
        raise ValueError("these probes require an isolated service-auth template")
    if runtime.is_durable:
        for key, value in TIMINGS.items():
            setattr(runtime.executor, f"_{key}", value)
    runtime.poll_seconds = 0.2

    def wrap(hook):
        async def instrumented(value, context):
            payload = copy.deepcopy(value)
            controls = payload.pop("_runtime_e2e", {}) if isinstance(payload, dict) else {}
            if not isinstance(controls, dict) or set(controls) - {
                "delay_seconds",
                "fail",
                "crash_once",
            }:
                raise ValueError("invalid runtime E2E controls")
            delay = float(controls.get("delay_seconds", 0))
            if not 0 <= delay <= 45:
                raise ValueError("E2E delay must be between 0 and 45 seconds")

            async def marker(kind):
                await context.emit(
                    {
                        "type": f"runtime.e2e.{kind}",
                        "invocation_id": context.invocation_id,
                        "session_id": context.session_id,
                        "attempt": context.attempt,
                        "pid": os.getpid(),
                        "boot_id": boot_id,
                        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )

            await marker("start")
            if delay:
                await asyncio.sleep(delay)
            try:
                result = await hook(payload, context)
                if controls.get("fail"):
                    raise RuntimeError("intentional isolated runtime E2E failure")
                if controls.get("crash_once") and context.attempt == 1:
                    # The marker is committed before process exit; recovery must use Lakebase,
                    # not an in-process exception handler or the ephemeral app filesystem.
                    await marker("crash")
                    os._exit(86)
                await marker("end")
                return result
            except Exception:
                await marker("error")
                raise

        return instrumented

    if app._invoke_hook is None or app._recovery_hook is None:
        raise ValueError("register the real invoke and recovery hooks before installing probes")
    app._invoke_hook = wrap(app._invoke_hook)
    app._recovery_hook = wrap(app._recovery_hook)

    @app.get("/api/runtime-e2e/sessions/{session_id}")
    async def session_state(session_id: str) -> dict[str, Any]:
        state = await runtime.get_invocation(session_id=session_id)
        return {"invocation": dataclasses.asdict(state) if state else None}

    @app.get("/api/runtime-e2e/sessions/{session_id}/events")
    async def session_events(session_id: str, after: int = 0) -> dict[str, Any]:
        events = await runtime.get_events(session_id=session_id, after_sequence=after)
        return {"events": [dataclasses.asdict(event) for event in events]}

    @app.get("/api/runtime-e2e/invocations/{invocation_id}")
    async def invocation_state(invocation_id: str) -> dict[str, Any]:
        state = await runtime.get_invocation(invocation_id)
        if state is None:
            raise HTTPException(404, "invocation not found")
        return dataclasses.asdict(state)

    @app.get("/api/runtime-e2e/invocations/{invocation_id}/wait")
    async def wait(invocation_id: str, timeout_seconds: float = Query(default=0.2, gt=0, le=5)):
        try:
            result = await runtime.wait(invocation_id, timeout=timeout_seconds)
            return {"status": "completed", "output": result}
        except TimeoutError:
            raise HTTPException(408, "runtime wait timed out") from None
        except InvocationNotFoundError:
            raise HTTPException(404, "invocation not found") from None
        except InvocationFailedError:
            raise HTTPException(500, "agent invocation failed") from None
