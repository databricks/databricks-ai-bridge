"""Developer terminal: authenticated App WebSocket to a session's sandbox."""

import asyncio
import base64
import hashlib
import logging
import os
import re
import shlex

from databricks.sdk import WorkspaceClient
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.lakebox import RemoteError
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, SessionBusy
from databricks_agentkit.sandbox.terminal import GrpcTerminal

LOG = logging.getLogger(__name__)


class TerminalRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=512)
    command: str | None = Field(default=None, min_length=1, max_length=32768)
    term: str = Field(default="xterm-256color", pattern=r"^[A-Za-z0-9_.+-]{1,64}$")
    columns: int = Field(default=80, ge=1, le=500)
    rows: int = Field(default=24, ge=1, le=500)


def authorize(headers, peer, workspace):
    """Verify workspace credentials and CAN_MANAGE for this App, before SSH access."""
    if headers.get("origin") is not None:
        raise HTTPException(403, "Debug terminals are available through the CLI")
    app_name = os.getenv("DATABRICKS_APP_NAME")
    if not app_name:
        if os.getenv("SANDBOX_ALLOW_LOCAL_DEBUG") == "true" and peer in {"127.0.0.1", "::1"}:
            return "local-developer"
        raise HTTPException(403, "Local debug requires SANDBOX_ALLOW_LOCAL_DEBUG=true and loopback")
    authorization = headers.get("x-agentbricks-debug-authorization", "")
    if not authorization.startswith("Bearer ") or not authorization[7:].strip():
        raise HTTPException(403, "Developer OAuth and CAN_MANAGE on this App are required")
    # Never use the App's own credentials to establish the developer's permissions.
    # This token is used only for these reads and is never logged or persisted.
    developer = WorkspaceClient(
        host=workspace.config.host,
        token=authorization[7:],
        auth_type="pat",
    )
    me = developer.current_user.me()
    groups = {group.display for group in me.groups or [] if group.display}
    permissions = developer.apps.get_permissions(app_name)
    for entry in permissions.access_control_list or []:
        matches = (
            bool(me.user_name)
            and entry.user_name == me.user_name
            or bool(me.user_name)
            and entry.service_principal_name == me.user_name
            or bool(entry.group_name)
            and entry.group_name in groups
        )
        if matches and any(
            permission.permission_level.value == "CAN_MANAGE"
            for permission in entry.all_permissions or []
            if permission.permission_level is not None
        ):
            return me.user_name
    raise HTTPException(403, "Debugging requires CAN_MANAGE on this App")


def open_session(state, environment, request):
    """Resolve/resume existing compute under the same lock used by invocations."""
    agent_id = state.environments.session_agent(environment, request.session_id)
    agent = state.control.agent(agent_id)
    orchestrator = state.orchestrator
    key = orchestrator.execution_session_id(scoped_session(agent_id, request.session_id))
    with orchestrator.sandboxes.invocation(key):
        runtime = orchestrator.sandboxes.runtime(key)
        row = runtime["sandbox"]
        if not row or not row["sandbox_id"]:
            raise HTTPException(
                404, "Session has no sandbox; send its first message before debugging"
            )
        if row["status"] not in {"ready", "stopped"}:
            raise HTTPException(
                409, "Session installation is not ready; finish or retry its invocation"
            )
        try:
            sandbox = orchestrator.sandboxes.get_or_create(
                key,
                source_hash=row["source_hash"],
                allow_create=False,
            ).sandbox
        except RuntimeError as exc:
            raise HTTPException(404, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        digest = row["source_hash"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid source hash in session mapping")
        deployment = hashlib.sha256(orchestrator.namespace.encode()).hexdigest()[:16]
        relative = f".agentbricks-prototype/{deployment}/{digest}/source"
        if artifact := agent.get("execution_artifact"):
            artifact_hash = artifact["sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", artifact_hash):
                raise ValueError("Invalid artifact hash in agent revision")
            relative = f".agentbricks-prototype/{deployment}/{digest}/{artifact_hash}/artifact"
            setup = f'cd -- "$HOME"/{relative} && . ../agent-env/bin/activate'
        else:
            setup = f'cd -- "$HOME"/{relative} && . .venv/bin/activate'
        # A separate SSH adapter keeps the invocation HTTPS transport unchanged.
        lakebox = state.debug_lakebox
        try:
            with lakebox.connect(sandbox) as remote:
                remote.run(
                    f'test -f "$HOME"/{relative.rsplit("/", 1)[0]}/ready.json && {setup}',
                    timeout=30,
                )
        except RemoteError as exc:
            raise HTTPException(
                409, "Installed agent files are missing; recover the session before debugging"
            ) from exc
        command = (
            f"bash -c {shlex.quote(setup + ' && exec bash -c ' + shlex.quote(request.command))}"
            if request.command is not None
            else None
        )
        transport = getattr(state, "debug_transport", "ssh")
        if transport == "https":
            client = channel = GrpcTerminal(
                orchestrator.lakebox.workspace, row["sandbox_id"], setup, request
            )
        else:
            client, channel = lakebox.open_terminal(
                sandbox, command, term=request.term, columns=request.columns, rows=request.rows
            )
        try:
            if command is None and transport == "ssh":
                channel.sendall((setup + "\n").encode())
            orchestrator.sandboxes.record(key, digest, row["sandbox_id"], "ready")
        except BaseException:
            client.close()
            raise
    return (
        client,
        channel,
        {
            "type": "ready",
            "session_id": request.session_id,
            "agent_id": agent_id,
            "sandbox_id": row["sandbox_id"],
            "source_directory": f"$HOME/{relative}",
            "transport": transport,
        },
    )


async def relay(websocket, channel):
    async def output():
        while True:
            for stream, ready, receive in (
                ("stdout", channel.recv_ready, channel.recv),
                ("stderr", channel.recv_stderr_ready, channel.recv_stderr),
            ):
                if ready():
                    chunk = receive(32768)
                    if chunk:
                        await websocket.send_json(
                            {"type": stream, "data": base64.b64encode(chunk).decode("ascii")}
                        )
            if (
                (channel.exit_status_ready() or channel.closed)
                and not channel.recv_ready()
                and not channel.recv_stderr_ready()
            ):
                code = channel.recv_exit_status()
                await websocket.send_json({"type": "exit", "code": code if code >= 0 else 255})
                return
            await asyncio.sleep(0.02)

    async def input_():
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "stdin":
                data = base64.b64decode(message.get("data", ""), validate=True)
                if len(data) > 32768:
                    raise ValueError("Terminal input exceeds 32768 bytes")
                await run_in_threadpool(channel.sendall, data)
            elif message.get("type") == "resize":
                columns, rows = int(message["columns"]), int(message["rows"])
                if not 1 <= columns <= 500 or not 1 <= rows <= 500:
                    raise ValueError("Invalid terminal dimensions")
                await run_in_threadpool(channel.resize_pty, width=columns, height=rows)
            elif message.get("type") == "eof":
                await run_in_threadpool(channel.shutdown_write)
            else:
                raise ValueError("Unknown terminal message")

    tasks = {asyncio.create_task(output()), asyncio.create_task(input_())}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        # Stop output before closing, so cleanup cannot manufacture an exit event.
        for task in tasks:
            task.cancel()
        # Closing also unblocks a sender whose SSH window or input queue is full.
        await run_in_threadpool(channel.close)
        await asyncio.gather(*tasks, return_exceptions=True)


async def terminal(websocket: WebSocket, environment: str):
    state = websocket.app.state
    client = None
    try:
        principal = await run_in_threadpool(
            authorize,
            websocket.headers,
            websocket.client.host if websocket.client else "",
            state.orchestrator.lakebox.workspace,
        )
    except Exception:  # fail closed; no credential-bearing exception reaches the socket/logs
        await websocket.close(code=1008, reason="Developer OAuth and App CAN_MANAGE are required")
        return
    await websocket.accept()
    try:
        request = TerminalRequest.model_validate(
            await asyncio.wait_for(websocket.receive_json(), 15)
        )
        client, channel, metadata = await run_in_threadpool(
            open_session, state, environment, request
        )
        LOG.info(
            "Debug terminal opened principal=%s session=%s sandbox=%s transport=%s",
            principal,
            request.session_id,
            metadata["sandbox_id"],
            metadata["transport"],
        )
        await websocket.send_json(metadata)
        await relay(websocket, channel)
    except WebSocketDisconnect:
        pass
    except (HTTPException, SessionBusy, ExecutionIdentityConflict, KeyError) as exc:
        await websocket.send_json(
            {"type": "error", "message": exc.detail if isinstance(exc, HTTPException) else str(exc)}
        )
    except (ValueError, ValidationError):
        await websocket.send_json({"type": "error", "message": "Invalid terminal request"})
    except Exception as exc:
        # In particular, never echo SDK request headers or developer credentials.
        LOG.warning("Debug terminal failed: %s", type(exc).__name__)
        await websocket.send_json(
            {"type": "error", "message": f"Terminal connection failed ({type(exc).__name__})"}
        )
    finally:
        if client is not None:
            await run_in_threadpool(client.close)
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass
