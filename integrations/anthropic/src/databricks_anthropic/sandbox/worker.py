"""Claude Managed Agents worker that runs *inside* a Databricks Sandbox.

The launcher uploads this file into the sandbox and starts it detached; it must stay
self-contained (stdlib + ``anthropic`` + ``databricks-sdk``, plus ``mcp``/``httpx`` when
MCP servers are configured) because ``databricks-anthropic`` itself is not installed there.

Environment (set by the launcher under ``env -i`` with an allowlist; no secrets):

  ANTHROPIC_ENVIRONMENT_ID  self-hosted environment
  ANTHROPIC_WORK_ID / ANTHROPIC_SESSION_ID              the claimed work item
  CMA_SECRETS_FILE          0600 JSON with ANTHROPIC_WORK_SECRET only: the work item's secret, reduced by
                            the launcher to its sessions token, which acts on this one session (heartbeat,
                            events, attached memory stores) and nothing else. Read and deleted at startup,
                            before any tool runs, so it is not in this process's environment (which the
                            agent could read via /proc/<pid>/environ). It stays in this process's memory, and
                            tools run as the same Linux user. Never logged. The environment key is never
                            given to the worker; it refuses to start if one is present.
  CMA_ANTHROPIC_BASE_URL    Anthropic API URL (https; its own name, so a stray ANTHROPIC_BASE_URL is ignored)
  DBX_SANDBOX_NAME          this sandbox (heartbeat + self-stop)
  CMA_WORKDIR               agent working directory
  CMA_MAX_IDLE              seconds without session activity before the worker returns
  CMA_HEARTBEAT_SECONDS     self-exec heartbeat interval; only exec API calls count as
                            sandbox activity, so this keeps the sandbox up while the
                            worker runs and lets inactivity_timeout stop it if it dies
  CMA_ON_EXIT               "delete" (ephemeral, default) or "stop" (resumable): what to do with this
                            sandbox when the worker returns, after memory stores are flushed
  CMA_FINISHING_MARKER      written when this worker starts finishing (export, then delete or stop), so the
                            launcher replaces it instead of treating it as still serving the session
  CMA_REPLACING_MARKER      if this file exists on exit, a newer worker is replacing this one: skip the
                            delete/stop step so the replacement keeps its sandbox
  CMA_EXPORT_VOLUME         optional UC Volume folder; files under CMA_EXPORT_DIR are copied to
                            <CMA_EXPORT_VOLUME>/<session id>/ before the sandbox is deleted/stopped.
                            If the export fails the sandbox is stopped (not deleted) so nothing is lost
  CMA_MCP_SERVERS           JSON list of {"name", "url"}: Databricks MCP servers whose
                            tools are served as custom tools, authenticated with the
                            sandbox's built-in credential (the sandbox creator's identity)
  CMA_MCP_TIMEOUT_SECONDS   per-call MCP read timeout
  CMA_GENIE_WAIT_TIMEOUT    seconds genie_ask_and_wait polls before giving up (default 300)

A server entry with ``"wait_tool": true`` (Genie One) also gets ``genie_ask_and_wait``: it
calls ``genie_ask`` and polls ``genie_poll_response`` *inside the sandbox* until Genie
finishes, so the model makes one tool call instead of one per poll.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import signal
import threading
from datetime import timedelta
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.lib.environments import EnvironmentWorker

log = logging.getLogger("databricks_anthropic.worker")

# Display-name marker on a sandbox kept because its export failed; must equal dispatcher.KEPT_MARKER.
KEPT_MARKER = "[export failed, kept]"

# Partner attribution, fixed by the connector; must equal databricks_anthropic._useragent.
PRODUCT_NAME = "ClaudeManagedAgents-DatabricksSandbox"
PRODUCT_VERSION = "0.1.0"
UPSTREAM_ENV = {
    "DATABRICKS_SDK_UPSTREAM": PRODUCT_NAME,
    "DATABRICKS_SDK_UPSTREAM_VERSION": PRODUCT_VERSION,
    "DATABRICKS_CLI_UPSTREAM": PRODUCT_NAME,
    "DATABRICKS_CLI_UPSTREAM_VERSION": PRODUCT_VERSION,
}


def _workspace_client() -> Any:
    """The sandbox's built-in credential, with the connector's User-Agent."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(product=PRODUCT_NAME, product_version=PRODUCT_VERSION)


def _heartbeat(stop: threading.Event, name: str, interval: float) -> None:
    w = _workspace_client()
    while not stop.wait(interval):
        try:
            w.sandbox.execute_command_sync(name, "/bin/true")
        except Exception as e:  # noqa: BLE001 - keep beating; the next call may succeed
            log.warning("heartbeat failed: %s: %s", type(e).__name__, e)


def _export_outputs(session_id: str) -> bool:
    """Copy CMA_EXPORT_DIR to <CMA_EXPORT_VOLUME>/<session_id>/ with the sandbox credential. True on success."""
    volume, source = os.environ.get("CMA_EXPORT_VOLUME"), os.environ.get("CMA_EXPORT_DIR")
    if not volume or not source or not os.path.isdir(source):
        return True  # nothing to export
    if os.path.islink(source):  # the agent controls the workdir: don't follow outputs -> /
        log.warning("export: %s is a symlink; skipping export", source)
        return True
    w, count = _workspace_client(), 0
    try:
        for root, _dirs, files in os.walk(source, followlinks=False):
            for filename in files:
                local = os.path.join(root, filename)
                if os.path.islink(local) or not os.path.isfile(local):
                    log.warning("export: skipping %s (not a regular file)", local)
                    continue
                relative = os.path.relpath(local, source)
                with open(local, "rb") as f:
                    w.files.upload(f"{volume}/{session_id}/{relative}", f, overwrite=True)
                count += 1
        log.info("exported %d file(s) from %s to %s/%s/", count, source, volume, session_id)
        return True
    except Exception as e:  # noqa: BLE001
        log.error(
            "export to %s failed after %d file(s): %s: %s", volume, count, type(e).__name__, e
        )
        return False


def _being_replaced() -> bool:
    marker = os.environ.get("CMA_REPLACING_MARKER")
    return bool(marker and os.path.exists(marker))


def _mark_finishing() -> None:
    marker = os.environ.get("CMA_FINISHING_MARKER")
    if marker:
        with contextlib.suppress(OSError):
            with open(marker, "w") as f:
                f.write(str(os.getpid()))


def _mark_kept(w: Any, name: str, session_id: str) -> None:
    """Tag a kept sandbox in its display name (tags are immutable) so cleanup keeps it until retention."""
    from databricks.sdk.common.types.fieldmask import FieldMask
    from databricks.sdk.service.sandbox import Sandbox

    try:
        w.sandbox.update_sandbox(
            name,
            Sandbox(display_name=f"Claude session {session_id} {KEPT_MARKER}"),
            FieldMask(field_mask=["display_name"]),
        )
    except Exception as e:  # noqa: BLE001 - best effort; the sandbox is still stopped, not deleted
        log.warning("could not mark %s as kept: %s: %s", name, type(e).__name__, e)


def _finish(name: str, session_id: str, *, attached: bool = True) -> None:
    """Export outputs, then delete (ephemeral) or stop (resumable) this sandbox.

    ``attached=False`` (start-up failed before the session was served): leave the sandbox alone, so the
    launcher can read this worker's log and delete it.
    """
    _mark_finishing()
    if not attached:
        log.error("worker failed before serving the session: keeping %s for the launcher", name)
        return
    action = os.environ.get("CMA_ON_EXIT", "delete")
    if _being_replaced():
        # A newer worker for this session is starting in this sandbox: leave the sandbox alone.
        log.info("replaced by a newer worker: keeping %s", name)
        action = "keep"
    exported = _export_outputs(session_id)
    kept = False
    if action == "delete" and not exported:
        log.warning(
            "export failed: stopping %s instead of deleting it, so its files are kept", name
        )
        action, kept = "stop", True
    # The export can take a while: a replacement may have started meanwhile. Check again right before acting.
    if action != "keep" and _being_replaced():
        log.info("replaced by a newer worker during export: keeping %s", name)
        action = "keep"
    try:
        w = _workspace_client()
        if kept and action == "stop":
            _mark_kept(w, name, session_id)
        if action == "delete":
            log.info("deleting %s (ephemeral)", name)
            w.sandbox.delete_sandbox(name)
        elif action == "stop":
            log.info("stopping %s (resumable; files kept)", name)
            w.sandbox.stop_sandbox(name)
    except Exception as e:  # noqa: BLE001
        log.warning("%s of %s failed: %s: %s", action, name, type(e).__name__, e)


def _mcp_major() -> int:
    from importlib.metadata import version

    return int(version("mcp").split(".")[0])


def _mcp_http_module() -> Any:
    """The HTTP library the installed mcp transport is built on: httpx2 for mcp 2.x, httpx for 1.x."""
    if _mcp_major() >= 2:
        import httpx2

        return httpx2
    import httpx  # ty: ignore[unresolved-import] - mcp 1.x only

    return httpx


def _mcp_read_timeout(seconds: float) -> Any:
    """mcp 2.x takes float seconds for read_timeout_seconds; 1.x takes a timedelta."""
    return seconds if _mcp_major() >= 2 else timedelta(seconds=seconds)


GENIE_TERMINAL = {"completed", "incomplete", "failed", "cancelled"}


def _mcp_payload(result: Any) -> dict[str, Any]:
    """JSON payload of a CallToolResult (mcp 1.x or 2.x field names), always as a dict.

    A JSON object is returned as is; any other JSON value becomes ``{"items": value}`` and text that
    isn't JSON becomes ``{"text": text}``, so callers can always use ``.get``.
    """
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        return structured if isinstance(structured, dict) else {"items": structured}
    text = "\n".join(
        getattr(c, "text", "") for c in (result.content or []) if getattr(c, "type", "") == "text"
    )
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"text": text}
    return value if isinstance(value, dict) else {"items": value}


def _mcp_error(result: Any) -> str | None:
    """Error text of a failed CallToolResult (mcp 1.x ``isError`` / 2.x ``is_error``), else None."""
    if not (getattr(result, "is_error", None) or getattr(result, "isError", None)):
        return None
    text = "\n".join(
        getattr(c, "text", "") for c in (result.content or []) if getattr(c, "type", "") == "text"
    )
    return text or "MCP tool call failed"


def _clip(value: Any, limit: int) -> Any:
    return (
        value[:limit] + "...[truncated]" if isinstance(value, str) and len(value) > limit else value
    )


def _genie_wait_tool(session: Any) -> Any:
    from anthropic import beta_async_tool

    poll_s = float(os.environ.get("CMA_GENIE_POLL_SECONDS", "3"))
    timeout_s = float(os.environ.get("CMA_GENIE_WAIT_TIMEOUT", "300"))

    @beta_async_tool
    async def genie_ask_and_wait(question: str, conversation_id: str | None = None) -> str:
        """Ask Databricks Genie One a data question and wait for its final answer.

        Args:
            question: The natural-language data question.
            conversation_id: Optional id of an earlier Genie conversation, for follow-up questions.
        """
        arguments: dict[str, Any] = {"question": question}
        if conversation_id:
            arguments["conversation_id"] = conversation_id
        result = await session.call_tool("genie_ask", arguments)
        if (error := _mcp_error(result)) is not None:
            return json.dumps({"status": "error", "error": _clip(error, 4000)})
        first = _mcp_payload(result)
        conv, resp = first.get("conversation_id"), first.get("response_id")
        status, final, polls, error = first.get("status"), first, 0, None
        if status not in GENIE_TERMINAL and not (conv and resp):
            # Nothing to poll (an immediate answer or an unexpected shape): return what Genie sent.
            status = "completed"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while status not in GENIE_TERMINAL and conv and resp and loop.time() < deadline:
            await asyncio.sleep(poll_s)
            result = await session.call_tool(
                "genie_poll_response", {"conversation_id": conv, "response_id": resp}
            )
            polls += 1
            if (error := _mcp_error(result)) is not None:
                status = "error"
                break
            final = _mcp_payload(result)
            status = final.get("status")
        if status not in GENIE_TERMINAL and status != "error":
            status = f"timed_out_after_{int(timeout_s)}s"
            if conv and resp:  # don't leave the Genie query running
                with contextlib.suppress(Exception):
                    await session.call_tool(
                        "genie_cancel_response", {"conversation_id": conv, "response_id": resp}
                    )
        log.info("genie_ask_and_wait status=%s polls=%d", status, polls)
        answer = {
            "status": status,
            "conversation_id": conv,
            "response_id": resp,
            "final_answer": _clip(final.get("final_answer"), 8000),
            "deep_link": final.get("deep_link"),
            "query_items": [
                {"item_id": q.get("item_id"), "sql": _clip(q.get("sql"), 4000)}
                for q in (final.get("query_items") or [])[:10]
            ],
        }
        if error is not None:
            answer["error"] = _clip(error, 4000)
        if answer["final_answer"] is None and ("text" in final or "items" in final):
            # Not Genie's usual shape: pass the content through rather than drop it.
            raw = final.get("text") if "text" in final else json.dumps(final["items"])
            answer["content"] = _clip(raw, 8000)
        return json.dumps(answer)  # fields are clipped, so this stays valid JSON

    return genie_ask_and_wait


async def _open_mcp_tools(
    stack: contextlib.AsyncExitStack, servers: list[dict[str, str]]
) -> list[Any]:
    """Connect to each Databricks MCP server and convert its tools for the worker."""
    if not servers:
        return []
    from anthropic.lib.tools.mcp import async_mcp_tool
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client  # mcp >= 1.24

    http_lib = _mcp_http_module()
    config = _workspace_client().config

    class _DatabricksAuth(http_lib.Auth):
        # The sandbox credential rotates hourly: resolve headers per request.
        def auth_flow(self, request):
            request.headers.update(config.authenticate())
            request.headers["User-Agent"] = config.user_agent
            yield request

    timeout = float(os.environ.get("CMA_MCP_TIMEOUT_SECONDS", "120"))
    tools: list[Any] = []
    for server in servers:
        http = await stack.enter_async_context(
            http_lib.AsyncClient(auth=_DatabricksAuth(), timeout=http_lib.Timeout(timeout))
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(server["url"], http_client=http)
        )
        session = await stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=_mcp_read_timeout(timeout))
        )
        await session.initialize()
        listed = await session.list_tools()
        exclude = set(server.get("exclude", []))
        converted = [async_mcp_tool(t, session) for t in listed.tools if t.name not in exclude]
        if server.get("wait_tool"):
            converted.append(_genie_wait_tool(session))
        log.info(
            "MCP %s: serving %s", server.get("name", server["url"]), [t.name for t in converted]
        )
        tools.extend(converted)
    return tools


def _sessions_token(secret: str | None) -> str | None:
    """The sessions token inside the work secret (URL-safe base64 JSON). Never log it."""
    if not secret:
        return None
    try:
        parsed = json.loads(base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4)))
    except (ValueError, TypeError):
        return None
    token = parsed.get("sessions_token") if isinstance(parsed, dict) else None
    return token if isinstance(token, str) and token else None


def _session_credential(secrets: dict[str, str]) -> tuple[str, str]:
    """(work secret, sessions token); fails closed on an environment key or a missing sessions token."""
    if "ANTHROPIC_ENVIRONMENT_KEY" in secrets or os.environ.get("ANTHROPIC_ENVIRONMENT_KEY"):
        # The SDK would fall back to it; a sandbox must only ever hold the per-session token.
        raise RuntimeError("an environment key reached the sandbox; refusing to start")
    work_secret = secrets.get("ANTHROPIC_WORK_SECRET") or ""
    token = _sessions_token(work_secret)
    if token is None:
        raise RuntimeError("the work secret carries no sessions token; refusing to start")
    return work_secret, token


def _load_secrets() -> dict[str, str]:
    """Read and delete CMA_SECRETS_FILE (written 0600 by the launcher) before any tool can run."""
    path = os.environ.pop("CMA_SECRETS_FILE", "")
    if not path:
        raise RuntimeError(
            "CMA_SECRETS_FILE is not set; the launcher passes secrets through this file"
        )
    try:
        with open(path) as f:
            return json.load(f)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


async def main() -> None:
    name = os.environ["DBX_SANDBOX_NAME"]
    stop = threading.Event()
    attached = (
        False  # True once the session is being served; a failure before that keeps the sandbox
    )
    try:
        work_secret, sessions_token = _session_credential(_load_secrets())
        workdir = os.environ.get("CMA_WORKDIR", "/home/sandbox-agent/agent-workspace")
        os.makedirs(workdir, exist_ok=True)
        base_url = os.environ.get("CMA_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        if not base_url.startswith("https://"):
            raise RuntimeError(f"CMA_ANTHROPIC_BASE_URL must use https, not {base_url!r}")
        servers = json.loads(os.environ.get("CMA_MCP_SERVERS") or "[]")
        os.environ.update(
            UPSTREAM_ENV
        )  # partner attribution for SDK/CLI calls from this process and its tools
        log.info("databricks user-agent: %s", _workspace_client().config.user_agent)

        interval = float(os.environ.get("CMA_HEARTBEAT_SECONDS", "30"))
        threading.Thread(target=_heartbeat, args=(stop, name, interval), daemon=True).start()
        log.info(
            "worker starting session=%s work=%s sandbox=%s workdir=%s mcp_servers=%d",
            os.environ.get("ANTHROPIC_SESSION_ID"),
            os.environ.get("ANTHROPIC_WORK_ID"),
            name,
            workdir,
            len(servers),
        )
        async with contextlib.AsyncExitStack() as stack:
            client = await stack.enter_async_context(
                AsyncAnthropic(auth_token=sessions_token, base_url=base_url)
            )
            mcp_tools = await _open_mcp_tools(stack, servers)
            tools = None
            if mcp_tools:
                from anthropic.lib.tools.agent_toolset import beta_agent_toolset_20260401

                def tools(ctx):
                    return [*beta_agent_toolset_20260401(ctx), *mcp_tools]

            worker = EnvironmentWorker(
                client,
                workdir=workdir,
                tools=tools,
                max_idle=float(os.environ.get("CMA_MAX_IDLE", "300")),
            )
            attached = True
            task = asyncio.create_task(
                worker.handle_item(
                    work_id=os.environ["ANTHROPIC_WORK_ID"],
                    environment_id=os.environ["ANTHROPIC_ENVIRONMENT_ID"],
                    session_id=os.environ["ANTHROPIC_SESSION_ID"],
                    work_secret=work_secret,
                )
            )
            # Cancel (not kill) on SIGTERM so memory files upload before exit.
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signum, task.cancel)
            with contextlib.suppress(asyncio.CancelledError):
                await task
        log.info("worker finished")
    finally:
        # Covers startup failures too, so an ephemeral sandbox is always cleaned up.
        stop.set()
        _finish(name, os.environ.get("ANTHROPIC_SESSION_ID", "unknown-session"), attached=attached)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(main())
