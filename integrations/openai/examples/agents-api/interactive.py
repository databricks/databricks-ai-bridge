# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.36", "databricks-sdk>=0.153.0"]
# ///
"""Interactive demo: converse with an OpenAI Agents API agent that runs in a Databricks sandbox.

The application owns the whole lifecycle locally: it creates a self-hosted Agents API
session, provisions a Databricks sandbox and launches ``codex exec-server`` inside it
(all outbound), then drops you into a prompt. Whatever you type is sent to the agent as
a turn; the agent decides which shell commands to run in the sandbox, and this script
streams back the command it ran, the command output, and the agent's reply.

Run it (no venv needed, uv resolves the inline deps). Provide two OpenAI keys via
OPENAI_API_KEY (application key) and OPENAI_EXECUTOR_API_KEY (Agents-tab environment
key), or via --app-key-file / --executor-key-file:

    OPENAI_API_KEY=... OPENAI_EXECUTOR_API_KEY=... uv run interactive.py --profile <profile>

Type an instruction and press enter. Slash commands: /help, /sh <cmd>, /info, /verbose, /quit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import shlex
import sys
import uuid
from pathlib import Path

from databricks.sdk import WorkspaceClient
from openai import APIStatusError, AsyncOpenAI, AuthenticationError, NotFoundError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import (  # noqa: E402  (local import after sys.path setup)
    WORK_DIR,
    SandboxCommandError,
    confirm_executor,
    delete_sandbox,
    prepare_workspace,
    provision_sandbox,
    run_command,
    start_executor,
    stop_executor,
)

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(text: str) -> str:
    return _c("2", text)


def bold(text: str) -> str:
    return _c("1", text)


def cyan(text: str) -> str:
    return _c("36", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


HELP = """\
Type a natural-language instruction; the agent runs commands in the sandbox and replies.
The sandbox is authenticated to your Databricks workspace, so the agent can query Unity
Catalog and SQL warehouses as you.
Slash commands:
  /demo           run a canned analysis of samples.nyctaxi and save a chart (then /pull it)
  /pull <path>    copy a file out of the sandbox to ./pulled/ (path relative to the work dir)
  /sh <command>   run a command directly in the sandbox (ground truth, bypasses the agent)
  /info           show session id, sandbox name, model, work dir
  /verbose        toggle showing the agent's reasoning summary
  /help           show this help
  /quit           end the session and tear down (Ctrl-D does the same)
Ctrl-C cancels the in-flight turn without quitting."""


def sandbox_egress_ip(workspace_client, name: str) -> str:
    """Best-effort: the sandbox's public egress IP (what the OpenAI org would allowlist)."""
    try:
        out = run_command(
            workspace_client,
            name,
            "curl -s --max-time 10 https://api.ipify.org || true",
            timeout_s=20,
            check=False,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


# Item types already surfaced as their own stream (the text reply and reasoning summary);
# labeling these again would just duplicate what is already on screen.
_UNLABELED_ITEMS = {"message", "reasoning", "command_execution"}


def _tool_label(item) -> str | None:
    """A short bracketed label for a non-shell agent tool call, or None to stay quiet.

    Keeps the demo honest: shell commands print as `$ ...`; every other tool the agent
    invokes (function call, MCP call, web search) still shows on screen instead of passing
    silently. Reasoning and the text reply have their own streams, so they are skipped.
    """
    if item is None:
        return None
    item_type = getattr(item, "type", None)
    if item_type is None or item_type in _UNLABELED_ITEMS:
        return None
    if item_type == "function_call":
        return f"[function call: {getattr(item, 'name', '?')}]"
    if item_type == "mcp_call":
        server = getattr(item, "server_label", None)
        name = getattr(item, "name", "?")
        return f"[mcp call: {server}/{name}]" if server else f"[mcp call: {name}]"
    if item_type == "web_search_call":
        action = getattr(item, "action", None)
        query = getattr(action, "query", None) if action is not None else None
        return f"[web search: {query}]" if query else "[web search]"
    return f"[{item_type}]"


def _handle_event(event, state: dict, *, verbose: bool) -> bool:
    """Render one streamed session event. Return True when the turn is done.

    Shared by the input stream and teardown logic so the agent's commands, their
    output, and its reply are surfaced the same way everywhere.
    """
    kind = event.type
    if kind == "agent.session.turn.item.added":
        item = getattr(event, "item", None)
        if item is not None and getattr(item, "type", None) == "command_execution":
            command = getattr(item, "command", "") or ""
            if command:
                print("\n" + dim("$ " + command), flush=True)
            state["replying"] = False
        else:
            label = _tool_label(item)
            if label is not None:
                print("\n" + dim(label), flush=True)
                state["replying"] = False
    elif kind == "agent.output.command_execution_output.delta":
        print(dim(event.delta), end="", flush=True)
    elif kind == "agent.session.turn.item.done":
        item = getattr(event, "item", None)
        if item is not None and getattr(item, "type", None) == "command_execution":
            code = getattr(item, "exit_code", None)
            if code is not None:
                print(dim(f"[exit {code}]"), flush=True)
    elif kind == "agent.session.turn.reasoning_summary_text.delta":
        if verbose:
            print(dim(event.delta), end="", flush=True)
    elif kind == "agent.session.turn.output_text.delta":
        if not state["replying"]:
            print("\n" + bold(green("agent> ")), end="", flush=True)
            state["replying"] = True
        print(event.delta, end="", flush=True)
    elif kind in (
        "agent.session.turn.completed",
        "agent.session.turn.failed",
        "agent.session.turn.cancelled",
    ):
        turn = getattr(event, "turn", None)
        if turn is not None and getattr(turn, "subagent_id", None) is not None:
            return False
        print(flush=True)
        if kind == "agent.session.turn.failed":
            raise RuntimeError("turn failed")
        if kind == "agent.session.turn.cancelled":
            print(yellow("[turn cancelled]"), flush=True)
        return True
    elif kind in ("error", "agent.session.failed", "agent.session.environment.failed"):
        raise RuntimeError(f"session failed: {kind}")
    return False


async def converse(client, session_id: str, user_input: str, *, verbose: bool) -> None:
    """Submit one turn on an idle session and stream it to completion."""
    state = {"replying": False}
    async with asyncio.timeout(900):
        async with client.beta.agents.sessions.stream(session_id, input=user_input) as events:
            async for event in events:
                if _handle_event(event, state, verbose=verbose):
                    return
    raise RuntimeError("stream ended without a completed turn")


async def _wait_until_idle(client, session_id: str, *, timeout_s: int = 180) -> str | None:
    """Poll session status until it leaves in_progress, or timeout. Never hangs.

    openai 3.36 has no turn-cancel API and a new turn needs an idle session, so after a
    failed or interrupted turn we wait for the one still running server-side to finish.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    status = None
    while loop.time() < deadline:
        try:
            session = await client.beta.agents.sessions.retrieve(session_id)
        except Exception:
            return status
        status = session.status
        if status != "in_progress":
            return status
        await asyncio.sleep(1)
    return status


async def handle_input(client, session_id: str, user_input: str, *, verbose: bool) -> None:
    """Run one user turn, recovering the session to idle if it errors or is interrupted.

    Does not auto-resend on error: the turn may have run server-side despite a dropped
    stream, so re-sending could double-execute. It only makes the session ready again.
    """
    try:
        await converse(client, session_id, user_input, verbose=verbose)
        return
    except KeyboardInterrupt:
        print(yellow("\n[detaching; the turn keeps running server-side]"), flush=True)
    except Exception as error:
        if "idle session" in str(error):
            print(yellow("\n[a previous turn is still running; wait a moment, then resend]"), flush=True)
        else:
            print(red(f"\nturn error: {error}"), flush=True)
    status = await _wait_until_idle(client, session_id)
    if status == "in_progress":
        print(dim("[turn still running; your next message will wait for it to finish]"), flush=True)


async def _cleanup_session(client, session_id: str) -> None:
    """Delete the session at teardown, tolerating a turn still in progress.

    Not common.delete_session: that calls sessions.cancel, which openai 3.36 removed.
    """
    await _wait_until_idle(client, session_id, timeout_s=60)
    for attempt in range(5):
        try:
            await client.beta.agents.sessions.delete(session_id)
            return
        except NotFoundError:
            return
        except APIStatusError as error:
            if getattr(error, "status_code", None) == 409 and attempt < 4:
                await _wait_until_idle(client, session_id, timeout_s=30)
                continue
            raise


PULLED_DIR = ROOT / "pulled"

DEMO_PROMPT = (
    "You are already authenticated to Databricks in this sandbox (the `databricks` CLI and "
    "the databricks-sdk WorkspaceClient work with no setup). Using the Databricks SDK, run "
    "SQL on the 'Serverless Starter Warehouse' SQL warehouse to compute the average "
    "fare_amount by pickup hour of day from samples.nyctaxi.trips. Load the result into a "
    "pandas DataFrame, plot a bar chart of average fare by hour with matplotlib, and save it "
    f"to {WORK_DIR}/fare_by_hour.png. Then tell me which hour has the highest average fare. "
    "pandas, matplotlib and databricks-sdk are preinstalled; pip install anything else you need."
)


def _pull(workspace_client, name: str, remote: str, local_name: str | None = None) -> None:
    """Copy a file out of the sandbox to ./pulled/ via base64 over execute_command_sync."""
    path = remote if remote.startswith("/") else f"{WORK_DIR}/{remote}"
    try:
        resp = run_command(workspace_client, name, f"base64 -w0 {shlex.quote(path)}", check=False)
    except Exception as error:
        print(red(f"pull failed: {error}"), flush=True)
        return
    if resp.exit_code != 0:
        print(red(f"pull failed (exit {resp.exit_code}): {resp.stderr.strip() or path}"), flush=True)
        return
    try:
        data = base64.b64decode(resp.stdout.strip())
    except Exception as error:
        print(red(f"could not decode {path}: {error}"), flush=True)
        return
    if getattr(resp, "truncated", False):
        print(yellow("warning: sandbox output was truncated; pulled file may be incomplete"), flush=True)
    PULLED_DIR.mkdir(exist_ok=True)
    local = PULLED_DIR / (local_name or Path(path).name)
    local.write_bytes(data)
    print(green(f"pulled {path} -> {local}  ({len(data)} bytes)"), flush=True)


def _run_sh(workspace_client, name: str, command: str) -> None:
    try:
        resp = run_command(workspace_client, name, command, check=False)
    except Exception as error:  # surface, do not crash the REPL
        print(red(f"sandbox error: {error}"), flush=True)
        return
    if resp.stdout:
        print(resp.stdout, end="" if resp.stdout.endswith("\n") else "\n", flush=True)
    if resp.stderr:
        print(dim(resp.stderr), end="" if resp.stderr.endswith("\n") else "\n", flush=True)
    print(dim(f"[exit {resp.exit_code}]"), flush=True)


async def repl(client, session_id: str, workspace_client, name: str, model: str, *, verbose: bool) -> None:
    while True:
        try:
            line = await asyncio.to_thread(input, cyan("you> "))
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            name_part, _, rest = line[1:].partition(" ")
            cmd = name_part.lower()
            if cmd in ("quit", "q", "exit"):
                return
            if cmd in ("help", "h", "?"):
                print(HELP, flush=True)
            elif cmd == "info":
                print(
                    dim(
                        f"session={session_id}\nsandbox={name}\nmodel={model}\nwork_dir={WORK_DIR}\n"
                        f"verbose={verbose}"
                    ),
                    flush=True,
                )
            elif cmd == "verbose":
                verbose = not verbose
                print(dim(f"verbose = {verbose}"), flush=True)
            elif cmd == "sh":
                if not rest.strip():
                    print(dim("usage: /sh <command>"), flush=True)
                else:
                    _run_sh(workspace_client, name, rest)
            elif cmd == "pull":
                parts = rest.split()
                if not parts:
                    print(dim("usage: /pull <sandbox-path> [local-name]"), flush=True)
                else:
                    _pull(workspace_client, name, parts[0], parts[1] if len(parts) > 1 else None)
            elif cmd == "demo":
                await handle_input(client, session_id, DEMO_PROMPT, verbose=verbose)
            else:
                print(dim(f"unknown command: /{cmd} (try /help)"), flush=True)
            continue
        await handle_input(client, session_id, line, verbose=verbose)


def _banner(session_id: str, name: str, model: str) -> None:
    print()
    print(bold("Databricks sandbox  x  OpenAI Agents API  -  interactive demo"))
    print(dim(f"  session {session_id}"))
    print(dim(f"  sandbox {name}  (model {model}, cwd {WORK_DIR})"))
    print(dim("  the sandbox is authenticated to your Databricks workspace"))
    print(dim("  try /demo for a live samples.nyctaxi analysis, /help for commands, /quit to exit"))
    print()


def _resolve_key(env_name: str, file_path: str | None) -> str | None:
    """Resolve a key from the env var, else a --*-key-file path.

    Reading from a file keeps the secret out of argv and the shell history.
    """
    if os.environ.get(env_name):
        return os.environ[env_name]
    if file_path:
        path = Path(file_path).expanduser()
        if path.exists():
            return path.read_text().strip()
    return None


async def main(args) -> None:
    app_key = _resolve_key("OPENAI_API_KEY", args.app_key_file)
    if not app_key:
        raise SystemExit(
            "no OpenAI application key: set OPENAI_API_KEY or pass --app-key-file "
            "(drives session creation)"
        )
    executor_key = _resolve_key("OPENAI_EXECUTOR_API_KEY", args.executor_key_file)
    if not executor_key:
        raise SystemExit(
            "no OpenAI environment key: set OPENAI_EXECUTOR_API_KEY or pass --executor-key-file. "
            "It must be an environment key created on the OpenAI Agents tab (every other "
            "permission None); a general API key is rejected with "
            "'missing required scope api.agents.environments.connect'"
        )

    workspace_client = WorkspaceClient(profile=args.profile)
    name = None
    pid = None
    session = None
    async with AsyncOpenAI(api_key=app_key, timeout=120) as client:
        try:
            print(dim("Creating OpenAI self-hosted session..."), flush=True)
            try:
                session = await client.beta.agents.sessions.create(
                    agent={"model": args.model},
                    environment={"type": "self_hosted", "workspace_directory": WORK_DIR},
                )
            except AuthenticationError:
                raise SystemExit(
                    "OpenAI rejected the application key (401). Provide a current OPENAI_API_KEY "
                    "for the org whose Agents API is enabled."
                )
            print(green(f"  session {session.id}"), flush=True)

            print(dim(f"Provisioning sandbox {args.sandbox_id}..."), flush=True)
            name = provision_sandbox(
                workspace_client, args.sandbox_id, inactivity_timeout_s=args.inactivity_timeout
            )
            prepare_workspace(workspace_client, name)
            print(green(f"  {name} RUNNING"), flush=True)

            egress_ip = sandbox_egress_ip(workspace_client, name)
            print(dim(f"  sandbox egress IP: {egress_ip}"), flush=True)

            print(dim("Launching codex exec-server in the sandbox..."), flush=True)
            pid = start_executor(
                workspace_client,
                name,
                environment_id=session.environment.id,
                remote_url=session.environment.remote_url,
                executor_api_key=executor_key,
            )
            try:
                confirm_executor(workspace_client, name, pid)
            except SandboxCommandError:
                print(red("codex exec-server did not stay connected to OpenAI."), flush=True)
                print(
                    "  If your OpenAI org enforces an IP allowlist, add the sandbox egress IP "
                    f"({egress_ip}) on the org (platform.openai.com settings), or disable the "
                    "allowlist for the test org, then re-run.",
                    flush=True,
                )
                raise
            print(green("  executor connected"), flush=True)

            _banner(session.id, name, args.model)
            await repl(client, session.id, workspace_client, name, args.model, verbose=args.verbose)
        finally:
            if session is not None or name is not None:
                print(dim("\nTearing down..."), flush=True)
            try:
                if pid is not None:
                    stop_executor(workspace_client, name, pid)
            finally:
                try:
                    if session is not None:
                        await _cleanup_session(client, session.id)
                finally:
                    if name is not None and args.keep_sandbox:
                        print(dim(f"  kept {name} (omit --keep-sandbox to delete on exit)"), flush=True)
                    elif name is not None:
                        delete_sandbox(workspace_client, name)
                        print(dim(f"  deleted {name}"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks CLI config profile for the sandbox APIs. Defaults to "
        "DATABRICKS_CONFIG_PROFILE, else the SDK's default auth.",
    )
    parser.add_argument(
        "--sandbox-id",
        default=f"openai-agent-demo-{uuid.uuid4().hex[:8]}",
        help="Sandbox id to create or reuse. Defaults to a unique openai-agent-demo-<uuid> so "
        "each run gets a fresh sandbox and never reuses a stale one.",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--inactivity-timeout", type=int, default=900)
    parser.add_argument(
        "--app-key-file",
        help="File holding the OpenAI application key. Overridden by OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--executor-key-file",
        help="File holding the OpenAI environment key (passed to codex as CODEX_API_KEY). "
        "Overridden by OPENAI_EXECUTOR_API_KEY.",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="Keep the sandbox on exit (default deletes it). Keeping reuses a stable egress IP.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show the agent's reasoning summary")
    asyncio.run(main(parser.parse_args()))
