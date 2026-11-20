"""Interactive developer terminal over the launcher's authenticated SSH bridge."""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import signal
import sys
import threading
from contextlib import contextmanager
from urllib.parse import quote, urlsplit, urlunsplit

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator


def websocket_url(launcher, environment):
    parsed = urlsplit(launcher.url)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise AgentCliError("Remote sandbox debug connections require HTTPS.")
    path = parsed.path.rstrip("/") + f"/api/environments/{quote(environment, safe='')}/ssh"
    return urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, "", ""))


@contextmanager
def raw_terminal(enabled):
    if not enabled:
        yield
        return
    if os.name != "posix":
        raise AgentCliError("Interactive SSH requires a POSIX terminal; use --command instead.")
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _write(stream, data):
    target = getattr(stream, "buffer", None)
    if target is None:
        stream.write(data.decode("utf-8", errors="replace"))
    else:
        target.write(data)
    stream.flush()


def connect(launcher, environment, session_id, *, command=None):
    """Return the remote exit code; keep credentials and SSH keys out of persisted state."""
    try:
        from websockets.exceptions import ConnectionClosed, InvalidStatus
        from websockets.sync.client import connect as websocket_connect
    except ImportError as exc:
        raise AgentCliError(
            "Install databricks-agentbricks[sandbox] to use environment ssh."
        ) from exc

    url = websocket_url(launcher, environment)
    headers = {"X-Routing-Key": session_id}
    if launcher.authenticate:
        authorization = WorkspaceOAuthAuthenticator(launcher.profile).authorization_header()
        headers.update(
            {"Authorization": authorization, "X-AgentBricks-Debug-Authorization": authorization}
        )
    elif urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise AgentCliError("Hosted debugging requires --app or --url with --auth.")
    interactive = command is None
    if interactive and not sys.stdin.isatty():
        raise AgentCliError("Use an interactive terminal, or pass --command for a single command.")
    columns, rows = shutil.get_terminal_size((80, 24))
    request = {
        "session_id": session_id,
        "command": command,
        "columns": min(columns, 500),
        "rows": min(rows, 500),
        "term": os.getenv("TERM", "xterm-256color"),
    }
    try:
        # WebSocket headers are never placed in a URL, output, or on disk.
        with websocket_connect(
            url,
            additional_headers=headers,
            open_timeout=30,
            close_timeout=5,
            max_size=131072,
        ) as socket:
            socket.send(json.dumps(request))
            message = json.loads(socket.recv(timeout=240))
            if message.get("type") == "error":
                raise AgentCliError(message["message"])
            if message.get("type") != "ready":
                raise AgentCliError("Sandbox launcher returned an invalid SSH handshake.")
            sys.stderr.write(
                f"Connected to {message['sandbox_id']} (session {session_id}, {message.get('transport', 'ssh')})\n"
                f"Source: {message['source_directory']}\n"
            )
            sys.stderr.flush()
            return _terminal(socket, interactive)
    except InvalidStatus as exc:
        status = exc.response.status_code
        raise AgentCliError(
            f"Sandbox SSH connection refused (HTTP {status}).",
            hint="Use an OAuth profile with CAN_MANAGE on this App and deploy the session-ssh runtime. "
            "For a local launcher, explicitly enable SANDBOX_ALLOW_LOCAL_DEBUG=true.",
        ) from None
    except ConnectionClosed:
        raise AgentCliError(
            "Sandbox SSH connection closed before the remote command exited."
        ) from None
    except (TimeoutError, OSError):
        raise AgentCliError(
            "Could not connect to the sandbox SSH bridge before the timeout."
        ) from None


def _terminal(socket, interactive):
    done = threading.Event()
    resized = threading.Event()
    failure = []

    def send_input():
        try:
            descriptor = sys.stdin.fileno()
            while not done.is_set():
                if resized.is_set():
                    resized.clear()
                    columns, rows = shutil.get_terminal_size((80, 24))
                    socket.send(
                        json.dumps(
                            {"type": "resize", "columns": min(columns, 500), "rows": min(rows, 500)}
                        )
                    )
                readable, _, _ = select.select([descriptor], [], [], 0.1)
                if not readable:
                    continue
                data = os.read(descriptor, 16384)
                if not data:
                    socket.send(json.dumps({"type": "eof"}))
                    return
                socket.send(
                    json.dumps({"type": "stdin", "data": base64.b64encode(data).decode("ascii")})
                )
        except Exception as exc:  # communicate a background transport error to the foreground
            if not done.is_set():
                failure.append(type(exc).__name__)
                socket.close()

    def resize(signum, frame):
        # A signal handler must not acquire the WebSocket's transport locks.
        resized.set()

    sender = None
    old_resize = None
    try:
        with raw_terminal(interactive):
            if interactive:
                old_resize = signal.signal(signal.SIGWINCH, resize)
                sender = threading.Thread(target=send_input, daemon=True)
                sender.start()
            else:
                # Commands run without a PTY or stdin; this preserves exact stdout,
                # stderr and exit status for scripts, including commands waiting for EOF.
                socket.send(json.dumps({"type": "eof"}))
            for raw in socket:
                message = json.loads(raw)
                kind = message.get("type")
                if kind in {"stdout", "stderr"}:
                    _write(
                        sys.stdout if kind == "stdout" else sys.stderr,
                        base64.b64decode(message["data"], validate=True),
                    )
                elif kind == "exit":
                    return int(message["code"])
                elif kind == "error":
                    raise AgentCliError(message["message"])
            hint = f"Input forwarding failed ({failure[0]})." if failure else None
            raise AgentCliError("Sandbox SSH connection closed without an exit status.", hint=hint)
    finally:
        done.set()
        if old_resize is not None:
            signal.signal(signal.SIGWINCH, old_resize)
        if sender is not None:
            sender.join(timeout=0.5)
