"""Workspace Lakebox lifecycle with HTTPS or SSH execution; no custom image."""

import base64
import fcntl
import logging
import re
import shlex
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import paramiko
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from databricks.sdk.errors import DatabricksError, NotFound, ResourceAlreadyExists
from paramiko.auth_strategy import AuthStrategy, InMemoryPrivateKey
from paramiko.common import cMSG_CHANNEL_REQUEST
from paramiko.message import Message

ROOT = "/api/2.0/lakebox"


class RemoteError(RuntimeError):
    pass


class PinFirstHostKey(paramiko.MissingHostKeyPolicy):
    """Persist the first key without Paramiko's MD5 fingerprint logging (FIPS)."""

    def __init__(self, path):
        self.path = path

    def missing_host_key(self, client, hostname, key):
        client.get_host_keys().add(hostname, key.get_name(), key)
        client.save_host_keys(str(self.path))


class GatewayKeyAuth(AuthStrategy):
    def __init__(self, username, key):
        super().__init__(ssh_config=None)
        self.username, self.key = username, key

    def get_sources(self):
        yield InMemoryPrivateKey(self.username, self.key)


class Lakebox:
    def __init__(
        self, workspace, state_dir: Path, namespace: str, transport="ssh", *, idle_timeout=900
    ):
        self.workspace = workspace
        self.state_dir = state_dir
        self.namespace = namespace
        if not isinstance(idle_timeout, int) or not 60 <= idle_timeout <= 86400:
            raise ValueError("Sandbox idle timeout must be 60–86400 seconds")
        self.idle_timeout = idle_timeout
        self._configured_timeouts = set()
        if transport not in {"ssh", "https"}:
            raise ValueError("Sandbox transport must be ssh or https")
        self.transport = transport
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if transport == "https":
            return
        self._ssh_lock = threading.Lock()
        self._registered = False
        self.key = state_dir / "sandbox_ed25519"
        self.known_hosts = state_dir / "known_hosts"
        with (state_dir / "key.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.key.exists():
                material = Ed25519PrivateKey.generate().private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.OpenSSH,
                    serialization.NoEncryption(),
                )
                self.key.touch(mode=0o600)
                self.key.write_bytes(material)
            self.known_hosts.touch(mode=0o600, exist_ok=True)

    def api(self, method, path, body=None, query=None):
        workspace_id = self.workspace.config.workspace_id or self.workspace.get_workspace_id()
        return self.workspace.api_client.do(
            method,
            ROOT + path,
            body=body,
            query=query,
            headers={"X-Databricks-Org-Id": str(workspace_id)},
        )

    def register_key(self):
        with self._ssh_lock:
            if self._registered:
                return
            key = paramiko.Ed25519Key.from_private_key_file(str(self.key))
            public_key = f"{key.get_name()} {key.get_base64()}"
            try:
                self.api(
                    "POST",
                    "/ssh-keys",
                    {"public_key": public_key, "name": f"{self.namespace}-prototype"},
                )
            except ResourceAlreadyExists:
                pass
            self._registered = True

    def find_named(self, name):
        matches, page = [], None
        while True:
            query = {"page_size": 100, **({"page_token": page} if page else {})}
            result = self.api("GET", "/sandboxes", query=query)
            matches.extend(sb for sb in result.get("sandboxes", []) if sb.get("name") == name)
            page = result.get("nextPageToken")
            if not page:
                break
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous sandbox name {name}; inspect before retrying")
        return matches[0] if matches else None

    def create(self, name):
        return self.api("POST", "/sandboxes", {"sandbox": {"name": name}})

    def get(self, sandbox_id):
        try:
            return self.api("GET", "/sandboxes/" + quote(sandbox_id, safe=""))
        except NotFound:
            return None

    def start(self, sandbox_id):
        return self.api(
            "POST",
            f"/sandboxes/{quote(sandbox_id, safe='')}/start",
            {"sandbox_id": sandbox_id},
        )

    def set_idle_timeout(self, sandbox_id):
        if sandbox_id in self._configured_timeouts:
            return
        self.api(
            "PATCH",
            f"/sandboxes/{quote(sandbox_id, safe='')}",
            {
                "sandbox_id": sandbox_id,
                "idle_timeout": f"{self.idle_timeout}s",
                "no_autostop": False,
            },
        )

        self._configured_timeouts.add(sandbox_id)

    def _open_client(self, sandbox):
        self.register_key()
        host = sandbox.get("gatewayHost")
        if not host or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
            raise RuntimeError("Sandbox response is missing a valid gatewayHost")
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.load_host_keys(str(self.known_hosts))
        # Trust on first use, then pin. The gateway comes from the authenticated API.
        client.set_missing_host_key_policy(PinFirstHostKey(self.known_hosts))
        try:
            client.connect(
                host,
                port=2222,
                username=sandbox["sandboxId"],
                auth_strategy=GatewayKeyAuth(
                    sandbox["sandboxId"],
                    paramiko.Ed25519Key.from_private_key_file(str(self.key)),
                ),
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
        except BaseException:
            client.close()
            raise
        return client

    def open_terminal(self, sandbox, command=None, *, term="xterm-256color", columns=80, rows=24):
        """Open an SSH channel; the caller owns and closes the returned client.

        The gateway does not acknowledge channel requests. Send want_reply=False
        rather than using Paramiko's blocking exec/get_pty/invoke_shell helpers.
        """
        client = self._open_client(sandbox)
        try:
            transport = client.get_transport()
            transport.set_keepalive(30)
            channel = transport.open_session(timeout=30)
            if command is None:
                request = Message()
                request.add_byte(cMSG_CHANNEL_REQUEST)
                request.add_int(channel.remote_chanid)
                request.add_string("pty-req")
                request.add_boolean(False)
                request.add_string(term)
                for value in (columns, rows, 0, 0):
                    request.add_int(value)
                request.add_string(b"")
                transport._send_user_message(request)
            request = Message()
            request.add_byte(cMSG_CHANNEL_REQUEST)
            request.add_int(channel.remote_chanid)
            request.add_string("shell" if command is None else "exec")
            request.add_boolean(False)
            if command is not None:
                request.add_string(command)
            transport._send_user_message(request)
            return client, channel
        except BaseException:
            client.close()
            raise

    @contextmanager
    def connect(self, sandbox):
        if self.transport == "https":
            yield HttpsRemote(self.workspace, sandbox["sandboxId"])
            return
        remote = Remote(self._open_client(sandbox), lambda: self._open_client(sandbox))
        remote.workspace, remote.sandbox_id = self.workspace, sandbox["sandboxId"]
        try:
            yield remote
        finally:
            remote.client.close()


class Remote:
    workspace: Any  # Attached by Lakebox.connect, also used by streaming readers.
    sandbox_id: str

    def __init__(self, client, reconnect):
        self.client = client
        self.reconnect = reconnect
        self.used = False

    def run(self, command, data=b"", timeout=600, *, on_stdout=None, stream=False):
        # Lakebox currently permits one session channel per gateway connection.
        if self.used:
            self.client.close()
            self.client = self.reconnect()
        self.used = True
        command = (
            f"timeout --signal=TERM --kill-after=10s {timeout}s bash -lc {shlex.quote(command)}"
        )
        transport = self.client.get_transport()
        channel = transport.open_session(timeout=30)
        # Current Lakebox forwards output and exit status but not CHANNEL_SUCCESS.
        # Send standard exec with want_reply=False so Paramiko does not wait for
        # that acknowledgement. This is the only use of its transport internals.
        request = Message()
        request.add_byte(cMSG_CHANNEL_REQUEST)
        request.add_int(channel.remote_chanid)
        request.add_string("exec")
        request.add_boolean(False)
        request.add_string(command)
        transport._send_user_message(request)
        stdin = channel.makefile_stdin("wb")
        chunks, errors = bytearray(), bytearray()
        deadline = time.monotonic() + timeout + 15
        # Drain output while sending stdin, avoiding a full SSH window deadlock.
        failure = []

        def send():
            try:
                stdin.write(data)
                stdin.flush()
                channel.shutdown_write()
            except (OSError, EOFError, paramiko.SSHException) as exc:
                failure.append(exc)

        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        while True:
            while channel.recv_ready():
                part = channel.recv(65536)
                chunks.extend(part)
                if on_stdout:
                    on_stdout(part)
                if len(chunks) > 1024 * 1024:
                    del chunks[: len(chunks) - 1024 * 1024]
            while channel.recv_stderr_ready():
                errors.extend(channel.recv_stderr(65536))
                if len(errors) > 100_000:
                    del errors[: len(errors) - 100_000]
            if (
                channel.exit_status_ready()
                and not channel.recv_ready()
                and not channel.recv_stderr_ready()
            ):
                break
            if time.monotonic() > deadline:
                channel.close()
                raise TimeoutError("Sandbox command exceeded its deadline")
            time.sleep(0.01)
        sender.join(timeout=1)
        if failure:
            raise RemoteError("Failed to send sandbox command input") from failure[0]
        code = channel.recv_exit_status()
        channel.close()
        if code:
            # Installation logs can be large; return a bounded diagnostic, never env dumps.
            raise RemoteError(
                f"Sandbox command exited {code}: {errors.decode(errors='replace')[-4000:]}"
            )
        return chunks.decode()


class HttpsRemote:
    """Buffered exec-sync for work; streaming ExecuteCommand follows attempt files."""

    def __init__(self, workspace, sandbox_id):
        self.workspace, self.sandbox_id = workspace, sandbox_id

    def _exec(self, cmd, args, timeout):
        workspace_id = self.workspace.config.workspace_id or self.workspace.get_workspace_id()
        result = self.workspace.api_client.do(
            "POST",
            f"/api/2.0/sandbox-exec/sandboxes/{quote(self.sandbox_id, safe='')}/exec-sync",
            body={
                "name": f"sandboxes/{self.sandbox_id}",
                "cmd": cmd,
                "args": args,
                # The public synchronous API has a 290-second server ceiling.
                "execution_timeout": f"{min(timeout, 240)}s",
            },
            headers={"X-Databricks-Org-Id": str(workspace_id)},
        )
        if result.get("status") == "EXECUTE_COMMAND_STATUS_TIMED_OUT":
            raise TimeoutError("Sandbox HTTPS command exceeded its deadline")
        if (
            result.get("status") != "EXECUTE_COMMAND_STATUS_COMPLETED"
            or result.get("exit_code") != 0
        ):
            raise RemoteError(
                f"Sandbox HTTPS command failed ({result.get('status')}, "
                f"exit {result.get('exit_code')}): {result.get('stderr', '')[-4000:]}"
            )
        if result.get("truncated"):
            raise RemoteError("Sandbox HTTPS output was truncated; result is incomplete")
        return result.get("stdout", "")

    def run(self, command, data=b"", timeout=600, *, stream=False):
        if stream:
            if data:
                raise ValueError("Streaming execution does not accept staged stdin")
            return self._stream_exec(command, timeout)
        if not data:
            return self._exec("/bin/bash", ["-lc", command], timeout)
        # exec-sync has no stdin field. Stage bounded chunks in a private file,
        # then feed that file to the process. Split encoded argv elements below
        # Linux's per-argument limit; offsets make an HTTP retry of a chunk safe.
        input_path = f"/tmp/agentbricks-input-{uuid4().hex}"
        writer = (
            "import base64,os,sys; "
            "fd=os.open(sys.argv[1],os.O_CREAT|os.O_WRONLY,0o600); "
            "os.lseek(fd,int(sys.argv[2]),os.SEEK_SET); "
            "f=os.fdopen(fd,'wb'); "
            "f.write(base64.b64decode(''.join(sys.argv[3:]))); f.close()"
        )
        runner = (
            "import subprocess,sys; "
            "f=open(sys.argv[1],'rb'); "
            "r=subprocess.run(['/bin/bash','-lc',sys.argv[2]],stdin=f); "
            "f.close(); sys.exit(r.returncode)"
        )
        try:
            for offset in range(0, len(data), 192 * 1024):
                encoded = base64.b64encode(data[offset : offset + 192 * 1024]).decode("ascii")
                arguments = [encoded[i : i + 48 * 1024] for i in range(0, len(encoded), 48 * 1024)]
                self._exec("python3", ["-c", writer, input_path, str(offset), *arguments], 30)
            return self._exec("python3", ["-c", runner, input_path, command], timeout)
        finally:
            try:
                self._exec("/bin/rm", ["-f", input_path], 10)
            except (DatabricksError, RemoteError, TimeoutError):
                # Preserve the command failure if cleanup also fails.
                logging.getLogger(__name__).warning("Could not remove sandbox command input file")

    def _stream_exec(self, command, timeout):
        """Execute long artifact operations without exec-sync's HTTP read timeout.

        No automatic command replay. A remote timeout bounds work if the stream
        disconnects; an explicit installation retry reconciles its ready marker.
        """
        from databricks_agentkit.sandbox.streaming import ExecuteCommand

        client = ExecuteCommand(self.workspace, self.sandbox_id)
        command = (
            f"timeout --signal=TERM --kill-after=10s {timeout}s bash -lc {shlex.quote(command)}"
        )
        output, errors = bytearray(), bytearray()
        exit_code = None
        try:
            for response in client.execute("/bin/bash", ["-lc", command], timeout + 15):
                event = response.process_event
                kind = event.WhichOneof("event")
                if kind == "data":
                    if event.data.WhichOneof("output") == "stdout":
                        output.extend(event.data.stdout)
                        if len(output) > 1024 * 1024:
                            raise RemoteError("Sandbox artifact command output exceeded 1 MiB")
                    elif event.data.WhichOneof("output") == "stderr":
                        errors.extend(event.data.stderr)
                        del errors[:-12000]
                elif kind == "end":
                    exit_code = event.end.exit_code if event.end.HasField("exit_code") else None
            if exit_code != 0:
                raise RemoteError(
                    f"Sandbox artifact command failed (exit {exit_code}): {errors.decode(errors='replace')[-4000:]}"
                )
            return output.decode()
        finally:
            client.close()
