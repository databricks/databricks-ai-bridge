"""Follow attempt files through the sandbox preview streaming ExecuteCommand API.

Only the reader uses streaming gRPC. The foreground agent command is independent
of that connection, and browser disconnects do not stop either worker. This is a
text-preview protocol, not MAR's Conversation Store commit/replay protocol.
"""

import base64
import json
import logging
import shlex
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

import grpc

from databricks_agentkit.sandbox import _command_pb2 as wire

LOG = logging.getLogger(__name__)
RPC = "/databricks.sandbox.SandboxService/ExecuteCommand"


class StreamDecoder:
    """Decode JSONL across byte/UTF-8 boundaries and reject malformed records."""

    def __init__(self, emit):
        self.emit = emit
        self.pending = b""
        self.ended = threading.Event()
        self.first_delta_at = None

    def feed(self, data):
        if self.ended.is_set() and data:
            raise ValueError("Data after sandbox stream end")
        self.pending += data
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if len(line) > 1024 * 1024:
                raise ValueError("Sandbox stream record exceeds 1 MiB")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Invalid sandbox stream record")
            if record.get("type") == "delta" and isinstance(record.get("delta"), str):
                if self.first_delta_at is None:
                    self.first_delta_at = record.get("at")
                self.emit(record["delta"])
            elif record.get("type") == "end" and record.get("status") in {"completed", "failed"}:
                if self.pending:
                    raise ValueError("Data after sandbox stream end")
                self.ended.set()
            else:
                raise ValueError("Invalid sandbox stream record")
        if len(self.pending) > 1024 * 1024:
            raise ValueError("Sandbox stream record exceeds 1 MiB")


class ExecuteCommand:
    """Small preview API binding, matching the sandbox SDK's wire contract.

    Keep this private until the public SDK/API is available. Request credentials
    are resolved at call time using the WorkspaceClient's refreshable provider.
    No private SDK wheel or Universe checkout is needed in the orchestrator App.
    """

    def __init__(self, workspace, sandbox_id):
        self.workspace, self.sandbox_id = workspace, sandbox_id
        host = urlsplit(workspace.config.host)
        if not host.hostname or host.scheme != "https":
            raise ValueError("Streaming ExecuteCommand requires an HTTPS workspace host")
        self.channel = grpc.secure_channel(
            host.netloc,
            grpc.ssl_channel_credentials(),
            options=[("grpc.primary_user_agent", "agentbricks-sandbox-prototype")],
        )
        self.rpc = self.channel.unary_stream(
            RPC,
            request_serializer=wire.ExecuteCommandRequest.SerializeToString,
            response_deserializer=wire.ExecuteCommandResponse.FromString,
        )

    def execute(self, cmd, args, timeout):
        auth = {key.lower(): value for key, value in self.workspace.config.authenticate().items()}
        return self.rpc(
            wire.ExecuteCommandRequest(cmd=cmd, args=args),
            metadata=[*auth.items(), ("x-databricks-sandbox-name", self.sandbox_id)],
            timeout=timeout,
        )

    def close(self):
        self.channel.close()


class FileFollower:
    """One bounded tail process per attempt, with deterministic remote cleanup.

    Canceling a gRPC stream does not terminate its sandbox command. Kill the
    reader explicitly, and also give it a remote timeout in case the App dies.
    Never retry ExecuteCommand blindly: a lost reply can leave a live process.
    """

    def __init__(self, remote, path, emit, timeout=300):
        self.remote, self.path = remote, path
        self.timeout = timeout
        self.pid_path = path + ".reader-" + uuid4().hex
        self.decoder = StreamDecoder(emit)
        self.offset = 0
        self.error = None
        self.stream = None
        self.client = None
        self.thread = None
        self.pid = 0

    def __enter__(self):
        # Resolve credentials before starting the agent. RPC errors can still
        # arrive asynchronously; the foreground command is independent of them.
        self.client = ExecuteCommand(self.remote.workspace, self.remote.sandbox_id)
        command = (
            'umask 077; echo $$ > "$2"; '
            f"exec timeout --signal=TERM --kill-after=5s {self.timeout + 15}s "
            'tail -c +1 -F -s 0.05 -- "$1"'
        )
        try:
            self.stream = self.client.execute(
                "/bin/bash",
                ["-c", command, "agentbricks-tail", self.path, self.pid_path],
                self.timeout + 25,
            )
            self.thread = threading.Thread(target=self._read, name="sandbox-tail", daemon=True)
            self.thread.start()
        except BaseException:
            self.client.close()
            raise
        return self

    def _read(self):
        assert self.stream is not None
        try:
            for response in self.stream:
                event = response.process_event
                kind = event.WhichOneof("event")
                if kind == "start":
                    self.pid = event.start.pid
                elif kind == "data" and event.data.WhichOneof("output") == "stdout":
                    data = event.data.stdout
                    self.decoder.feed(data)
                    self.offset += len(data)
                    if self.decoder.ended.is_set():
                        return
                elif kind == "end":
                    raise RuntimeError("Sandbox tail exited before the attempt ended")
            if not self.decoder.ended.is_set():
                raise RuntimeError("Sandbox tail disconnected before the attempt ended")
        except Exception as exc:
            self.error = exc

    def finish(self):
        """Called after the foreground command finishes, including agent failures."""
        assert self.stream is not None and self.thread is not None
        if not self.decoder.ended.wait(timeout=1):
            self.stream.cancel()
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise TimeoutError("Sandbox stream reader did not stop")
            # One catch-up read of the completed file, never a file polling loop.
            # offset only advances after a successful decoder callback.
            if isinstance(self.error, (ValueError, TypeError)):
                raise self.error
            read = (
                "import base64,sys; f=open(sys.argv[1],'rb'); "
                "f.seek(int(sys.argv[2])); print(base64.b64encode(f.read()).decode())"
            )
            rest = self.remote.run(
                f"python3 -c {shlex.quote(read)} {shlex.quote(self.path)} {self.offset}",
                timeout=15,
            )
            # The byte offset may split a UTF-8 character. The buffered REST API
            # returns text, so carry catch-up bytes as base64 rather than decoding
            # an incomplete character at that transport boundary.
            self.decoder.feed(base64.b64decode(rest.strip(), validate=True))
            if self.error:
                LOG.warning("Sandbox stream disconnected; caught up from the attempt file")
        if not self.decoder.ended.is_set():
            raise RuntimeError("Sandbox attempt stream ended without a terminal record")

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.cancel()
        if self.thread:
            self.thread.join(timeout=5)
        if self.client:
            self.client.close()
        # Verify the random attempt path is still in the process argv before
        # signaling a saved PID; a finished process's PID might have been reused.
        cleanup = (
            "import os,signal,sys; from pathlib import Path; "
            "p=Path(sys.argv[1]); pid=int(p.read_text()) if p.exists() else int(sys.argv[3]); "
            "cmd=Path('/proc/'+str(pid)+'/cmdline'); "
            "args=cmd.read_bytes().split(b'\\0') if cmd.exists() else []; "
            "os.kill(pid,signal.SIGTERM) if os.fsencode(sys.argv[2]) in args else None; "
            "p.unlink(missing_ok=True)"
        )
        try:
            self.remote.run(
                f"python3 -c {shlex.quote(cleanup)} {shlex.quote(self.pid_path)} "
                f"{shlex.quote(self.path)} {self.pid}",
                timeout=15,
            )
        except Exception:
            LOG.warning("Could not stop sandbox tail; its remote timeout remains active")


class LivePreview:
    """Bounded, subscriber-local preview rebuilt from an immutable attempt file.

    SQL contains only the descriptor. Slow clients coalesce snapshots in memory;
    disconnecting closes their tail process without touching agent execution.
    """

    def __init__(self, remote, source, attempt):
        self.source, self.attempt = source, attempt
        self.condition = threading.Condition()
        self.text, self.sequence = "", 0
        self.follower = FileFollower(remote, source["path"], self.emit)

    def emit(self, delta):
        with self.condition:
            self.text += delta[: max(0, 256 * 1024 - len(self.text))]
            self.sequence += 1
            self.condition.notify_all()

    def __enter__(self):
        self.follower.__enter__()
        return self

    def __exit__(self, *exc):
        self.follower.__exit__(*exc)

    def snapshot(self):
        with self.condition:
            first = self.follower.decoder.first_delta_at
            started = self.source.get("started_at")
            return {
                "attempt": self.attempt,
                "text": self.text,
                "sequence": self.sequence,
                "first_token_ms": round(max(0, first - started) * 1000, 2)
                if isinstance(first, (int, float)) and isinstance(started, (int, float))
                else None,
            }

    def wait(self, sequence, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence != sequence, timeout=timeout)
        # Coalesce bursts without putting database writes on the text path.
        time.sleep(0.02)
