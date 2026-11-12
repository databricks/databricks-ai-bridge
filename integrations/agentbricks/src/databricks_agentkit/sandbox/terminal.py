"""Sandbox terminal over authenticated gRPC/TLS for Apps without SSH egress."""

import logging
import queue
import shlex
import threading
from urllib.parse import urlsplit

import grpc

from databricks_agentkit.sandbox import _command_pb2 as wire
from databricks_agentkit.sandbox.lakebox import HttpsRemote

LOG = logging.getLogger(__name__)
ROOT = "/databricks.sandbox.SandboxService/"


class GrpcTerminal:
    """Channel operations shared with the SSH relay; closes its remote process tree."""

    def __init__(self, workspace, sandbox_id, setup, request):
        self.workspace, self.sandbox_id = workspace, sandbox_id
        self._closed = threading.Event()
        self._finished = threading.Event()
        self._error = None
        self._exit_code = None
        self._stdout, self._stderr = queue.Queue(maxsize=64), queue.Queue(maxsize=64)
        self._input = queue.Queue(maxsize=64)
        self._pty = request.command is None
        host = urlsplit(workspace.config.host)
        if host.scheme != "https" or not host.netloc:
            raise ValueError("Sandbox terminal requires an HTTPS workspace")
        self._channel = grpc.secure_channel(host.netloc, grpc.ssl_channel_credentials())
        self._stream = None
        self._input_call = None
        self.command_id, self.pid = None, None
        try:
            execute = self._channel.unary_stream(
                ROOT + "ExecuteCommand",
                request_serializer=wire.ExecuteCommandRequest.SerializeToString,
                response_deserializer=wire.ExecuteCommandResponse.FromString,
            )
            shell = (
                "exec /bin/bash --noprofile --norc -i"
                if self._pty
                else "exec /bin/bash -c " + shlex.quote(request.command) + " < /dev/null"
            )
            command = f"{setup} && export TERM={shlex.quote(request.term)} && {shell}"
            message = wire.ExecuteCommandRequest(cmd="/bin/bash", args=["-lc", command])
            if self._pty:
                message.pty.CopyFrom(wire.PtyConfig(cols=request.columns, rows=request.rows))
            self._stream = execute(message, metadata=self._metadata(), timeout=3600)
            startup_timeout = threading.Timer(30, self._stream.cancel)
            startup_timeout.daemon = True
            startup_timeout.start()
            try:
                first = next(self._stream)
            finally:
                startup_timeout.cancel()
            if first.process_event.WhichOneof("event") != "start":
                raise RuntimeError("Sandbox terminal did not report a process start")
            self.command_id = first.process_event.start.command_id
            self.pid = first.process_event.start.pid
            if not self.command_id or not self.pid or self.pid <= 0:
                raise RuntimeError("Sandbox terminal returned an invalid process identity")
            if self._pty:
                input_rpc = self._channel.stream_unary(
                    ROOT + "StreamInput",
                    request_serializer=wire.StreamInputRequest.SerializeToString,
                    response_deserializer=wire.StreamInputResponse.FromString,
                )
                self._input_call = input_rpc.future(
                    self._inputs(), metadata=self._metadata(), timeout=3600
                )
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
        except BaseException:
            self.close()
            raise

    @property
    def closed(self):
        return self._closed.is_set()

    def _metadata(self):
        return [
            *((key.lower(), value) for key, value in self.workspace.config.authenticate().items()),
            ("x-databricks-sandbox-name", self.sandbox_id),
        ]

    def _inputs(self):
        yield wire.StreamInputRequest(start=wire.StreamInputStart(command_id=self.command_id))
        while not self.closed:
            try:
                message = self._input.get(timeout=0.1)
            except queue.Empty:
                continue
            yield message
            if message.WhichOneof("event") == "close":
                return

    def _read(self):
        try:
            assert self._stream is not None
            for response in self._stream:
                event = response.process_event
                kind = event.WhichOneof("event")
                if kind == "data":
                    stream = event.data.WhichOneof("output")
                    if stream is None:
                        continue
                    output = self._stderr if stream == "stderr" else self._stdout
                    data = getattr(event.data, stream)
                    for offset in range(0, len(data), 32768):
                        while not self.closed:
                            try:
                                output.put(data[offset : offset + 32768], timeout=0.1)
                                break
                            except queue.Full:
                                continue
                elif kind == "end":
                    # Daemon output tasks can emit final chunks after EndEvent.
                    # Drain the stream to EOF before exposing the exit status.
                    if event.end.HasField("exit_code"):
                        self._exit_code = event.end.exit_code
                if self.closed:
                    return
            if self._exit_code is None:
                raise RuntimeError("Sandbox terminal stream ended without an exit status")
        except Exception as exc:
            if not self.closed:
                self._error = type(exc).__name__
        finally:
            self._finished.set()

    def recv_ready(self):
        return not self._stdout.empty()

    def recv_stderr_ready(self):
        return not self._stderr.empty()

    def recv(self, size):
        return self._stdout.get_nowait()

    def recv_stderr(self, size):
        return self._stderr.get_nowait()

    def exit_status_ready(self):
        if self._input_call is not None and self._input_call.done() and not self._finished.is_set():
            failure = self._input_call.exception()
            if failure is not None:
                raise RuntimeError("Sandbox terminal input stream failed")
        return self._finished.is_set()

    def recv_exit_status(self):
        if self._error:
            raise RuntimeError(f"Sandbox terminal output failed ({self._error})")
        return self._exit_code if self._exit_code is not None else 255

    def _send(self, message):
        while not self.closed:
            try:
                self._input.put(message, timeout=0.1)
                return
            except queue.Full:
                continue
        raise OSError("Terminal closed")

    def sendall(self, data):
        self._send(wire.StreamInputRequest(data=wire.StreamInputData(data=data)))

    def shutdown_write(self):
        if self._pty:
            self.sendall(b"\x04")
        # Pipe commands already have stdin redirected from /dev/null.

    def resize_pty(self, *, width, height):
        if self._pty and not self._finished.is_set():
            # The preview API has no resize RPC. Changing the existing PTY's
            # window sends SIGWINCH to its foreground process group. The daemon
            # creates root-owned PTYs before dropping UID. Resolve only this
            # process's terminal, then use the sandbox's passwordless sudo.
            HttpsRemote(self.workspace, self.sandbox_id)._exec(
                "/bin/bash",
                [
                    "-c",
                    'terminal=$(/bin/ps -p "$1" -o tty=) && '
                    '[[ "$terminal" =~ ^pts/[0-9]+$ ]] && '
                    'exec /usr/bin/sudo -n /bin/stty -F "/dev/$terminal" rows "$2" cols "$3"',
                    "agentbricks-resize",
                    str(self.pid),
                    str(height),
                    str(width),
                ],
                10,
            )

    def close(self):
        if self.closed:
            return
        self._closed.set()
        try:
            if self.command_id and self._exit_code is None:
                terminate = self._channel.unary_unary(
                    ROOT + "TerminateCommand",
                    request_serializer=wire.TerminateCommandRequest.SerializeToString,
                    response_deserializer=wire.TerminateCommandResponse.FromString,
                )
                terminate(
                    wire.TerminateCommandRequest(command_id=self.command_id),
                    metadata=self._metadata(),
                    timeout=10,
                )
        except Exception as exc:
            LOG.warning("Could not terminate debug terminal: %s", type(exc).__name__)
        finally:
            if self._stream is not None:
                self._stream.cancel()
            if self._input_call is not None:
                self._input_call.cancel()
            self._channel.close()
            reader = getattr(self, "_reader", None)
            if reader is not None:
                reader.join(timeout=1)
