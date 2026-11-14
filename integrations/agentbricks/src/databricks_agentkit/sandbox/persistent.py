"""Private, session-local worker protocol; only Python's stdlib runs remotely.

The lightweight client runs through exec-sync. The resident process uses the
agent's installed interpreter and a single asyncio loop. An invocation is sent
once; reconnecting or restarting is only permitted before submission.
"""

import hashlib

from databricks_agentkit.sandbox.runner import CALL_SCRIPT

DAEMON_SCRIPT = (
    CALL_SCRIPT
    + r"""
import contextvars
import fcntl
import signal
import socket
import sys
import threading
import traceback
from uuid import uuid4

state_dir = Path(sys.argv[1])
entrypoint, address, identity = sys.argv[2:5]
run_seconds = float(sys.argv[5])
max_invocations = int(sys.argv[6])
worker_id = uuid4().hex
count = 0
os.umask(0o077)

def send(connection, value):
    connection.sendall(json.dumps(value).encode() + b"\n")

def terminate_worker():
    # Popen starts a dedicated process group. Kill tool/MCP children together
    # with the worker before a replacement can acquire the execution lease.
    os.killpg(os.getpgrp(), signal.SIGKILL)

def request_shutdown(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, request_shutdown)

async def invoke_request(request, cleanup):
    if request["op"] == "prepare":
        # Startup resources belong to the worker lifetime, not a user turn.
        return await prepare_handler(state_dir, entrypoint)
    existing = asyncio.all_tasks()
    try:
        return await call_handler(request["request"], state_dir, entrypoint, request.get("stream_path"))
    finally:
        pending = asyncio.all_tasks() - existing
        for task in pending:
            task.cancel()
        if pending:
            done, remaining = await asyncio.wait(pending, timeout=1)
            await asyncio.gather(*done, return_exceptions=True)
            # Keep the successful handler result, then retire the whole worker
            # if an invocation-owned task ignored cancellation.
            cleanup["retire"] = bool(remaining or asyncio.all_tasks() - existing)

with (state_dir / "worker.lock").open("a") as lease:
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(75)
    Path(address).unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX) as server, asyncio.Runner() as loop:
        server.bind(address)
        server.listen(4)
        # The sandbox owns idle shutdown. Keep imports warm for its lifetime;
        # a separate worker timer leaves running compute with a cold handler.
        try:
            while count < max_invocations:
                connection, _ = server.accept()
                retire = False
                with connection:
                    connection.settimeout(5)
                    try:
                        with connection.makefile("rb") as incoming:
                            line = incoming.readline(8 * 1024 * 1024 + 1)
                        if len(line) > 8 * 1024 * 1024 or not line.endswith(b"\n"):
                            raise ValueError("Invalid worker request")
                        request = json.loads(line)
                        if request.get("op") == "ping":
                            send(connection, {"identity": identity, "id": worker_id, "pid": os.getpid(),
                                "prepared": _preparation is not None, "invocation_count": count})
                            continue
                        if request.get("op") == "shutdown":
                            send(connection, {"stopped": True})
                            break
                        if request.get("op") not in {"invoke", "prepare"} or request.get("identity") != identity:
                            raise ValueError("Worker identity mismatch")
                        # Independent of worker version: old and new processes
                        # cannot execute a session concurrently after a timeout.
                        with (state_dir / "execution.lock").open("a") as execution:
                            fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            reused = count > 0 or _preparation is not None
                            if request["op"] == "invoke":
                                count += 1
                            # Also bounds synchronous handlers that block the loop.
                            watchdog = threading.Timer(run_seconds, terminate_worker)
                            watchdog.daemon = True
                            watchdog.start()
                            try:
                                cleanup = {"retire": False}
                                result = loop.run(invoke_request(request, cleanup), context=contextvars.Context())
                                retire = cleanup["retire"]
                            finally:
                                watchdog.cancel()
                            send(connection, {
                                "ok": True, "result": result,
                                "worker": {"mode": "persistent", "id": worker_id,
                                    "pid": os.getpid(), "reused": reused,
                                    "invocation_count": count, "identity": identity,
                                    "retiring": retire or count >= max_invocations},
                            })
                    except BaseException as error:
                        traceback.print_exc()
                        retire = True
                        try:
                            send(connection, {"ok": False,
                                "error": (type(error).__name__ + ": " + str(error))[:4000]})
                        except OSError:
                            pass
                # An exception or lost reply can leave handler state uncertain.
                # Preserve disk state, but retire this process before another turn.
                if retire:
                    break
        finally:
            Path(address).unlink(missing_ok=True)
            # Keep telemetry draining out of the response path. Bound even a
            # synchronous custom shutdown hook; watchdog failures still kill
            # immediately and cannot promise a graceful export.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            shutdown_watchdog = threading.Timer(10, terminate_worker)
            shutdown_watchdog.daemon = True
            shutdown_watchdog.start()
            try:
                loop.run(shutdown_handler(state_dir))
            except BaseException:
                traceback.print_exc()
            finally:
                terminate_worker()
"""
)

WORKER_VERSION = hashlib.sha256(DAEMON_SCRIPT.encode()).hexdigest()

CLIENT_SCRIPT = (
    "DAEMON_SCRIPT = "
    + repr(DAEMON_SCRIPT)
    + "\nWORKER_VERSION = "
    + repr(WORKER_VERSION)
    + "\n"
    + r"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

os.umask(0o077)
state_dir = Path(sys.argv[1]).resolve()
entrypoint, python = sys.argv[2:4]
stream_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--stream-file" else None
state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
key = hashlib.sha256(str(state_dir).encode()).hexdigest()[:32]
directory = Path("/tmp") / ("ab-worker-" + key)
directory.mkdir(mode=0o700, exist_ok=True)
address = str(directory / "rpc.sock")
identity = hashlib.sha256(json.dumps([str(state_dir), entrypoint, python, WORKER_VERSION]).encode()).hexdigest()

def rpc(payload, timeout):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(timeout)
        connection.connect(address)
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as reply:
            line = reply.readline(8 * 1024 * 1024 + 1)
        if len(line) > 8 * 1024 * 1024 or not line.endswith(b"\n"):
            raise RuntimeError("Worker disconnected or returned an invalid response")
        return json.loads(line)

def probe():
    try:
        return rpc({"op": "ping"}, .5)
    except (OSError, ValueError, RuntimeError):
        return None

def ensure_worker():
    with (state_dir / "worker-start.lock").open("a") as startup:
        fcntl.flock(startup, fcntl.LOCK_EX)
        existing = probe()
        if existing and existing.get("identity") == identity:
            return
        if existing:
            # The listener handles ping only between invocations. A busy worker
            # does not acknowledge this upgrade and cannot be replaced underneath a run.
            rpc({"op": "shutdown"}, 2)
        deadline = time.monotonic() + 15
        while True:
            with (state_dir / "worker.lock").open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Session worker is busy or unhealthy; invocation was not submitted")
            time.sleep(.05)
        with (state_dir / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [python, "-u", "-c", DAEMON_SCRIPT, str(state_dir), entrypoint,
                    address, identity, "210", "100"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ready = probe()
            if ready and ready.get("identity") == identity:
                return
            if process.poll() is not None:
                raise RuntimeError("Session worker failed to start; inspect worker.log")
            time.sleep(.05)
        raise TimeoutError("Session worker did not become ready; invocation was not submitted")

if "--probe" in sys.argv[4:]:
    print(json.dumps(probe()), flush=True)
    sys.exit(0)

ensure_worker()
request = json.load(sys.stdin)
try:
    # Never retry after submission: an agent may already have performed side effects.
    response = rpc({"op": "prepare" if "--prepare" in sys.argv[4:] else "invoke", "identity": identity, "request": request,
        "stream_path": stream_path}, 225)
except (OSError, ValueError, RuntimeError) as error:
    raise RuntimeError("Worker request failed after submission; outcome may be unknown. "
        "Inspect the invocation before explicitly retrying.") from error
if not response.get("ok"):
    raise RuntimeError(response.get("error", "Worker invocation failed"))
print(json.dumps(response), flush=True)
"""
)
