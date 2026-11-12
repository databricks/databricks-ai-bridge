"""Standard-library shim executed with the installed agent's Python interpreter."""

CALL_SCRIPT = r"""
import asyncio
import importlib
import inspect
import json
import os
from pathlib import Path

_preparation = None
_handler = None

def load_handler(entrypoint):
    global _handler
    module, name = entrypoint.split(":", 1)
    _handler = getattr(importlib.import_module(module), name)
    if not callable(_handler):
        raise TypeError("Agent entrypoint must be callable")
    return _handler

async def shutdown_handler(state_dir):
    # Do not import an unused handler during shutdown. Hooks own only worker-lifetime
    # resources; session/checkpoint commits still complete inside the invocation.
    hook = getattr(_handler, "shutdown_session", None)
    if hook:
        result = hook(Path(state_dir))
        if inspect.isawaitable(result):
            await result

async def prepare_handler(state_dir: Path, entrypoint: str):
    global _preparation
    if _preparation is None:
        handler = load_handler(entrypoint)
        hook = getattr(handler, "prepare_session", None)
        details = hook(Path(state_dir)) if hook else None
        if inspect.isawaitable(details):
            details = await details
        _preparation = {"level": "initialized" if hook else "handler-imported", "details": details}
        json.dumps(_preparation)
    return _preparation

async def call_handler(request, state_dir: Path, entrypoint: str, stream_path=None):
    stream = None
    if stream_path:
        path = Path(stream_path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stream = os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w")

    def write_record(record):
        if stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    def emit(delta):
        if not isinstance(delta, str):
            raise TypeError("emit expects a text delta")
        # Split large deltas so every UTF-8 JSONL record stays below 1 MiB.
        for offset in range(0, len(delta), 65536):
            write_record({"type": "delta", "delta": delta[offset:offset + 65536]})

    try:
        handler = load_handler(entrypoint)
        kwargs = {"emit": emit} if "emit" in inspect.signature(handler).parameters else {}
        result = handler(request, Path(state_dir), **kwargs)
        if inspect.isawaitable(result):
            result = await result
        json.dumps(result)
    except BaseException:
        write_record({"type": "end", "status": "failed"})
        raise
    else:
        write_record({"type": "end", "status": "completed"})
        return result
    finally:
        if stream:
            stream.close()
"""

SCRIPT = (
    CALL_SCRIPT
    + r"""
import sys

request = json.load(sys.stdin)
stream_path = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--stream-file" else None
async def main():
    try:
        return await call_handler(request, Path(sys.argv[1]), sys.argv[2], stream_path)
    finally:
        # One-shot processes have no later worker lifetime in which to export traces.
        # Preserve the agent outcome if an optional shutdown hook fails.
        try:
            await shutdown_handler(Path(sys.argv[1]))
        except Exception:
            import traceback
            traceback.print_exc()

result = asyncio.run(main())
print(json.dumps(result), flush=True)
"""
)
