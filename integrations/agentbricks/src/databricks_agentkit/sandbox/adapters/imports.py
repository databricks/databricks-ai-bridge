"""Bounded file read-ahead for first-party Python imports on cold sandbox disks."""

import csv
import importlib.util
import json
import os
import stat
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

MAX_FILES = 24000
MAX_BYTES = 256 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_prefetched = False


def _inventory(prefix):
    # New installations already have warm source files. Reading them again
    # competes with compilation; only prefetch native extensions until learned.
    library = Path(sysconfig.get_path("purelib"))
    if not library.is_relative_to(prefix):
        return []
    paths = set()
    deadline, scanned = time.monotonic() + 1, 0
    for index, record in enumerate(library.glob("*.dist-info/RECORD")):
        if index >= 512 or time.monotonic() >= deadline:
            break
        try:
            size = record.stat().st_size
            scanned += size
            if scanned > 8 * MAX_MANIFEST_BYTES:
                break
            if size > MAX_MANIFEST_BYTES:
                continue
            with record.open(newline="") as handle:
                for row in csv.reader(handle):
                    if time.monotonic() >= deadline:
                        return sorted(paths)
                    if not row:
                        continue
                    path = library / row[0]
                    if path.suffix == ".so" and ".." not in path.parts:
                        paths.add(str(path.relative_to(prefix)))
                    if len(paths) >= MAX_FILES:
                        return sorted(paths)
        except (OSError, ValueError, csv.Error):
            continue
    return sorted(paths)


def _paths(manifest, prefix):
    try:
        if manifest.stat().st_size <= MAX_MANIFEST_BYTES:
            document = json.loads(manifest.read_text())
            paths = document["files"]
            if document["prefix"] == str(prefix) and isinstance(paths, list):
                return paths[:MAX_FILES]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return _inventory(prefix)


@contextmanager
def warm_imports(state_dir, timings):
    """Read code concurrently with normal imports; never execute/cache module objects."""
    global _prefetched
    if _prefetched or os.getenv("AGENTBRICKS_IMPORT_PREFETCH", "1") == "0":
        yield
        return
    _prefetched = True
    started = time.perf_counter()
    prefix = Path(sys.prefix).resolve()
    manifest = state_dir / "import-files.json"
    stopped, lock = threading.Event(), threading.Lock()
    remaining = MAX_BYTES
    deadline = time.monotonic() + 10

    def read(relative):
        nonlocal remaining
        try:
            if stopped.is_set() or time.monotonic() >= deadline or not isinstance(relative, str):
                return
            path = (prefix / relative).resolve()
            if not path.is_relative_to(prefix) or path.suffix not in {".py", ".pyc", ".so"}:
                return
            if path.suffix == ".py":
                cached = Path(importlib.util.cache_from_source(str(path)))
                if cached.is_file():
                    path = cached.resolve()
            if not path.is_relative_to(prefix):
                return
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                return
            size = metadata.st_size
            with lock:
                if size > MAX_FILE_BYTES or size > remaining:
                    return
                remaining -= size
            with path.open("rb") as handle:
                while size and not stopped.is_set() and time.monotonic() < deadline:
                    block = handle.read(min(size, 64 * 1024))
                    if not block:
                        break
                    size -= len(block)
        except (OSError, ValueError):
            pass

    pool = None
    try:
        pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent-import")
        for path in _paths(manifest, prefix):
            if time.monotonic() >= deadline:
                break
            pool.submit(read, path)
    except Exception:
        # Optional acceleration must not prevent the normal import path.
        pass
    setup = time.perf_counter() - started
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        finished = time.perf_counter()
        stopped.set()
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
        if succeeded:
            # Remember actual imports on retained disk; /tmp is cleared on Stop.
            try:
                paths = set()
                for module in tuple(sys.modules.values()):
                    attributes = vars(module) if module is not None else {}
                    name = attributes.get("__cached__") or attributes.get("__file__")
                    if isinstance(name, str):
                        path = Path(name)
                        if path.is_relative_to(prefix) and path.suffix in {".py", ".pyc", ".so"}:
                            paths.add(str(path.relative_to(prefix)))
                state_dir.mkdir(parents=True, exist_ok=True)
                temporary = manifest.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"prefix": str(prefix), "files": sorted(paths)[:MAX_FILES]})
                )
                temporary.replace(manifest)
            except (OSError, TypeError, ValueError):
                pass
        timings["import_cache"] = round((setup + time.perf_counter() - finished) * 1000, 2)
