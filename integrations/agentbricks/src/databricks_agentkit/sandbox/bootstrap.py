"""Executed by the sandbox's Python, before the agent environment exists."""

# Kept as a standalone script so bootstrap needs only Python's standard library.
SCRIPT = r"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

root = Path(sys.argv[1])
digest = sys.argv[2]
metrics = {}

def phase(name, args, **kwargs):
    start = time.perf_counter()
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)
    metrics[name] = round((time.perf_counter() - start) * 1000, 2)

with (root / "bootstrap.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    marker = root / "ready.json"
    if marker.exists():
        print(json.dumps({"reused": True, **json.loads(marker.read_text())}))
        sys.exit(0)
    start = time.perf_counter()
    archive = root / "source.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError("Source archive digest mismatch")
    source = root / "source"
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    metrics["extract"] = round((time.perf_counter() - start) * 1000, 2)
    start = time.perf_counter()
    uv = shutil.which("uv")
    if uv is None:
        tools = root / "tools"
        phase("toolchain_venv", [sys.executable, "-m", "venv", str(tools)])
        phase("toolchain_download", [str(tools / "bin/python"), "-m", "pip", "install", "uv==0.13.5"])
        uv = str(tools / "bin/uv")
    metrics["toolchain"] = round((time.perf_counter() - start) * 1000, 2)
    env = {**os.environ, "UV_CACHE_DIR": str(root / "uv-cache"), "UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"}
    # Index selection belongs to the published project, not the sandbox image.
    # uv uses public PyPI when the project does not configure another index.
    for key in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL"):
        env.pop(key, None)
    phase("dependency_install", [uv, "sync", "--locked", "--no-dev", "--no-install-project", "--python", "3.12"], cwd=source, env=env)
    phase("project_build", [uv, "build", "--wheel", "--out-dir", str(root / "wheels")], cwd=source, env=env)
    wheels = list((root / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one agent wheel")
    python = str(source / ".venv/bin/python")
    phase("project_install", [uv, "pip", "install", "--python", python, "--no-deps", str(wheels[0])], env=env)
    entrypoint = sys.argv[3] if len(sys.argv) > 3 else "sandbox_agent.worker:run"
    phase("import_check", [python, "-c", "import importlib,sys; m,n=sys.argv[1].split(':'); assert callable(getattr(importlib.import_module(m),n))", entrypoint], cwd=root)
    result = {"source_hash": digest, "timings_ms": metrics, "python": python, "uv": uv}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(marker)
    print(json.dumps({"reused": False, **result}))
"""
