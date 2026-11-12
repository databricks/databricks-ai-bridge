"""Remote artifact build and installation, using the sandbox's stock Python.

The SDK is supplied by Lakebox and uses that sandbox's own credentials. The
agent virtualenv, installers and package wheels are independent of that SDK.
"""

COMMON = r"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time

MAX_BYTES = 450 * 1024 * 1024
UV_VERSION = "0.13.5"
PIP_VERSION = "25.3"
metrics = {}

def phase(name, args, **kwargs):
    start = time.perf_counter()
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)
    metrics[name] = round((time.perf_counter() - start) * 1000, 2)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def target():
    return {
        "python": platform.python_version(), "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag, "platform": sysconfig.get_platform(),
        "soabi": sysconfig.get_config_var("SOABI"), "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "os_release": Path("/etc/os-release").read_text(),
    }

def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)

def clean_env(root):
    # Ignore per-machine uv/pip settings; use only the project's locked inputs.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    return {**env, "UV_CACHE_DIR": str(root / "cache"), "UV_LINK_MODE": "copy",
        "UV_NO_PROGRESS": "1", "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "PIP_INDEX_URL": "https://pypi.org/simple", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}

def import_check(python, entrypoint, root):
    phase("import_check", [str(python), "-c",
        "import importlib,sys; m,n=sys.argv[1].split(':'); assert callable(getattr(importlib.import_module(m),n))",
        entrypoint], cwd=root)
"""

BUILD = (
    COMMON
    + r"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import importlib.metadata
import traceback

root = Path(sys.argv[1]).resolve()
request = json.loads((root / "request.json").read_text())
source_hash, entrypoint = request["source_hash"], request["entrypoint"]
try:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Artifact builds require sandbox Python 3.12")
    archive = root / "source.tar.gz"
    if digest(archive) != source_hash:
        raise ValueError("Source archive digest mismatch")
    source = root / "source"
    source.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    env = clean_env(root)
    # Bootstrap pinned tooling once in the builder, never in a session sandbox.
    tools = root / "tools"
    system_uv = shutil.which("uv")
    if not system_uv:
        raise ValueError("The default sandbox must supply uv for builder tooling bootstrap")
    phase("toolchain_venv", [system_uv, "venv", "--python", sys.executable, str(tools)])
    python = str(tools / "bin/python")
    phase("toolchain_install", [system_uv, "pip", "install", "--python", python, "uv==" + UV_VERSION, "pip==" + PIP_VERSION], env=env)
    uv = str(tools / "bin/uv")
    payload = root / "payload"
    wheels, installer = payload / "wheels", payload / "installer"
    wheels.mkdir(parents=True)
    installer.mkdir()
    # Runtime bindings belong beside the installed virtualenv, where existing
    # first-party adapters discover their project configuration.
    if (source / "agent.toml").is_file():
        shutil.copyfile(source / "agent.toml", payload / "agent.toml")
    # uv selects extras, markers and locked local wheel sources for this project.
    lock = payload / "dependencies.lock"
    phase("export_lock", [uv, "export", "--locked", "--no-dev", "--no-emit-project",
        "--no-editable", "--output-file", str(lock)], cwd=source, env=env)
    phase("dependency_download", [python, "-m", "pip", "download", "--require-hashes",
        "--no-deps", "--only-binary=:all:", "-r", str(lock), "-d", str(wheels)], cwd=source, env=env)
    phase("installer_download", [python, "-m", "pip", "download", "--no-deps",
        "--only-binary=:all:", "uv==" + UV_VERSION, "-d", str(installer)], env=env)
    import tomllib
    project = tomllib.loads((source / "pyproject.toml").read_text())
    requirements = project.get("build-system", {}).get("requires")
    if not requirements:
        raise ValueError("An explicit build-system.requires is required")
    build_in = root / "build.in"
    build_in.write_text("\n".join(requirements) + "\n")
    build_lock = payload / "build-requirements.lock"
    phase("build_lock", [uv, "pip", "compile", "--generate-hashes", str(build_in), "-o", str(build_lock)], env=env)
    build_env = root / "build-env"
    phase("build_venv", [uv, "venv", "--python", sys.executable, str(build_env)], env=env)
    phase("build_dependencies", [uv, "pip", "install", "--python", str(build_env / "bin/python"),
        "--require-hashes", "-r", str(build_lock)], env=env)
    project_wheels = root / "project-wheels"
    phase("project_build", [uv, "build", "--wheel", "--no-build-isolation", "--python",
        str(build_env / "bin/python"), "--out-dir", str(project_wheels)], cwd=source, env=env)
    agent_wheels = list(project_wheels.glob("*.whl"))
    if len(agent_wheels) != 1:
        raise ValueError("Expected exactly one agent wheel")
    agent_wheel = wheels / agent_wheels[0].name
    shutil.copyfile(agent_wheels[0], agent_wheel)
    # First check the full closure against the wheelhouse, then check the exact
    # runtime command with an empty cache and dependency resolution disabled.
    offline = {**env, "UV_OFFLINE": "1", "UV_NO_INDEX": "1", "UV_PYTHON_DOWNLOADS": "never"}
    validation = root / "validation"
    phase("validation_venv", [uv, "venv", "--python", sys.executable, str(validation)], env=offline)
    phase("closure_check", [uv, "pip", "install", "--python", str(validation / "bin/python"),
        "--no-cache", "--offline", "--no-index", "--find-links", str(wheels), str(agent_wheel)], env=offline)
    shutil.rmtree(validation)
    install = payload / "install.txt"
    install.write_text("\n".join("./wheels/" + p.name + " --hash=sha256:" + digest(p)
        for p in sorted(wheels.glob("*.whl"))) + "\n")
    phase("validation_clean_venv", [uv, "venv", "--python", sys.executable, str(validation)], env=offline)
    phase("offline_install", [uv, "pip", "install", "--python", str(validation / "bin/python"),
        "--no-cache", "--offline", "--no-index", "--no-deps", "--require-hashes", "-r", str(install)], cwd=payload, env=offline)
    import_check(validation / "bin/python", entrypoint, root)
    inventory = json.loads(subprocess.check_output([str(validation / "bin/python"), "-c",
        "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"], cwd=root))
    manifest = {"format": "agentbricks.wheelhouse.v1", "source_hash": source_hash,
        "lock_hash": digest(source / "uv.lock"), "entrypoint": entrypoint, "target": target(),
        "installer": UV_VERSION, "pip": PIP_VERSION, "inventory": inventory,
        "files": {str(p.relative_to(payload)): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(payload.rglob("*")) if p.is_file()}}
    atomic(payload / "artifact.json", manifest)
    artifact = root / "execution.tar"
    with tarfile.open(artifact, "w") as bundle:
        for p in sorted(payload.rglob("*")):
            if p.is_file():
                bundle.add(p, arcname=str(p.relative_to(payload)), recursive=False)
    if artifact.stat().st_size > MAX_BYTES:
        raise ValueError("Execution artifact exceeds the 450 MiB Workspace storage limit")
    result = {"format": manifest["format"], "path": request["artifact_path"],
        "sha256": digest(artifact), "bytes": artifact.stat().st_size,
        "source_hash": source_hash, "entrypoint": entrypoint, "target": manifest["target"],
        "timings_ms": metrics, "packages": len(inventory), "lock_hash": manifest["lock_hash"]}
    workspace = WorkspaceClient()
    start = time.perf_counter()
    with artifact.open("rb") as content:
        workspace.workspace.upload(request["artifact_path"], content, format=ImportFormat.RAW, overwrite=False)
    result["timings_ms"]["artifact_upload"] = round((time.perf_counter() - start) * 1000, 2)
    atomic(root / "build-result.json", {"ok": True, "artifact": result})
except BaseException as exc:
    traceback.print_exc()
    atomic(root / "build-result.json", {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:2000]})
    sys.exit(1)
"""
)

FETCH = (
    COMMON
    + r"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat
root = Path(sys.argv[1])
artifact = json.loads(sys.argv[2])
if target() != artifact["target"]:
    raise ValueError("Execution artifact is incompatible with this sandbox; rebuild the environment")
root.mkdir(parents=True, exist_ok=True)
path = root / "execution.tar"
start = time.perf_counter()
h = hashlib.sha256()
size = 0
with WorkspaceClient().workspace.download(artifact["path"], format=ExportFormat.AUTO) as response, path.open("wb") as out:
    while chunk := response.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_BYTES or size > artifact["bytes"]:
            raise ValueError("Execution artifact exceeds its declared size")
        h.update(chunk)
        out.write(chunk)
if size != artifact["bytes"] or h.hexdigest() != artifact["sha256"]:
    path.unlink(missing_ok=True)
    raise ValueError("Execution artifact archive digest mismatch")
print(json.dumps({"artifact_download": round((time.perf_counter() - start) * 1000, 2)}))
"""
)

CHECK_READY = (
    COMMON
    + r"""
root = Path(sys.argv[1])
artifact = json.loads(sys.argv[2])
if target() != artifact["target"]:
    raise ValueError("Execution artifact is incompatible with this sandbox; rebuild the environment")
marker = root / "ready.json"
if marker.exists():
    saved = json.loads(marker.read_text())
    if saved.get("artifact_hash") != artifact["sha256"]:
        raise ValueError("Installed artifact differs from the pinned revision")
    print(json.dumps(saved))
"""
)

INSTALL = (
    COMMON
    + r"""
import fcntl
import zipfile
root = Path(sys.argv[1]).resolve()
artifact = json.loads(sys.argv[2])
with (root / "bootstrap.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    marker = root / "ready.json"
    if target() != artifact["target"]:
        raise ValueError("Execution artifact is incompatible with this sandbox; rebuild the environment")
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("artifact_hash") != artifact["sha256"]:
            raise ValueError("Installed artifact differs from the pinned revision")
        print(json.dumps({"reused": True, **saved}))
        sys.exit(0)
    if target() != artifact["target"]:
        raise ValueError("Execution artifact is incompatible with this sandbox; rebuild the environment")
    archive = root / "execution.tar"
    if archive.stat().st_size != artifact["bytes"] or digest(archive) != artifact["sha256"]:
        raise ValueError("Execution artifact archive digest mismatch")
    start = time.perf_counter()
    payload = root / "artifact"
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > MAX_BYTES:
            raise ValueError("Execution artifact expanded size limit exceeded")
        names = set()
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or member.name in names:
                raise ValueError("Unsafe execution artifact member")
            names.add(member.name)
        bundle.extractall(payload, filter="data")
    manifest = json.loads((payload / "artifact.json").read_text())
    if (manifest["format"] != "agentbricks.wheelhouse.v1" or manifest["target"] != target()
        or manifest["source_hash"] != artifact["source_hash"] or manifest["entrypoint"] != artifact["entrypoint"]):
        raise ValueError("Execution artifact metadata mismatch")
    if names != set(manifest["files"]) | {"artifact.json"}:
        raise ValueError("Execution artifact inventory mismatch")
    for name, expected in manifest["files"].items():
        p = payload / name
        if p.stat().st_size != expected["bytes"] or digest(p) != expected["sha256"]:
            raise ValueError("Execution artifact file digest mismatch: " + name)
    metrics["artifact_verify_extract"] = round((time.perf_counter() - start) * 1000, 2)
    uv_wheels = list((payload / "installer").glob("uv-*.whl"))
    if len(uv_wheels) != 1:
        raise ValueError("Expected one pinned uv installer wheel")
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    uv = tools / "uv"
    with zipfile.ZipFile(uv_wheels[0]) as wheel:
        scripts = [n for n in wheel.namelist() if n.endswith(".data/scripts/uv")]
        if len(scripts) != 1:
            raise ValueError("uv wheel has no executable")
        with wheel.open(scripts[0]) as src, uv.open("wb") as dest:
            shutil.copyfileobj(src, dest)
    uv.chmod(0o700)
    env = {**clean_env(root), "UV_OFFLINE": "1", "UV_NO_INDEX": "1", "UV_PYTHON_DOWNLOADS": "never"}
    venv = root / "agent-env"
    if venv.exists():
        shutil.rmtree(venv)
    phase("agent_venv", [str(uv), "venv", "--python", sys.executable, str(venv)], env=env)
    phase("offline_install", [str(uv), "pip", "install", "--python", str(venv / "bin/python"),
        "--no-cache", "--offline", "--no-index", "--no-deps", "--require-hashes",
        "-r", str(payload / "install.txt")], cwd=payload, env=env)
    installed = json.loads(subprocess.check_output([str(venv / "bin/python"), "-c",
        "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"], cwd=root))
    if installed != manifest["inventory"]:
        raise ValueError("Installed package inventory differs from the validated artifact")
    if (payload / "agent.toml").is_file():
        shutil.copyfile(payload / "agent.toml", root / "agent.toml")
    result = {"source_hash": artifact["source_hash"], "artifact_hash": artifact["sha256"],
        "timings_ms": metrics, "python": str(venv / "bin/python"), "uv": str(uv)}
    atomic(marker, result)
    print(json.dumps({"reused": False, **result}))
"""
)
