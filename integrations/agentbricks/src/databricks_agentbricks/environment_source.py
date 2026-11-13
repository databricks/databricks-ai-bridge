"""Bounded project uploads for sandbox environments; never execute local agent code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from databricks_agentbricks.errors import AgentCliError

MAX_BYTES = 20 * 1024 * 1024
MAX_FILES = 1000
IGNORED = {"__pycache__", "build", "dist", "node_modules"}


def files_digest(files: dict[str, bytes]) -> str:
    entries = [[name, hashlib.sha256(data).hexdigest()] for name, data in sorted(files.items())]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def repository_url(value: str) -> str | None:
    """Record repository provenance without transport credentials or local paths."""
    if "://" not in value:
        match = re.fullmatch(r"(?:[^/@:]+@)?([^/:]+):(.+)", value)
        if not match:
            return None
        value = f"https://{match[1]}/{match[2]}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https", "ssh", "git"} or not parsed.hostname:
        return None
    host = parsed.hostname
    if parsed.scheme in {"http", "https"} and parsed.port:
        host += f":{parsed.port}"
    return f"https://{host}{parsed.path}".removesuffix(".git")


def git_project_files(source: Path, revision: str) -> tuple[dict[str, bytes], dict]:
    """Read committed blobs, without checking out files or applying export filters."""

    def git(*args: str, optional: bool = False) -> bytes:
        try:
            result = subprocess.run(
                ["git", "--no-replace-objects", "-C", str(source), *args],
                capture_output=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentCliError("Could not read the local Git repository.") from exc
        if result.returncode and not optional:
            raise AgentCliError(
                "Could not resolve Git source; use a local commit, tag, or branch containing the project.",
                hint="Fetch the revision into your local repository before creating the environment.",
            )
        return result.stdout if result.returncode == 0 else b""

    commit = (
        git("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()
    )
    prefix = git("rev-parse", "--show-prefix").decode().removesuffix("\n").rstrip("/")
    tree = f"{commit}:{prefix}" if prefix else commit
    files = {}
    size = 0
    for entry in git("ls-tree", "--full-tree", "-r", "-z", tree).split(b"\0"):
        if not entry:
            continue
        info, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = info.decode().split()
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentCliError("Git source filenames must be UTF-8.") from exc
        path = PurePosixPath(name)
        if any(part.startswith(".") or part in IGNORED for part in path.parts) or path.suffix in {
            ".pyc",
            ".pem",
            ".key",
            ".p12",
        }:
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise AgentCliError(f"Git source symlinks and submodules are unsupported: {name}")
        length = int(git("cat-file", "-s", object_id).strip())
        if size + length > MAX_BYTES or len(files) >= MAX_FILES:
            raise AgentCliError("Environment source is limited to 20 MiB and 1000 files.")
        files[name] = git("cat-file", "blob", object_id)
        size += length
    for required in ("pyproject.toml", "uv.lock"):
        if required not in files:
            raise AgentCliError(
                f"Git revision must include {required} in the selected project folder."
            )
    remote = git("config", "--get", "remote.origin.url", optional=True).decode().strip()
    return files, {
        "commit": commit,
        "requested_revision": revision,
        "repository_url": repository_url(remote),
        "subdirectory": prefix,
        "files_digest": files_digest(files),
    }


def environment_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise AgentCliError("Environment names use 1–64 letters, digits, underscores, or hyphens.")
    return value


def entrypoint_name(value: str) -> str:
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    if not re.fullmatch(rf"{identifier}(?:\.{identifier})*:{identifier}", value):
        raise AgentCliError("Entrypoint must be an installed Python module:function.")
    return value


def workspace_path(value: str) -> str:
    if not value.startswith("/") or ".." in value.split("/") or "\x00" in value:
        raise AgentCliError("Choose an absolute Workspace folder without '..'.")
    value = str(PurePosixPath(value))
    if value.startswith("/Workspace/"):
        value = value[len("/Workspace") :]
    if value in {"/", "/Workspace", "/Volumes"} or value.startswith("/Volumes/"):
        raise AgentCliError("Choose a Workspace source folder, not a root or Volume.")
    return value


def project_files(source: Path, *, require_lock: bool = True) -> dict[str, bytes]:
    """Read a source snapshot before making any remote changes."""
    files: dict[str, bytes] = {}
    size = 0
    for directory, folders, names in os.walk(source, followlinks=False):
        folders[:] = sorted(n for n in folders if not n.startswith(".") and n not in IGNORED)
        for name in [*folders, *sorted(names)]:
            path = Path(directory) / name
            if name.startswith("."):
                continue
            if path.is_symlink():
                raise AgentCliError(f"Source symlinks are unsupported: {path.relative_to(source)}")
            if not path.is_file() or path.suffix in {".pyc", ".pem", ".key", ".p12"}:
                continue
            with path.open("rb") as stream:
                contents = stream.read(MAX_BYTES - size + 1)
            size += len(contents)
            if size > MAX_BYTES or len(files) >= MAX_FILES:
                raise AgentCliError("Environment source is limited to 20 MiB and 1000 files.")
            files[path.relative_to(source).as_posix()] = contents
    for required in ("pyproject.toml", "uv.lock") if require_lock else ("pyproject.toml",):
        if required not in files:
            raise AgentCliError(f"Agent source must include {required}.")
    return files
