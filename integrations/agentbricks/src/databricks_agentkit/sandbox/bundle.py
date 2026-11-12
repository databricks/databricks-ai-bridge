"""Small, deterministic source archives. Never copy the App's environment."""

import gzip
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_FILES = 1000
IGNORED_DIRECTORIES = {"__pycache__", "dist", "build", "node_modules"}


def included_path(relative: str) -> bool:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError("Source entries must be relative paths without '..'")
    return not any(part.startswith(".") or part in IGNORED_DIRECTORIES for part in parts) and (
        Path(relative).suffix not in {".pyc", ".pem", ".key", ".p12"}
    )


@dataclass(frozen=True)
class Bundle:
    data: bytes
    digest: str
    file_count: int


def bundle_contents(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        return {
            entry.name: archive.extractfile(entry).read() for entry in archive if entry.isfile()
        }


def files_digest(files: dict[str, bytes]) -> str:
    entries = [[name, hashlib.sha256(data).hexdigest()] for name, data in sorted(files.items())]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def build_bundle(source: Path) -> Bundle:
    files = {}
    size = 0
    for directory, folders, names in os.walk(source, followlinks=False):
        folders[:] = sorted(
            name for name in folders if not name.startswith(".") and name not in IGNORED_DIRECTORIES
        )
        for name in [*folders, *sorted(names)]:
            path = Path(directory) / name
            relative = path.relative_to(source)
            if name.startswith("."):
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlinks are unsupported: {relative}")
            if path.is_file() and included_path(relative.as_posix()):
                with path.open("rb") as stream:
                    data = stream.read(MAX_SOURCE_BYTES - size + 1)
                size += len(data)
                if size > MAX_SOURCE_BYTES or len(files) >= MAX_SOURCE_FILES:
                    raise ValueError("Source limit is 20 MiB and 1000 files")
                files[relative.as_posix()] = data
    return bundle_files(files)


def bundle_files(files: dict[str, bytes]) -> Bundle:
    files = {name: data for name, data in files.items() if included_path(name)}
    for required in ("pyproject.toml", "uv.lock"):
        if required not in files:
            raise ValueError(f"Agent source must include {required}")
    if sum(len(data) for data in files.values()) > MAX_SOURCE_BYTES:
        raise ValueError("Prototype source limit is 20 MiB uncompressed")
    if len(files) > MAX_SOURCE_FILES:
        raise ValueError("Source limit is 1000 files")
    stream = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, contents in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(contents))
    data = stream.getvalue()
    return Bundle(data, hashlib.sha256(data).hexdigest(), len(files))
