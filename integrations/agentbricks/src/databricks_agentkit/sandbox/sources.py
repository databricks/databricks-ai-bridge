"""Read source files through platform APIs without importing agent code in the App."""

import base64
import re
from pathlib import Path, PurePosixPath

from databricks.sdk.service.workspace import ExportFormat, ObjectType

from databricks_agentkit.sandbox.bundle import (
    MAX_SOURCE_BYTES,
    MAX_SOURCE_FILES,
    Bundle,
    build_bundle,
    bundle_files,
    included_path,
)

DEFAULT_ENTRYPOINT = "sandbox_agent.worker:run"


def validate_entrypoint(value: str) -> str:
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    if not re.fullmatch(rf"{identifier}(?:\.{identifier})*:{identifier}", value):
        raise ValueError("Entrypoint must be an installed Python module:function")
    return value


def source_path(kind: str, path: str) -> str:
    if not path.startswith("/") or ".." in path.split("/") or "\x00" in path:
        raise ValueError("Choose an absolute source directory without '..'")
    path = str(PurePosixPath(path))
    if kind == "workspace":
        if path.startswith("/Workspace/"):
            path = path[len("/Workspace") :]
        if path == "/" or path.startswith("/Volumes/") or path == "/Workspace":
            raise ValueError("Choose a Workspace source folder, not a workspace root or Volume")
    elif kind == "volume":
        if not path.startswith("/Volumes/") or len(PurePosixPath(path).parts) < 5:
            raise ValueError("Volume paths start with /Volumes/<catalog>/<schema>/<volume>")
    else:
        raise ValueError("Source kind must be workspace or volume")
    return path


class SourceLoader:
    def __init__(self, workspace, bundled_source: Path):
        self.workspace = workspace
        self.bundled_source = bundled_source

    def load(self, kind: str, path: str = "") -> Bundle:
        if kind == "bundled":
            return build_bundle(self.bundled_source)
        root = source_path(kind, path)
        if kind == "workspace":
            status = self.workspace.workspace.get_status(root)
            if status.object_type != ObjectType.DIRECTORY:
                raise ValueError("Workspace source must be a directory of source files")
        files = {}
        size = 0
        visited = set()

        def visit(directory: str, depth: int = 0):
            nonlocal size
            if depth > 32 or len(visited) >= MAX_SOURCE_FILES:
                raise ValueError("Source directory limit exceeded")
            if directory in visited:
                raise ValueError("Source directory appeared more than once")
            visited.add(directory)
            entries = (
                self.workspace.workspace.list(directory)
                if kind == "workspace"
                else self.workspace.files.list_directory_contents(directory)
            )
            for entry in entries:
                item = entry.path
                # Check both containment and direct parenthood before following API paths.
                if not item or PurePosixPath(item).parent != PurePosixPath(directory):
                    raise ValueError("Source API returned an entry outside the requested directory")
                relative = PurePosixPath(item).relative_to(root).as_posix()
                if not included_path(relative):
                    continue
                is_directory = (
                    entry.object_type == ObjectType.DIRECTORY
                    if kind == "workspace"
                    else entry.is_directory
                )
                if is_directory:
                    visit(item, depth + 1)
                    continue
                if kind == "workspace" and entry.object_type != ObjectType.FILE:
                    raise ValueError(f"Use Workspace files, not notebooks or libraries: {relative}")
                if len(files) >= MAX_SOURCE_FILES:
                    raise ValueError("Source limit is 1000 files")
                remaining = MAX_SOURCE_BYTES - size
                declared_size = entry.size if kind == "workspace" else entry.file_size
                if declared_size is not None and declared_size > remaining:
                    raise ValueError("Source limit is 20 MiB uncompressed")
                if kind == "workspace":
                    result = self.workspace.workspace.export(item, format=ExportFormat.AUTO)
                    if result.content is None:
                        raise ValueError(f"Workspace file had no content: {relative}")
                    if len(result.content) > 4 * ((remaining + 2) // 3):
                        raise ValueError("Source limit is 20 MiB uncompressed")
                    contents = base64.b64decode(result.content, validate=True)
                else:
                    response = self.workspace.files.download(item)
                    if response.contents is None:
                        raise ValueError(f"Volume file had no content: {relative}")
                    with response.contents as stream:
                        contents = stream.read(remaining + 1)
                size += len(contents)
                if size > MAX_SOURCE_BYTES:
                    raise ValueError("Source limit is 20 MiB uncompressed")
                files[relative] = contents

        visit(root)
        return bundle_files(files)
