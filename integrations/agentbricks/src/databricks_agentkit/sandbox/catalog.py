"""Published revisions live in Workspace files; Lakebase stores execution state only."""

import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache

from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import ExportFormat, ImportFormat, ObjectType

from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.sources import source_path


class WorkspaceCatalog:
    def __init__(self, workspace, root):
        self.workspace = workspace
        self.root = source_path("workspace", root).rstrip("/")
        self._cached_revision = lru_cache(maxsize=128)(self._revision)
        self._cached_archive = lru_cache(maxsize=4)(self._archive)

    def _path(self, kind, name, suffix=".json"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("Invalid publication identifier")
        return f"{self.root}/{kind}/{name}{suffix}"

    def _read(self, path, limit=1024 * 1024):
        with self.workspace.workspace.download(path, format=ExportFormat.AUTO) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Publication exceeds size limit")
        return data

    def _publish(self, path, data):
        try:
            existing = self._read(path, max(len(data), 1024 * 1024))
        except NotFound:
            self.workspace.workspace.mkdirs(path.rsplit("/", 1)[0])
            try:
                self.workspace.workspace.upload(
                    path, data, format=ImportFormat.RAW, overwrite=False
                )
                return
            except Exception:
                # Reconcile concurrent publication or a lost successful upload reply.
                existing = self._read(path, max(len(data), 1024 * 1024))
        if existing != data:
            raise ValueError("Published revision differs; publish a new revision")

    def get(self, kind, name):
        try:
            return json.loads(self._read(self._path(kind, name)))
        except NotFound as exc:
            raise KeyError("Publication not found; publish source with deploy --sandbox") from exc

    def put(self, kind, name, value):
        self._publish(self._path(kind, name), json.dumps(value, sort_keys=True).encode())
        return value

    def list(self, kind):
        try:
            entries = list(self.workspace.workspace.list(f"{self.root}/{kind}"))
        except NotFound:
            return []
        values = [
            self.get(kind, entry.path.rsplit("/", 1)[-1][:-5])
            for entry in entries
            if entry.object_type == ObjectType.FILE and entry.path and entry.path.endswith(".json")
        ]
        return sorted(values, key=lambda value: value["created_at"], reverse=True)[:100]

    def create_agent(
        self, name, kind, path, entrypoint, bundle, *, git_source=None, execution_artifact=None
    ):
        parts = [name, kind, path, entrypoint, bundle.digest]
        if git_source:
            parts.append(git_source)
        if execution_artifact:
            parts.append(execution_artifact)
        agent_id = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]
        archive = self._path("sources", agent_id, ".tar.gz")
        self._publish(archive, bundle.data)
        try:
            return self.agent(agent_id)
        except KeyError:
            pass
        value = dict(
            id=agent_id,
            name=name,
            source_kind=kind,
            source_path=path,
            entrypoint=entrypoint,
            source_hash=bundle.digest,
            file_count=bundle.file_count,
            bundle_bytes=len(bundle.data),
            source_archive=archive,
            git_source=git_source,
            execution_artifact=execution_artifact,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            return self.put("revisions", agent_id, value)
        except ValueError:
            # Concurrent identical publications can differ only in creation time.
            existing = self.agent(agent_id)
            if {k: v for k, v in existing.items() if k != "created_at"} != {
                k: v for k, v in value.items() if k != "created_at"
            }:
                raise
            return existing

    def _revision(self, agent_id):
        return self.get("revisions", agent_id)

    def _archive(self, agent_id, digest):
        data = self._read(self._path("sources", agent_id, ".tar.gz"), 21 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Published source archive digest mismatch")
        return data

    def agent(self, agent_id, *, with_bundle=False):
        value = dict(self._cached_revision(agent_id))
        if with_bundle:
            # Derive the archive path rather than trusting an arbitrary manifest URI.
            value["bundle_data"] = self._cached_archive(agent_id, value["source_hash"])
        return value

    def bundle(self, agent_id):
        value = self.agent(agent_id, with_bundle=True)
        return Bundle(value["bundle_data"], value["source_hash"], value["file_count"])
