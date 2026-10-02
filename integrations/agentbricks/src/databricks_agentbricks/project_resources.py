"""Local provisioning receipts. A binding or a matching name is never proof of ownership."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any

from databricks_agentbricks.errors import AgentCliError

_RECEIPTS = pathlib.Path(".agentbricks/resources.json")
_KINDS = {"deployment", "memory_store", "session_store"}


def load_receipts(source: pathlib.Path) -> list[dict[str, Any]]:
    path = source / _RECEIPTS
    if not path.exists():
        return []
    try:
        document = json.loads(path.read_text())
        if document.get("schema_version") != 1 or not isinstance(document.get("resources"), list):
            raise ValueError("unsupported receipt schema")
        for item in document["resources"]:
            if (
                not isinstance(item, dict)
                or item.get("kind") not in _KINDS
                or not all(
                    isinstance(item.get(key), str) and item[key] for key in ("host", "name", "id")
                )
                or not isinstance(item.get("created"), bool)
            ):
                raise ValueError("invalid resource receipt")
        return document["resources"]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise AgentCliError(f"Could not read resource receipts at {path}: {exc}.") from exc


def save_receipts(source: pathlib.Path, resources: list[dict[str, Any]]) -> None:
    path = source / _RECEIPTS
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = output.name
            json.dump({"schema_version": 1, "resources": resources}, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        if temporary:
            pathlib.Path(temporary).unlink(missing_ok=True)
        raise AgentCliError(
            f"Could not save provisioning receipts at {path}: {exc}.",
            hint="The cloud operation may have succeeded. Retain this project's receipts and inspect status before retrying.",
        ) from exc


def record_resource(
    source: pathlib.Path,
    *,
    host: str,
    kind: str,
    name: str,
    resource_id: str,
    created: bool,
    managed_runtime: bool = False,
) -> None:
    """Record successful create/reuse immediately, including deployments that later fail."""
    resources = load_receipts(source)
    host = host.rstrip("/")
    previous = next(
        (r for r in resources if (r["host"], r["kind"], r["name"]) == (host, kind, name)),
        None,
    )
    receipt = {
        "host": host,
        "kind": kind,
        "name": name,
        "id": resource_id,
        "created": created
        or bool(previous and previous["id"] == resource_id and previous["created"]),
        "managed_runtime": managed_runtime,
        "cleanup": "active",
    }
    if previous is not None:
        resources.remove(previous)
    resources.append(receipt)
    save_receipts(source, resources)
