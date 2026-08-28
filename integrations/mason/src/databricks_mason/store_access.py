"""Shared Lakebase access plumbing for granting a deployed app's service principal store access.

A managed store (session or memory) is backed by a per-store Lakebase database in a shared,
service-managed project. The store's tables are owned by whoever created the store (the deploying
user), so giving the deployed app's service principal access takes TWO steps:
  1. a `postgres` app resource so the SP gets a Lakebase role and CONNECT on the database, and
  2. a table GRANT (issued over psql as the store owner) so the SP can read/write the tables.
With only (1) the SP connects but hits "permission denied" on the tables; with only (2) it can't
connect at all. Both are best-effort — deploy proceeds and reports if either step can't be applied.

This module holds the store-agnostic mechanics; `session_store_access` and `memory_store_access`
supply the per-store project/schema/table specifics.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from databricks_mason.errors import AgentCliError


def _databricks(
    args: list[str],
    profile: Optional[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    result = subprocess.run(cmd, text=True, capture_output=capture, cwd=cwd)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else None
        raise AgentCliError(f"`{' '.join(cmd)}` failed (exit {result.returncode})", hint=detail)
    return result


@dataclass(frozen=True)
class LakebaseBackend:
    """A per-store Lakebase database: its shared project/branch, its endpoint, and its schema/tables."""

    project: str
    branch: str
    endpoint_id: str
    database: str
    schema: str
    tables: tuple[str, ...]
    resource_name: str  # the app-resource name (must be unique across an app's resources)

    @property
    def branch_path(self) -> str:
        return f"projects/{self.project}/branches/{self.branch}"

    @property
    def database_path(self) -> str:
        return f"{self.branch_path}/databases/{self.database}"

    @property
    def endpoint_path(self) -> str:
        return f"{self.branch_path}/endpoints/{self.endpoint_id}"

    def postgres_resource(self) -> dict:
        """The `postgres` app-resource entry that grants the SP a Lakebase role + CONNECT."""
        return {
            "name": self.resource_name,
            "postgres": {
                "branch": self.branch_path,
                "database": self.database_path,
                "permission": "CAN_CONNECT_AND_CREATE",
            },
        }


def apply_postgres_resources(
    app: str, backends: list[LakebaseBackend], profile: Optional[str]
) -> Optional[str]:
    """Bind each backend's database as a `postgres` app resource in one update.

    `apps update --json` replaces the whole resources array, so all needed backends must be applied
    together. Returns None on success or a human-readable reason on failure.
    """
    payload = {"resources": [b.postgres_resource() for b in backends]}
    result = _databricks(
        ["apps", "update", app, "--json", json.dumps(payload)], profile, capture=True, check=False
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"


def _resolve_pg_host(backend: LakebaseBackend, profile: Optional[str]) -> Optional[str]:
    """Read the backend branch's read-write endpoint host, or None if it can't be resolved."""
    result = _databricks(
        ["postgres", "get-endpoint", backend.endpoint_path, "-o", "json"],
        profile,
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        hosts = json.loads(result.stdout).get("status", {}).get("hosts", {})
    except (json.JSONDecodeError, AttributeError):
        return None
    return hosts.get("host") if isinstance(hosts, dict) else None


def _mint_token(backend: LakebaseBackend, profile: Optional[str]) -> Optional[str]:
    """Mint an owner OAuth token for the backend endpoint (used as the psql password)."""
    result = _databricks(
        ["postgres", "generate-database-credential", backend.endpoint_path, "-o", "json"],
        profile,
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("token")
    except json.JSONDecodeError:
        return None


def grant_tables(
    backend: LakebaseBackend, sp_client_id: str, owner: str, profile: Optional[str]
) -> Optional[str]:
    """Grant the app's SP read/write on the backend's tables (owner-issued, over psql).

    Returns None on success, or a human-readable reason if the grant couldn't be applied. Deploy
    proceeds regardless — the app runs, but the store's durable path fails until the SP has access.
    """
    if shutil.which("psql") is None:
        return "psql not found on PATH (needed to grant the app SP access to the store)."
    host = _resolve_pg_host(backend, profile)
    if not host:
        return "could not resolve the store's Lakebase endpoint."
    token = _mint_token(backend, profile)
    if not token:
        return "could not mint a Lakebase credential for the store (need store ownership)."

    # The SP's Postgres role is its application id, verbatim. Qualify tables with their schema (the
    # SP's search_path may not include it), and add default privileges so tables the service creates
    # later are covered too.
    qualified = ", ".join(f"{backend.schema}.{t}" for t in backend.tables)
    grants = (
        f'GRANT USAGE ON SCHEMA {backend.schema} TO "{sp_client_id}";'
        f' GRANT SELECT, INSERT, UPDATE, DELETE ON {qualified} TO "{sp_client_id}";'
        f" ALTER DEFAULT PRIVILEGES IN SCHEMA {backend.schema}"
        f' GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{sp_client_id}";'
    )
    conn = f"host={host} port=5432 dbname={backend.database} user={owner} sslmode=require"
    result = subprocess.run(
        ["psql", conn, "-v", "ON_ERROR_STOP=1", "-c", grants],
        env={**os.environ, "PGPASSWORD": token},
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"
