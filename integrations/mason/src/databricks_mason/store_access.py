"""Shared Lakebase access plumbing for granting a deployed app's service principal store access.

A managed store (session or memory) is backed by a per-store Lakebase database in a shared,
service-managed project. The store's tables are owned by whoever created the store (the deploying
user), so giving the deployed app's service principal access takes TWO steps:
  1. a `postgres` app resource so the SP gets a Lakebase role and CONNECT on the database, and
  2. a table GRANT (issued over a Postgres connection as the store owner) so the SP can read/write
     the tables.
With only (1) the SP connects but hits "permission denied" on the tables; with only (2) it can't
connect at all. Both are best-effort — deploy proceeds and reports if either step can't be applied.

This module holds the store-agnostic mechanics. Session, memory, and durability modules supply the
backend-specific project/schema/table details.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

import psycopg

from databricks_mason.errors import AgentCliError

_MASON_POSTGRES_RESOURCE_NAMES = frozenset({"postgres", "postgres-memory"})


def _databricks(
    args: list[str],
    profile: Optional[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
    action: Optional[str] = None,
) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    result = subprocess.run(cmd, text=True, capture_output=capture, cwd=cwd)
    if check and result.returncode != 0:
        # Mason drives the `databricks apps` CLI as an implementation detail; surface a failure in
        # Mason's own terms (`action`) rather than echoing the raw subcommand and --profile, which
        # leaks the underlying tool at the customer. The captured stderr still rides along as the
        # hint so debugging isn't lost.
        detail = (result.stderr or result.stdout or "").strip() if capture else None
        raise AgentCliError(action or "A Databricks CLI command failed.", hint=detail)
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


def _current_app_resources(app: str, profile: Optional[str]) -> list[dict]:
    """Read the app's existing resources array (empty list if it can't be read)."""
    result = _databricks(["apps", "get", app, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return []
    try:
        resources = json.loads(result.stdout or "{}").get("resources", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return resources if isinstance(resources, list) else []


def apply_postgres_resources(
    app: str, backends: list[LakebaseBackend], profile: Optional[str]
) -> Optional[str]:
    """Bind each backend's database as a `postgres` app resource in one update.

    The resources update replaces the whole array, so send the complete set: read the app's current
    resources, drop the names Mason manages so re-deploys update rather than duplicate them, keep
    every other resource, and put the selected durability backend first. Databricks Apps maps the
    first Postgres resource to the standard ``PG*`` environment variables.
    Returns None on success or a human-readable reason on failure.
    """
    ours = [b.postgres_resource() for b in backends]
    preserved = [
        r
        for r in _current_app_resources(app, profile)
        if isinstance(r, dict) and r.get("name") not in _MASON_POSTGRES_RESOURCE_NAMES
    ]
    payload = {"update_mask": "resources", "app": {"resources": ours + preserved}}
    result = _databricks(
        ["apps", "create-update", app, "--json", json.dumps(payload)],
        profile,
        capture=True,
        check=False,
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
    """Grant the app's SP read/write on the backend's tables (owner-issued, over a pg connection).

    Uses psycopg (a bundled dependency) rather than the psql binary so no system Postgres client is
    required. Returns None on success, or a human-readable reason if the grant couldn't be applied.
    Deploy proceeds regardless — the app runs, but the store's durable path fails until the SP has
    access.
    """
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
    try:
        # The password token is scoped to this endpoint; the SP's role name is its client id.
        with psycopg.connect(
            host=host,
            port=5432,
            dbname=backend.database,
            user=owner,
            password=token,
            sslmode="require",
            autocommit=True,
        ) as conn:
            # Encode to bytes: the GRANT is composed at runtime (not a LiteralString), which the
            # str overload of execute() rejects; the bytes overload takes a composed query as-is.
            conn.execute(grants.encode())
    except psycopg.Error as exc:
        return str(exc).strip() or "unknown error"
    return None
