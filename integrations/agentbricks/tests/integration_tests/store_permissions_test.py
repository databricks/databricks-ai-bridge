"""Opt-in permission CLI lifecycle with three existing identities in one workspace.

See CONTRIBUTING.md for the profile and principal configuration. Creates only test stores,
removes their grants, and deletes the stores; does not provision identities or change flags.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from databricks_agentkit import AgentKitClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENTBRICKS_PERMISSION_INTEGRATION_TESTS") != "1",
    reason="Opt in with RUN_AGENTBRICKS_PERMISSION_INTEGRATION_TESTS=1 and three workspace profiles",
)


@pytest.fixture
def identities():
    from databricks.sdk import WorkspaceClient

    names = {
        role: os.environ[f"AGENTBRICKS_PERMISSION_TEST_{role.upper()}_PROFILE"]
        for role in ("creator", "admin", "grantee")
    }
    principal = os.environ["AGENTBRICKS_PERMISSION_TEST_GRANTEE_PRINCIPAL"]
    principal_type = os.environ.get(
        "AGENTBRICKS_PERMISSION_TEST_PRINCIPAL_TYPE", "SERVICE_PRINCIPAL"
    )
    assert all(names.values()) and principal, (
        "Configure creator/admin/grantee profiles and the grantee principal"
    )
    assert principal_type in {"USER", "SERVICE_PRINCIPAL"}
    clients = {role: WorkspaceClient(profile=profile) for role, profile in names.items()}
    assert len({(w.config.host or "").rstrip("/") for w in clients.values()}) == 1
    users = {role: w.current_user.me() for role, w in clients.items()}
    assert len({user.id for user in users.values()}) == 3, (
        "Profiles must authenticate as distinct identities"
    )
    for role, user in users.items():
        admin = any(group.display == "admins" for group in (user.groups or []))
        assert admin == (role == "admin"), f"{role} has unexpected workspace admin membership"
    assert users["grantee"].user_name == principal, (
        "Grantee principal must match its authenticated identity"
    )
    return names, clients, principal, principal_type


@pytest.mark.parametrize("kind", ["session", "memory"])
def test_permission_cli_lifecycle(identities, kind):
    profiles, clients, principal, principal_type = identities
    creator = AgentKitClient(clients["creator"])
    store = getattr(creator, f"{kind}_stores").create(f"cli-permissions-{uuid4().hex[:12]}")
    cli = Path(sys.executable).with_name("agentbricks")
    root = "sessions" if kind == "session" else "memory"
    principal_args = ["--user" if principal_type == "USER" else "--service-principal", principal]
    key = f"{kind}_store_permissions"

    def run(role, operation, *args, denied=False):
        result = subprocess.run(
            [
                str(cli),
                "--profile",
                profiles[role],
                "--output",
                "json",
                root,
                "stores",
                "permissions",
                operation,
                store.name,
                *args,
            ],
            env={**os.environ, "AGENTBRICKS_DISABLE_TELEMETRY": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        if denied:
            assert result.returncode != 0, result.stdout
            assert json.loads(result.stderr)["error"]["code"] == "PERMISSION_DENIED"
            return {}
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def permission(role):
        return run(role, "get").get("permission", "PERMISSION_LEVEL_UNSPECIFIED")

    try:
        assert permission("creator") == "WRITE"
        assert permission("grantee") == "PERMISSION_LEVEL_UNSPECIFIED"
        assert run("creator", "list", "--all").get(key, []) == []
        run("admin", "grant", *principal_args, "--permission", "READ")
        assert permission("grantee") == "READ"
        grants = run("creator", "list", "--all")[key]
        assert [(g["principal"], g["permission"]) for g in grants] == [
            ({"principal_type": principal_type, "name": principal}, "READ")
        ]
        run("admin", "grant", *principal_args, "--permission", "WRITE")
        run("creator", "grant", *principal_args, "--permission", "READ")
        assert permission("grantee") == "WRITE"
        run("grantee", "list", denied=True)
        run("grantee", "grant", *principal_args, "--permission", "WRITE", denied=True)
        run("grantee", "revoke", *principal_args, "--yes", denied=True)
        run("creator", "revoke", *principal_args, "--yes")
        run("admin", "revoke", *principal_args, "--yes")
        assert permission("grantee") == "PERMISSION_LEVEL_UNSPECIFIED"
        assert run("admin", "list", "--all").get(key, []) == []
    finally:
        try:
            store.revoke_permission(principal, principal_type=principal_type)
        finally:
            store.delete()
