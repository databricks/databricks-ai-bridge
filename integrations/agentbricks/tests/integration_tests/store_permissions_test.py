"""Opt-in permission and data-access CLI tests with three identities in one workspace.

See CONTRIBUTING.md for configuration. All store operations use the installed CLI;
SDK calls only verify existing identities. Tests never provision identities or change flags.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENTBRICKS_PERMISSION_INTEGRATION_TESTS") != "1",
    reason="Opt in with RUN_AGENTBRICKS_PERMISSION_INTEGRATION_TESTS=1 and three workspace profiles",
)

_LOG = logging.getLogger(__name__)
_NO_ACCESS = "PERMISSION_LEVEL_UNSPECIFIED"


@pytest.fixture(scope="module")
def identities():
    from databricks.sdk import WorkspaceClient

    profiles = {
        role: os.environ[f"AGENTBRICKS_PERMISSION_TEST_{role.upper()}_PROFILE"]
        for role in ("creator", "admin", "grantee")
    }
    principal = os.environ["AGENTBRICKS_PERMISSION_TEST_GRANTEE_PRINCIPAL"]
    principal_type = os.environ.get(
        "AGENTBRICKS_PERMISSION_TEST_PRINCIPAL_TYPE", "SERVICE_PRINCIPAL"
    )
    assert all(profiles.values()) and principal
    assert principal_type in {"USER", "SERVICE_PRINCIPAL"}
    clients = {role: WorkspaceClient(profile=profile) for role, profile in profiles.items()}
    assert len({(w.config.host or "").rstrip("/") for w in clients.values()}) == 1
    users = {role: w.current_user.me() for role, w in clients.items()}
    assert len({user.id for user in users.values()}) == 3
    for role, user in users.items():
        admin = any(group.display == "admins" for group in (user.groups or []))
        assert admin == (role == "admin"), f"{role} has unexpected workspace admin membership"
    assert users["grantee"].user_name == principal
    principals = {
        role: ("--user" if "@" in user.user_name else "--service-principal", user.user_name)
        for role, user in users.items()
    }
    assert principals["grantee"][0] == (
        "--user" if principal_type == "USER" else "--service-principal"
    )
    return profiles, principals


@pytest.fixture(scope="module")
def cli(identities):
    profiles, _ = identities
    executable = Path(sys.executable).with_name("agentbricks")
    assert executable.is_file(), "Install the package to test its console entry point"
    # Explicit profiles must not inherit a different caller's environment credentials.
    env = {key: value for key, value in os.environ.items() if not key.startswith("DATABRICKS_")}
    env.update(
        {
            "AGENTBRICKS_DISABLE_TELEMETRY": "1",
            "AGENTBRICKS_DISABLE_AGENT_HINT": "1",
            **{
                key: os.environ[key]
                for key in ("DATABRICKS_CONFIG_FILE", "DATABRICKS_CLI_PATH")
                if key in os.environ
            },
        }
    )

    def run(role, *args, error=None, usage=False, output="json"):
        command = [str(executable), "--profile", profiles[role], "--output", output, *args]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        _LOG.info(
            "%s %s -> exit=%s stdout=%s stderr=%s",
            role,
            " ".join(args),
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )
        if usage:
            assert result.returncode == 2, result.stderr
            return result.stderr
        if error:
            assert result.returncode != 0, result.stdout
            data = json.loads(result.stderr)
            assert data["error"]["code"] in error, data
            return data
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout) if output == "json" else result.stdout

    return run


def _root(store):
    return "sessions" if store["kind"] == "session" else "memory"


def _permission(cli, store, role, operation, *args, **kwargs):
    return cli(
        role, _root(store), "stores", "permissions", operation, store["name"], *args, **kwargs
    )


def _level(cli, store, role):
    return _permission(cli, store, role, "get").get("permission", _NO_ACCESS)


def _grants(cli, store, role, *args):
    data = _permission(cli, store, role, "list", *args)
    return data.get(f"{store['kind']}_store_permissions", [])


def _read(cli, store, role, **kwargs):
    if store["kind"] == "session":
        return cli(
            role,
            "sessions",
            "get",
            store["seed"].rsplit("/", 1)[-1],
            "--store",
            store["display_name"],
            **kwargs,
        )
    return cli(role, "memory", "entries", "get", store["seed"], **kwargs)


def _write(cli, store, role, **kwargs):
    if store["kind"] == "session":
        args = [
            "sessions",
            "create",
            "--store",
            store["display_name"],
            "--actor-id",
            "cli-permissions",
        ]
    else:
        args = [
            "memory",
            "entries",
            "create",
            "--store",
            store["name"],
            "--actor-id",
            "cli-permissions",
            "--path",
            f"/cli-permissions/{uuid4().hex}",
            "--content",
            "Synthetic CLI permission validation",
        ]
    data = cli(role, *args, **kwargs)
    if "error" not in data:
        store["children"].append(data["name"])
    return data


@pytest.fixture(scope="module")
def stores(cli, identities):
    _, principals = identities
    created = []
    try:
        for creator in ("admin", "creator"):
            for kind in ("session", "memory"):
                display_name = f"cli-permissions-{uuid4().hex[:12]}"
                root = "sessions" if kind == "session" else "memory"
                data = cli(creator, root, "stores", "create", "--name", display_name)
                store = {
                    "kind": kind,
                    "creator": creator,
                    "name": data["name"],
                    "display_name": display_name,
                    "children": [],
                }
                created.append(store)
                store["seed"] = _write(cli, store, creator)["name"]
        yield created
    finally:
        failures = []
        for store in reversed(created):
            try:
                for role in ("grantee", "creator" if store["creator"] == "admin" else "admin"):
                    _permission(cli, store, "admin", "revoke", *principals[role], "--yes")
                for child in reversed(store["children"]):
                    args = (
                        [
                            "sessions",
                            "delete",
                            child.rsplit("/", 1)[-1],
                            "--store",
                            store["display_name"],
                            "--yes",
                        ]
                        if store["kind"] == "session"
                        else ["memory", "entries", "delete", child, "--yes"]
                    )
                    cli(store["creator"], *args)
                cli(store["creator"], _root(store), "stores", "delete", store["name"], "--yes")
                # Store deletion is asynchronous; require eventual disappearance.
                deadline = time.monotonic() + 30
                while True:
                    try:
                        _permission(
                            cli,
                            store,
                            "admin",
                            "get",
                            error={"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"},
                        )
                        break
                    except AssertionError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(1)
            except (AssertionError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{store['name']}: {exc}")
        assert not failures, "Cleanup failed: " + "; ".join(failures)


@pytest.mark.parametrize(
    "index", range(4), ids=["admin-session", "admin-memory", "creator-session", "creator-memory"]
)
def test_permission_cli_lifecycle(cli, identities, stores, index):
    _, principals = identities
    store = stores[index]
    creator = store["creator"]
    target = principals["grantee"]
    denied = {"PERMISSION_DENIED"}
    invalid = {"INVALID_PARAMETER_VALUE", "BAD_REQUEST"}
    key = f"{store['kind']}_store_permissions"

    assert _level(cli, store, creator) == "WRITE"
    assert _level(cli, store, "grantee") == _NO_ACCESS
    assert _grants(cli, store, creator, "--all") == []
    assert _grants(cli, store, "admin", "--all") == []
    if store["kind"] == "memory":
        assert (
            cli(creator, "memory", "stores", "permissions", "get", store["display_name"])[
                "permission"
            ]
            == "WRITE"
        )
    assert "WRITE" in _permission(cli, store, creator, "get", output="text")
    seed = _read(cli, store, creator)
    assert seed["name"] == store["seed"]
    if store["kind"] == "memory":
        assert seed["content"] == "Synthetic CLI permission validation"

    for level in (_NO_ACCESS, "READ", "WRITE"):
        if level != _NO_ACCESS:
            caller = creator if level == "READ" else "admin"
            _permission(cli, store, caller, "grant", *target, "--permission", level)
            assert _level(cli, store, "grantee") == level
            grants = _grants(cli, store, creator, "--all")
            assert [(g["principal"], g["permission"]) for g in grants] == [
                (
                    {
                        "principal_type": "USER" if target[0] == "--user" else "SERVICE_PRINCIPAL",
                        "name": target[1],
                    },
                    level,
                )
            ]
            assert _grants(cli, store, "admin", "--all") == grants
            assert _read(cli, store, "grantee")["name"] == store["seed"]
        else:
            _read(cli, store, "grantee", error=denied)
        _write(cli, store, "grantee", **({} if level == "WRITE" else {"error": denied}))
        _permission(cli, store, "grantee", "list", error=denied)
        _permission(cli, store, "grantee", "grant", *target, "--permission", "WRITE", error=denied)
        _permission(cli, store, "grantee", "revoke", *target, "--yes", error=denied)

    _permission(cli, store, creator, "grant", *target, "--permission", "WRITE")
    _permission(cli, store, creator, "grant", *target, "--permission", "READ")
    assert _level(cli, store, "grantee") == "WRITE"
    secondary = "creator" if creator == "admin" else "admin"
    _permission(cli, store, creator, "grant", *principals[secondary], "--permission", "READ")
    assert _level(cli, store, secondary) == "READ"
    expected = {target[1]: "WRITE", principals[secondary][1]: "READ"}
    full = _grants(cli, store, creator, "--all", "--page-size", "1")
    assert {g["principal"]["name"]: g["permission"] for g in full} == expected
    if principals[secondary][0] == "--user":
        assert (
            next(g for g in full if g["principal"]["name"] == principals[secondary][1])[
                "principal"
            ]["principal_type"]
            == "USER"
        )
    first = _permission(cli, store, creator, "list", "--page-size", "1")
    assert len(first[key]) == 1 and first["next_page_token"]
    second = _permission(
        cli, store, creator, "list", "--page-size", "1", "--page-token", first["next_page_token"]
    )
    assert not second.get("next_page_token")
    assert {g["principal"]["name"] for g in first[key] + second[key]} == set(expected)
    assert len(first[key] + second[key]) == 2
    other = next(s for s in stores if s["name"] != store["name"])
    _permission(
        cli, other, "admin", "list", "--page-token", first["next_page_token"], error=invalid
    )
    _permission(cli, store, creator, "list", "--page-token", "invalid-token", error=invalid)

    _permission(cli, store, "admin", "revoke", *target, "--yes")
    assert _level(cli, store, "grantee") == _NO_ACCESS
    _read(cli, store, "grantee", error=denied)
    _write(cli, store, "grantee", error=denied)
    assert {g["principal"]["name"] for g in _grants(cli, store, creator, "--all")} == {
        principals[secondary][1]
    }
    _permission(cli, store, creator, "revoke", *target, "--yes")
    assert _level(cli, store, secondary) == "READ"
    _permission(cli, store, creator, "grant", *target, "--permission", "WRITE")
    assert _level(cli, store, "grantee") == "WRITE"
    _write(cli, store, "grantee")
    _permission(cli, store, creator, "revoke", *target, "--yes")
    assert _level(cli, store, "grantee") == _NO_ACCESS
    _permission(cli, store, creator, "revoke", *principals[secondary], "--yes")
    assert _level(cli, store, secondary) == _NO_ACCESS
    assert _grants(cli, store, "admin", "--all") == []
    _permission(cli, store, "admin", "revoke", *principals[creator], "--yes", error=invalid)
    assert _level(cli, store, creator) == "WRITE"


@pytest.mark.parametrize("kind", ["session", "memory"])
def test_permission_cli_store_isolation(cli, identities, stores, kind):
    _, principals = identities
    target = principals["grantee"]
    a, b = [s for s in stores if s["kind"] == kind]
    _permission(cli, a, "admin", "grant", *target, "--permission", "READ")
    _permission(cli, b, "admin", "grant", *target, "--permission", "WRITE")
    _permission(cli, a, "admin", "revoke", *target, "--yes")
    assert _level(cli, a, "grantee") == _NO_ACCESS
    assert _level(cli, b, "grantee") == "WRITE"
    assert _read(cli, b, "grantee")["name"] == b["seed"]
    _write(cli, b, "grantee")
    _permission(cli, b, "admin", "revoke", *target, "--yes")


@pytest.mark.parametrize("kind", ["session", "memory"])
def test_permission_cli_validation_and_missing_store(cli, identities, stores, kind):
    _, principals = identities
    target = principals["grantee"]
    store = next(s for s in stores if s["kind"] == kind)
    for args in (
        ["grant", "--permission", "READ"],
        ["grant", *target],
        ["grant", *target, "--permission", "NONE"],
        ["grant", *target, "--user", "test@example.com", "--permission", "READ"],
        ["list", "--page-size", "-1"],
        ["list", "--all", "--page-token", "invalid-token"],
    ):
        _permission(cli, store, "admin", args[0], *args[1:], usage=True)
    assert _level(cli, store, "grantee") == _NO_ACCESS
    missing = {"kind": kind, "name": f"{kind}-stores/{uuid4()}"}
    for operation, args in (
        ("get", []),
        ("list", []),
        ("grant", [*target, "--permission", "READ"]),
        ("revoke", [*target, "--yes"]),
    ):
        _permission(
            cli, missing, "admin", operation, *args, error={"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}
        )
