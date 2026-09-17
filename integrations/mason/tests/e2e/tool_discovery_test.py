"""Read-only live discovery through an installed Mason CLI (no mocks or deployment)."""

import json
import logging
import os
import pathlib
import subprocess
import sys
from importlib.metadata import distribution

import pytest


@pytest.fixture
def live_mason(tmp_path):
    profile = os.environ.get("MASON_E2E_PROFILE")
    if not profile:
        pytest.skip("set MASON_E2E_PROFILE for read-only live MCP discovery")
    mason = pathlib.Path(sys.executable).with_name("mason")
    assert mason.is_file(), "install the built Mason wheel beside the test interpreter"
    package = distribution("databricks-mason")
    provenance = json.loads(package.read_text("direct_url.json") or "{}")
    assert "archive_info" in provenance, "install a built wheel, not an editable checkout"
    assert (
        pathlib.Path(str(package.locate_file("databricks_mason")))
        .resolve()
        .is_relative_to(pathlib.Path(sys.prefix).resolve())
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)

    def run(*args):
        result = subprocess.run(
            [str(mason), "--profile", profile, "--output", "json", *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert list(tmp_path.iterdir()) == [], "discovery must not create an agent project"
        return json.loads(result.stdout)

    return run


def test_live_default_discovery_includes_mcp_filter_inventory(live_mason):
    discovered = live_mason("tools", "list")
    mcp_only = live_mason("tools", "list", "--kind", "mcp")
    assert discovered["schema_version"] == 2
    assert discovered["complete"] is True
    assert discovered["errors"] == []
    assert discovered["mcp_schema"] == "system.ai"
    assert {tool["kind"] for tool in discovered["available_tools"]} >= {
        "sandbox",
        "uc-function",
        "genie-one",
        "genie-agent",
    }
    actual = {tool["name"] for tool in discovered["available_tools"] if tool["kind"] == "mcp"}
    assert actual == {tool["name"] for tool in mcp_only["available_tools"]}
    logging.getLogger(__name__).info(
        "Live default discovery: %s MCP services plus 4 built-in recipes", len(actual)
    )


@pytest.mark.parametrize("explicit_schema", [False, True])
def test_live_mcp_filter_returns_valid_inventory(live_mason, explicit_schema):
    scope = ["--schema", os.environ.get("MASON_E2E_SCHEMA", "system.ai")] if explicit_schema else []
    discovered = live_mason("tools", "list", "--kind", "mcp", *scope)
    assert discovered["complete"] is True
    assert discovered["errors"] == []
    assert discovered["mcp_schema"] == (scope[-1] if scope else "system.ai")
    assert all(tool["kind"] == "mcp" for tool in discovered["available_tools"])
    assert [tool["name"] for tool in discovered["available_tools"]] == sorted(
        tool["name"] for tool in discovered["available_tools"]
    )
    for tool in discovered["available_tools"]:
        assert tool["add_command"] == (
            "mason tools add sandbox --scope table:catalog.schema.table"
            if tool["name"] == "system.ai.sandbox"
            else f"mason tools add mcp {tool['name']}"
        )
    logging.getLogger(__name__).info(
        "Live MCP discovery: %s services in %s",
        len(discovered["available_tools"]),
        discovered["mcp_schema"],
    )
