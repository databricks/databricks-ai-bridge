"""Functional smoke tests: the REAL installed `mason` binary runs each workspace-free command.

These complement the in-process CliRunner unit tests, which already own detailed behavior (arg
handling, help text, agent.toml effects). CliRunner imports the command modules in-process, so it
can't catch a regression that only shows up under the real console script — a broken entry point, a
packaging/dependency gap, an import that fails only when installed. This runs the `mason` on PATH,
next to the interpreter running the tests: CI installs the built wheel and runs pytest from that
venv (so CI exercises the shipped artifact), while a local `uv run` provides the editable install.
Everything runs with an isolated HOME and no Databricks config, so no workspace is used.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import tomli

# Every top-level command; `<group> --help` proves each module imports and wires up as installed.
_COMMANDS = (
    "login",
    "logout",
    "init",
    "dev",
    "memory",
    "sessions",
    "tracing",
    "deploy",
    "deployments",
    "endpoint",
    "tools",
)


@pytest.fixture
def run_mason(tmp_path: pathlib.Path):
    mason = pathlib.Path(sys.executable).with_name("mason")
    if not mason.is_file():
        pytest.skip("requires the mason CLI on PATH")
    home = tmp_path / "home"
    home.mkdir()
    empty_cfg = tmp_path / "empty.databrickscfg"
    empty_cfg.write_text("")
    # Isolate HOME so login/logout can't touch the real ~/.mason; empty config file so no real
    # profile is reachable. env= replaces the environment wholesale (no ambient DATABRICKS_* leaks).
    env = {
        "PATH": f"{mason.parent}:/usr/bin:/bin",
        "HOME": str(home),
        "DATABRICKS_CONFIG_FILE": str(empty_cfg),
    }

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [str(mason), *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        if check:
            assert result.returncode == 0, (
                f"`mason {' '.join(args)}` exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result

    return run


def test_cli_and_every_command_help_load(run_mason) -> None:
    assert "Usage" in run_mason("--help").stdout
    for command in _COMMANDS:
        run_mason(command, "--help")


@pytest.mark.parametrize(
    "extra",
    [
        ["--framework", "langgraph"],
        ["--framework", "openai"],
        ["--framework", "langgraph", "--server", "custom"],
    ],
)
def test_init_scaffolds(run_mason, tmp_path: pathlib.Path, extra) -> None:
    dest = tmp_path / ("proj_" + "_".join(token.lstrip("-") for token in extra))
    run_mason("init", *extra, str(dest))
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "app.yaml").is_file()
    assert (dest / "agent.toml").is_file()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("output", ["text", "json"])
@pytest.mark.parametrize(
    "args",
    [
        ["sandbox", "--scope", "table:catalog.schema.table"],
        ["uc-function", "catalog.schema.function"],
        ["genie-one"],
        ["genie-agent", "0" * 32],
    ],
)
def test_tools_offline_add_review_manifest_remove(
    run_mason, tmp_path, framework, output, args
) -> None:
    project = tmp_path / "agent"
    run_mason("init", "--framework", framework, str(project))
    manifest = project / "agent.toml"
    original = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file() and path != manifest
    }
    for changed in (True, False):
        added = run_mason(
            "--output", output, "tools", "add", *args, "--name", "tested", "--source", str(project)
        )
        if output == "json":
            payload = json.loads(added.stdout)
            assert payload["manifest"] == str(manifest)
            assert payload["changed"] is changed
            assert payload["changed_files"] == ([str(manifest)] if changed else [])
        else:
            assert f"Review {manifest}" in added.stdout
            assert "configured managed tools and MCP bindings" in added.stdout
            if not changed:
                assert "already configured" in added.stdout
        assert [tool["id"] for tool in tomli.loads(manifest.read_text())["tools"]] == ["tested"]
    run_mason("tools", "remove", "tested", "--source", str(project))
    assert tomli.loads(manifest.read_text()).get("tools", []) == []
    assert {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file() and path != manifest
    } == original


@pytest.mark.parametrize("kind", ["sandbox", "uc-function", "genie-one", "genie-agent"])
def test_tools_local_discovery_without_project_or_auth(run_mason, kind):
    result = run_mason("-o", "json", "tools", "list", "--kind", kind)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["complete"] is True
    assert payload["mcp_schema"] is None
    assert payload["errors"] == []
    assert [tool["kind"] for tool in payload["available_tools"]] == [kind]


def test_tools_text_discovery_suggests_mcp_filters(run_mason):
    result = run_mason("tools", "list", "--kind", "genie-one")
    assert "mason tools list --kind mcp" in result.stdout
    assert "mason tools list --kind mcp --schema catalog.schema" in result.stdout


def test_tools_discovery_without_auth_is_explicitly_incomplete(run_mason):
    result = run_mason("-o", "json", "tools", "list", check=False)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert payload["mcp_schema"] == "system.ai"
    assert [tool["kind"] for tool in payload["available_tools"]] == [
        "sandbox",
        "uc-function",
        "genie-one",
        "genie-agent",
    ]
    assert payload["errors"]


def test_tools_help_and_removed_configured_route(run_mason):
    for path in [
        ("tools",),
        ("tools", "list"),
        ("tools", "add"),
        ("tools", "add", "sandbox"),
        ("tools", "add", "mcp"),
        ("tools", "add", "uc-function"),
        ("tools", "add", "genie-one"),
        ("tools", "add", "genie-agent"),
    ]:
        text = " ".join(run_mason(*path, "--help").stdout.split())
        assert "agent.toml" in text
        assert "list configured tools" not in text
        assert "mason mcp list" not in text
    listing_help = " ".join(run_mason("tools", "list", "--help").stdout.split())
    for expected in (
        "--kind",
        "--schema",
        "system.ai",
        "catalog.schema",
        "genie-one",
        "genie-agent",
        "No agent project",
    ):
        assert expected in listing_help
    assert "--source DIRECTORY" not in listing_help
    assert run_mason("tools", "list", "--source", ".", check=False).returncode != 0
    assert run_mason("tools", "mcp", "list", check=False).returncode != 0
    assert run_mason("mcp", "--help", check=False).returncode != 0


def test_mcp_add_without_auth_does_not_change_manifest(run_mason, tmp_path: pathlib.Path) -> None:
    project = tmp_path / "agent"
    run_mason("init", "--framework", "langgraph", str(project))
    manifest = project / "agent.toml"
    before = manifest.read_bytes()

    result = run_mason(
        "--output",
        "json",
        "tools",
        "add",
        "mcp",
        "system.ai.web_search",
        "--source",
        str(project),
        check=False,
    )

    assert result.returncode == 1
    assert "Could not initialize Databricks auth" in result.stderr
    assert manifest.read_bytes() == before


def test_tracing_unbind_and_rebind(run_mason, tmp_path: pathlib.Path) -> None:
    project = tmp_path / "agent"
    run_mason("init", "--framework", "langgraph", str(project))
    run_mason("tracing", "unbind", "--source", str(project))
    # No --experiment-name/--experiment-id => re-enable the per-project default; needs no workspace.
    run_mason("tracing", "bind", "--source", str(project))


def test_logout_runs_cleanly(run_mason) -> None:
    # Isolated HOME: there's no saved selection to forget, but it must still exit cleanly.
    run_mason("logout")
