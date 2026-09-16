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

import pathlib
import subprocess
import sys

import pytest

# Every top-level command; `<group> --help` proves each module imports and wires up as installed.
_COMMANDS = (
    "login",
    "logout",
    "init",
    "dev",
    "memory",
    "sessions",
    "mcp",
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
            [str(mason), *args], env=env, capture_output=True, text=True, timeout=60
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


def test_tools_add_list_remove(run_mason, tmp_path: pathlib.Path) -> None:
    project = tmp_path / "agent"
    run_mason("init", "--framework", "langgraph", str(project))
    run_mason(
        "tools",
        "add",
        "mcp",
        "system.ai.web_search",
        "--name",
        "websearch",
        "--source",
        str(project),
    )
    listing = run_mason("tools", "list", "--source", str(project)).stdout
    assert "websearch" in listing or "web_search" in listing
    run_mason("tools", "remove", "websearch", "--source", str(project))


def test_tracing_disable_and_reenable(run_mason, tmp_path: pathlib.Path) -> None:
    project = tmp_path / "agent"
    run_mason("init", "--framework", "langgraph", str(project))
    run_mason("tracing", "disable", "--source", str(project))
    # No --experiment => re-enable the per-project default; needs no workspace.
    run_mason("tracing", "configure", "--source", str(project))


def test_logout_runs_cleanly(run_mason) -> None:
    # Isolated HOME: there's no saved selection to forget, but it must still exit cleanly.
    run_mason("logout")
