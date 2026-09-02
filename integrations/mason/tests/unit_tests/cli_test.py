"""Unit tests for the mason command tree and its help-discovery contract."""

from __future__ import annotations

import click
from click.testing import CliRunner

from databricks_mason import cli


def _command_paths(group: click.Group, prefix: tuple[str, ...] = ()):
    for name, command in group.commands.items():
        path = (*prefix, name)
        yield path
        if isinstance(command, click.Group):
            yield from _command_paths(command, path)


def test_sessions_verbs_are_flat_no_redundant_subgroup():
    names = set(cli.sessions.commands)
    # Session verbs are direct subcommands of `sessions` (no `mason sessions sessions`).
    assert {"create", "list", "get", "update", "delete", "fork"} <= names
    assert "sessions" not in names
    # Sub-resources remain their own groups.
    assert {"stores", "items"} <= names


def test_root_registers_supported_commands():
    names = set(cli.mason.commands)
    assert {
        "login",
        "logout",
        "memory",
        "sessions",
        "tracing",
        "deploy",
        "deployments",
        "mcp",
        "tools",
    } <= names
    assert "help" not in names
    assert "add-sandbox" not in names


def test_tools_add_only_manages_databricks_integrations():
    add = cli.tools.commands["add"]

    assert isinstance(add, click.Group)
    assert set(add.commands) == {"sandbox", "mcp", "uc-function"}


def test_nested_command_help_shows_usage_options_and_examples():
    result = CliRunner().invoke(cli.mason, ["tools", "add", "sandbox", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: mason tools add sandbox [OPTIONS]" in result.output
    assert "--scope TEXT" in result.output
    assert "Examples:" in result.output
    assert "mason tools add sandbox --scope table:samples.nyctaxi.trips" in result.output


def test_tools_help_explains_add_workflow():
    result = CliRunner().invoke(cli.mason, ["tools", "--help"])

    assert result.exit_code == 0, result.output
    assert "Manage Databricks integrations selected in agent code." in result.output
    assert "Add a sandbox, MCP service, or UC function." in result.output
    assert "List Databricks integrations configured for this agent." in result.output
    assert "mason tools add --help" in result.output
    assert "mason tools add mcp system.ai.web_search" in result.output


def test_tools_add_help_explains_types_and_project_targeting():
    result = CliRunner().invoke(cli.mason, ["tools", "add", "--help"])

    assert result.exit_code == 0, result.output
    assert "Subcommands target the current directory by default" in result.output
    assert "Pass --source PATH to target another project." in result.output
    for example in (
        "mason tools add sandbox --scope table:samples.nyctaxi.trips",
        "mason tools add mcp system.ai.web_search",
        "mason tools add uc-function catalog.schema.lookup_ticket",
    ):
        assert example in result.output


def test_help_examples_recommend_the_default_happy_path():
    runner = CliRunner()
    expected_examples = {
        (): (
            "mason login --profile my-workspace",
            "mason init my-agent",
            "cd my-agent",
            "mason dev",
            "mason deploy my-agent",
        ),
        ("init",): ("mason init my-agent",),
        ("dev",): ("mason dev",),
        ("deploy",): ("mason deploy my-agent",),
    }

    for path, examples in expected_examples.items():
        result = runner.invoke(cli.mason, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        for example in examples:
            assert example in result.output, (path, example)


def test_every_command_has_an_example_in_option_help():
    runner = CliRunner()

    for path in ((), *_command_paths(cli.mason)):
        result = runner.invoke(cli.mason, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        assert "Examples:" in result.output, path
