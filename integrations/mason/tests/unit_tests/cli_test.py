"""Unit tests for the mason command tree and its help-discovery contract."""

from __future__ import annotations

import click
from click.testing import CliRunner

from databricks_mason import cli
from databricks_mason import help as help_mod


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
    assert "Manage tools configured in an agent project's agent.toml." in result.output
    assert "Add a sandbox, MCP service, UC function, or Python tool." in result.output
    assert "Remove a tool binding from this agent." in result.output
    assert "List tools configured for this agent." in result.output
    assert "mason tools add --help" in result.output
    assert "mason tools add mcp system.ai.web_search" in result.output
    assert "mason tools remove mcp system.ai.web_search" in result.output


def test_tools_remove_help_shows_id_and_project_targeting():
    result = CliRunner().invoke(cli.mason, ["tools", "remove", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: mason tools remove [OPTIONS] TOOL_ID [MCP_SERVICE]" in result.output
    assert "--source DIRECTORY" in result.output
    assert "mason tools remove mcp system.ai.web_search" in result.output
    assert "mason tools remove web_search" in result.output


def test_tools_add_help_explains_types_and_project_targeting():
    result = CliRunner().invoke(cli.mason, ["tools", "add", "--help"])

    assert result.exit_code == 0, result.output
    assert "Subcommands target the current directory by default" in result.output
    assert "Pass --source PATH to target another project." in result.output
    for example in (
        "mason tools add sandbox --scope table:samples.nyctaxi.trips",
        "mason tools add mcp system.ai.web_search",
        "mason tools add uc-function catalog.schema.lookup_ticket",
        "mason tools add python lookup-ticket",
    ):
        assert example in result.output


def test_help_examples_recommend_the_default_happy_path():
    runner = CliRunner()
    expected_examples = {
        (): (
            "mason login --profile <profile>",
            "mason init my-agent",
            "cd my-agent",
            "mason dev",
            "mason deploy my-agent",
        ),
        ("init",): ("mason init my-agent",),
        ("dev",): ("mason dev",),
        ("memory",): (
            "mason memory stores create --display-name agent-memory",
            "mason memory bind agent-memory",
        ),
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


def test_root_examples_render_inline_comments():
    # Short commands carry an aligned inline comment so first-time readers know what each does.
    result = CliRunner().invoke(cli.mason, ["--help"])

    assert result.exit_code == 0, result.output
    assert "mason init my-agent" in result.output
    assert "# scaffold a new agent project" in result.output
    # inline: command and its comment on the same line
    line = next(ln for ln in result.output.splitlines() if "mason init my-agent" in ln)
    assert "# scaffold a new agent project" in line


def test_inline_comments_are_column_aligned():
    # Every inline comment in a group starts at the same column (the `#` lines up).
    result = CliRunner().invoke(cli.mason, ["mcp", "--help"])

    assert result.exit_code == 0, result.output
    hash_columns = {ln.index("#") for ln in result.output.splitlines() if "  # " in ln}
    assert len(hash_columns) == 1, result.output


def test_long_commands_stack_the_comment_above():
    # A command too long to inline puts its comment on the preceding line so nothing wraps.
    epilog = help_mod._example_epilog(
        (("mason x " + "y" * help_mod._INLINE_COMMENT_MAX, "does a long thing"),)
    )
    lines = epilog.splitlines()
    comment_i = next(i for i, ln in enumerate(lines) if "# does a long thing" in ln)
    # comment sits on its own line, immediately above the command
    assert lines[comment_i].strip() == "# does a long thing"
    assert lines[comment_i + 1].strip().startswith("mason x")


def test_group_comment_layout_is_uniform():
    # If any command in a group must stack, the whole group stacks (no mixed inline/stacked).
    epilog = help_mod._example_epilog(
        (
            ("mason short", "inline-able"),
            ("mason " + "z" * help_mod._INLINE_COMMENT_MAX, "forces stacking"),
        )
    )
    # no command line carries a trailing inline comment
    assert not any(ln.strip().startswith("mason") and " # " in ln for ln in epilog.splitlines())
