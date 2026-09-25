"""Unit tests for the ab command tree and its help-discovery contract."""

from __future__ import annotations

import click
from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli
from databricks_agentbricks.cli import help as help_mod


def _command_paths(group: click.Group, prefix: tuple[str, ...] = ()):
    for name, command in group.commands.items():
        path = (*prefix, name)
        yield path
        if isinstance(command, click.Group):
            yield from _command_paths(command, path)


def test_sessions_verbs_are_flat_no_redundant_subgroup():
    names = set(cli.sessions.commands)
    # Session verbs are direct subcommands of `sessions` (no `ab sessions sessions`).
    assert {"create", "list", "get", "update", "delete", "fork"} <= names
    assert "sessions" not in names
    # Sub-resources remain their own groups.
    assert {"stores", "items"} <= names


def test_root_registers_supported_commands():
    names = set(cli.agentbricks.commands)
    assert {
        "login",
        "logout",
        "init",
        "dev",
        "memory",
        "dreamer",
        "sessions",
        "tracing",
        "deploy",
        "deployments",
        "endpoint",
        "tools",
    } <= names
    assert "mcp" not in names
    assert "durability" not in names
    assert "help" not in names
    assert "add-sandbox" not in names


def test_root_command_name_and_version_default_to_ab():
    runner = CliRunner()
    help_result = runner.invoke(cli.agentbricks, ["--help"])
    version_result = runner.invoke(cli.agentbricks, ["--version"])

    assert help_result.exit_code == 0, help_result.output
    assert help_result.output.startswith("Usage: ab ")
    assert version_result.exit_code == 0, version_result.output
    assert version_result.output.startswith("ab, version ")


def test_root_help_describes_the_product_and_links_out():
    # The root page should say what Agent Bricks is in plain language (not lead with internal API detail)
    # and point a reader to docs + support, per CLI help best practices.
    result = CliRunner().invoke(cli.agentbricks, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Agent Bricks is a CLI for building and deploying" in result.output
    assert "AgentBricks is a CLI" not in result.output
    assert "building and deploying custom AI agents" in result.output
    assert "Databricks" in result.output
    # No internal API path in the user-facing description.
    assert "2.0/agents" not in result.output
    # Docs and issues links appear (root only).
    assert help_mod._DOCS_URL in result.output
    assert help_mod._ISSUES_URL in result.output


def test_group_help_has_no_raw_api_paths():
    # Group descriptions should read for users, not expose REST endpoints.
    runner = CliRunner()
    for path in ((), *_command_paths(cli.agentbricks)):
        result = runner.invoke(cli.agentbricks, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        # The Docs/Issues footer legitimately carries the repo URL on the root page; the offending
        # pattern we guard against is the raw API path that used to lead group descriptions.
        assert "/api/2.0/agents" not in result.output, path


def test_nested_command_help_shows_usage_options_and_examples():
    result = CliRunner().invoke(cli.agentbricks, ["tools", "add", "sandbox", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: ab tools add sandbox [OPTIONS]" in result.output
    assert "--scope TEXT" in result.output
    assert "EXAMPLES" in result.output
    assert "ab tools add sandbox --scope table:samples.nyctaxi.trips" in result.output


def test_tools_help_explains_add_workflow():
    result = CliRunner().invoke(cli.agentbricks, ["tools", "--help"])

    assert result.exit_code == 0, result.output
    assert "Tools are what let an agent act" in result.output
    # The CLI-addable tool types are described on the group page (Python tools are code-first,
    # written directly in the project — see #509 upstream — so they are not a `tools add` type).
    for tool_type in ("sandbox", "mcp", "uc-function"):
        assert tool_type in result.output
    assert "ab tools add --help" in result.output
    assert "ab tools add mcp system.ai.web_search" in result.output
    assert "ab tools remove mcp system.ai.web_search" in result.output
    assert "system.ai.python_exec" not in result.output


def test_tools_remove_help_shows_id_and_project_targeting():
    result = CliRunner().invoke(cli.agentbricks, ["tools", "remove", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: ab tools remove [OPTIONS] TOOL_ID [MCP_SERVICE]" in result.output
    assert "--source DIRECTORY" in result.output
    assert "ab tools remove mcp system.ai.web_search" in result.output
    assert "ab tools remove web_search" in result.output
    assert "system.ai.python_exec" not in result.output


def test_tools_add_help_explains_types_and_project_targeting():
    result = CliRunner().invoke(cli.agentbricks, ["tools", "add", "--help"])

    assert result.exit_code == 0, result.output
    assert "Subcommands target the current directory by default" in result.output
    assert "Pass --source PATH to target another project." in result.output
    for example in (
        "ab tools add sandbox --scope table:samples.nyctaxi.trips",
        "ab tools add mcp system.ai.web_search",
        "ab tools add uc-function catalog.schema.lookup_ticket",
    ):
        assert example in result.output
    assert "system.ai.python_exec" not in result.output
    # `ab tools add python` was removed (Python tools are code-first); the subcommand must not be
    # advertised. Checked as the command invocation, not a bare "python" substring.
    assert "ab tools add python" not in result.output
    assert "\n  python " not in result.output  # no `python` row in the add-group command list


def test_help_examples_recommend_the_default_happy_path():
    runner = CliRunner()
    expected_examples = {
        (): (
            "ab login --profile <profile>",
            "ab init my-agent",
            "cd my-agent",
            "ab dev",
            "ab deploy my-agent",
        ),
        ("init",): ("ab init my-agent",),
        ("dev",): ("ab dev",),
        ("memory",): (
            "ab memory stores create --display-name agent-memory",
            "ab memory bind agent-memory",
        ),
        ("deploy",): ("ab deploy my-agent",),
    }

    for path, examples in expected_examples.items():
        result = runner.invoke(cli.agentbricks, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        for example in examples:
            assert example in result.output, (path, example)


def test_every_command_has_an_example_in_option_help():
    runner = CliRunner()

    for path in ((), *_command_paths(cli.agentbricks)):
        result = runner.invoke(cli.agentbricks, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        # The root leads with a numbered "Getting started" path; subcommands show "Examples".
        # Headings render uppercased and grayed (facelift §6); CliRunner strips the color codes.
        expected = "GETTING STARTED" if path == () else "EXAMPLES"
        assert expected in result.output, path


def test_root_examples_render_inline_comments():
    # Short commands carry an aligned inline comment so first-time readers know what each does.
    result = CliRunner().invoke(cli.agentbricks, ["--help"])

    assert result.exit_code == 0, result.output
    assert "ab init my-agent" in result.output
    assert "# scaffold a new agent project" in result.output
    # inline: command and its comment on the same line
    line = next(ln for ln in result.output.splitlines() if "ab init my-agent" in ln)
    assert "# scaffold a new agent project" in line


def test_inline_comments_are_column_aligned():
    # Every inline comment in a group starts at the same column (the `#` lines up).
    result = CliRunner().invoke(cli.agentbricks, ["tools", "--help"])

    assert result.exit_code == 0, result.output
    hash_columns = {ln.index("#") for ln in result.output.splitlines() if "  # " in ln}
    assert len(hash_columns) == 1, result.output


def test_long_commands_stack_the_comment_above():
    # A command too long to inline puts its comment on the preceding line so nothing wraps.
    epilog = help_mod._example_epilog(
        (("ab x " + "y" * help_mod._INLINE_COMMENT_MAX, "does a long thing"),)
    )
    lines = [click.unstyle(ln) for ln in epilog.splitlines()]  # drop the graying color codes
    comment_i = next(i for i, ln in enumerate(lines) if "# does a long thing" in ln)
    # comment sits on its own line, immediately above the command
    assert lines[comment_i].strip() == "# does a long thing"
    assert lines[comment_i + 1].strip().startswith("ab x")


def test_group_comment_layout_is_uniform():
    # If any command in a group must stack, the whole group stacks (no mixed inline/stacked).
    epilog = help_mod._example_epilog(
        (
            ("ab short", "inline-able"),
            ("ab " + "z" * help_mod._INLINE_COMMENT_MAX, "forces stacking"),
        )
    )
    # no command line carries a trailing inline comment
    assert not any(ln.strip().startswith("ab") and " # " in ln for ln in epilog.splitlines())


def test_memory_search_uses_canonical_page_size_option():
    entries = cli.memory.commands["entries"]
    assert isinstance(entries, click.Group)
    search = entries.commands["search"]
    parameter_names = {parameter.name for parameter in search.params}

    assert "page_size" in parameter_names
    assert "limit" not in parameter_names


def test_session_delete_has_force_option():
    delete = cli.sessions.commands["delete"]

    assert "force" in {parameter.name for parameter in delete.params}


# --- did-you-mean & grouped listing (facelift §1, §4) ------------------------


def test_unknown_command_suggests_close_match():
    result = CliRunner().invoke(cli.agentbricks, ["tolls", "list"])
    assert result.exit_code != 0
    # Diagnostic grammar: an `error:` keyword and a `help:` line with a ranked suggestion.
    assert "error: unknown command `tolls`" in result.output
    assert "did you mean `tools`" in result.output


def test_doubled_invocation_is_named_directly():
    result = CliRunner().invoke(cli.agentbricks, ["ab", "login"])
    assert result.exit_code != 0
    assert "you typed `ab` twice" in result.output


def test_unknown_nested_command_suggests_within_group():
    result = CliRunner().invoke(cli.agentbricks, ["memory", "storx"])
    assert result.exit_code != 0
    assert "unknown command `storx`" in result.output
    assert "did you mean `stores`" in result.output


def test_bad_option_uses_diagnostic_grammar():
    # Click's own usage errors (bad/unknown option) render in the agentbricks diagnostic grammar, not
    # Click's stock `Error: …` / `Usage:` block.
    result = CliRunner().invoke(cli.agentbricks, ["init", "--nope", "x"])
    assert result.exit_code != 0
    # Click's message wording varies across versions ("No such option: --nope" vs "'--nope'"), so
    # assert on the parts we own: the `error:` keyword, the offending option, and the `help:` line.
    assert "error: No such option" in result.output
    assert "--nope" in result.output
    assert "help: run `ab init --help`" in result.output
    # Not Click's default framing.
    assert "Try 'ab" not in result.output


def test_missing_argument_uses_diagnostic_grammar():
    result = CliRunner().invoke(cli.agentbricks, ["memory", "stores", "get"])
    assert result.exit_code != 0
    assert "error: Missing argument" in result.output
    assert "help: run `ab memory stores get --help`" in result.output


def test_unknown_command_without_close_match_points_to_help():
    result = CliRunner().invoke(cli.agentbricks, ["zzzzz"])
    assert result.exit_code != 0
    assert "unknown command `zzzzz`" in result.output
    assert "ab --help" in result.output


def test_root_help_shows_numbered_getting_started_path():
    result = CliRunner().invoke(cli.agentbricks, ["--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "GETTING STARTED" in out
    # Read just the numbered block (login/init/deploy also appear in the prose description above).
    block = out[out.index("GETTING STARTED") :]
    block = block[: block.index("Not authenticated")]
    numbered = [ln.strip() for ln in block.splitlines() if ln.strip()[:1].isdigit()]
    # The path is numbered and ordered: login (1) → init (2) → cd (3) → dev (4) → deploy (5).
    assert numbered[0].startswith("1") and "ab login --profile" in numbered[0]
    assert numbered[1].startswith("2") and "ab init my-agent" in numbered[1]
    assert numbered[4].startswith("5") and "ab deploy my-agent" in numbered[4]


def test_help_dims_headings_and_descriptions_not_names():
    # Headings and the description column recede via the terminal's adaptive dim attribute (SGR 2),
    # while command/option names keep full intensity. Assert on the raw ANSI (color forced on).
    import click

    ctx = cli.agentbricks.make_context("ab", [], resilient_parsing=True)
    ctx.color = True
    raw = cli.agentbricks.get_help(ctx)
    dim = "\x1b[2m"  # SGR 2 = faint/dim, adaptive to the terminal's own foreground
    lines = raw.splitlines()

    # A section heading (uppercase, no colon) is dimmed.
    setup = next(ln for ln in lines if click.unstyle(ln).strip() == "SETUP")
    assert setup.startswith(dim)
    # A command row dims the description but not the name.
    login = next(ln for ln in lines if click.unstyle(ln).strip().startswith("login"))
    assert dim in login  # the description is dimmed
    assert not login.startswith(dim)  # the `login` name is at full intensity


def test_epilog_headings_align_flush_left_with_sections():
    # GETTING STARTED / EXAMPLES are epilog blocks; Click indents epilogs one level by default,
    # which left them 2 columns right of OPTIONS/SETUP. They should sit flush-left like real
    # sections, with their rows at the same 2-space column as command rows.
    result = CliRunner().invoke(cli.agentbricks, ["--help"])
    assert result.exit_code == 0, result.output

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    lines = result.output.splitlines()
    heading = {"OPTIONS", "SETUP", "GETTING STARTED"}
    for name in heading:
        line = next(ln for ln in lines if ln.strip() == name)
        assert indent(line) == 0, (name, line)
    # A numbered getting-started row aligns with an OPTIONS/command row at column 2.
    row = next(ln for ln in lines if ln.strip().startswith("1  ab login"))
    assert indent(row) == 2, row


def test_root_help_dims_capability_descriptions_but_not_labels_or_prose():
    import click

    ctx = cli.agentbricks.make_context("ab", [], resilient_parsing=True)
    ctx.color = True
    raw = cli.agentbricks.get_help(ctx)
    dim = "\x1b[2m"
    lines = raw.splitlines()

    # In a capabilities row the label (left column) stays full-weight; only the description recedes.
    models = next(ln for ln in lines if click.unstyle(ln).strip().startswith("Models"))
    assert not models.startswith(dim)  # the "Models" label is not dimmed
    assert models.index("Models") < models.index(dim)  # dim begins at the description, after label
    # The prose introducing and following the block stays at full weight.
    intro = next(ln for ln in lines if "combine the platform's capabilities" in click.unstyle(ln))
    assert dim not in intro
    closing = next(ln for ln in lines if "provisions and wires these" in click.unstyle(ln))
    assert dim not in closing


def test_root_help_groups_commands_by_intent():
    result = CliRunner().invoke(cli.agentbricks, ["--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    # The command list reads as a workflow: SETUP → DEVELOP → SHIP, in that order.
    assert "SETUP" in out and "DEVELOP" in out and "SHIP" in out
    assert out.index("SETUP") < out.index("DEVELOP") < out.index("SHIP")
    # Commands land under their section.
    setup_to_develop = out[out.index("SETUP") : out.index("DEVELOP")]
    assert "login" in setup_to_develop and "init" in setup_to_develop
    ship_onward = out[out.index("SHIP") :]
    assert "deploy" in ship_onward
