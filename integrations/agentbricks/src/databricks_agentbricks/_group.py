"""A `click.Group` subclass carrying the facelift's `--help` upgrades.

1. **"Did you mean?" on unknown commands.** Click's default dead-ends with
   ``Error: No such command 'foo'.``. `AgentBricksGroup` turns that into the CLI's diagnostic grammar —
   ``error: unknown command `foo``` — plus up to three ranked suggestions and a ``--help`` pointer,
   so a typo resolves itself instead of stranding the user.

2. **Grouped command listing.** On the root group, commands are listed by intent
   (SETUP / DEVELOP / SHIP) instead of one flat alphabetical dump — the fix for the "blank-page
   problem". Subgroups keep Click's default single-section listing.

3. **Grayed `--help`.** `AgentBricksHelpFormatter` renders section headings and description text in the
   muted secondary color (and headings uppercased, colon-less), reserving the terminal's default
   foreground for the command/option names — so the page reads as scannable structure, not a wall
   of equal-weight text. Installed on every command via `AgentBricksContext`.
"""

from __future__ import annotations

import difflib
import inspect
from typing import Iterable, Optional

import click

from databricks_agentbricks.errors import AgentCliError

# Ordered intent sections for the root command list. Each command name must be a real subcommand;
# any not registered is silently skipped, so this stays robust if the command set changes.
CommandSections = tuple[tuple[str, tuple[str, ...]], ...]


def dim(text: str) -> str:
    """Recede `text` for `--help` output (the facelift graying of headings and descriptions).

    Uses the terminal's adaptive "dim" attribute (ANSI SGR 2) rather than a fixed gray hex. A fixed
    shade tuned for the design mock's dark background washes out on a light terminal (and on
    terminals without 24-bit color), which made the help hard to read; SGR 2 dims relative to the
    reader's own foreground, so secondary text stays legible on both light and dark backgrounds
    while the command/option names — left at full intensity — still stand out.

    ``click.style`` emits the code on a TTY and ``click.echo`` strips it when output is piped or
    ``NO_COLOR`` is set. Shared with `help.py` so the epilog (getting-started / examples) recedes to
    match the formatter-rendered sections.
    """
    return click.style(text, dim=True)


class _FlushEpilog:
    """Render the epilog flush-left instead of one indent level in.

    Click's default `format_epilog` wraps the epilog in `with formatter.indentation():`, so an
    epilog section heading (our `GETTING STARTED` / `EXAMPLES`) lands two columns to the right of
    the real sections (`OPTIONS` / `SETUP` / …). We treat those blocks as first-class grayed
    sections, so they should sit flush-left like the others, with their content at the same 2-space
    column as command rows. Dropping the extra `indentation()` does exactly that.
    """

    epilog: Optional[str]

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(inspect.cleandoc(self.epilog))


class AgentBricksGroup(_FlushEpilog, click.Group):
    """Group with typo suggestions and (optionally) an intent-grouped command listing."""

    # Set only on the root group. When present, `format_commands` renders these sections instead of
    # Click's flat "Commands:" list.
    command_sections: Optional[CommandSections] = None

    def resolve_command(self, ctx: click.Context, args: list[str]):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # Click raises `UsageError("No such command '<name>'.")`; re-render it in the CLI's
            # diagnostic grammar with suggestions. The offending token is the first arg.
            name = args[0] if args else ""
            raise self._unknown_command_error(ctx, name) from None

    def _unknown_command_error(self, ctx: click.Context, name: str) -> AgentCliError:
        invocation = ctx.command_path or "agentbricks"
        # Repeating the executable as a command is a common paste typo — call it out directly.
        program = invocation.split(maxsplit=1)[0]
        if ctx.parent is None and name == program:
            return AgentCliError(
                f"unknown command `{name}`",
                hint=f"you typed `{name}` twice — drop the extra `{name}` (run `{invocation} --help`)",
            )
        suggestions = difflib.get_close_matches(name, self.list_commands(ctx), n=3, cutoff=0.5)
        if suggestions:
            quoted = ", ".join(f"`{s}`" for s in suggestions)
            hint = f"did you mean {quoted}? (run `{invocation} --help` to see all commands)"
        else:
            hint = f"run `{invocation} --help` to see available commands"
        return AgentCliError(f"unknown command `{name}`", hint=hint)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not self.command_sections:
            super().format_commands(ctx, formatter)
            return
        listed: set[str] = set()
        for label, names in self.command_sections:
            rows: list[tuple[str, str]] = []
            for name in names:
                command = self.get_command(ctx, name)
                if command is None or command.hidden:
                    continue
                listed.add(name)
                rows.append((name, command.get_short_help_str(limit=60)))
            if rows:
                with formatter.section(label):
                    formatter.write_dl(rows)
        # Any command not placed in a section still needs to appear, so nothing goes missing if a
        # new command is added without updating the section map.
        leftovers: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name in listed:
                continue
            command = self.get_command(ctx, name)
            if command is None or command.hidden:
                continue
            leftovers.append((name, command.get_short_help_str(limit=60)))
        if leftovers:
            with formatter.section("Other commands"):
                formatter.write_dl(leftovers)


class AgentBricksHelpFormatter(click.HelpFormatter):
    """A help formatter that grays the page's structure.

    Section headings render uppercase, colon-less, and in the muted secondary color; description
    text (the second column of every option/command list) renders in the same secondary color. The
    command and option *names* keep the terminal's default foreground, so the eye lands on what's
    runnable. Command docstrings (the intro prose) are left untouched so the most important text
    stays at full contrast.
    """

    def write_heading(self, heading: str) -> None:
        self.write(f"{'':>{self.current_indent}}{dim(heading.upper())}\n")

    def write_dl(
        self,
        rows: Iterable[tuple[str, str]],
        col_max: int = 30,
        col_spacing: int = 2,
    ) -> None:
        # Gray only the description column. Click ≥8.4 measures column widths and wraps with
        # `term_len` (ANSI-aware), so styling the description doesn't disturb alignment or wrapping.
        grayed = [(term, dim(desc) if desc else desc) for term, desc in rows]
        super().write_dl(grayed, col_max=col_max, col_spacing=col_spacing)


class AgentBricksCommand(_FlushEpilog, click.Command):
    """A leaf command whose `EXAMPLES` epilog renders flush-left, like the section headings."""


class AgentBricksContext(click.Context):
    """Context that renders help through `AgentBricksHelpFormatter`."""

    formatter_class = AgentBricksHelpFormatter


def _usage_error_show(self: click.exceptions.UsageError, file=None) -> None:
    """Render a Click usage error (bad/unknown option, missing or extra argument) in the agentbricks
    diagnostic grammar, matching the errors we raise ourselves.

    Click's own `UsageError.show` prints `Error: <msg>` plus a `Usage:` / `Try … --help` block.
    `format_message()` already includes Click's own "Did you mean …?" for a mistyped option, so we
    keep the message verbatim and add the agentbricks `error:` keyword and an indented `help:` pointer.
    """
    # Local imports avoid a module-load cycle (render/errors import from click, not from here).
    from rich.console import Console

    from databricks_agentbricks import errors, render

    invocation = self.ctx.command_path if self.ctx is not None else "agentbricks"
    message = self.format_message()
    hint = f"run `{invocation} --help` to see the available options and arguments"
    if errors._OUTPUT_MODE == "json":
        import json

        click.echo(json.dumps({"error": {"message": message, "hint": hint}}, indent=2), err=True)
        return
    render.diagnostic("error", message, help=hint, con=Console(stderr=True))


def apply_group_class(root: click.Group, sections: CommandSections) -> None:
    """Upgrade the whole command tree in place: `AgentBricksGroup` on every group (typo suggestions +
    the intent-grouped root listing) and `AgentBricksContext` on every command (grayed `--help`), and
    route Click's own usage errors through the agentbricks diagnostic grammar.

    Reassigning ``__class__`` upgrades the already-built Click groups (their instance layout is
    compatible), so this works without rewiring how commands are declared across the codebase.
    """
    _upgrade(root)
    assert isinstance(root, AgentBricksGroup)  # _upgrade just set root.__class__ = AgentBricksGroup
    root.command_sections = sections
    # Override once: every UsageError subclass (NoSuchOption, BadParameter, MissingParameter, …)
    # inherits `show`, so parse-time errors render like the ones we raise. Keeps Click's `main()`
    # intact (exit codes, --help, broken-pipe/abort handling), unlike wrapping the entry point.
    click.exceptions.UsageError.show = _usage_error_show  # type: ignore[method-assign]


def _upgrade(command: click.Command) -> None:
    # Every command — groups and leaves — renders help through AgentBricksContext, and every epilog
    # renders flush-left (AgentBricksGroup / AgentBricksCommand both mix in _FlushEpilog).
    command.context_class = AgentBricksContext
    if isinstance(command, click.Group):
        if not isinstance(command, AgentBricksGroup):
            command.__class__ = AgentBricksGroup
        for child in command.commands.values():
            _upgrade(child)
    elif not isinstance(command, AgentBricksCommand):
        command.__class__ = AgentBricksCommand
