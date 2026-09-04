"""Terminal presentation layer for the Mason CLI.

Rendering is centralized so every `list`/`get`/`create` command looks consistent.
Functions accept an optional `console` for testability; default is stdout.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence

import click
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Databricks brand accent.
ACCENT = "#FF3621"
MUTED = "grey62"

_stdout = Console()


def console() -> Console:
    return _stdout


@contextmanager
def status(message: str, con: Optional[Console] = None) -> Iterator[None]:
    """Show an animated spinner with ``message`` while a slow call runs, then clear it.

    Wraps ``rich``'s console status; a no-op spinner (no TTY) still runs the body. Use around
    network work like store provisioning so the CLI doesn't look hung. Under ``-o json`` the spinner
    is skipped so no control characters leak into machine-readable output.
    """
    from databricks_mason import errors  # local import avoids a cycle at module load

    con = con or _stdout
    if errors._OUTPUT_MODE == "json":
        yield
        return
    with con.status(message, spinner="dots"):
        yield


@contextmanager
def progress(message: str, con: Optional[Console] = None) -> Iterator[None]:
    """Like ``status``, but first prints a persistent line so feedback survives the spinner.

    ``status`` clears itself on exit and only animates on a TTY, so a long, silent wait (e.g. waiting
    for app compute) can look like a hang in terminals where the spinner doesn't render. This prints
    a durable "• message" line up front, then runs the spinner underneath it. Skipped under
    ``-o json`` so machine output stays clean.
    """
    from databricks_mason import errors  # local import avoids a cycle at module load

    con = con or _stdout
    if errors._OUTPUT_MODE == "json":
        yield
        return
    con.print(f"[{MUTED}]•[/] {message}")
    with con.status(message, spinner="dots"):
        yield


# --- small helpers -----------------------------------------------------------


def field(obj: dict, name: str) -> Any:
    """Read a field tolerating snake_case or camelCase JSON keys."""
    if name in obj:
        return obj[name]
    parts = name.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return obj.get(camel)


def emit_json(data: Any) -> None:
    """Print raw JSON for `--output json` (stable, pipe-friendly)."""
    click.echo(json.dumps(data, indent=2, default=str))


def confirm_destroy(target: str, *, assume_yes: bool) -> None:
    """Guard a destructive action with a confirmation prompt.

    No-op when `assume_yes` is set (the `--yes/-y` flag, for scripts). Otherwise prompts
    on the terminal and aborts unless the user answers yes. A non-interactive stdin (a
    pipe with no `--yes`) answers no and aborts, which is the safe default.
    """
    if assume_yes:
        return
    if not click.confirm(f"Delete {target}? This cannot be undone.", default=False):
        raise click.Abort()


def status_pill(status: Optional[str]) -> Text:
    """Create a colored ●/○ status indicator."""
    value = (status or "").strip().upper()
    if value in {"ACTIVE", "RUNNING", "READY"}:
        return Text("● ", style="green") + Text(value.title(), style="green")
    if value in {"PENDING", "CREATING", "STARTING", "DEPLOYING"}:
        return Text("○ ", style="yellow") + Text(value.title(), style="yellow")
    if value in {"DISABLED", "DELETED", "STOPPED", "ERROR", "FAILED"}:
        return Text("⨯ ", style="red") + Text(value.title(), style="red")
    return Text("● ", style=MUTED) + Text(value.title() or "Unknown", style=MUTED)


def hyperlink(text: str, url: Optional[str]) -> Text:
    """A terminal hyperlink (OSC 8): renders `text`, opens `url` on click; plain text if no url.

    Supported terminals show `text` as clickable so the full URL needn't fit on screen; others
    fall back to the plain text. Use `-o json` for the raw URL where a terminal lacks OSC 8.
    """
    if not url:
        return Text(text)
    return Text(text, style=f"{ACCENT} underline link {url}")


# --- list view ---------------------------------------------------------------


def resource_table(
    title: str,
    columns: Sequence[tuple[str, Literal["default", "left", "center", "right", "full"]]],
    rows: Iterable[Sequence[Any]],
    *,
    subtitle: Optional[str] = None,
    con: Optional[Console] = None,
) -> None:
    """Render a titled list table.

    `columns` is a sequence of (header, justify) where justify is left/right/center.
    """
    con = con or _stdout
    rows = list(rows)

    con.print()
    con.print(Text(title, style=f"bold {ACCENT}"))
    if subtitle:
        con.print(Text(subtitle, style=MUTED))

    table = Table(box=box.SIMPLE_HEAD, expand=False, pad_edge=False, show_edge=False)
    for header, justify in columns:
        table.add_column(header.upper(), justify=justify, header_style=f"bold {MUTED}")
    for row in rows:
        table.add_row(*[_cell(v) for v in row])
    con.print(table)

    con.print(Text(f"{len(rows)} item{'s' if len(rows) != 1 else ''}", style=MUTED))


def _cell(value: Any) -> Any:
    if isinstance(value, Text):
        return value
    if value is None:
        return Text("—", style=MUTED)
    return str(value)


# --- detail view (aig-endpoint.png) ------------------------------------------


def detail(
    breadcrumb: str,
    name: str,
    fields: dict[str, Any],
    *,
    status: Optional[str] = None,
    snippets: Optional[Sequence[tuple[str, str, str]]] = None,
    con: Optional[Console] = None,
) -> None:
    """Render a resource detail page.

    `breadcrumb` is the section label ("Agent Memory"); `fields` is an ordered
    key -> value map for the details rail. `snippets` is a sequence of
    (label, lexer, code) blocks shown under a "Starter code" panel.
    """
    con = con or _stdout

    con.print()
    con.print(Text(f"{breadcrumb}  ›  ", style=MUTED) + Text(name, style="bold"))
    if status is not None:
        con.print(status_pill(status))
    con.print()

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=MUTED, justify="left")
    grid.add_column(justify="left")
    for key, value in fields.items():
        grid.add_row(key, _cell(value))
    con.print(grid)

    if snippets:
        con.print()
        con.print(
            Panel(
                _snippet_group(snippets),
                title="Starter code",
                title_align="left",
                border_style=MUTED,
                box=box.ROUNDED,
            )
        )


def _snippet_group(snippets: Sequence[tuple[str, str, str]]) -> RenderableType:
    parts: list[RenderableType] = []
    for i, (label, lexer, code) in enumerate(snippets):
        if i:
            parts.append(Text())
        parts.append(Text(label, style=f"bold {MUTED}"))
        parts.append(Syntax(code.strip(), lexer, background_color="default", word_wrap=True))
    return Group(*parts)


# --- success / next steps (the "Set up" cards) -------------------------------


def success(
    title: str,
    *,
    fields: Optional[dict[str, Any]] = None,
    next_steps: "Optional[Sequence[str | tuple[str, str]]]" = None,
    con: Optional[Console] = None,
) -> None:
    """A green success panel with optional details and a Next steps list.

    Each next step is either a ``(command, description)`` pair — the command is highlighted and
    the description explains what it does — or a bare string for a non-command instruction (e.g.
    ``"Open http://localhost:8000"``), rendered as plain prose. Commands are shown so they can be
    copied and run in a terminal.
    """
    con = con or _stdout
    body: list[RenderableType] = [Text("✓ ", style="green") + Text(title, style="bold")]

    if fields:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style=MUTED)
        grid.add_column()
        for key, value in fields.items():
            grid.add_row(key, _cell(value))
        body.append(grid)

    if next_steps:
        has_command = any(isinstance(step, tuple) for step in next_steps)
        # When any step is a command, define the `$` marker so the convention is self-explanatory.
        header = Text("Next steps", style=f"bold {MUTED}")
        if has_command:
            header += Text("   ($ = run in your terminal)", style=MUTED)
        body.append(header)
        # A two-column grid aligns every command's description at the same offset.
        steps = Table.grid(padding=(0, 2))
        steps.add_column()
        steps.add_column(style=MUTED)
        for step in next_steps:
            if isinstance(step, tuple):
                # A `$ ` prompt prefix marks a runnable command (both in accent).
                command, description = step
                steps.add_row(Text("$ ", style=ACCENT) + Text(command, style=ACCENT), description)
            else:
                # Prose (e.g. "Open <url>") gets a muted bullet, not a command prompt.
                steps.add_row(Text("• ", style=MUTED) + Text(step), "")
        body.append(steps)

    con.print()
    con.print(Panel(Group(*body), border_style="green", box=box.ROUNDED))
