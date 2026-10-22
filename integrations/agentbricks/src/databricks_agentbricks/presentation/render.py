"""Terminal presentation layer for the Agent Bricks CLI.

Rendering is centralized so every `list`/`get`/`create` command looks consistent.
Functions accept an optional `console` for testability; default is stdout.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

import click
from rich import box
from rich.console import Console, Group, RenderableType
from rich.measure import Measurement
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from databricks_agentbricks.presentation import theme
from databricks_agentbricks.reporting import HEARTBEAT_INTERVAL_S, Heartbeat, format_elapsed

# Semantic styles from the facelift palette (see `theme.py`). Color carries meaning, never
# decoration: LINK (blue) for URLs, COMMAND (cyan) for the actionable next step, SUCCESS (green)
# for the ✔ and success boxes, and the mid-tone greys for descriptions/metadata. There is no
# decorative brand accent — the design reserves every color for a signal. These are raw hex/style
# strings, not themed names, so they render on any `Console` a caller passes in — not just the
# module's own themed stdout.
SECONDARY = theme.SECONDARY  # descriptions, section labels
DIM = theme.DIM  # metadata: timestamps, key labels, breadcrumbs
FAINT = theme.FAINT  # tree branches, placeholders
SUCCESS = theme.GREEN
LINK = theme.BLUE
COMMAND = theme.CYAN
WARNING = theme.AMBER
ERROR = theme.RED

# Severity-keyword styles for `diagnostic`: error=red, warning=amber, note=blue.
_SEVERITY_STYLE = {
    "error": f"bold {theme.RED}",
    "warning": f"bold {theme.AMBER}",
    "note": f"bold {theme.BLUE}",
}

_stdout = Console(theme=theme.AGENTBRICKS_THEME)
# Progress, heartbeats, and relayed command output when stdout must stay clean (`-o json`, logs).
_stderr = Console(theme=theme.AGENTBRICKS_THEME, stderr=True)
# How often the heartbeat thread wakes to refresh a spinner or check whether a heartbeat is due.
_HEARTBEAT_POLL_S = 1.0


def console() -> Console:
    return _stdout


def stderr_console() -> Console:
    return _stderr


def _plain_progress(con: Console) -> bool:
    """Whether progress must be plain lines: under ``-o json`` or when ``con`` isn't a terminal."""
    from databricks_agentbricks import errors  # local import avoids a cycle at module load

    return errors._OUTPUT_MODE == "json" or not con.is_terminal


@contextmanager
def _every(poll_s: float, on_tick: Callable[[], None]) -> Iterator[None]:
    """Call ``on_tick`` every ``poll_s`` seconds on a daemon thread while the body runs."""
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(poll_s):
            on_tick()

    thread = threading.Thread(target=run, name="agentbricks-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


@contextmanager
def _plain_heartbeats(
    message: str, err: Console, clock: Callable[[], float], interval: float
) -> Iterator[None]:
    """Print ``• <phase>… M:SS`` to ``err`` every ``interval`` seconds while the body runs."""
    beat = Heartbeat(message, start=clock(), interval=interval)

    def tick() -> None:
        line = beat.tick(clock())
        if line:
            err.print(Text(line), soft_wrap=True)

    with _every(min(_HEARTBEAT_POLL_S, interval), tick):
        yield


@contextmanager
def _spinner_elapsed(spinner: Status, label: str, clock: Callable[[], float]) -> Iterator[None]:
    """Keep a TTY spinner's text at ``label`` plus the elapsed ``M:SS`` while the body runs."""
    start = clock()

    def tick() -> None:
        elapsed = format_elapsed(clock() - start)
        spinner.update(Text(f"{label} {elapsed}" if label else elapsed, style=DIM))

    with _every(_HEARTBEAT_POLL_S, tick):
        yield


@contextmanager
def status(
    message: str,
    con: Optional[Console] = None,
    *,
    err: Optional[Console] = None,
    clock: Callable[[], float] = time.monotonic,
    interval: float = HEARTBEAT_INTERVAL_S,
) -> Iterator[None]:
    """Show an animated spinner with ``message`` while a slow call runs, then clear it.

    Use around network work like store provisioning so the CLI doesn't look hung. On a TTY the
    spinner text carries the elapsed time. Without a TTY, or under ``-o json``, there is no spinner
    (so no control characters leak into captured or machine-readable output); instead a step that
    runs past ``interval`` prints a plain ``• <phase>… M:SS`` heartbeat line to stderr each
    interval, so a log or a coding agent can tell the CLI is still working.
    """
    con = con or _stdout
    if _plain_progress(con):
        with _plain_heartbeats(message, err or _stderr, clock, interval):
            yield
        return
    with con.status(message, spinner="dots") as spinner:
        with _spinner_elapsed(spinner, message, clock):
            yield


@contextmanager
def progress(
    message: str,
    con: Optional[Console] = None,
    *,
    err: Optional[Console] = None,
    clock: Callable[[], float] = time.monotonic,
    interval: float = HEARTBEAT_INTERVAL_S,
) -> Iterator[None]:
    """Like ``status``, but first prints a persistent line so feedback survives the spinner.

    ``status`` clears itself on exit and only animates on a TTY, so a long, silent wait (e.g. waiting
    for app compute) can look like a hang in terminals where the spinner doesn't render. This prints
    a durable "• message" line up front, then runs a bare elapsed-time spinner beneath it on a TTY,
    or ``status``'s plain stderr heartbeats otherwise. Under ``-o json`` the up-front line goes to
    stderr too, so stdout stays a single JSON document.
    """
    from databricks_agentbricks import errors  # local import avoids a cycle at module load

    con = con or _stdout
    err = err or _stderr
    (err if errors._OUTPUT_MODE == "json" else con).print(f"[{DIM}]•[/] {message}")
    if _plain_progress(con):
        with _plain_heartbeats(message, err, clock, interval):
            yield
        return
    # Empty label: the persistent line above already carries the message, so the spinner
    # underneath is just the animated glyph and the elapsed time — no duplicated sentence.
    with con.status("", spinner="dots") as spinner:
        with _spinner_elapsed(spinner, "", clock):
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
    """Create a colored ●/○ status indicator, using the palette's semantic colors."""
    value = (status or "").strip().upper()
    if value in {"ACTIVE", "RUNNING", "READY"}:
        return Text("● ", style=SUCCESS) + Text(value.title(), style=SUCCESS)
    if value in {"PENDING", "CREATING", "STARTING", "DEPLOYING"}:
        return Text("○ ", style=WARNING) + Text(value.title(), style=WARNING)
    if value in {"DISABLED", "DELETED", "STOPPED", "ERROR", "FAILED"}:
        return Text("⨯ ", style=ERROR) + Text(value.title(), style=ERROR)
    return Text("● ", style=DIM) + Text(value.title() or "Unknown", style=DIM)


def warning(message: str, con: Optional[Console] = None) -> None:
    """Print a yellow ⚠ warning line (non-fatal; the command keeps running)."""
    (con or _stdout).print(Text("⚠ ", style="yellow") + Text(message, style="yellow"))


def hyperlink(text: str, url: Optional[str]) -> Text:
    """A terminal hyperlink (OSC 8): renders `text`, opens `url` on click; plain text if no url.

    Blue + underline, because the palette reserves blue for links/URLs. Supported terminals show
    `text` as clickable so the full URL needn't fit on screen; others fall back to the plain text.
    Use `-o json` for the raw URL where a terminal lacks OSC 8.
    """
    if not url:
        return Text(text)
    return Text(text, style=f"{LINK} underline link {url}")


def diagnostic(
    severity: Literal["error", "warning", "note"],
    message: str,
    *,
    code: Optional[str] = None,
    help: Optional[str] = None,
    con: Optional[Console] = None,
) -> None:
    """Render one cargo/uv-style diagnostic.

    A color-coded severity keyword, a plain-language message, and an optional indented ``help:``
    line carrying the fix::

        warning: active virtualenv ignored — `VIRTUAL_ENV` doesn't match `.venv`
          help: pass --active to target the active environment

    ``code`` renders cargo's ``error[CODE]:`` form (errors only). The severity keyword is colored
    (error=red, warning=amber, note=blue); the message and fix use the default foreground so they
    stay readable on any terminal background.
    """
    con = con or _stdout
    keyword = f"{severity}[{code}]" if code and severity == "error" else severity
    # `Text.assemble` styles each span independently; `Text(a, style=s) + Text(b)` would instead make
    # `s` the whole line's container style, coloring the message/fix too (and dimming the fix so far
    # it's hard to read). Only the severity keyword and the `help:` label carry weight/color; the
    # message and the fix stay at the terminal's default foreground so they're fully legible.
    con.print(Text.assemble((keyword, _SEVERITY_STYLE[severity]), (f": {message}", "")))
    if help:
        con.print(Text.assemble(("  help:", "bold"), (f" {help}", "")))


# --- list view ---------------------------------------------------------------


def resource_table(
    title: str,
    columns: Sequence[tuple[str, Literal["default", "left", "center", "right", "full"]]],
    rows: Iterable[Sequence[Any]],
    *,
    subtitle: Optional[str] = None,
    footer: Optional[str] = None,
    con: Optional[Console] = None,
    no_wrap: Optional[Sequence[int]] = None,
) -> None:
    """Render a titled list table.

    `columns` is a sequence of (header, justify) where justify is left/right/center.
    `no_wrap` marks identifier columns, which are never cut off: piped output renders the table at
    its full width so each row stays on one line (and greps cleanly), and a terminal wraps a
    too-long identifier instead of ending it in an ellipsis. The other columns narrow to fit, on one
    line.
    `footer` replaces the trailing item count.
    """
    con = con or _stdout
    rows = list(rows)
    no_wrap_cols = set(no_wrap or ())

    con.print()
    con.print(Text(title, style="bold"))
    if subtitle:
        con.print(Text(subtitle, style=SECONDARY))

    # Dim, uppercase headers; columns size to content so they always line up.
    table = Table(box=box.SIMPLE_HEAD, expand=False, pad_edge=False, show_edge=False)
    # Reserve each identifier column's full content width so Rich narrows the others instead, as
    # long as the identifiers fit side by side; otherwise they fold onto the next line.
    reserved = {
        i: max([len(columns[i][0])] + [_cell_len(row[i]) for row in rows]) for i in no_wrap_cols
    }
    reserve = sum(reserved.values()) + 3 * len(columns) <= con.width
    for i, (header, justify) in enumerate(columns):
        table.add_column(
            header.upper(),
            justify=justify,
            header_style=f"bold {DIM}",
            overflow="fold" if i in no_wrap_cols else "ellipsis",
            # Next to identifier columns the others shorten on one line: wrapping a column narrowed
            # to a few characters would spread each row over blank-looking lines.
            no_wrap=bool(no_wrap_cols) and i not in no_wrap_cols,
            min_width=reserved.get(i) if reserve else None,
        )
    for row in rows:
        table.add_row(*[_cell(v) for v in row])
    if no_wrap_cols and not con.is_terminal:
        # Piped output has no screen to fit: print at the table's natural width. `print(width=)`
        # can't exceed the console's width, so widen the console itself for this one table.
        natural = Measurement.get(con, con.options.update_width(1_000_000), table).maximum
        saved_width = con._width
        con.width = max(con.width, natural)
        try:
            con.print(table)
        finally:
            con._width = saved_width
    else:
        con.print(table)

    if footer is None:
        footer = f"{len(rows)} item{'s' if len(rows) != 1 else ''}"
    con.print(Text(footer, style=DIM))


def more_pages_footer(shown: int, next_page_token: Optional[str]) -> Optional[str]:
    """The stores-list footer saying another page exists and how to get it, or None on the last."""
    if not next_page_token:
        return None
    return f"{shown} shown; more available: --page-token {next_page_token}, or use --all / --name"


def _cell_len(value: Any) -> int:
    if value is None:
        return 1  # em-dash placeholder
    return len(value.plain if isinstance(value, Text) else str(value))


def _cell(value: Any) -> Any:
    if isinstance(value, Text):
        return value
    if value is None:
        return Text("—", style=DIM)
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
    con.print(Text(f"{breadcrumb}  ›  ", style=DIM) + Text(name, style="bold"))
    if status is not None:
        con.print(status_pill(status))
    con.print()

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=DIM, justify="left")
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
                border_style=DIM,
                box=box.ROUNDED,
            )
        )


def _snippet_group(snippets: Sequence[tuple[str, str, str]]) -> RenderableType:
    parts: list[RenderableType] = []
    for i, (label, lexer, code) in enumerate(snippets):
        if i:
            parts.append(Text())
        parts.append(Text(label, style=f"bold {DIM}"))
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
    """A green success panel with optional details and a Next step(s) list.

    Each next step is either a ``(command, description)`` pair — the description is shown in
    secondary on its own line, with the command in cyan (the actionable accent) on the line below —
    or a bare string for a non-command instruction (e.g. ``"Open http://localhost:8000"``), rendered
    as prose. Commands are **copy-safe**: no ``$`` prompt prefix and a full-width line to
    themselves, so they paste straight into a shell. The heading is "Next step" or "Next steps",
    chosen automatically from the count.
    """
    con = con or _stdout
    body: list[RenderableType] = [Text("✔ ", style=SUCCESS) + Text(title, style="bold")]

    if fields:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style=DIM)
        grid.add_column()
        for key, value in fields.items():
            grid.add_row(key, _cell(value))
        body.append(grid)

    if next_steps:
        # Singular vs plural from the count, per the design.
        heading = "Next step" if len(next_steps) == 1 else "Next steps"
        body.append(Text(heading, style=SECONDARY))
        for step in next_steps:
            if isinstance(step, tuple):
                # The description leads; the cyan command gets the full width of the line below.
                command, description = step
                body.append(Text(description, style=SECONDARY))
                # Fold rather than crop a long, space-free token (e.g. a JSON body) at the border.
                body.append(Text(command, style=COMMAND, overflow="fold"))
            else:
                # Prose (e.g. "Open <url>") renders plainly — no command accent, no prompt marker.
                body.append(Text(step))

    con.print()
    con.print(Panel(Group(*body), border_style=SUCCESS, box=box.ROUNDED))
