"""Unit tests for the timefmt and render presentation helpers."""

from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console
from rich.status import Status

from databricks_agentbricks.presentation import render
from databricks_agentbricks.reporting import Heartbeat, format_elapsed
from databricks_agentkit import timefmt


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=200, no_color=True), buf


def test_relative_phrasing_matches_mock():
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    assert timefmt.relative(now - timedelta(days=13), now=now) == "13 days ago"
    assert timefmt.relative(now - timedelta(hours=14), now=now) == "14 hours ago"
    assert timefmt.relative(now - timedelta(hours=1), now=now) == "An hour ago"
    assert timefmt.relative(now - timedelta(days=1), now=now) == "A day ago"
    assert timefmt.relative(now - timedelta(days=35), now=now) == "A month ago"
    assert timefmt.relative(None, now=now) == "—"


def test_parse_timestamp_millis_and_rfc3339():
    dt = timefmt.parse_timestamp(1_755_100_000_000)
    assert dt is not None and dt.tzinfo is not None
    rfc3339 = timefmt.parse_timestamp("2026-08-15T01:29:00Z")
    assert rfc3339 is not None and rfc3339.year == 2026
    assert timefmt.parse_timestamp("") is None
    assert timefmt.parse_timestamp("not-a-date") is None


def test_status_pill_colors():
    assert "Active" in render.status_pill("ACTIVE").plain
    assert render.status_pill("ACTIVE").plain.startswith("●")
    assert "Pending" in render.status_pill("PENDING").plain
    assert "Disabled" in render.status_pill("DISABLED").plain


def test_hyperlink_carries_url_and_plain_text():
    link = render.hyperlink("my-app", "https://example.com/app")
    assert link.plain == "my-app"  # shows the label, not the raw URL
    assert "link https://example.com/app" in str(link.style)


def test_warning_prints_yellow_message():
    con, buf = _console()
    render.warning("careful now", con=con)
    out = buf.getvalue()
    assert "careful now" in out
    assert "⚠" in out


def test_hyperlink_without_url_is_plain():
    link = render.hyperlink("my-app", None)
    assert link.plain == "my-app"
    assert not link.style


def test_field_snake_and_camel():
    assert render.field({"display_name": "x"}, "display_name") == "x"
    assert render.field({"displayName": "y"}, "display_name") == "y"
    assert render.field({}, "missing") is None


def test_resource_table_renders_title_and_count():
    con, buf = _console()
    render.resource_table(
        "Managed Memory Stores",
        [("Name", "left"), ("Created", "left")],
        [["acme", "13 days ago"]],
        con=con,
    )
    out = buf.getvalue()
    assert "Managed Memory Stores" in out
    assert "acme" in out
    assert "1 item" in out


def test_resource_table_keeps_no_wrap_column_full_width_when_narrow():
    buf = io.StringIO()
    con = Console(file=buf, width=60, no_color=True)
    long_name = "memory-store-with-a-long-id"
    render.resource_table(
        "Stores",
        [("Name", "left"), ("Resource name", "left"), ("Description", "left")],
        [["demo", long_name, "some long description text here"]],
        con=con,
        no_wrap=[1],
    )
    out = buf.getvalue()
    assert long_name in out  # resource name reserved full-width, never truncated
    assert "DESCRIPTION" in out.upper()  # no column is dropped; it just narrows/wraps


def test_success_next_steps_render_command_and_description():
    con, buf = _console()
    render.success(
        "Logged in",
        next_steps=[
            ("agentbricks init my-agent", "Scaffold a new agent project"),
            "Open http://localhost:8000 to chat with it",
        ],
        con=con,
    )
    out = buf.getvalue()
    # Plural heading when there is more than one step.
    assert "Next steps" in out
    # Commands are copy-safe: no `$` prompt prefix, so they paste straight into a shell.
    assert "agentbricks init my-agent" in out
    assert "$ agentbricks init my-agent" not in out
    assert "$ = run in your terminal" not in out
    assert "Scaffold a new agent project" in out
    # A bare-string step renders as prose (no command accent, no bullet).
    assert "Open http://localhost:8000 to chat with it" in out


def test_success_next_steps_lead_with_description_above_command():
    con, buf = _console()
    render.success(
        "Logged in",
        next_steps=[
            "Open http://localhost:8000 to chat with it",
            ("agentbricks init my-agent", "Scaffold a new agent project"),
            ("agentbricks dev", "Run it locally"),
        ],
        con=con,
    )
    rows = [line.strip("│ ") for line in buf.getvalue().splitlines() if line.startswith("│")]
    # Every command sits alone on the line under its description, so it copies without the gloss.
    steps = rows[rows.index("Next steps") + 1 :]
    assert steps == [
        "Open http://localhost:8000 to chat with it",
        "Scaffold a new agent project",
        "agentbricks init my-agent",
        "Run it locally",
        "agentbricks dev",
    ]


def test_success_single_next_step_uses_singular_heading():
    con, buf = _console()
    render.success(
        "Deployed",
        next_steps=[("agentbricks deployments logs my-agent", "stream the logs")],
        con=con,
    )
    out = buf.getvalue()
    assert "Next step" in out
    assert "Next steps" not in out  # singular when there is exactly one step


def test_success_prose_only_next_steps():
    con, buf = _console()
    render.success("Done", next_steps=["Set DATABRICKS_CONFIG_PROFILE in .env"], con=con)
    out = buf.getvalue()
    assert "Next step" in out  # singular for one step
    assert "$ = run in your terminal" not in out  # no command -> no marker legend
    assert "Set DATABRICKS_CONFIG_PROFILE in .env" in out


def test_diagnostic_renders_severity_keyword_and_help_line():
    con, buf = _console()
    render.diagnostic(
        "warning",
        "active virtualenv ignored",
        help="pass --active to target the active environment",
        con=con,
    )
    out = buf.getvalue()
    assert "warning: active virtualenv ignored" in out
    # The fix lives on an indented `help:` line, cargo/uv style.
    assert "  help: pass --active to target the active environment" in out


def test_diagnostic_error_uses_cargo_code_form():
    con, buf = _console()
    render.diagnostic("error", "store not found", code="NOT_FOUND", con=con)
    # cargo's `error[CODE]:` form keeps the machine-relevant code visible in the human line.
    assert "error[NOT_FOUND]: store not found" in buf.getvalue()


def test_diagnostic_colors_only_the_keyword_leaving_message_and_fix_readable(monkeypatch):
    # Regression: styling must not bleed onto the message/fix. Only the severity keyword is colored
    # and the `help:` label bold; the message and fix stay at the default foreground so they are
    # fully legible on any terminal. The style resets right after the keyword/label.
    import io

    from rich.console import Console

    from databricks_agentbricks.presentation.theme import AGENTBRICKS_THEME

    # Test ANSI styling independently of the test runner's terminal settings.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=120,
        force_terminal=True,
        color_system="standard",
        no_color=False,
        theme=AGENTBRICKS_THEME,
    )
    render.diagnostic("error", "boom happened", help="do the fix", con=con)
    raw = buf.getvalue()
    assert "error\x1b[0m: boom happened" in raw  # reset lands after the keyword; message is default
    assert "help:\x1b[0m do the fix" in raw  # reset lands after the label; fix is default


def test_detail_renders_breadcrumb_status_and_snippet():
    con, buf = _console()
    render.detail(
        "Agent Memory",
        "acme",
        {"Store ID": "abc"},
        status="ACTIVE",
        snippets=[("curl", "bash", "curl https://x")],
        con=con,
    )
    out = buf.getvalue()
    assert "Agent Memory" in out and "acme" in out
    assert "Active" in out
    assert "Starter code" in out


def test_status_skips_spinner_in_json_mode(monkeypatch):
    # Under -o json the spinner must not render, so no control chars pollute machine output.
    from databricks_agentbricks import errors

    con, buf = _console()
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    with render.status("working…", con=con):
        pass
    assert buf.getvalue() == ""  # nothing emitted in json mode


def test_status_renders_spinner_in_text_mode(monkeypatch):
    from databricks_agentbricks import errors

    # A TTY-backed console renders the spinner; a plain StringIO console is not a terminal, so
    # assert the body still runs and the call is a no-op-safe context manager.
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    con, buf = _console()
    ran = []
    with render.status("working…", con=con):
        ran.append(True)
    assert ran == [True]  # body runs whether or not the spinner is visible


def test_progress_prints_persistent_line_in_text_mode(monkeypatch):
    # The persistent line must survive even on a non-TTY console (where the spinner doesn't render),
    # so a long wait always leaves visible feedback.
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    con, buf = _console()
    with render.progress("Waiting for app compute…", con=con):
        pass
    assert "Waiting for app compute…" in buf.getvalue()


def test_progress_keeps_stdout_clean_in_json_mode(monkeypatch):
    from databricks_agentbricks import errors

    con, buf = _console()
    err, err_buf = _console()
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    with render.progress("Waiting for app compute…", con=con, err=err):
        pass
    assert buf.getvalue() == ""  # stdout carries only the command's JSON document
    assert err_buf.getvalue() == "• Waiting for app compute…\n"


# --- heartbeats --------------------------------------------------------------


class _FakeClock:
    """A monotonic clock that advances ``step`` seconds on every read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)


def test_heartbeat_ticks_once_per_interval():
    beat = Heartbeat("Waiting for agent compute to start…", start=100.0)

    assert beat.phase == "Waiting for agent compute to start"
    assert beat.tick(114.9) is None
    assert beat.tick(115.0) == "• Waiting for agent compute to start… 0:15"
    assert beat.tick(120.0) is None
    assert beat.tick(130.2) == "• Waiting for agent compute to start… 0:30"
    # A late tick reports once, then resumes on the interval grid.
    assert beat.tick(181.0) == "• Waiting for agent compute to start… 1:21"
    assert beat.tick(189.0) is None
    assert beat.tick(190.0) == "• Waiting for agent compute to start… 1:30"


def test_heartbeat_picks_up_a_phase_already_in_progress():
    # A waiter that joins 30 s into a phase reports next at 0:45, timed from the phase's start.
    beat = Heartbeat("Deploying the agent source", start=0.0, now=30.0)

    assert beat.tick(44.0) is None
    assert beat.tick(45.0) == "• Deploying the agent source… 0:45"


def test_format_elapsed():
    assert format_elapsed(0) == "0:00"
    assert format_elapsed(45.9) == "0:45"
    assert format_elapsed(100) == "1:40"
    assert format_elapsed(-3) == "0:00"


def test_progress_prints_heartbeats_to_stderr_without_a_tty(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    monkeypatch.setattr(render, "_HEARTBEAT_POLL_S", 0.001)
    con, buf = _console()
    err, err_buf = _console()
    clock = _FakeClock(step=5.0)  # every heartbeat-thread poll moves the clock 5 s
    with render.progress("Uploading the agent source…", con=con, err=err, clock=clock):
        _wait_for(lambda: err_buf.getvalue().count("\n") >= 3)

    assert buf.getvalue() == "• Uploading the agent source…\n"
    lines = err_buf.getvalue().splitlines()
    assert lines[:3] == [
        "• Uploading the agent source… 0:15",
        "• Uploading the agent source… 0:30",
        "• Uploading the agent source… 0:45",
    ]


def test_status_prints_heartbeats_only_to_stderr_in_json_mode(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    monkeypatch.setattr(render, "_HEARTBEAT_POLL_S", 0.001)
    con, buf = _console()
    err, err_buf = _console()
    with render.status("Reconciling Runtime Store…", con=con, err=err, clock=_FakeClock(5.0)):
        _wait_for(lambda: "0:15" in err_buf.getvalue())

    assert buf.getvalue() == ""
    assert err_buf.getvalue().splitlines()[0] == "• Reconciling Runtime Store… 0:15"


def test_short_step_prints_no_heartbeat(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    con, _ = _console()
    err, err_buf = _console()
    with render.status("Reconciling memory store 'm'…", con=con, err=err):
        pass
    assert err_buf.getvalue() == ""


def test_tty_progress_shows_elapsed_in_spinner_without_extra_lines(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    monkeypatch.setattr(render, "_HEARTBEAT_POLL_S", 0.001)
    updates: list[str] = []
    original_update = Status.update

    def record_update(self, status=None, **kwargs):
        if status is not None:
            updates.append(str(status))
        return original_update(self, status, **kwargs)

    monkeypatch.setattr(Status, "update", record_update)
    buf = io.StringIO()
    con = Console(file=buf, width=200, force_terminal=True, no_color=True)
    err, err_buf = _console()
    with render.progress(
        "Waiting for agent compute to start…", con=con, err=err, clock=_FakeClock(5)
    ):
        _wait_for(lambda: "0:15" in updates)

    assert "0:15" in updates
    assert err_buf.getvalue() == ""  # no heartbeat lines on a TTY
    assert buf.getvalue().count("• Waiting for agent compute to start…") == 1


def test_reporter_note_prints_through_a_live_tty_spinner(monkeypatch):
    # A busy-App retry notice arrives while the deploy's spinner is live; rich only keeps a line
    # clear of the spinner's redraw when it goes through the sys.stderr proxy the spinner installs.
    from databricks_agentbricks import errors
    from databricks_agentbricks.presentation.reporter import ClickReporter

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    buf = io.StringIO()
    con = Console(file=buf, width=200, force_terminal=True, no_color=True)
    notice = "App is busy (another operation in progress); retrying in 5s…"
    with render.progress("Deploying the agent source…", con=con):
        ClickReporter().note(notice)

    assert notice in buf.getvalue()


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (45.9, "45s"), (60, "1m00s"), (72, "1m12s"), (125, "2m05s")],
)
def test_format_duration(seconds, expected):
    assert render.format_duration(seconds) == expected


def test_progress_with_done_message_prints_the_total_wait(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    now = [100.0]
    con, buf = _console()
    err, _ = _console()
    with render.progress(
        "Waiting for agent compute to start…",
        con=con,
        err=err,
        clock=lambda: now[0],
        interval=1000,
        done_message="Agent compute is running",
    ):
        now[0] = 164.0
    lines = buf.getvalue().splitlines()
    assert lines[0].endswith("Waiting for agent compute to start…")
    assert lines[-1] == "  Agent compute is running (waited 1m04s)"


def test_progress_with_done_message_prints_nothing_more_on_failure(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    con, buf = _console()
    with pytest.raises(RuntimeError):
        with render.progress("Waiting…", con=con, done_message="Agent compute is running"):
            raise RuntimeError("timed out")
    assert "Agent compute is running" not in buf.getvalue()


def test_progress_done_message_goes_to_stderr_in_json_mode(monkeypatch):
    from databricks_agentbricks import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    con, buf = _console()
    err, err_buf = _console()
    with render.progress("Waiting…", con=con, err=err, done_message="Agent compute is running"):
        pass
    assert buf.getvalue() == ""
    assert err_buf.getvalue().splitlines()[-1] == "  Agent compute is running (waited 0s)"
