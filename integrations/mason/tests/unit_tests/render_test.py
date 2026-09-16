"""Unit tests for the timefmt and render presentation helpers."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from databricks_mason import render, timefmt


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
            ("mason init my-agent", "Scaffold a new agent project"),
            "Open http://localhost:8000 to chat with it",
        ],
        con=con,
    )
    out = buf.getvalue()
    # Plural heading when there is more than one step.
    assert "Next steps" in out
    # Commands are copy-safe: no `$` prompt prefix, so they paste straight into a shell.
    assert "mason init my-agent" in out
    assert "$ mason init my-agent" not in out
    assert "$ = run in your terminal" not in out
    assert "Scaffold a new agent project" in out
    # A bare-string step renders as prose (no command accent, no bullet).
    assert "Open http://localhost:8000 to chat with it" in out


def test_success_single_next_step_uses_singular_heading():
    con, buf = _console()
    render.success(
        "Deployed",
        next_steps=[("mason deployments logs my-agent", "stream the logs")],
        con=con,
    )
    out = buf.getvalue()
    assert "Next step" in out
    assert "Next steps" not in out  # singular when there is exactly one step


@pytest.mark.parametrize("width", [40, 80, 200])
@pytest.mark.parametrize("terminal", [False, True])
def test_success_commands_are_standalone_unwrapped_lines(width: int, terminal: bool):
    buf = io.StringIO()
    con = Console(file=buf, width=width, no_color=True, force_terminal=terminal)
    command = "mason endpoint invoke --url http://localhost:8000 --path /invocations " + (
        '--json \'{"input":[{"role":"user","content":"hi"}]}\''
    )
    render.success("Started", next_steps=[(command, "Send a test request")], con=con)

    lines = buf.getvalue().splitlines()
    # No stripping, joining, or removal of frame characters: copy the entire printed line.
    assert command in lines
    command_index = lines.index(command)
    assert lines[command_index - 1] == "Send a test request"
    assert any(line.startswith("╰") for line in lines[:command_index])


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


def test_diagnostic_colors_only_the_keyword_leaving_message_and_fix_readable():
    # Regression: styling must not bleed onto the message/fix. Only the severity keyword is colored
    # and the `help:` label bold; the message and fix stay at the default foreground so they are
    # fully legible on any terminal. The style resets right after the keyword/label.
    import io

    from rich.console import Console

    from databricks_mason.theme import MASON_THEME

    buf = io.StringIO()
    con = Console(file=buf, width=120, force_terminal=True, theme=MASON_THEME)
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
    from databricks_mason import errors

    con, buf = _console()
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    with render.status("working…", con=con):
        pass
    assert buf.getvalue() == ""  # nothing emitted in json mode


def test_status_renders_spinner_in_text_mode(monkeypatch):
    from databricks_mason import errors

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
    from databricks_mason import errors

    monkeypatch.setattr(errors, "_OUTPUT_MODE", "text")
    con, buf = _console()
    with render.progress("Waiting for app compute…", con=con):
        pass
    assert "Waiting for app compute…" in buf.getvalue()


def test_progress_skips_output_in_json_mode(monkeypatch):
    from databricks_mason import errors

    con, buf = _console()
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    with render.progress("Waiting for app compute…", con=con):
        pass
    assert buf.getvalue() == ""  # nothing emitted in json mode
