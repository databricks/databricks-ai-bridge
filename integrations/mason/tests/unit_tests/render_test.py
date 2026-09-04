"""Unit tests for the timefmt and render presentation helpers."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

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
    # The header defines the `$` marker when at least one step is a command.
    assert "$ = run in your terminal" in out
    # Commands get a `$ ` prompt prefix and show their description.
    assert "$ mason init my-agent" in out
    assert "Scaffold a new agent project" in out
    # A bare-string step renders as prose with a `•` bullet (not a command prompt).
    assert "• Open http://localhost:8000 to chat with it" in out


def test_success_prose_only_next_steps_omit_terminal_hint():
    con, buf = _console()
    render.success("Done", next_steps=["Set DATABRICKS_CONFIG_PROFILE in .env"], con=con)
    out = buf.getvalue()
    assert "Next steps" in out
    assert "$ = run in your terminal" not in out  # no command -> no marker legend
    assert "Set DATABRICKS_CONFIG_PROFILE in .env" in out


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
