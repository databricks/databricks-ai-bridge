"""Credential examples are synthetic; scrubbing does not make error text non-sensitive."""

import pytest

from databricks_agentbricks.cli.failure_text import MAX_FAILURE_BYTES, scrub_failure_text


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer synthetic-secret",
        "{'Authorization': 'Basic synthetic-secret'}",
        "Cookie: session=synthetic-secret",
        '"client_secret": "synthetic-secret"',
        "DATABRICKS_TOKEN=synthetic-secret",
        "password='synthetic-secret'",
        "api-key: synthetic-secret",
        "Bearer synthetic-secret",
        "https://name:synthetic-secret@example.com/path",
        "https://example.com/path?unrecognized=synthetic-secret",
        "-----BEGIN PRIVATE KEY-----\nsynthetic-secret\n-----END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\nsynthetic-secret",
        "Authorization: Be\x1b[31marer synthetic-secret",
    ],
)
def test_credentials_are_scrubbed(value):
    result = scrub_failure_text("Connection failed\n" + value)
    assert "synthetic-secret" not in result
    assert "Connection failed" in result
    assert "[REDACTED]" in result


@pytest.mark.parametrize(
    "value",
    ["dapi" + "a" * 32, "ghp_" + "b" * 36, "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln"],
)
def test_recognizable_standalone_tokens_are_scrubbed(value):
    assert scrub_failure_text(f"Rejected {value}") == "Rejected [REDACTED]"


def test_customer_content_remains_sensitive_and_unknown_errors_need_no_enum():
    value = "NovelServiceError: customer_chosen_table cannot be resolved"
    assert scrub_failure_text(value) == value


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/private-project/state.json",
        "/Users/private-user/project/app.py",
        "/home/private-user/project/app.py",
        r"C:\Users\private-user\app.py",
        "C:/Users/private-user/app.py",
        "/Workspace/Users/private-user@example.com/project",
        "/Volumes/private-catalog/schema/volume/file.csv",
        "~/.private-config/token.json",
        "./private-project/app.py",
        "../private-project/app.py",
        ".private-cache/project/state.json",
        "private-project/app.py",
    ],
)
def test_paths_are_scrubbed_without_losing_the_error(path):
    assert scrub_failure_text(f"Cannot load '{path}': missing dependency") == (
        "Cannot load '[REDACTED_PATH]': missing dependency"
    )


def test_email_is_scrubbed_without_losing_the_error():
    assert scrub_failure_text("Access denied for private-user@example.com") == (
        "Access denied for [REDACTED_EMAIL]"
    )


def test_single_component_endpoint_path_is_preserved():
    assert scrub_failure_text("POST /telemetry-ext failed") == "POST /telemetry-ext failed"


def test_unicode_bound_preserves_utf8_and_marks_truncation():
    value = "Dependency failed: " + "界" * 10000
    result = scrub_failure_text(value)
    assert len(result.encode("utf-8")) <= MAX_FAILURE_BYTES
    assert result.startswith("Dependency failed: ")
    assert result.endswith("[truncated]")


def test_bound_does_not_leave_partial_assignment_secret():
    result = scrub_failure_text("token=" + "x" * 100000)
    assert "xxx" not in result
    assert len(result.encode()) <= MAX_FAILURE_BYTES


def test_control_sequences_are_removed():
    assert scrub_failure_text("\x1b[31mfailed\x1b[0m\x00\r\nnext") == "failed\nnext"
