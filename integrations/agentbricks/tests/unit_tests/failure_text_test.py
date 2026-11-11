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


def test_customer_paths_remain_sensitive_and_unknown_errors_need_no_enum():
    value = "NovelServiceError: /home/customer/project/custom_module.py cannot import package_x"
    assert scrub_failure_text(value) == value


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
