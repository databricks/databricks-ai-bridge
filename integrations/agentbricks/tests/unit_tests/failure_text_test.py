"""Credential examples are synthetic; scrubbing does not make error text non-sensitive."""

import json
import time

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
        'Authorization: "synthetic-secret\ncontinued"',
    ],
)
def test_credentials_are_scrubbed(value):
    result = scrub_failure_text("Connection failed\n" + value)
    assert "synthetic-secret" not in result
    assert "Connection failed" in result
    assert "[REDACTED]" in result


@pytest.mark.parametrize("scheme", ["postgres", "jdbc:postgresql", "ssh", "redis+tls"])
@pytest.mark.parametrize("password", ["synthetic-secret", "synthetic-secret@synthetic-tail"])
def test_url_credentials_are_scrubbed_for_any_scheme(scheme, password):
    result = scrub_failure_text(f"Connection failed: {scheme}://name:{password}@host/db")
    assert "synthetic-secret" not in result
    assert "synthetic-tail" not in result
    assert "Connection failed" in result
    assert "[REDACTED]" in result


@pytest.mark.parametrize(
    "name", ["secretKey", "tokenValue", "access_token_value", "apiKeyValue", "private-token-value"]
)
def test_credential_marker_can_appear_anywhere_in_the_assignment_name(name):
    result = scrub_failure_text(f"Request failed: data={{'{name}': 'synthetic-opaque-value'}}")
    assert "synthetic-opaque-value" not in result
    assert "Request failed" in result


@pytest.mark.parametrize(
    "value",
    [
        r'{"client_secret": "synthetic-prefix\"synthetic-tail"}',
        r"{'password': 'synthetic-prefix\'synthetic-tail'}",
        'client_secret="synthetic-prefix\\\nsynthetic-tail"',
        'password="synthetic-prefix synthetic-tail\\',
    ],
)
def test_quoted_credentials_respect_escapes_and_incomplete_values(value):
    result = scrub_failure_text(value)
    assert "synthetic-prefix" not in result
    assert "synthetic-tail" not in result


def test_escaped_backslash_keeps_following_error_text():
    assert scrub_failure_text(r'password="synthetic-prefix\\" after') == "[REDACTED] after"


@pytest.mark.parametrize("prefix", ["b", "B", "r", "u", "br", "rb", "f", "fr", "rf"])
@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("separator", [" ", ",", ";", "\n"])
def test_prefixed_credential_literals_are_scrubbed_completely(prefix, quote, separator):
    value = f"{prefix}{quote}{separator}synthetic-opaque-value{quote}"
    assert scrub_failure_text(f"password={value}\nConnection failed") == (
        "[REDACTED]\nConnection failed"
    )


def test_prose_apostrophes_do_not_hide_the_following_error():
    value = r"Error: app doesn't exist in workspace; see .\logs\x.log; Final error: quota exceeded"
    assert scrub_failure_text(value) == value


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize(
    "value",
    [
        json.dumps({"client_secret": "synthetic-opaque-value"}),
        json.dumps({"credentials": [{"value": "synthetic-opaque-value"}]}),
        json.dumps({"Authorization": ["synthetic-opaque-value"]}),
        r'{"\u0063lient_secret": ["synthetic-opaque-value"]}',
    ],
)
def test_encoded_values_are_omitted_with_their_credential_context(value, depth):
    for _ in range(depth):
        value = json.dumps({"message": value})
    result = scrub_failure_text(f"Request failed: {value}\nConnection failed")
    assert "synthetic-opaque-value" not in result
    assert result.startswith("Request failed: ")
    assert result.endswith("\nConnection failed")


@pytest.mark.parametrize(
    "name", ["credentials", "Authorization", "Proxy-Authorization", "Cookie", "Set-Cookie"]
)
@pytest.mark.parametrize(
    "value",
    [
        '{"client_secret": "synthetic-opaque-value"}',
        '{"client_secret":"synthetic-opaque-value"}',
        '{\n  "nested": {"value": "synthetic-opaque-value"}\n}',
        '[{"value": "synthetic-opaque-value"}]',
        '["synthetic-opaque-value"]',
        '[\n  "synthetic-opaque-value"\n]',
        r"""{'value': 'synthetic-opaque-value\'with}braces'}""",
    ],
)
def test_credential_containers_are_scrubbed_without_losing_following_error(name, value):
    result = scrub_failure_text(f"{name}: {value}\nConnection failed")
    assert result == "[REDACTED]\nConnection failed"


@pytest.mark.parametrize(
    "name", ["credentials", "Authorization", "Proxy-Authorization", "Cookie", "Set-Cookie"]
)
@pytest.mark.parametrize(
    "value", ['{"value": "synthetic-opaque-value"', '["synthetic-opaque-value"}']
)
def test_incomplete_or_malformed_credential_container_is_omitted(name, value):
    assert scrub_failure_text(f"Request failed: {name}={value}") == ("Request failed: [REDACTED]")


@pytest.mark.parametrize("marker", ["|", "|-", "|2", ">", ">+", ">-"])
@pytest.mark.parametrize("name", ["client_secret", "Authorization"])
def test_yaml_credential_blocks_are_omitted_conservatively(marker, name):
    assert (
        scrub_failure_text(
            f"Request failed: {name}: {marker}\n  synthetic-opaque-value\n  continued-value"
        )
        == "Request failed: [REDACTED]"
    )


@pytest.mark.parametrize("name", ["credentials", "Authorization", "client_secret"])
@pytest.mark.parametrize(
    "value",
    [
        "\n  client_secret: synthetic-opaque-value",
        " \n  - synthetic-opaque-value\n  - another-opaque-value",
        "\n  nested:\n    value: synthetic-opaque-value",
    ],
)
def test_multiline_credential_containers_are_omitted_conservatively(name, value):
    assert scrub_failure_text(f"Request failed: {name}:{value}") == "Request failed: [REDACTED]"


@pytest.mark.parametrize("separator", [": ", "="])
@pytest.mark.parametrize("name", ["token", "client_secret"])
@pytest.mark.parametrize("prefix", ["Bearer ", "Basic ", "multiple words "])
def test_unquoted_credentials_are_removed_through_the_end_of_the_line(name, separator, prefix):
    assert scrub_failure_text(
        f"{name}{separator}{prefix}synthetic-opaque-value\nConnection failed"
    ) == ("[REDACTED]\nConnection failed")


@pytest.mark.parametrize("pattern", ["a-", "a/", "http://", "eyJ-", "eyJ-."])
def test_dense_delimiters_do_not_delay_failure_reporting(pattern):
    start = time.monotonic()
    result = scrub_failure_text(pattern * (64000 // len(pattern)))
    assert time.monotonic() - start < 1.0
    assert len(result.encode()) <= MAX_FAILURE_BYTES


@pytest.mark.parametrize(
    "value",
    ["dapi" + "a" * 32, "ghp_" + "b" * 36, "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln"],
)
def test_recognizable_standalone_tokens_are_scrubbed(value):
    assert scrub_failure_text(f"Rejected {value}") == "Rejected [REDACTED]"


@pytest.mark.parametrize("prefix", ["error-", "outer.", "outer.error-"])
def test_jwt_redaction_preserves_surrounding_context(prefix):
    assert scrub_failure_text(f"{prefix}eyJheader.payload.signature") == f"{prefix}[REDACTED]"


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


@pytest.mark.parametrize("suffix", ["", "@localhost", "@example.com"])
def test_long_identifier_does_not_delay_failure_reporting(suffix):
    start = time.monotonic()
    result = scrub_failure_text("a" * 60000 + suffix)
    assert time.monotonic() - start < 1.0
    assert len(result.encode()) <= MAX_FAILURE_BYTES
    if suffix == "@example.com":
        assert result == "[REDACTED_EMAIL]"


def test_single_component_endpoint_path_is_preserved():
    assert scrub_failure_text("POST /telemetry-ext failed") == "POST /telemetry-ext failed"


def test_unicode_bound_preserves_utf8_and_marks_truncation():
    value = "Dependency failed: " + "界" * 10000 + "\nFinal error: package_missing"
    result = scrub_failure_text(value)
    assert len(result.encode("utf-8")) <= MAX_FAILURE_BYTES
    assert result.startswith("Dependency failed: ")
    assert "[truncated]" in result
    assert result.endswith("Final error: package_missing")


def test_bound_does_not_leave_partial_assignment_secret():
    result = scrub_failure_text("token=" + "x" * 100000)
    assert "xxx" not in result
    assert len(result.encode()) <= MAX_FAILURE_BYTES


def test_control_sequences_are_removed():
    assert scrub_failure_text("\x1b[31mfailed\x1b[0m\x00\r\nnext") == "failed\nnext"
