"""Tests for caller-facing descriptions of failed invocations."""

import builtins
import sys
from typing import Any, cast

import httpx
import openai
import pytest
import requests
from databricks.sdk.errors import platform

from databricks_agentkit.runtime.errors import (
    MAX_MESSAGE_LENGTH,
    InvocationError,
    describe_failure,
    redact,
)

_RATE_LIMIT_MESSAGE = (
    "Exceeded workspace input tokens per minute rate limit for databricks-claude-sonnet-4-5"
)
_REQUEST = httpx.Request("POST", "https://example.cloud.databricks.com/serving-endpoints/chat")
# openai reads only status, headers, and request, whichever httpx distribution it was built on.
_OPENAI_REQUEST = cast(Any, _REQUEST)


def openai_status_error(cls, status_code, *, headers=None, body=None):
    response = cast(Any, httpx.Response(status_code, headers=headers or {}, request=_REQUEST))
    return cls(f"Error code: {status_code} - {body}", response=response, body=body)


def message_of(exc: BaseException, secrets: tuple[str, ...] = ()) -> str:
    message = describe_failure(exc, secrets).error["message"]
    assert isinstance(message, str)
    return message


def rate_limit_error(headers=None):
    return openai_status_error(
        openai.RateLimitError,
        429,
        headers=headers,
        body={"error_code": "REQUEST_LIMIT_EXCEEDED", "message": _RATE_LIMIT_MESSAGE},
    )


def test_model_rate_limit_is_described_with_upstream_details():
    failure = describe_failure(rate_limit_error({"retry-after": "7"}))

    assert failure.status_code == 429
    assert failure.retry_after == 7
    assert failure.error == {
        "code": "UPSTREAM_RATE_LIMITED",
        "type": "openai.RateLimitError",
        "message": _RATE_LIMIT_MESSAGE,
        "upstream_status": 429,
        "upstream_code": "REQUEST_LIMIT_EXCEEDED",
        "retry_after": 7,
    }


def test_rate_limit_without_retry_after_header_omits_it():
    failure = describe_failure(rate_limit_error())

    assert failure.status_code == 429
    assert failure.retry_after is None
    assert "retry_after" not in failure.error


def test_retry_after_ms_header_rounds_up_to_seconds():
    assert describe_failure(rate_limit_error({"retry-after-ms": "1200"})).retry_after == 2


def test_retry_after_http_date_is_not_forwarded():
    headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert describe_failure(rate_limit_error(headers)).retry_after is None


def test_rate_limit_wrapped_as_cause_is_found():
    try:
        try:
            raise rate_limit_error({"retry-after": "3"})
        except openai.RateLimitError as exc:
            raise RuntimeError("graph step failed") from exc
    except RuntimeError as wrapped:
        failure = describe_failure(wrapped)

    assert failure.status_code == 429
    assert failure.error["type"] == "openai.RateLimitError"
    assert failure.error["retry_after"] == 3


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup requires Python 3.11")
def test_rate_limit_inside_exception_group_is_found():
    exception_group = getattr(builtins, "ExceptionGroup")  # noqa: B009 - absent on Python 3.10
    group = exception_group("tools failed", [ValueError("other"), rate_limit_error()])

    failure = describe_failure(group)

    assert failure.status_code == 429
    assert failure.error["upstream_code"] == "REQUEST_LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    ("exc", "status_code", "code", "upstream_status"),
    [
        (
            openai_status_error(
                openai.InternalServerError,
                503,
                body={"message": "Service unavailable", "code": "UNAVAILABLE"},
            ),
            502,
            "UPSTREAM_ERROR",
            503,
        ),
        (openai.APITimeoutError(request=_OPENAI_REQUEST), 504, "UPSTREAM_TIMEOUT", None),
        (httpx.ReadTimeout("timed out", request=_REQUEST), 504, "UPSTREAM_TIMEOUT", None),
        (httpx.ConnectTimeout("timed out", request=_REQUEST), 504, "UPSTREAM_TIMEOUT", None),
        (requests.exceptions.ConnectTimeout("timed out"), 504, "UPSTREAM_TIMEOUT", None),
        (
            httpx.HTTPStatusError(
                "Not Found",
                request=_REQUEST,
                response=httpx.Response(404, request=_REQUEST),
            ),
            502,
            "UPSTREAM_ERROR",
            404,
        ),
        (platform.TooManyRequests("slow down"), 429, "UPSTREAM_RATE_LIMITED", 429),
        (platform.DeadlineExceeded("deadline"), 504, "UPSTREAM_TIMEOUT", 504),
        (platform.NotFound("no such endpoint"), 502, "UPSTREAM_ERROR", 404),
        (KeyError("messages"), 500, "AGENT_ERROR", None),
        (RuntimeError("boom"), 500, "AGENT_ERROR", None),
    ],
)
def test_failures_map_to_status_codes(exc, status_code, code, upstream_status):
    failure = describe_failure(exc)

    assert failure.status_code == status_code
    assert failure.error["code"] == code
    assert failure.error.get("upstream_status") == upstream_status


def test_agent_bug_reports_builtin_type_and_message():
    failure = describe_failure(KeyError("messages"))

    assert failure.error == {"code": "AGENT_ERROR", "type": "KeyError", "message": "'messages'"}


def test_databricks_sdk_error_keeps_error_code_and_retry_after():
    exc = platform.TooManyRequests(
        "Too many requests", error_code="REQUEST_LIMIT_EXCEEDED", retry_after_secs=4
    )

    failure = describe_failure(exc)

    assert failure.error["type"] == "databricks.sdk.errors.platform.TooManyRequests"
    assert failure.error["upstream_code"] == "REQUEST_LIMIT_EXCEEDED"
    assert failure.retry_after == 4


def test_invocation_error_is_returned_as_described():
    failure = describe_failure(RuntimeError("boom"))

    assert describe_failure(InvocationError(failure)) is failure


def test_persisted_response_marks_agent_source():
    assert describe_failure(RuntimeError("boom")).response() == {
        "error": {"code": "AGENT_ERROR", "type": "RuntimeError", "message": "boom"},
        "status_code": 500,
        "source": "agent",
    }


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer abc.def-123",
        # Fake credentials, assembled so secret scanners don't flag the source.
        "dapi" + "0123456789abcdef" * 2,
        ".".join(["eyJ" + "hbGciOiJIUzI1NiJ9", "eyJ" + "zdWIiOiJ1c2VyIn0", "c2lnbmF0dXJl"]),
        "token=s3cr3t-value",
        "api_key: 'sk-live-123456'",
        "password=hunter22",
    ],
)
def test_known_credential_shapes_are_redacted(secret):
    message = message_of(RuntimeError(f"call failed with {secret} attached"))

    assert "[REDACTED]" in message
    for fragment in (
        "abc.def-123",
        "0123456789abcdef",
        "c2lnbmF0dXJl",
        "s3cr3t",
        "sk-live",
        "hunter22",
    ):
        assert fragment not in message


def test_request_token_and_env_secrets_are_redacted(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-secret-value-123")
    exc = RuntimeError("request opaque-user-token failed; env env-secret-value-123 leaked")

    message = message_of(exc, ("opaque-user-token",))

    assert message == "request [REDACTED] failed; env [REDACTED] leaked"


def test_short_env_values_are_not_treated_as_secrets(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "true")

    assert message_of(RuntimeError("flag is true")) == "flag is true"


def test_long_messages_are_truncated_and_tracebacks_dropped():
    exc = RuntimeError(
        "first line\nsecond line\n"
        + "x" * 2000
        + '\nTraceback (most recent call last):\n  File "agent.py"'
    )

    message = message_of(exc)

    assert len(message) <= MAX_MESSAGE_LENGTH
    assert message.startswith("first line second line xxx")
    assert message.endswith("…")
    assert "Traceback" not in redact("bad\nTraceback (most recent call last):\n  File 'x'")


def test_secret_cut_by_truncation_is_still_redacted():
    message = message_of(RuntimeError("x" * 290 + " Bearer abcdefghijklmnop"))

    assert "abcdefgh" not in message
    assert len(message) <= MAX_MESSAGE_LENGTH
