"""Unit tests for AgentCliError rendering across output modes."""

from __future__ import annotations

import json

import pytest

from databricks_mason import errors
from databricks_mason.errors import AgentCliError, wrap_api_error


@pytest.fixture(autouse=True)
def _reset_output_mode():
    # Keep the process-global output mode from leaking between tests.
    yield
    errors.set_output_mode("text")


def test_error_renders_json_object_in_json_mode(capsys):
    errors.set_output_mode("json")
    AgentCliError("store not found", error_code="NOT_FOUND", hint="check the id").show()
    captured = capsys.readouterr()
    # JSON errors go to stderr so stdout stays a clean success channel.
    payload = json.loads(captured.err)
    assert payload == {
        "error": {"message": "store not found", "code": "NOT_FOUND", "hint": "check the id"}
    }


def test_error_json_omits_absent_code_and_hint(capsys):
    errors.set_output_mode("json")
    AgentCliError("boom").show()
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"error": {"message": "boom"}}


def test_error_renders_text_in_text_mode(capsys):
    errors.set_output_mode("text")
    AgentCliError("boom", error_code="NOT_FOUND").show()
    err = capsys.readouterr().err
    assert "boom" in err and "NOT_FOUND" in err
    # Not JSON in text mode.
    with pytest.raises(json.JSONDecodeError):
        json.loads(err)


class _NoDetailError(RuntimeError):
    """Mimics a databricks-sdk error with a code but no message (str() == 'None')."""

    error_code = "CANCELLED"


def test_message_less_error_surfaces_code_not_none():
    # The SDK stringifies a message-less DatabricksError as the literal "None"; the wrapped
    # error must surface the code instead of a bare `None`.
    mapped = wrap_api_error(_NoDetailError(None))
    assert mapped.error_code == "CANCELLED"
    assert "None" not in mapped.message
    assert "CANCELLED" in mapped.message


def test_transient_error_gets_retry_hint():
    mapped = wrap_api_error(_NoDetailError(None))
    assert mapped.hint is not None
    assert "transient" in mapped.hint.lower()


def test_non_transient_error_keeps_original_message():
    class NotFoundError(RuntimeError):
        error_code = "NOT_FOUND"

    mapped = wrap_api_error(NotFoundError("store not found"))
    assert mapped.message == "store not found"
    assert mapped.hint is None


def test_retry_timeout_surfaces_original_service_error():
    # When the SDK exhausts its retry budget it raises TimeoutError chaining the last underlying
    # error; the wrapped error must surface that original service error, not the retry wrapper.
    class ResourceExhausted(RuntimeError):
        error_code = "RESOURCE_EXHAUSTED"

    original = ResourceExhausted("Workspace has reached the maximum of 50 managed memory stores")
    timeout = TimeoutError("Timed out after 0:01:00")
    timeout.__cause__ = original

    mapped = wrap_api_error(timeout)
    assert mapped.error_code == "RESOURCE_EXHAUSTED"
    assert "maximum of 50" in mapped.message
    assert "Timed out" not in mapped.message


def test_retry_timeout_without_cause_falls_through():
    mapped = wrap_api_error(TimeoutError("Timed out after 0:01:00"))
    assert "Timed out" in mapped.message
