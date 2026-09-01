"""Unit tests for AgentCliError rendering across output modes."""

from __future__ import annotations

import json

import pytest

from databricks_mason import errors
from databricks_mason.errors import AgentCliError


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
