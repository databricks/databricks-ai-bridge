"""Unit tests for AgentCliError rendering across output modes."""

from __future__ import annotations

import json

import pytest

from databricks_mason import errors
from databricks_mason.errors import AgentCliError, auth_hint


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


def test_auth_hint_steers_to_a_single_mason_login(monkeypatch):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    hint = auth_hint("my-workspace")
    assert "mason login --profile my-workspace" in hint
    assert "databricks auth profiles" in hint
    # No stray-token note when the env var is absent.
    assert "DATABRICKS_TOKEN" not in hint


def test_auth_hint_uses_placeholder_without_profile(monkeypatch):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    assert "mason login --profile <name>" in auth_hint()


def test_auth_hint_flags_a_stray_token(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiXXXX")
    hint = auth_hint("my-workspace")
    # Calls out the override that the SDK message alone rarely makes obvious...
    assert "unset DATABRICKS_TOKEN" in hint
    # ...while still funneling to the one command.
    assert "mason login --profile my-workspace" in hint
