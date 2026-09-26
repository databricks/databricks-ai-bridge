"""Offline contract tests for the deployed declarative-auth evidence matrix."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from types import ModuleType

import pytest


def _load_matrix_module() -> ModuleType:
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"
    spec = importlib.util.spec_from_file_location("auth_scope_matrix_e2e", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auth_scope_matrix_accepts_complete_redacted_evidence(tmp_path: pathlib.Path):
    source_sha = "a" * 40
    marker = f"AUTH_SCOPE_E2E_{source_sha[:12]}"
    rows = []
    for framework in ("langgraph", "openai"):
        for scope_source in ("explicit", "combined"):
            required = ["sql"] if scope_source == "explicit" else ["ai-gateway", "sql"]
            rows.append(
                {
                    "framework": framework,
                    "scope_source": scope_source,
                    "status": "pass",
                    "required_scopes": required,
                    "configured_scopes": required,
                    "effective_scopes": [*required, "iam.current-user:read"],
                    "user_sql_marker": "AGENTBRICKS_USER_SQL_OK",
                    "app_sql_denied": True,
                    "web_search_verified": scope_source == "combined",
                    "invocation_status": "completed",
                    "freshness_marker": marker,
                }
            )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": source_sha,
                "wheel_sha256": "b" * 64,
                "rows": rows,
                "cleanup": {"apps_deleted": True, "sql_asset_deleted": True},
            }
        ),
        encoding="utf-8",
    )
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"

    result = subprocess.run(
        [sys.executable, str(script), "--verify-evidence", str(evidence)],
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed, 0 failed, 0 skipped" in result.stdout


def test_auth_scope_matrix_rejects_missing_negative_control(tmp_path: pathlib.Path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": "a" * 40,
                "wheel_sha256": "b" * 64,
                "rows": [],
                "cleanup": {"apps_deleted": True, "sql_asset_deleted": True},
            }
        ),
        encoding="utf-8",
    )
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"

    result = subprocess.run(
        [sys.executable, str(script), "--verify-evidence", str(evidence)],
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "4 skipped" in result.stdout


def test_invocation_readiness_retries_502_and_503(tmp_path: pathlib.Path):
    matrix = _load_matrix_module()
    outcomes = iter(
        [
            matrix.InvocationHTTPError(502, "App Not Available"),
            matrix.InvocationHTTPError(503, "App starting"),
            ({"status": "completed"}, 200),
        ]
    )
    attempts = 0

    def invoke():
        nonlocal attempts
        attempts += 1
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    transcript = matrix.Transcript(tmp_path / "commands.log")

    response, status = matrix._invoke_with_readiness_retry(
        "invoke-sql-langgraph-explicit",
        invoke,
        transcript,
        timeout=1,
        retry_interval=0,
    )

    assert (response, status) == ({"status": "completed"}, 200)
    assert attempts == 3
    log = transcript.path.read_text(encoding="utf-8")
    assert "attempt 1 | HTTP 502 | retrying" in log
    assert "attempt 2 | HTTP 503 | retrying" in log


def test_invocation_readiness_does_not_retry_functional_4xx(tmp_path: pathlib.Path):
    matrix = _load_matrix_module()
    attempts = 0

    def invoke():
        nonlocal attempts
        attempts += 1
        raise matrix.InvocationHTTPError(400, "Invalid invocation")

    with pytest.raises(matrix.InvocationHTTPError, match="HTTP 400"):
        matrix._invoke_with_readiness_retry(
            "invoke-sql-langgraph-explicit",
            invoke,
            matrix.Transcript(tmp_path / "commands.log"),
            timeout=1,
            retry_interval=0,
        )

    assert attempts == 1
