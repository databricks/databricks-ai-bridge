"""Offline contract tests for the deployed declarative-auth evidence matrix."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


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
