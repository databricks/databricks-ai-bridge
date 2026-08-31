"""Unit tests for the Mason tool-matrix evidence contract."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import tomli

_MATRIX_PATH = pathlib.Path(__file__).parents[1] / "e2e" / "tool_matrix.py"
_VALIDATION_CHECKS = (
    "undeclared_warning",
    "broken_entrypoint_rejection",
    "valid_contract_check",
    "direct_custom_tool_run",
)


@pytest.fixture(scope="module")
def tool_matrix() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mason_e2e_tool_matrix", _MATRIX_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_evidence() -> dict[str, Any]:
    app_names = {
        "cli": "mason-tools-langgraph-cl-test",
        "direct": "mason-tools-langgraph-di-test",
    }
    rows = [
        {
            "framework": framework,
            "authoring": authoring,
            "runtime": runtime,
            "tool_kind": tool_kind,
            "status": "pass",
            "app_name": app_names[authoring] if runtime == "deploy" else None,
        }
        for framework in ("langgraph",)
        for authoring in ("cli", "direct")
        for runtime in ("dev", "deploy")
        for tool_kind in ("sandbox", "mcp", "python", "uc_function")
    ]
    validation_checks = [
        {
            "framework": "langgraph",
            "authoring": "cli",
            "check": check,
            "status": "pass",
            "command": "mason tools check",
            "return_code": 0,
            "stdout": "{}",
            "stderr": "",
        }
        for check in _VALIDATION_CHECKS
    ]
    cleanup = [
        {
            "resource_kind": "app",
            "resource": name,
            "status": "pass",
            "command": f"databricks apps delete {name}",
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }
        for name in app_names.values()
    ]
    cleanup.append(
        {
            "resource_kind": "uc_function",
            "resource": "main.mason_agent_tools_e2e.marker",
            "status": "pass",
            "command": "DROP FUNCTION IF EXISTS main.mason_agent_tools_e2e.marker",
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }
    )
    return {
        "uc_function": "main.mason_agent_tools_e2e.marker",
        "rows": rows,
        "validation_checks": validation_checks,
        "cleanup": cleanup,
    }


def _write_evidence(tmp_path: pathlib.Path, document: dict[str, Any]) -> pathlib.Path:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    return evidence


def test_literal_python_activation_appends_exact_manifest_record(
    tmp_path: pathlib.Path, tool_matrix: ModuleType
):
    manifest = tmp_path / "agent.toml"
    manifest.write_text(
        """schema_version = 1

[agent]
framework = "langgraph"
""",
        encoding="utf-8",
    )

    tool_matrix.append_python_activation(manifest)

    document = tomli.loads(manifest.read_text(encoding="utf-8"))
    assert document["tools"] == [
        {
            "id": "matrix-marker",
            "source": {
                "kind": "python",
                "entrypoint": "agent.tools.matrix_marker:matrix_marker",
            },
        }
    ]


def test_verify_evidence_accepts_all_required_validation_checks(
    tmp_path: pathlib.Path, tool_matrix: ModuleType
):
    evidence = _write_evidence(tmp_path, _complete_evidence())

    assert tool_matrix.verify_evidence(evidence) == 0


@pytest.mark.parametrize("missing_check", _VALIDATION_CHECKS)
def test_verify_evidence_rejects_each_missing_validation_check(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, missing_check: str
):
    document = _complete_evidence()
    document["validation_checks"] = [
        check for check in document["validation_checks"] if check["check"] != missing_check
    ]
    evidence = _write_evidence(tmp_path, document)

    assert tool_matrix.verify_evidence(evidence) == 1


@pytest.mark.parametrize("failed_check", _VALIDATION_CHECKS)
def test_verify_evidence_rejects_each_unsuccessful_validation_check(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, failed_check: str
):
    document = _complete_evidence()
    for check in document["validation_checks"]:
        if check["check"] == failed_check:
            check["status"] = "fail"
            break
    evidence = _write_evidence(tmp_path, document)

    assert tool_matrix.verify_evidence(evidence) == 1


def test_deploy_tracks_app_for_cleanup_before_deploy_command(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    runner = tool_matrix.Runner("profile", tmp_path, tmp_path / "mason.whl")
    case = tool_matrix.ProjectCase("langgraph", "cli", tmp_path / "project", "new-app")
    monkeypatch.setattr(
        runner,
        "run_long",
        lambda *args, **kwargs: (_ for _ in ()).throw(tool_matrix.MatrixError("deploy failed")),
    )
    monkeypatch.setattr(runner, "_record_runtime_failure", lambda *args, **kwargs: None)

    runner.deploy(case)

    assert runner.apps == ["new-app"]


def test_e2e_bundles_the_sdk_wheel_for_generated_dev_and_deploy_projects(
    tmp_path: pathlib.Path, tool_matrix: ModuleType
):
    wheel = tmp_path / "databricks_mason-0.1.0.dev0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "agent"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    runner = tool_matrix.Runner("profile", tmp_path / "output", wheel)

    runner._install_sdk_wheel(project)

    bundled = project / ".mason" / "sdk" / wheel.name
    assert bundled.read_bytes() == b"wheel"
    document = tomli.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["tool"]["uv"]["sources"]["databricks-mason"] == {
        "path": f".mason/sdk/{wheel.name}"
    }


def test_cleanup_records_app_delete_failure_as_observable_evidence(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    wheel = tmp_path / "mason.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path, wheel)
    runner.apps = ["undeletable-app"]
    runner.uc_function = None
    monkeypatch.setattr(
        runner,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "PERMISSION_DENIED"),
    )

    runner.cleanup()

    assert len(runner.cleanup_checks) == 1
    check = runner.cleanup_checks[0]
    assert check.resource_kind == "app"
    assert check.resource == "undeletable-app"
    assert check.status == "fail"
    assert check.return_code == 1
    assert "PERMISSION_DENIED" in (check.error or "")


@pytest.mark.parametrize("mutation", ["missing", "failed"])
def test_verify_evidence_requires_successful_cleanup_proof(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, mutation: str
):
    document = _complete_evidence()
    if mutation == "missing":
        document["cleanup"].pop()
    else:
        document["cleanup"][0]["status"] = "fail"
        document["cleanup"][0]["return_code"] = 1
    evidence = _write_evidence(tmp_path, document)

    assert tool_matrix.verify_evidence(evidence) == 1


@pytest.mark.parametrize("resource_kind", ["app", "uc_function"])
def test_verify_evidence_requires_cleanup_identity_for_every_created_resource(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, resource_kind: str
):
    document = _complete_evidence()
    if resource_kind == "app":
        missing_app = "mason-tools-langgraph-di-test"
        for row in document["rows"]:
            if row.get("app_name") == missing_app:
                row["app_name"] = None
        document["cleanup"] = [
            check for check in document["cleanup"] if check["resource"] != missing_app
        ]
    else:
        document["uc_function"] = None
        document["cleanup"] = [
            check for check in document["cleanup"] if check["resource_kind"] != "uc_function"
        ]

    evidence = _write_evidence(tmp_path, document)

    assert tool_matrix.verify_evidence(evidence) == 1


def test_main_persists_cleanup_proof_before_returning_zero(
    tmp_path: pathlib.Path, tool_matrix: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.output = tmp_path
            self.cleaned = False
            self.transcript = SimpleNamespace(write=lambda message: events.append(message))

        def bootstrap(self):
            pass

        def select_warehouse(self, override):
            pass

        def create_uc_function(self, schema):
            pass

        def create_projects(self):
            return []

        def _write_evidence(self):
            events.append("write")
            document = _complete_evidence()
            if not self.cleaned:
                document.pop("cleanup")
            _write_evidence(tmp_path, document)

        def cleanup(self):
            events.append("cleanup")
            self.cleaned = True

    args = SimpleNamespace(
        verify_evidence=None,
        profile="profile",
        output=tmp_path,
        wheel=tmp_path / "mason.whl",
        template_repo=None,
        template_ref=None,
        app_auth_profile=None,
        warehouse_id=None,
        uc_schema="main.mason_agent_tools_e2e",
        keep_resources=False,
    )
    monkeypatch.setattr(tool_matrix, "parse_args", lambda: args)
    monkeypatch.setattr(tool_matrix, "Runner", FakeRunner)

    exit_code = tool_matrix.main()

    assert exit_code == 0
    assert json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))["cleanup"]
    assert events[-2:] == ["cleanup", "write"]
