"""Static, offline onboarding checks for ``mason doctor``."""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

import databricks_mason.cli.doctor as doctor_module
from databricks_mason.cli.app import mason

_SOURCE_WIRING = (
    "from databricks_mason import AgentApp\n"
    "from databricks_mason.langgraph import checkpointer\n"
    "app = AgentApp()\n"
    "app.invoke(lambda value, context: None)\n"
    "state = checkpointer()\n"
)


def _python_evidence(source: str):
    evidence, error = doctor_module._python_evidence(source.encode())
    assert error is None
    assert evidence is not None
    return evidence


def _write_source_wiring(path: pathlib.Path) -> bytes:
    content = _SOURCE_WIRING.encode()
    path.write_bytes(content)
    return content


def _write_onboarded_project(root: pathlib.Path, framework: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".mason").mkdir()
    (root / "agent.toml").write_text(
        f'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "mason"\n',
        encoding="utf-8",
    )
    (root / ".mason" / "project.toml").write_text(
        f'schema_version = 1\nframework = "{framework}"\ntemplate = "agent-{framework}"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "existing-agent"\n'
        f'dependencies = ["databricks-mason[{framework}]>=0.2"]\n',
        encoding="utf-8",
    )
    (root / "app.yaml").write_text('command: ["uv", "run", "start-server"]\n', encoding="utf-8")
    adapter = "checkpointer" if framework == "langgraph" else "session_store"
    (root / "runtime").mkdir()
    (root / "runtime" / "main.py").write_text(
        "from databricks_mason import AgentApp\n"
        "\n"
        "def invoke(value, context):\n"
        "    return None\n"
        "\n"
        "app = AgentApp()\n"
        "app.invoke(invoke)\n",
        encoding="utf-8",
    )
    (root / "agent").mkdir()
    (root / "agent" / "agent.py").write_text(
        f"from databricks_mason.{framework} import {adapter}\n\nstate = {adapter}()\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_fully_onboarded_framework_reports_success(tmp_path: pathlib.Path, framework: str):
    project = tmp_path / framework
    _write_onboarded_project(project, framework)

    result = CliRunner().invoke(mason, ["doctor", str(project)])

    assert result.exit_code == 0, result.output
    assert f"Framework: {framework}" in result.output
    assert "7/7 checks passed — onboarded" in result.output
    assert "[fail]" not in result.output


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_fresh_generated_project_reports_success(tmp_path: pathlib.Path, framework: str):
    project = tmp_path / framework
    init_result = CliRunner().invoke(
        mason,
        ["init", "--framework", framework, "--disable-chat-app", str(project)],
    )

    assert init_result.exit_code == 0, init_result.output
    result = CliRunner().invoke(mason, ["doctor", str(project)])
    assert result.exit_code == 0, result.output
    assert "7/7 checks passed — onboarded" in result.output


def test_plain_repository_is_a_normal_failed_report(tmp_path: pathlib.Path):
    result = CliRunner().invoke(mason, ["doctor", str(tmp_path)])

    assert result.exit_code == 1
    assert result.exception is not None
    assert "Framework: unknown" in result.output
    assert "[fail] agent_manifest:" in result.output
    assert "not onboarded" in result.output
    assert "mason init --framework <langgraph|openai> --existing <directory>" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("relative_path", "body", "check_id"),
    [
        ("agent.toml", "[agent\n", "agent_manifest"),
        (".mason/project.toml", "schema_version = [\n", "project_metadata"),
    ],
)
def test_malformed_evidence_becomes_a_failed_check(
    tmp_path: pathlib.Path, relative_path: str, body: str, check_id: str
):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / relative_path).write_text(body, encoding="utf-8")

    result = CliRunner().invoke(mason, ["doctor", str(tmp_path)])

    assert result.exit_code == 1
    assert f"[fail] {check_id}: Could not parse" in result.output
    assert "Traceback" not in result.output


def test_semantically_invalid_manifest_becomes_a_failed_check(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    with (tmp_path / "agent.toml").open("a", encoding="utf-8") as manifest:
        manifest.write(
            "\n[[tools]]\n"
            'id = "duplicate"\n'
            'source = { kind = "mcp", service = "system.ai.first" }\n'
            "\n[[tools]]\n"
            'id = "duplicate"\n'
            'source = { kind = "mcp", service = "system.ai.second" }\n'
        )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    report = json.loads(result.output)
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["agent_manifest"]["status"] == "fail"
    assert "tool ids must be unique" in checks["agent_manifest"]["detail"]
    # The independently readable server field remains useful even when another manifest section is
    # invalid.
    assert checks["mason_server"]["status"] == "pass"


def test_json_report_has_stable_shape_and_failure_exit_code(tmp_path: pathlib.Path):
    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report == {
        "directory": str(tmp_path.resolve()),
        "framework": None,
        "onboarded": False,
        "checks": report["checks"],
        "summary": {"total": 7, "passed": 0, "failed": 7},
        "next_step": "mason init --framework <langgraph|openai> --existing <directory>",
    }
    assert [check["id"] for check in report["checks"]] == [
        "agent_manifest",
        "mason_server",
        "project_metadata",
        "mason_dependency",
        "app_command",
        "agent_app",
        "framework_adapter",
    ]
    assert all(set(check) == {"id", "status", "detail"} for check in report["checks"])
    assert {check["status"] for check in report["checks"]} == {"fail"}


def test_directory_defaults_to_current_directory(tmp_path: pathlib.Path, monkeypatch):
    _write_onboarded_project(tmp_path, "openai")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(mason, ["--output", "json", "doctor"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["directory"] == str(tmp_path.resolve())
    assert report["framework"] == "openai"
    assert report["onboarded"] is True


def test_source_wiring_aliases_are_supported(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / "runtime" / "main.py").write_text(
        "import databricks_mason as mason_runtime\n"
        "\n"
        "def handler(value, context):\n"
        "    return None\n"
        "\n"
        "application = mason_runtime.AgentApp()\n"
        "application.invoke(handler)\n",
        encoding="utf-8",
    )
    (tmp_path / "agent" / "agent.py").write_text(
        "from databricks_mason.langgraph import checkpointer as mason_checkpointer\n"
        "\n"
        "state = mason_checkpointer()\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/wiring.py",
        "wiring_test.py",
        "example/wiring.py",
        "examples/wiring.py",
        "old/wiring.py",
        "stale/wiring.py",
    ],
)
def test_test_only_source_evidence_is_ignored(tmp_path: pathlib.Path, relative_path: str):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / "runtime" / "main.py").unlink()
    (tmp_path / "agent" / "agent.py").unlink()
    evidence_path = tmp_path / relative_path
    evidence_path.parent.mkdir(exist_ok=True)
    evidence_path.write_text(
        "from databricks_mason import AgentApp\n"
        "from databricks_mason.langgraph import checkpointer\n"
        "\n"
        "app = AgentApp()\n"
        "app.invoke(lambda value, context: None)\n"
        "state = checkpointer()\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["agent_app"]["status"] == "fail"
    assert checks["framework_adapter"]["status"] == "fail"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from databricks_mason import AgentApp\napp = AgentApp()\napp.invoke(handler)\n",
            True,
        ),
        (
            "from databricks_mason import AgentApp\n"
            "app = object()\n"
            "app.invoke(handler)\n"
            "app = AgentApp()\n",
            False,
        ),
        (
            "from databricks_mason import AgentApp\n"
            "app = AgentApp()\n"
            "app.invoke(handler)\n"
            "app = object()\n",
            True,
        ),
        (
            "from databricks_mason import AgentApp\nAgentApp().invoke(handler)\n",
            False,
        ),
        (
            "from databricks_mason import AgentApp\n"
            "app = AgentApp()\n"
            "@app.invoke\n"
            "def handler(value, context):\n"
            "    return None\n",
            True,
        ),
        (
            "from databricks_mason.runtime import AgentApp as RuntimeApp\n"
            "app = RuntimeApp()\n"
            "app.invoke(handler)\n",
            True,
        ),
    ],
)
def test_agent_app_registration_uses_source_order_and_bound_instances(source: str, expected: bool):
    assert _python_evidence(source).agent_app_invoked is expected


@pytest.mark.parametrize(
    "source",
    [
        (
            "from databricks_mason import AgentApp\n"
            "def wire(AgentApp):\n"
            "    app = AgentApp()\n"
            "    app.invoke(handler)\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "def wire():\n"
            "    app = AgentApp()\n"
            "    app.invoke(handler)\n"
            "    from fake_runtime import AgentApp\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "AgentApp = lambda: object()\n"
            "app = AgentApp()\n"
            "app.invoke(handler)\n"
        ),
    ],
)
def test_agent_app_import_is_not_used_after_lexical_shadowing(source: str):
    assert _python_evidence(source).agent_app_invoked is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "adapter()\n"
            "adapter = lambda: None\n",
            True,
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "adapter = lambda: None\n"
            "adapter()\n",
            False,
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "def adapter():\n"
            "    return None\n"
            "adapter()\n",
            False,
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "def wire(adapter):\n"
            "    adapter()\n",
            False,
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "def wire():\n"
            "    adapter()\n"
            "    from fake_adapter import adapter\n",
            False,
        ),
    ],
)
def test_adapter_alias_counts_only_while_it_resolves_to_supported_symbol(
    source: str, expected: bool
):
    evidence = _python_evidence(source)
    assert bool(evidence.adapters) is expected


@pytest.mark.parametrize(
    "source",
    [
        (
            "from databricks_mason import AgentApp\n"
            "for AgentApp in factories:\n"
            "    app = AgentApp()\n"
            "    app.invoke(handler)\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "with manager() as AgentApp:\n"
            "    app = AgentApp()\n"
            "    app.invoke(handler)\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "app = AgentApp()\n"
            "registrations = [app.invoke(handler) for app in applications]\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "app = AgentApp()\n"
            "match payload:\n"
            '    case {"app": app}:\n'
            "        app.invoke(handler)\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "async def wire():\n"
            "    async for AgentApp in factories:\n"
            "        app = AgentApp()\n"
            "        app.invoke(handler)\n"
        ),
        (
            "from databricks_mason import AgentApp\n"
            "async def wire():\n"
            "    async with manager() as AgentApp:\n"
            "        app = AgentApp()\n"
            "        app.invoke(handler)\n"
        ),
    ],
)
def test_agent_app_is_shadowed_by_control_flow_bindings(source: str):
    assert _python_evidence(source).agent_app_invoked is False


@pytest.mark.parametrize(
    "source",
    [
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "for adapter in adapters:\n"
            "    adapter()\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "with manager() as adapter:\n"
            "    adapter()\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "values = [adapter() for adapter in adapters]\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "match payload:\n"
            '    case {"adapter": adapter}:\n'
            "        adapter()\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "async def wire():\n"
            "    async for adapter in adapters:\n"
            "        adapter()\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "async def wire():\n"
            "    async with manager() as adapter:\n"
            "        adapter()\n"
        ),
    ],
)
def test_adapter_is_shadowed_by_control_flow_bindings(source: str):
    assert not _python_evidence(source).adapters


@pytest.mark.parametrize(
    "source",
    [
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "for adapter in adapter():\n"
            "    pass\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "with adapter() as adapter:\n"
            "    pass\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "values = [item for item in adapter()]\n"
        ),
        (
            "from databricks_mason.langgraph import checkpointer as adapter\n"
            "match payload:\n"
            "    case _:\n"
            "        adapter()\n"
        ),
    ],
)
def test_adapter_calls_before_or_without_control_flow_rebinding_still_count(source: str):
    assert _python_evidence(source).adapters


def test_bound_agent_app_survives_unrelated_comprehension_target():
    evidence = _python_evidence(
        "from databricks_mason import AgentApp\n"
        "app = AgentApp()\n"
        "registrations = [app.invoke(handler) for item in items]\n"
    )

    assert evidence.agent_app_invoked is True


def test_agent_app_without_invoke_registration_fails(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "openai")
    (tmp_path / "runtime" / "main.py").write_text(
        "from databricks_mason import AgentApp\n\napp = AgentApp()\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["agent_app"]["status"] == "fail"
    assert checks["framework_adapter"]["status"] == "pass"


def test_nonexistent_mason_symbols_do_not_count_as_source_wiring(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / "runtime" / "main.py").write_text(
        "from databricks_mason.not_real import AgentApp\n"
        "\n"
        "app = AgentApp()\n"
        "app.invoke(lambda value, context: None)\n",
        encoding="utf-8",
    )
    (tmp_path / "agent" / "agent.py").write_text(
        "from databricks_mason.langgraph import imaginary_adapter\n\nstate = imaginary_adapter()\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["agent_app"]["status"] == "fail"
    assert checks["framework_adapter"]["status"] == "fail"


def test_source_scan_budget_makes_source_checks_incomplete(tmp_path: pathlib.Path, monkeypatch):
    _write_onboarded_project(tmp_path, "langgraph")
    monkeypatch.setattr(doctor_module, "_MAX_SCAN_ENTRIES", 1)

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["agent_app"]["status"] == "fail"
    assert checks["framework_adapter"]["status"] == "fail"
    assert "exceeded its limit" in checks["agent_app"]["detail"]
    assert "exceeded its limit" in checks["framework_adapter"]["detail"]


def test_directory_budget_processes_exact_limit_and_fails_on_next(
    tmp_path: pathlib.Path, monkeypatch
):
    _write_source_wiring(tmp_path / "main.py")
    monkeypatch.setattr(doctor_module, "_MAX_SCAN_DIRECTORIES", 1)

    exact = doctor_module._scan_sources(tmp_path)
    (tmp_path / "nested").mkdir()
    overflow = doctor_module._scan_sources(tmp_path)

    assert exact.truncated is False
    assert exact.agent_app == pathlib.Path("main.py")
    assert overflow.truncated is True


def test_entry_budget_processes_exact_limit_and_fails_on_next(tmp_path: pathlib.Path, monkeypatch):
    _write_source_wiring(tmp_path / "main.py")
    monkeypatch.setattr(doctor_module, "_MAX_SCAN_ENTRIES", 1)

    exact = doctor_module._scan_sources(tmp_path)
    (tmp_path / "README.md").write_text("extra", encoding="utf-8")
    overflow = doctor_module._scan_sources(tmp_path)

    assert exact.truncated is False
    assert exact.agent_app == pathlib.Path("main.py")
    assert overflow.truncated is True


def test_source_file_budget_processes_exact_limit_and_fails_on_next(
    tmp_path: pathlib.Path, monkeypatch
):
    _write_source_wiring(tmp_path / "main.py")
    monkeypatch.setattr(doctor_module, "_MAX_SOURCE_FILES", 1)

    exact = doctor_module._scan_sources(tmp_path)
    (tmp_path / "other.py").write_text("value = 1\n", encoding="utf-8")
    overflow = doctor_module._scan_sources(tmp_path)

    assert exact.truncated is False
    assert exact.agent_app == pathlib.Path("main.py")
    assert overflow.truncated is True
    assert all(
        check.status == "fail" for check in doctor_module._source_checks(overflow, "langgraph")
    )


def test_source_total_byte_budget_processes_exact_limit_and_fails_on_next(
    tmp_path: pathlib.Path, monkeypatch
):
    content = _write_source_wiring(tmp_path / "main.py")
    monkeypatch.setattr(doctor_module, "_MAX_SOURCE_TOTAL_BYTES", len(content))

    exact = doctor_module._scan_sources(tmp_path)
    (tmp_path / "other.py").write_text("value = 1\n", encoding="utf-8")
    overflow = doctor_module._scan_sources(tmp_path)

    assert exact.truncated is False
    assert exact.agent_app == pathlib.Path("main.py")
    assert overflow.truncated is True


def test_per_file_byte_budget_processes_exact_limit_and_fails_on_next_byte(
    tmp_path: pathlib.Path, monkeypatch
):
    source = tmp_path / "main.py"
    content = _write_source_wiring(source)
    monkeypatch.setattr(doctor_module, "_MAX_SOURCE_BYTES", len(content))

    exact = doctor_module._scan_sources(tmp_path)
    source.write_bytes(content + b"\n")
    overflow = doctor_module._scan_sources(tmp_path)

    assert exact.truncated is False
    assert exact.agent_app == pathlib.Path("main.py")
    assert overflow.truncated is True


def test_deep_expression_is_reported_as_unparseable_without_traceback(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    deep_expression = "value = " + "+".join(["1"] * 5_000) + "\n"
    (tmp_path / "agent" / "agent.py").write_text(deep_expression, encoding="utf-8")

    result = CliRunner().invoke(mason, ["doctor", str(tmp_path)])

    assert result.exit_code == 1
    assert "Python file(s) did not parse" in result.output
    assert "Traceback" not in result.output


def test_deploy_generated_app_command_placeholder_fails(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / "app.yaml").write_text(
        "command: [\"# TODO: set your run command, e.g. ['uvicorn', 'app:app']\"]\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["app_command"]["status"] == "fail"
    assert "placeholder" in checks["app_command"]["detail"]


@pytest.mark.parametrize(
    ("framework", "requirement"),
    [
        (
            "langgraph",
            "databricks.mason[LangGraph]>=0.2 ; python_version >= '3.10'",
        ),
        (
            "openai",
            "Databricks_Mason[OpenAI] @ https://example.invalid/mason.whl ; python_version < '1'",
        ),
    ],
)
def test_mason_dependency_supports_pep508_forms(
    tmp_path: pathlib.Path, framework: str, requirement: str
):
    _write_onboarded_project(tmp_path, framework)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "existing-agent"\ndependencies = [{json.dumps(requirement)}]\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["mason_dependency"]["status"] == "pass"


def test_malformed_mason_requirement_fails(tmp_path: pathlib.Path):
    _write_onboarded_project(tmp_path, "langgraph")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "existing-agent"\ndependencies = ["databricks-mason[langgraph"]\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    assert result.exit_code == 1
    checks = {check["id"]: check for check in json.loads(result.output)["checks"]}
    assert checks["mason_dependency"]["status"] == "fail"
    assert "invalid PEP 508 requirement" in checks["mason_dependency"]["detail"]


def test_unknown_framework_next_step_requires_explicit_choice(tmp_path: pathlib.Path):
    text_result = CliRunner().invoke(mason, ["doctor", str(tmp_path)])
    json_result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(tmp_path)])

    expected = "mason init --framework <langgraph|openai> --existing <directory>"
    assert text_result.exit_code == 1
    assert expected in text_result.output
    assert json_result.exit_code == 1
    assert json.loads(json_result.output)["next_step"] == expected


def test_text_output_sanitizes_and_truncates_repository_content(tmp_path: pathlib.Path):
    project = tmp_path / ("repo\n\x1b[31m" + ("a" * 160)) / ("b" * 100)
    _write_onboarded_project(project, "langgraph")
    (project / "app.yaml").write_text(
        'command: ["uv", "secret\\n\\u001b[31mcommand"]\n',
        encoding="utf-8",
    )

    text_result = CliRunner().invoke(mason, ["doctor", str(project)])
    json_result = CliRunner().invoke(mason, ["--output", "json", "doctor", str(project)])

    assert text_result.exit_code == 0, text_result.output
    assert "\x1b" not in text_result.output
    assert "secret" not in text_result.output
    assert str(project.resolve()) not in text_result.output
    assert r"\n\u001b" in text_result.output
    assert "…" in text_result.output
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["directory"] == str(project.resolve())


def test_doctor_is_registered_and_help_documents_directory():
    runner = CliRunner()

    root = runner.invoke(mason, ["--help"])
    command = runner.invoke(mason, ["doctor", "--help"])

    assert root.exit_code == 0, root.output
    assert "doctor" in root.output
    assert command.exit_code == 0, command.output
    assert "Usage: mason doctor [OPTIONS] [DIRECTORY]" in command.output
    assert "defaults to the current directory" in command.output
    assert "EXAMPLES" in command.output
    assert "mason doctor ." in command.output
