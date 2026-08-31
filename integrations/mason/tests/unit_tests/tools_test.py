"""Unit tests for manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest
from click.testing import CliRunner

from databricks_mason import tools as tools_mod
from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.project_config import write_project_metadata

tools = tools_mod.tools


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(tmp_path: pathlib.Path, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    write_project_metadata(project, framework=framework, template=f"agent-{framework}")
    AgentProject.create(project, framework=framework).write()
    return project


def _declare_python(project: pathlib.Path, tool_id: str, entrypoint: str) -> None:
    manifest = AgentProject.load(project)
    manifest.add_tool(ToolSpec.python(tool_id, entrypoint=entrypoint))
    manifest.write()


def test_add_help_only_offers_attachment_commands():
    result = CliRunner().invoke(tools, ["add", "--help"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "python" not in result.output
    for command in ("sandbox", "mcp", "uc-function"):
        assert command in result.output


def test_add_sandbox_only_updates_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    loaded = AgentProject.load(project)
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope[0].resource == "table:samples.nyctaxi.trips"
    assert (project / "agent" / "mcps.py").read_text(encoding="utf-8") == "ORIGINAL = True\n"


def test_generic_mcp_rejects_sandbox_scope(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--scope" in result.output
    assert AgentProject.load(project).tools == []


def test_add_mcp_and_uc_function_write_typed_manifest_records(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()

    mcp = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--name", "web", "--source", str(project)],
        obj=_Ctx(),
    )
    uc = runner.invoke(
        tools,
        [
            "add",
            "uc-function",
            "main.tools.lookup_ticket",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert mcp.exit_code == 0, mcp.output
    assert uc.exit_code == 0, uc.output
    loaded = AgentProject.load(project)
    assert [(tool.id, tool.source.kind) for tool in loaded.tools] == [
        ("web", "mcp"),
        ("lookup_ticket", "uc_function"),
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
        ["add", "mcp", "system.ai.web_search"],
        ["add", "uc-function", "main.tools.lookup_ticket"],
    ],
)
def test_add_rejects_framework_without_runtime_adapter(tmp_path: pathlib.Path, command: list[str]):
    project = _project(tmp_path, framework="openai")

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "supports only the 'langgraph' framework" in result.output
    assert AgentProject.load(project).tools == []


def test_add_is_idempotent_and_json_reports_changed_files(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    args = ["add", "mcp", "system.ai.web_search", "--source", str(project)]
    runner = CliRunner()

    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["changed"] is True
    assert first_payload["changed_files"] == [str(project / "agent.toml")]
    assert second_payload["changed"] is False
    assert second_payload["changed_files"] == []
    assert len(AgentProject.load(project).tools) == 1


def test_tools_list_emits_manifest_records_as_json(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tools"] == [
        {
            "id": "web_search",
            "kind": "mcp",
            "source": "system.ai.web_search",
        }
    ]


def test_discovery_finds_only_literal_top_level_decorators_and_subtracts_declared_tools(
    tmp_path: pathlib.Path,
):
    project = _project(tmp_path)
    _declare_python(project, "declared", "agent.tools.declared:declared")
    (project / "agent" / "tools" / "declared.py").write_text(
        "@tool\ndef declared():\n    return 'declared'\n", encoding="utf-8"
    )
    (project / "agent" / "tools" / "tickets.py").write_text(
        "raise AssertionError('discovery must not import this module')\n\n"
        "@tool\n"
        "def lookup_ticket():\n    return 'ticket'\n\n"
        "@tool()\n"
        "async def search_tickets():\n    return []\n\n"
        "@mason.tool\n"
        "def close_ticket():\n    return None\n\n"
        "alias = tool\n"
        "@alias\n"
        "def dynamic_alias():\n    return None\n\n"
        "def outer():\n"
        "    @tool\n"
        "    def nested():\n"
        "        return None\n",
        encoding="utf-8",
    )

    assert tools_mod.discover_python_tool_candidates(AgentProject.load(project)) == (
        "agent.tools.tickets:close_ticket",
        "agent.tools.tickets:lookup_ticket",
        "agent.tools.tickets:search_tickets",
    )


def test_tools_check_uses_result_file_and_attaches_child_logs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        result_path = pathlib.Path(command[command.index("--result-path") + 1])
        observed["command"] = command
        observed["result_path_existed"] = result_path.exists()
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "tools": [
                        {
                            "id": "lookup-ticket",
                            "entrypoint": "agent.tools.lookup:lookup_ticket",
                            "description": "Look up a ticket.",
                            "input_schema": {"type": "object"},
                            "fingerprint": "abc123",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return tools_mod._BoundedProcessResult(0, "import log\n", "debug log\n", False)

    monkeypatch.setattr(tools_mod, "_run_bounded_process", fake_run)

    result = CliRunner().invoke(
        tools,
        ["check", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "schema_version": 1,
        "ok": True,
        "tools": [
            {
                "id": "lookup-ticket",
                "entrypoint": "agent.tools.lookup:lookup_ticket",
                "description": "Look up a ticket.",
                "input_schema": {"type": "object"},
                "fingerprint": "abc123",
            }
        ],
        "warnings": [],
        "logs": {"stdout": "import log\n", "stderr": "debug log\n"},
    }
    observed_command = observed["command"]
    assert isinstance(observed_command, list)
    assert observed == {
        "command": [
            "uv",
            "run",
            "--project",
            str(project.resolve()),
            "python",
            "-m",
            "databricks_mason.python_runtime",
            "check",
            "lookup-ticket",
            "--result-path",
            observed_command[-1],
        ],
        "result_path_existed": True,
    }


def test_runtime_command_pins_selected_project_over_ambient_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project_path = _project(tmp_path)
    _declare_python(project_path, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
    project = AgentProject.load(project_path)
    ambient = tmp_path / "ambient-project"
    ambient.mkdir()
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(ambient))
    observed: dict[str, str] = {}

    def fake_run(command, **kwargs):
        observed["project_root"] = kwargs["env"]["MASON_PROJECT_ROOT"]
        result_path = pathlib.Path(command[command.index("--result-path") + 1])
        result_path.write_text(
            json.dumps({"schema_version": 1, "ok": True, "tools": []}), encoding="utf-8"
        )
        return tools_mod._BoundedProcessResult(0, "", "", False)

    monkeypatch.setattr(tools_mod, "_run_bounded_process", fake_run)

    payload = tools_mod._runtime_command(project, ["check", "lookup-ticket"])

    assert payload["ok"] is True
    assert observed == {"project_root": str(project.root)}


def test_bounded_process_times_out_kills_and_reaps_child(tmp_path: pathlib.Path):
    pid_path = tmp_path / "child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, time; "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
    ]
    started = time.monotonic()

    result = tools_mod._run_bounded_process(
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=0.5,
        log_limit_bytes=256,
    )

    assert result.timed_out is True
    assert time.monotonic() - started < 5
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_bounded_process_timeout_includes_inherited_child_pipes(tmp_path: pathlib.Path):
    grandchild_pid_path = tmp_path / "grandchild.pid"
    child_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('parent exited', flush=True)"
    )
    started = time.monotonic()

    result = tools_mod._run_bounded_process(
        [sys.executable, "-c", parent_code],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=0.5,
        log_limit_bytes=256,
    )

    assert result.timed_out is True
    assert time.monotonic() - started < 5
    assert "parent exited" in result.stdout


def test_bounded_process_drains_noisy_streams_and_marks_truncation(tmp_path: pathlib.Path):
    command = [
        sys.executable,
        "-c",
        "import sys; print('A' * 10000); print('B' * 12000, file=sys.stderr)",
    ]

    result = tools_mod._run_bounded_process(
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_limit_bytes=256,
    )

    assert result.returncode == 0
    assert result.timed_out is False
    assert "truncated" in result.stdout
    assert "truncated" in result.stderr
    assert len(result.stdout.encode()) < 512
    assert len(result.stderr.encode()) < 512


def test_runtime_command_surfaces_timeout_with_bounded_logs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project_path = _project(tmp_path)
    _declare_python(project_path, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
    project = AgentProject.load(project_path)
    timed_out = type(
        "TimedOut",
        (),
        {
            "returncode": -9,
            "stdout": "partial stdout",
            "stderr": "partial stderr",
            "timed_out": True,
        },
    )()
    monkeypatch.setattr(tools_mod, "_run_bounded_process", lambda *args, **kwargs: timed_out)

    payload = tools_mod._runtime_command(project, ["check", "lookup-ticket"])

    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    error = payload["error"]
    assert isinstance(error, str)
    assert "timed out" in error
    assert "terminated" in error
    assert payload["logs"] == {"stdout": "partial stdout", "stderr": "partial stderr"}


def test_tools_check_reports_broken_declared_entrypoint_as_a_hard_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    _declare_python(project, "broken", "agent.tools.missing:broken")

    def fake_run(command, **kwargs):
        result_path = pathlib.Path(command[command.index("--result-path") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": "Could not import module 'agent.tools.missing' for Python tool 'broken'.",
                }
            ),
            encoding="utf-8",
        )
        return tools_mod._BoundedProcessResult(1, "", "", False)

    monkeypatch.setattr(tools_mod, "_run_bounded_process", fake_run)

    result = CliRunner().invoke(tools, ["check", "broken", "--source", str(project)], obj=_Ctx())

    assert result.exit_code != 0
    assert "Could not import module 'agent.tools.missing'" in result.output


def test_tools_check_warns_for_undeclared_literal_tool_without_blocking(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    (project / "agent" / "tools" / "tickets.py").write_text(
        "@tool\ndef lookup_ticket():\n    return 'ticket'\n", encoding="utf-8"
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("an empty Python manifest must not spawn a runtime")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    runner = CliRunner()

    json_result = runner.invoke(tools, ["check", "--source", str(project)], obj=_Ctx(output="json"))
    text_result = runner.invoke(tools, ["check", "--source", str(project)], obj=_Ctx())

    warning = {
        "code": "MASON001",
        "entrypoint": "agent.tools.tickets:lookup_ticket",
        "message": (
            "agent.tools.tickets:lookup_ticket appears to be a decorated Python tool but is not "
            "active in agent.toml."
        ),
    }
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output) == {
        "schema_version": 1,
        "ok": True,
        "tools": [],
        "warnings": [warning],
        "logs": {},
    }
    assert text_result.exit_code == 0, text_result.output
    assert "MASON001" in text_result.output
    assert "agent.tools.tickets:lookup_ticket" in text_result.output


def test_tools_check_allows_an_empty_python_tool_set(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("an empty Python manifest must not spawn a runtime")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    result = CliRunner().invoke(tools, ["check", "--source", str(project)], obj=_Ctx(output="json"))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "schema_version": 1,
        "ok": True,
        "tools": [],
        "warnings": [],
        "logs": {},
    }


def test_tools_check_reports_syntax_errors_as_non_blocking_path_warnings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    (project / "agent" / "tools" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an empty Python manifest must not spawn a runtime")
        ),
    )

    result = CliRunner().invoke(tools, ["check", "--source", str(project)], obj=_Ctx(output="json"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["warnings"][0]["code"] == "MASON002"
    assert payload["warnings"][0]["path"] == "agent/tools/broken.py"


def test_discovery_does_not_follow_python_files_outside_the_project(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    external = tmp_path / "outside.py"
    external.write_text("@tool\ndef outside():\n    return None\n", encoding="utf-8")
    (project / "agent" / "tools" / "outside.py").symlink_to(external)

    assert tools_mod.discover_python_tool_candidates(AgentProject.load(project)) == ()


def test_tools_run_rejects_non_object_input_before_spawning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("invalid input must not spawn a runtime")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    result = CliRunner().invoke(
        tools,
        ["run", "lookup-ticket", "--input", "[]", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    assert "JSON object" in result.output


@pytest.mark.parametrize(
    ("input_json", "message"),
    [("{", "valid JSON"), ("[]", "JSON object")],
)
def test_tools_run_parent_input_errors_are_json_objects(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    input_json: str,
    message: str,
):
    project = _project(tmp_path)
    _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid input must not spawn a runtime")
        ),
    )

    result = CliRunner().invoke(
        tools,
        ["run", "lookup-ticket", "--input", input_json, "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert message in payload["error"]
    assert payload["logs"] == {}


@pytest.mark.parametrize("project_error", ["framework", "declaration"])
def test_tools_run_parent_project_errors_are_json_objects(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    project_error: str,
):
    framework = "openai" if project_error == "framework" else "langgraph"
    project = _project(tmp_path, framework=framework)
    if project_error == "framework":
        _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
        expected = "supports only the 'langgraph' framework"
    else:
        expected = "not declared in agent.toml"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent validation errors must not spawn a runtime")
        ),
    )

    result = CliRunner().invoke(
        tools,
        [
            "run",
            "lookup-ticket",
            "--input",
            "{}",
            "--source",
            str(project),
        ],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert expected in payload["error"]
    assert payload["logs"] == {}


def test_tools_run_reads_control_json_from_file_and_keeps_tool_output_as_logs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = _project(tmp_path)
    _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")

    def fake_run(command, **kwargs):
        result_path = pathlib.Path(command[command.index("--result-path") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "tool": "lookup-ticket",
                    "result": {"ticket_id": "INC-123"},
                }
            ),
            encoding="utf-8",
        )
        return tools_mod._BoundedProcessResult(0, "import log\ntool log\n", "trace log\n", False)

    monkeypatch.setattr(tools_mod, "_run_bounded_process", fake_run)

    result = CliRunner().invoke(
        tools,
        [
            "run",
            "lookup-ticket",
            "--input",
            '{"ticket_id":"INC-123"}',
            "--source",
            str(project),
        ],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "schema_version": 1,
        "ok": True,
        "tool": "lookup-ticket",
        "result": {"ticket_id": "INC-123"},
        "logs": {"stdout": "import log\ntool log\n", "stderr": "trace log\n"},
    }


@pytest.mark.parametrize(
    "command",
    [
        ["check"],
        ["run", "lookup-ticket", "--input", "{}"],
    ],
)
@pytest.mark.parametrize("failure", ["manifest", "framework"])
def test_tools_parent_project_errors_are_schema_v1_json_objects(
    tmp_path: pathlib.Path, command: list[str], failure: str
):
    project = _project(tmp_path, framework="openai" if failure == "framework" else "langgraph")
    if failure == "manifest":
        (project / "agent.toml").write_text("schema_version = [\n", encoding="utf-8")
        expected = "Could not read agent manifest"
    else:
        _declare_python(project, "lookup-ticket", "agent.tools.lookup:lookup_ticket")
        expected = "supports only the 'langgraph' framework"

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert expected in payload["error"]


def test_tools_check_manifest_error_remains_actionable_in_text(
    tmp_path: pathlib.Path,
):
    project = _project(tmp_path)
    (project / "agent.toml").write_text("schema_version = [\n", encoding="utf-8")

    result = CliRunner().invoke(tools, ["check", "--source", str(project)], obj=_Ctx())

    assert result.exit_code != 0
    assert "Could not read agent manifest" in result.output
