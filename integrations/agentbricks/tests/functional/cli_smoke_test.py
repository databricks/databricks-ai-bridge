"""Functional smoke tests: the REAL installed `agentbricks` binary runs each workspace-free command.

These complement the in-process CliRunner unit tests, which already own detailed behavior (arg
handling, help text, agent.toml effects). CliRunner imports the command modules in-process, so it
can't catch a regression that only shows up under the real console script — a broken entry point, a
packaging/dependency gap, an import that fails only when installed. This runs the `agentbricks` on PATH,
next to the interpreter running the tests: CI installs the built wheel and runs pytest from that
venv (so CI exercises the shipped artifact), while a local `uv run` provides the editable install.
Everything runs with an isolated HOME and no Databricks config, so no workspace is used.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import requires

import pytest
import tomli
from packaging.requirements import Requirement

# Every top-level command; `<group> --help` proves each module imports and wires up as installed.
_COMMANDS = (
    "login",
    "logout",
    "init",
    "doctor",
    "dev",
    "memory",
    "sessions",
    "tracing",
    "deploy",
    "deployments",
    "endpoint",
    "tools",
    "experimental",
)


@pytest.fixture
def run_agentbricks(tmp_path: pathlib.Path):
    agentbricks = pathlib.Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        if os.environ.get("AGENTBRICKS_WHEEL"):
            pytest.fail(
                f"wheel-backed functional tests require the installed script at {agentbricks}"
            )
        pytest.skip("requires the agentbricks CLI on PATH")
    home = tmp_path / "home"
    home.mkdir()
    empty_cfg = tmp_path / "empty.databrickscfg"
    empty_cfg.write_text("")
    # Isolate HOME so login/logout can't touch the real ~/.agentbricks; empty config file so no real
    # profile is reachable. env= replaces the environment wholesale (no ambient DATABRICKS_* leaks).
    env = {
        "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
        "HOME": str(home),
        "DATABRICKS_CONFIG_FILE": str(empty_cfg),
    }

    def run(
        *args: str, check: bool = True, env_override: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [str(agentbricks), *args],
            cwd=tmp_path,
            env={**env, **(env_override or {})},
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check:
            assert result.returncode == 0, (
                f"`agentbricks {' '.join(args)}` exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result

    return run


@pytest.mark.parametrize("profile_source", ["saved", "sdk"])
@pytest.mark.parametrize("command", ["memory-list", "slow-init", "default-init"])
def test_cli_posts_telemetry_to_local_receiver(
    run_agentbricks, tmp_path, profile_source, command
) -> None:
    uploads: queue.Queue[tuple[str, dict, str | None]] = queue.Queue()
    reads: queue.Queue[str] = queue.Queue()

    class _TelemetryReceiver(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _respond_json(self, value: dict) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/2.0/agents/memory-stores"):
                reads.put(self.path)
                self._respond_json({"managed_memory_stores": []})
            else:
                self._respond_json({})

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            if command == "slow-init":
                time.sleep(0.1)
            self._respond_json({})
            uploads.put((self.path, body, self.headers.get("Authorization")))

    with ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryReceiver) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            profile_name = "telemetry-smoke"
            config_file = tmp_path / ".databrickscfg"
            config_file.write_text(
                f"[{profile_name}]\nhost = http://127.0.0.1:{server.server_port}\n"
                "token = local-test-token\n"
            )
            agentbricks_home = tmp_path / "agentbricks-home"
            agentbricks_home.mkdir()
            env_override = {
                "DATABRICKS_CONFIG_FILE": str(config_file),
                "AGENTBRICKS_CONFIG_HOME": str(agentbricks_home),
            }
            if profile_source == "saved":
                (agentbricks_home / "config.json").write_text(json.dumps({"profile": profile_name}))
            else:
                env_override["DATABRICKS_CONFIG_PROFILE"] = profile_name

            if command == "slow-init":
                instrumentation = tmp_path / "slow-client"
                instrumentation.mkdir()
                (instrumentation / "sitecustomize.py").write_text(
                    "import time\n"
                    "from databricks_agentkit import _api_client\n"
                    "original = _api_client._workspace_client\n"
                    "def delayed_client(profile):\n"
                    "    time.sleep(1.15)\n"
                    "    return original(profile)\n"
                    "_api_client._workspace_client = delayed_client\n"
                )
                env_override["PYTHONPATH"] = str(instrumentation)
                arguments = [
                    "init",
                    str(tmp_path / "agent"),
                    "--framework",
                    "openai",
                    "--server",
                    "custom",
                ]
            elif command == "default-init":
                arguments = ["init", str(tmp_path / "agent")]
            else:
                arguments = ["memory", "stores", "list", "--page-size", "7"]
            result = run_agentbricks("-o", "json", *arguments, env_override=env_override)
            assert result.stdout
            if command == "memory-list":
                assert reads.get(timeout=5).startswith("/api/2.0/agents/memory-stores")
            path, body, authorization = uploads.get(timeout=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert path == "/telemetry-ext"
    assert authorization == "Bearer local-test-token"
    assert isinstance(body["uploadTime"], int)
    assert body["items"] == []
    assert len(body["protoLogs"]) == 1
    frontend_log = json.loads(body["protoLogs"][0])
    assert uuid.UUID(frontend_log["frontend_log_event_id"])
    log = frontend_log["entry"]["agentbricks_cli_log"]
    assert log["command_path"] == (
        "agentbricks memory stores list" if command == "memory-list" else "agentbricks init"
    )
    assert log["exit_code"] == 0
    expected_parameters: dict[str, dict[str, str | bool | int]] = {
        "agentbricks.output": {
            "name": "agentbricks.output",
            "choice_value": "AGENTBRICKS_CLI_JSON",
        },
    }
    if command != "memory-list":
        expected_parameters.update(
            {
                "agentbricks init.framework": {
                    "name": "agentbricks init.framework",
                    "choice_value": (
                        "AGENTBRICKS_CLI_OPENAI"
                        if command == "slow-init"
                        else "AGENTBRICKS_CLI_LANGGRAPH"
                    ),
                },
                "agentbricks init.server": {
                    "name": "agentbricks init.server",
                    "choice_value": (
                        "AGENTBRICKS_CLI_CUSTOM"
                        if command == "slow-init"
                        else "AGENTBRICKS_CLI_AGENTBRICKS"
                    ),
                },
                "agentbricks init.existing": {
                    "name": "agentbricks init.existing",
                    "bool_value": False,
                },
                "agentbricks init.disable_chat_app": {
                    "name": "agentbricks init.disable_chat_app",
                    "bool_value": False,
                },
                "agentbricks init.enable_chat_app": {
                    "name": "agentbricks init.enable_chat_app",
                    "bool_value": False,
                },
            }
        )
    else:
        expected_parameters["agentbricks memory stores list.all_"] = {
            "name": "agentbricks memory stores list.all_",
            "bool_value": False,
        }
        expected_parameters["agentbricks memory stores list.page_size"] = {
            "name": "agentbricks memory stores list.page_size",
            "int_value": 7,
        }
    assert {parameter["name"]: parameter for parameter in log["parameters"]} == expected_parameters


def test_cli_and_every_command_help_load(run_agentbricks) -> None:
    assert "Usage" in run_agentbricks("--help").stdout
    for command in _COMMANDS:
        run_agentbricks(command, "--help")


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_model_and_prompt_bindings_from_installed_cli(run_agentbricks, tmp_path, framework):
    """Exercise local binding commands through the shipped console script without auth."""
    project = tmp_path / "bound-agent"
    run_agentbricks("init", "--framework", framework, str(project))
    for role in ("router", "writer"):
        result = run_agentbricks(
            "-o",
            "json",
            "experimental",
            "models",
            "bind",
            f"main.test.{role}",
            "--role",
            role,
            "--default",
            "system.ai.claude-sonnet-4-5",
            "--source",
            str(project),
        )
        assert json.loads(result.stdout)["role"] == role
    run_agentbricks(
        "-o",
        "json",
        "experimental",
        "prompts",
        "bind",
        "main.test.writer_prompt",
        "--key",
        "writer",
        "--source",
        str(project),
    )
    manifest = tomli.loads((project / "agent.toml").read_text())
    assert set(manifest["experimental"]["model_services"]) == {"router", "writer"}
    assert manifest["experimental"]["prompts"] == {"writer": "main.test.writer_prompt"}
    for role in ("router", "writer"):
        run_agentbricks(
            "experimental",
            "models",
            "unbind",
            "--role",
            role,
            "--source",
            str(project),
        )
    run_agentbricks("experimental", "prompts", "unbind", "writer", "--source", str(project))
    manifest = tomli.loads((project / "agent.toml").read_text())
    assert "experimental" not in manifest
    assert manifest["agent"]["framework"] == framework


def test_installed_wheel_exposes_agentkit_sdk() -> None:
    from databricks_agentkit import AgentKitClient, MemoryStore, SessionStore

    assert AgentKitClient.__module__ == "databricks_agentkit.client"
    assert MemoryStore.__module__ == "databricks_agentkit.memory_store"
    assert SessionStore.__module__ == "databricks_agentkit.session_store"


@pytest.mark.parametrize(
    "extra",
    [
        ["--framework", "langgraph"],
        ["--framework", "openai"],
        ["--framework", "langgraph", "--server", "custom"],
    ],
)
def test_init_scaffolds(run_agentbricks, tmp_path: pathlib.Path, extra) -> None:
    dest = tmp_path / ("proj_" + "_".join(token.lstrip("-") for token in extra))
    run_agentbricks("init", *extra, str(dest))
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "app.yaml").is_file()
    assert (dest / "agent.toml").is_file()
    with (dest / "pyproject.toml").open("rb") as project_file:
        dependencies = tomli.load(project_file)["project"]["dependencies"]
    if "langgraph" in extra:
        assert "databricks-langchain==0.21.0" in dependencies
    else:
        assert not any(Requirement(dep).name == "databricks-langchain" for dep in dependencies)


def test_installed_langgraph_extra_pins_fixed_provider() -> None:
    requirements = [Requirement(dep) for dep in requires("databricks-agentbricks") or []]
    provider = next(dep for dep in requirements if dep.name == "databricks-langchain")
    assert str(provider.specifier) == "==0.21.0"
    assert provider.marker is not None
    assert provider.marker.evaluate({"extra": "langgraph"})
    assert not provider.marker.evaluate({"extra": "openai"})
    assert not provider.marker.evaluate({"extra": ""})


def test_existing_init_includes_migration_skill_from_wheel(
    run_agentbricks, tmp_path: pathlib.Path
) -> None:
    project = tmp_path / "existing-agent"
    project.mkdir()

    run_agentbricks("init", "--existing", "--framework", "langgraph", str(project))

    skill = project / "agent-bricks-migrate/SKILL.md"
    assert "name: agent-bricks-migrate" in skill.read_text()
    pointer = project / ".claude/skills/agent-bricks-migrate/SKILL.md"
    assert pointer.is_file()
    assert "../../../agent-bricks-migrate/SKILL.md" in pointer.read_text()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("output", ["text", "json"])
@pytest.mark.parametrize(
    "args",
    [
        ["sandbox", "--scope", "volume:catalog.schema.volume"],
        ["uc-function", "catalog.schema.function"],
        ["uc-function", "catalog.schema.function", "--auth", "user"],
        ["uc-function", "catalog.schema.function", "--auth", "app"],
        ["genie-one"],
        ["genie-agent", "0" * 32],
    ],
)
def test_tools_offline_add_review_manifest_remove(
    run_agentbricks, tmp_path, framework, output, args
) -> None:
    project = tmp_path / "agent"
    run_agentbricks("init", "--framework", framework, str(project))
    manifest = project / "agent.toml"
    original = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file() and path != manifest
    }
    for changed in (True, False):
        added = run_agentbricks(
            "--output", output, "tools", "add", *args, "--name", "tested", "--source", str(project)
        )
        if output == "json":
            payload = json.loads(added.stdout)
            assert payload["manifest"] == str(manifest)
            assert payload["changed"] is changed
            assert payload["changed_files"] == ([str(manifest)] if changed else [])
        else:
            assert f"Review {manifest}" in added.stdout
            assert "configured managed tools and MCP bindings" in added.stdout
            if not changed:
                assert "already configured" in added.stdout
        assert [tool["id"] for tool in tomli.loads(manifest.read_text())["tools"]] == ["tested"]
        expected_auth = args[args.index("--auth") + 1] if "--auth" in args else "user"
        assert tomli.loads(manifest.read_text())["tools"][0]["auth"] == expected_auth
    run_agentbricks("tools", "remove", "tested", "--source", str(project))
    assert tomli.loads(manifest.read_text()).get("tools", []) == []
    assert {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file() and path != manifest
    } == original


@pytest.mark.parametrize("kind", ["sandbox", "uc-function", "genie-one", "genie-agent"])
def test_tools_local_discovery_without_project_or_auth(run_agentbricks, kind):
    result = run_agentbricks("-o", "json", "tools", "list", "--kind", kind)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["complete"] is True
    assert payload["mcp_schema"] is None
    assert payload["errors"] == []
    assert [tool["kind"] for tool in payload["available_tools"]] == [kind]


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ("table:main.data.rows", "Table sandbox scopes are not supported"),
        ("workspace:/Workspace/Shared", "Workspace sandbox scopes are not supported"),
    ],
)
def test_sandbox_unsupported_scope_is_rejected_without_manifest_change(
    run_agentbricks, tmp_path, framework, scope, message
):
    project = tmp_path / "agent"
    run_agentbricks("init", "--framework", framework, str(project))
    manifest = project / "agent.toml"
    before = manifest.read_bytes()

    result = run_agentbricks(
        "tools",
        "add",
        "sandbox",
        "--scope",
        scope,
        "--source",
        str(project),
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert manifest.read_bytes() == before


def test_tools_text_discovery_suggests_mcp_filters(run_agentbricks):
    result = run_agentbricks("tools", "list", "--kind", "genie-one")
    assert "agentbricks tools list --kind mcp" in result.stdout
    assert "agentbricks tools list --kind mcp --schema catalog.schema" in result.stdout


def test_tools_discovery_without_auth_is_explicitly_incomplete(run_agentbricks):
    result = run_agentbricks("-o", "json", "tools", "list", check=False)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert payload["mcp_schema"] == "system.ai"
    assert [tool["kind"] for tool in payload["available_tools"]] == [
        "sandbox",
        "uc-function",
        "genie-one",
        "genie-agent",
    ]
    assert payload["errors"]


def test_tools_help_and_removed_configured_route(run_agentbricks):
    for path in [
        ("tools",),
        ("tools", "list"),
        ("tools", "add"),
        ("tools", "add", "sandbox"),
        ("tools", "add", "mcp"),
        ("tools", "add", "uc-function"),
        ("tools", "add", "genie-one"),
        ("tools", "add", "genie-agent"),
    ]:
        text = " ".join(run_agentbricks(*path, "--help").stdout.split())
        assert "agent.toml" in text
        assert "list configured tools" not in text
        assert "agentbricks mcp list" not in text
    listing_help = " ".join(run_agentbricks("tools", "list", "--help").stdout.split())
    for expected in (
        "--kind",
        "--schema",
        "system.ai",
        "catalog.schema",
        "genie-one",
        "genie-agent",
        "No agent project",
    ):
        assert expected in listing_help
    assert "--source DIRECTORY" not in listing_help
    assert run_agentbricks("tools", "list", "--source", ".", check=False).returncode != 0
    assert run_agentbricks("tools", "mcp", "list", check=False).returncode != 0
    assert run_agentbricks("mcp", "--help", check=False).returncode != 0


def test_mcp_add_without_auth_does_not_change_manifest(
    run_agentbricks, tmp_path: pathlib.Path
) -> None:
    project = tmp_path / "agent"
    run_agentbricks("init", "--framework", "langgraph", str(project))
    manifest = project / "agent.toml"
    before = manifest.read_bytes()

    result = run_agentbricks(
        "--output",
        "json",
        "tools",
        "add",
        "mcp",
        "system.ai.web_search",
        "--source",
        str(project),
        check=False,
    )

    assert result.returncode == 1
    assert "Could not initialize Databricks auth" in result.stderr
    assert manifest.read_bytes() == before


def test_tracing_unbind_then_bind_requires_an_experiment(
    run_agentbricks, tmp_path: pathlib.Path
) -> None:
    project = tmp_path / "agent"
    run_agentbricks("init", "--framework", "langgraph", str(project))
    # unbind removes the default binding `agentbricks init` wrote - a pure agent.toml edit, no workspace.
    run_agentbricks("tracing", "unbind", "--source", str(project))
    # bind now requires an experiment (the old no-arg "re-enable the default" is gone); with neither
    # flag it errors clearly - and does so before any workspace call, so this stays hermetic.
    result = run_agentbricks("tracing", "bind", "--source", str(project), check=False)
    assert result.returncode != 0
    assert "Pass --experiment-name or --experiment-id" in result.stderr


def test_logout_runs_cleanly(run_agentbricks) -> None:
    # Isolated HOME: there's no saved selection to forget, but it must still exit cleanly.
    run_agentbricks("logout")
