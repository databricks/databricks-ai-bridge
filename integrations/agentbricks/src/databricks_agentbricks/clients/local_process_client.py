"""Launch a project's agent directly in its own Python environment."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from dotenv import dotenv_values

from databricks_agentbricks.clients.local_ports import accepts_connections
from databricks_agentbricks.clients.process_runner import run_process
from databricks_agentbricks.diagnostics import CommandDiagnostics
from databricks_agentbricks.errors import AgentCliError


class LocalProcessClient:
    def __init__(
        self,
        profile: str | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = run_process,
        diagnostics: CommandDiagnostics | None = None,
    ) -> None:
        self.profile = profile
        self.runner = runner
        self.diagnostics = diagnostics or CommandDiagnostics()

    def run(
        self,
        source: Path,
        command: Sequence[str],
        environment: Mapping[str, str | None],
        *,
        prepare_environment: bool,
        port: int,
        on_ready: Callable[[], None] | None = None,
        quiet_status: Callable[[float], str] | None = None,
    ) -> None:
        source = source.resolve()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_NO_SYNC", "PORT"}
        }
        for key, value in dotenv_values(source / ".env").items():
            if value is not None:
                env.setdefault(key, value)
        for key, value in environment.items():
            if value is None:
                if not env.get(key):
                    raise AgentCliError(
                        f"Environment variable {key!r} needs a local value.",
                        hint="Set it in .env or your shell; deployed Apps resources are not resolved locally.",
                    )
            else:
                env[key] = value
        if self.profile:
            env["DATABRICKS_CONFIG_PROFILE"] = self.profile
        env.update(
            PORT=str(port),
            HOST="127.0.0.1",
            PYTHONUNBUFFERED="1",
            DATABRICKS_APP_URL=f"http://localhost:{port}",
        )
        venv = source / ".venv"
        env["UV_PROJECT_ENVIRONMENT"] = str(venv)
        self.diagnostics.starting_local_agent()
        try:
            if prepare_environment:
                if (source / "pyproject.toml").is_file():
                    self._build(["uv", "sync"], source, env)
                elif (source / "requirements.txt").is_file():
                    if not venv.exists():
                        self._build(["uv", "venv", str(venv)], source, env)
                    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    self._build(
                        ["uv", "pip", "install", "--python", str(python), "-r", "requirements.txt"],
                        source,
                        env,
                    )
            if venv.is_dir():
                env["VIRTUAL_ENV"] = str(venv)
                env["PATH"] = (
                    str(venv / ("Scripts" if os.name == "nt" else "bin"))
                    + os.pathsep
                    + env.get("PATH", "")
                )
            env["UV_NO_SYNC"] = "1"

            def ready() -> None:
                self.diagnostics.local_agent_listened()
                if on_ready is not None:
                    on_ready()

            def exited(code: int) -> None:
                self.diagnostics.subprocess_exit_code = code

            result = self.runner(
                list(command),
                cwd=str(source),
                env=env,
                ready_probe=lambda: accepts_connections(port),
                on_ready=ready,
                on_exit=exited,
                quiet_status=quiet_status,
            )
        except KeyboardInterrupt:
            self.diagnostics.parent_interrupted = True
            raise
        except FileNotFoundError as exc:
            raise AgentCliError(
                "Could not start the local command.",
                hint="Install the command executable and uv, or check [dev].command in agent.toml.",
            ) from exc
        self.diagnostics.subprocess_exit_code = result.returncode
        if result.returncode:
            raise self._failure(result)

    def _build(self, command: list[str], source: Path, environment: dict[str, str]) -> None:
        result = self.runner(command, cwd=str(source), env=environment)
        self.diagnostics.subprocess_exit_code = result.returncode
        if result.returncode:
            raise self._failure(result)

    def _failure(self, result: subprocess.CompletedProcess) -> AgentCliError:
        return AgentCliError(
            "The local agent stopped unexpectedly."
            if self.diagnostics.app_listened
            else "Could not start the agent locally.",
            hint="See the output above for the cause.",
            subprocess_exit_code=result.returncode,
            failure_detail=(result.stderr or "").strip() or (result.stdout or "").strip() or None,
        )
