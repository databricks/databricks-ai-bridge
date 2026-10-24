"""Run an app's manifest command locally on one port, the way the Apps runtime starts it.

``agentbricks dev`` runs the ``command`` and literal ``env`` from its prepared local manifest
directly, so the app listens on the selected port and nothing else: no proxy and no forwarding
port. The Databricks Apps runtime contract the templates rely on is reproduced here: ``PORT`` and
the ``DATABRICKS_APP_*`` variables, plus the selected profile's auth.

The process launcher is injected so tests can assert on the argv and env without starting one.
"""

from __future__ import annotations

import configparser
import os
import pathlib
import signal
import subprocess
from collections.abc import Callable, Mapping
from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.app_manifest import AppManifest

ProcessRunner = Callable[..., int]

# Seconds to let the app shut down after Ctrl-C before it is killed.
_STOP_TIMEOUT_S = 10


def _run_process(argv: list[str], *, cwd: str, env: Mapping[str, str]) -> int:
    """Run ``argv`` in the foreground and return its exit code.

    Output is inherited, so the app's logs stream as it runs. Ctrl-C reaches the child directly
    (same process group), so this only waits for it; a SIGTERM to us is forwarded, and a stop we
    forwarded counts as a clean exit.
    """
    proc = subprocess.Popen(argv, cwd=cwd, env=dict(env))
    stopped = False

    def _forward_stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True
        proc.terminate()

    try:
        previous = signal.signal(signal.SIGTERM, _forward_stop)
    except ValueError:  # not the main thread: leave SIGTERM handling alone
        previous = None
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return 0 if stopped else code


class LocalAppRunner:
    """Prepare and run a project's manifest command on one local port."""

    def __init__(
        self,
        profile: Optional[str],
        *,
        run_process: ProcessRunner = _run_process,
        warn: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self._profile = profile
        self._run_process = run_process
        self._warn = warn

    def run(
        self,
        source_dir: pathlib.Path,
        manifest_path: pathlib.Path,
        *,
        prepare_environment: bool,
        port: int,
    ) -> None:
        """Run the manifest's command in ``source_dir`` on ``port``, streaming its output."""
        manifest = AppManifest.parse(manifest_path.read_text(), source=manifest_path)
        command = manifest.command()
        if not command:
            raise AgentCliError(
                f"No command in {source_dir / 'app.yaml'}.",
                hint='Set one, e.g. command: ["uv", "run", "start-server"].',
            )
        manifest_env = self._manifest_env(manifest)
        if prepare_environment:
            env = self._environment(source_dir, manifest_env, port)
            for step in _prepare_steps(source_dir):
                if self._run_process(step, cwd=str(source_dir), env=env) != 0:
                    raise AgentCliError("Could not build the agent's environment.")
        # Built after preparing, so a newly created .venv is the one the command runs in.
        env = self._environment(source_dir, manifest_env, port)
        if self._run_process(command, cwd=str(source_dir), env=env) != 0:
            raise AgentCliError("Could not start the agent locally.")

    def _manifest_env(self, manifest: AppManifest) -> dict[str, str]:
        """The manifest's literal ``value`` entries; ``valueFrom`` resource refs are skipped."""
        env: dict[str, str] = {}
        for entry in manifest.raw_env():
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            if "value" in entry:
                env[str(entry["name"])] = str(entry["value"])
            elif "valueFrom" in entry:
                self._warn(f"Skipping env {entry['name']}: valueFrom isn't available locally.")
        return env

    def _environment(
        self, source_dir: pathlib.Path, manifest_env: Mapping[str, str], port: int
    ) -> dict[str, str]:
        # An activated venv or an exported PORT would point the app at the wrong packages or port.
        env = {
            key: value for key, value in os.environ.items() if key not in ("VIRTUAL_ENV", "PORT")
        }
        venv = source_dir.resolve() / ".venv"
        if venv.is_dir():
            env["VIRTUAL_ENV"] = str(venv)
            bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
            env["PATH"] = os.pathsep.join(filter(None, [str(bin_dir), env.get("PATH")]))
        env.update(manifest_env)
        env.update(
            {
                "PORT": str(port),
                "DATABRICKS_APP_PORT": str(port),
                "DATABRICKS_APP_NAME": source_dir.resolve().name,
                "DATABRICKS_APP_URL": f"http://localhost:{port}",
            }
        )
        if self._profile:
            env["DATABRICKS_CONFIG_PROFILE"] = self._profile
            settings = _profile_settings(self._profile)
            if host := settings.get("host"):
                env["DATABRICKS_HOST"] = host
            if workspace_id := settings.get("workspace_id"):
                env["DATABRICKS_WORKSPACE_ID"] = workspace_id
        return env


def _prepare_steps(source_dir: pathlib.Path) -> list[list[str]]:
    """The uv commands that build the project's ``.venv``, reusing one that already exists."""
    if (source_dir / "pyproject.toml").is_file():
        return [["uv", "sync"]]
    if (source_dir / "requirements.txt").is_file():
        return [
            ["uv", "venv", "--allow-existing"],
            ["uv", "pip", "install", "-r", "requirements.txt"],
        ]
    return []


def _profile_settings(profile: str) -> dict[str, str]:
    """The profile's section in the Databricks config file, or ``{}`` if it can't be read."""
    path = pathlib.Path(os.getenv("DATABRICKS_CONFIG_FILE", pathlib.Path.home() / ".databrickscfg"))
    # [DEFAULT] is an ordinary profile here, so don't let configparser merge it into every section.
    parser = configparser.ConfigParser(default_section="agentbricks:none", interpolation=None)
    try:
        parser.read(path)
    except (OSError, configparser.Error):
        return {}
    return dict(parser[profile]) if parser.has_section(profile) else {}
