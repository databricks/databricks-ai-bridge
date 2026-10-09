"""Manage the local MLflow server used by dev and local trace reads."""

from __future__ import annotations

import pathlib
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

_AGENTBRICKS_LOCAL_DIR = ".agentbricks"
_MLFLOW_SPEC = "mlflow>=3.10,<4"
_SERVER_PYTHON = "3.12"


@dataclass(frozen=True)
class LocalTracingStart:
    server: subprocess.Popen | None
    base_url: str | None
    environment: dict[str, str] = field(default_factory=dict)
    warning: str | None = None
    help: str | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _mlflow_server_argv(db: pathlib.Path, artifacts: pathlib.Path, port: int) -> list[str]:
    return [
        "uvx",
        "--python",
        _SERVER_PYTHON,
        "--from",
        _MLFLOW_SPEC,
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{db}",
        "--default-artifact-root",
        str(artifacts),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _wait_for_server(base_url: str, server: subprocess.Popen, timeout: float = 60.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


class LocalTracingClient:
    """Own the subprocess lifecycle for a project's local trace store."""

    def start_dev(self, source_dir: pathlib.Path) -> LocalTracingStart:
        agentbricks_dir = source_dir / _AGENTBRICKS_LOCAL_DIR
        try:
            agentbricks_dir.mkdir(exist_ok=True)
            db = (agentbricks_dir / "mlflow.db").resolve()
            artifacts = (agentbricks_dir / "mlartifacts").resolve()
            port = _free_port()
            with (agentbricks_dir / "mlflow-server.log").open("w") as log:
                server = subprocess.Popen(
                    _mlflow_server_argv(db, artifacts, port),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            return LocalTracingStart(
                None,
                None,
                warning=f"local tracing not started — {exc}",
                help="running without traces (is `uv` installed?)",
            )
        base_url = f"http://127.0.0.1:{port}"
        return LocalTracingStart(
            server,
            base_url,
            environment={
                "MLFLOW_TRACKING_URI": base_url,
                "MLFLOW_EXPERIMENT_NAME": source_dir.resolve().name,
            },
        )

    def start_read(self, db: pathlib.Path) -> LocalTracingStart:
        artifacts = (db.parent / "mlartifacts").resolve()
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        try:
            with (db.parent / "mlflow-read.log").open("w") as log:
                server = subprocess.Popen(
                    _mlflow_server_argv(db, artifacts, port),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            return LocalTracingStart(
                None,
                None,
                warning=f"could not read local traces - {exc}",
                help="is `uv` installed?",
            )
        if not _wait_for_server(base_url, server):
            self.stop(server)
            return LocalTracingStart(
                None,
                None,
                warning="local trace store did not come up",
                help="see .agentbricks/mlflow-read.log",
            )
        return LocalTracingStart(server, base_url)

    def stop(self, server: subprocess.Popen) -> None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def start_local_tracing_server(
    source_dir: pathlib.Path,
) -> tuple[subprocess.Popen | None, dict[str, str]]:
    result = LocalTracingClient().start_dev(source_dir)
    return result.server, result.environment


def stop_local_tracing_server(server: subprocess.Popen) -> None:
    LocalTracingClient().stop(server)


def _start_read_server(db: pathlib.Path) -> tuple[subprocess.Popen | None, Optional[str]]:
    result = LocalTracingClient().start_read(db)
    return result.server, result.base_url
