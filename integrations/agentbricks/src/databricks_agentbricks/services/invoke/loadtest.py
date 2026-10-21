"""Run Locust separately so its gevent patches do not affect the CLI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.transport import EndpointRequest


def run_loadtest(
    request: EndpointRequest,
    *,
    concurrency: int,
    duration: int,
    output: str,
    per_user_routing: bool,
) -> int:
    """Delegate scheduling, statistics, and failure reporting to Locust."""
    if find_spec("locust") is None:
        raise AgentCliError(
            "Locust is required for --loadtest.",
            hint="Install the loadtest extra: pip install 'databricks-agentbricks[loadtest]'",
        )
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(Path(__file__).with_name("_locustfile.py")),
        "--headless",
        "--users",
        str(concurrency),
        "--spawn-rate",
        str(concurrency),
        "--run-time",
        f"{duration}s",
        "--stop-timeout",
        str(math.ceil(request.timeout)),
        "--only-summary",
    ]
    if output == "json":
        command.append("--json")
    # Keep credentials and the payload out of process arguments and files.
    return subprocess.run(
        command,
        input=json.dumps({"request": asdict(request), "per_user_routing": per_user_routing}),
        text=True,
        check=False,
    ).returncode
