"""A running agent, a dev server or a deployed App, and how a test calls it over HTTP.

``AgentEndpoint.invoke`` POSTs straight to the agent's /api/invocations, so it is the one test step
that does not go through the CLI; ``agentbricks endpoint invoke`` is checked separately.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from harness.agentbricks_cli import BackgroundCommand
from harness.logs import last_lines, last_nonempty_line, log
from harness.run import E2E_MODEL, E2EError, now
from harness.workspace import Workspace


class AgentEndpoint:
    """Where a running agent answers, a dev server or a deployed App, and how to call it."""

    def __init__(
        self,
        base_url: str,
        headers: Callable[[], dict[str, str]],
        *,
        app_name: str | None = None,
        app: dict[str, Any] | None = None,
    ):
        self.base_url = base_url
        self.app_name = app_name
        self.app = app
        self._headers = headers

    @classmethod
    def local(cls, server: BackgroundCommand, port: int) -> AgentEndpoint:
        """The `agentbricks dev` server on ``port``, once it answers; fails if the server exits."""
        started = time.monotonic()
        next_tick = 60.0
        while True:
            if server.process.poll() is not None:
                raise E2EError(
                    f"{server.label} exited {server.process.returncode}: "
                    f"{last_lines(server.log_path, 30)}"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5):
                    break
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - started
            if elapsed > 1200:
                raise E2EError(
                    f"{server.label} did not become reachable: {last_lines(server.log_path, 30)}"
                )
            if elapsed >= next_tick:
                log(
                    f"tick {now():%H:%M} | {server.label} | starting | "
                    f"{last_nonempty_line(server.log_path)}"
                )
                next_tick += 60
            time.sleep(5)
        return cls(f"http://127.0.0.1:{port}", lambda: {})

    @classmethod
    def deployed(cls, workspace: Workspace, app_name: str) -> AgentEndpoint:
        """The App once it is up and has answered once."""
        workspace.check_app_auth()
        app = workspace.wait_for_app(app_name)
        url = str(app.get("url") or "").rstrip("/")
        if not url:
            raise E2EError(f"App {app_name} has no URL: {app}")
        endpoint = cls(url, lambda: workspace.app_headers, app_name=app_name, app=app)
        endpoint.warm_up()
        return endpoint

    @property
    def runtime(self) -> str:
        return "deploy" if self.app_name else "dev"

    def invoke(self, prompt: str, label: str = "invoke") -> str:
        """POST the prompt to /api/invocations and return the serialized response."""
        url = f"{self.base_url}/api/invocations"
        last: Exception | None = None
        invocation_id = str(uuid.uuid4())
        body = {
            "id": invocation_id,
            # session_id is a top-level invocation field; the adapter rejects it nested in input.
            "session_id": invocation_id,
            "input": {
                "model": E2E_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for attempt in range(1, 4):
            try:
                response = _monitored(
                    label, lambda: _http_json(url, body, self._headers()), timeout=360
                )
                return json.dumps(response, sort_keys=True, default=str)
            except Exception as exc:
                last = exc
                log(f"attempt {attempt}/3 | {label} | {exc}")
                if attempt < 3:
                    time.sleep(15)
        raise E2EError(f"{label} failed after 3 attempts: {last}")

    def warm_up(self, deadline_seconds: float = 180.0) -> None:
        """Invoke until the agent answers once. A new App's session store can drop the first calls,
        and a redeployed one restarts."""
        started = time.monotonic()
        while True:
            try:
                self.invoke("Reply with the single word OK.", label="warm-up")
                return
            except E2EError:
                if time.monotonic() - started > deadline_seconds:
                    raise


def _monitored(
    label: str, operation: Callable[[], dict[str, Any]], *, timeout: float
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started = time.monotonic()
        while True:
            try:
                return future.result(
                    timeout=min(60, max(1, timeout - (time.monotonic() - started)))
                )
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                log(f"tick {now():%H:%M} | {label} | running | {elapsed:.0f}s")
                if elapsed >= timeout:
                    raise E2EError(f"{label} timed out after {timeout:.0f}s") from None


def _http_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=340) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise E2EError(f"HTTP {exc.code} from {url}: {detail}") from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise E2EError(f"Invalid JSON from {url}: {payload[:2000]}") from exc
    if not isinstance(value, dict):
        raise E2EError(f"Expected object response from {url}, got {type(value).__name__}")
    return value
