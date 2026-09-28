#!/usr/bin/env python3
"""Verify header-only sessions against isolated, instrumented Databricks Apps.

Run explicitly; this is not a pytest test. The real template/framework/model and Lakebase store
must be installed with runtime_session_instrumentation.py. This runner creates invocation data
only: it does not provision, deploy, delete, restart, or retry failed HTTP requests.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import re
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from databricks.sdk import WorkspaceClient

SESSION_HEADER = "X-Databricks-Session-Id"
MODEL_MARKER = "RUNTIME_SESSION_HEADER_OK"
SOURCE_MODULES = ("app", "runtime", "store", "types", "durability.lakebase_runtime_store")
CASES = (
    "foreground",
    "foreground-stream",
    "background",
    "background-stream",
    "concurrent-fifo",
    "cross-session",
    "idempotency",
    "header-contract",
    "event-replay",
)
OPTIONAL_CASES = ("failure-queue", "healthy-heartbeat")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def report(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


async def complete_all(*awaitables):
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def sanitize(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if re.search(r"authorization|password|secret|access.token|refresh.token", key, re.I)
            else sanitize(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)Bearer\s+[^\s\"']+|dapi[a-zA-Z0-9]{20,}", "[REDACTED]", value)
    return value


def sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = {}
        data = []
        for line in block.splitlines():
            key, _, value = line.partition(":")
            if key == "data":
                data.append(value.lstrip())
            elif key in {"id", "event"}:
                fields[key] = value.lstrip()
        if data:
            fields["data"] = json.loads("\n".join(data))
            events.append(fields)
    return events


def wheel_provenance(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as wheel:
        sources = {
            name: hashlib.sha256(
                wheel.read(f"databricks_agentkit/runtime/{name.replace('.', '/')}.py")
            ).hexdigest()
            for name in SOURCE_MODULES
        }
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "installed_source_sha256": sources,
    }


class Runner:
    def __init__(self, args):
        self.args = args
        self.workspace = WorkspaceClient(profile=args.profile)
        self.run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.output = args.output / self.run_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.expected = wheel_provenance(args.expected_wheel)
        self.secrets: set[str] = set()
        self.serial = 0
        self.current: dict[str, Any] = {}
        self.results: list[dict[str, Any]] = []
        self.artifacts: dict[str, str] = {}
        self.boot_ids: set[str] = set()
        self.build_metadata: dict[str, Any] = {}

    def save(self, name: str, value: Any) -> None:
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sanitize(value, self.secrets), indent=2, sort_keys=True) + "\n")
        self.artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    async def request(self, method, path, body=None, session=None, expected=200, headers=()):
        self.serial += 1
        serial = self.serial
        started = time.monotonic()
        public_headers = list(headers)
        if session is not None:
            public_headers.append((SESSION_HEADER, session))
        record = {
            "case": self.current["case"],
            "method": method,
            "path": path,
            "headers": public_headers,
            "request": body,
        }
        try:
            auth = await asyncio.to_thread(self.workspace.config.authenticate)
            authorization = auth.get("Authorization")
            if not isinstance(authorization, str) or not authorization:
                raise AssertionError("profile returned no authorization")
            self.secrets.update((authorization, authorization.split(" ", 1)[-1]))
            # A fresh client prevents routing cookies from pinning the test to one replica.
            async with httpx.AsyncClient(follow_redirects=False, timeout=180) as client:
                response = await client.request(
                    method,
                    self.args.app_url.rstrip("/") + path,
                    json=body,
                    headers=[("Authorization", authorization), *public_headers],
                )
            content_type = response.headers.get("content-type", "")
            result = response.json() if "application/json" in content_type else response.text
            record.update(status=response.status_code, content_type=content_type, response=result)
            require(
                response.status_code == expected,
                f"{method} {path}: expected HTTP {expected}, got {response.status_code}; artifact {serial}",
            )
            return result
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            self.save(f"http/{serial:05}.json", record)

    def session(self):
        value = f"header-e2e-{uuid.uuid4()}"
        self.current["sessions"].append(value)
        return value

    def body(self, background=True, stream=False, delay=0):
        payload: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": f"Reply with exactly {MODEL_MARKER}. Do not use tools."}
            ]
        }
        if delay:
            payload["_runtime_e2e"] = {"delay_seconds": delay}
        value: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "background": background,
            "stream": stream,
            "input": payload,
        }
        self.current["invocations"].append(value["id"])
        return value

    async def submit(self, body, session, expected=None):
        return await self.request(
            "POST",
            "/api/invocations",
            body,
            session,
            expected=expected if expected is not None else (202 if body["background"] else 200),
        )

    async def state(self, invocation_id):
        return await self.request("GET", f"/api/runtime-e2e/invocations/{invocation_id}")

    async def events(self, session, after=0):
        result = await self.request(
            "GET", f"/api/runtime-e2e/sessions/{quote(session, safe='')}/events?after={after}"
        )
        for event in result["events"]:
            marker = event["event"]
            if marker.get("type", "").startswith("runtime.e2e."):
                require(
                    marker["session_id"] == session
                    and marker["invocation_id"] == event["invocation_id"]
                    and marker["attempt"] == event["attempt"],
                    "handler context differs from stored session/invocation/attempt",
                )
                self.boot_ids.add(marker["boot_id"])
        return result["events"]

    async def terminal(self, invocation_id, session, expected_status="COMPLETED"):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = await self.state(invocation_id)
            if state["status"] in {"COMPLETED", "FAILED"}:
                require(
                    state["status"] == expected_status,
                    f"{invocation_id}: expected {expected_status}, got {state['status']}",
                )
                require(state["session_id"] == session, "stored session differs")
                if expected_status == "COMPLETED":
                    require(
                        MODEL_MARKER in json.dumps(state["response"]), "real model output missing"
                    )
                    require(state["response"]["session_id"] == session, "framework session differs")
                require(state["attempt"] == 1, "unexpected recovery during normal execution")
                require(
                    "session_id" not in state["request"], "session duplicated in request envelope"
                )
                return state
            await asyncio.sleep(0.5)
        raise AssertionError(f"{invocation_id}: completion deadline exceeded")

    async def active(self, invocation_id):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = await self.state(invocation_id)
            if state["status"] == "ACTIVE":
                require(state["attempt"] == 1, "unexpected recovery before active observation")
                return state
            require(state["status"] == "QUEUED", "execution terminated before active observation")
            await asyncio.sleep(0.25)
        raise AssertionError(f"{invocation_id}: active observation deadline exceeded")

    @staticmethod
    def markers(events, kind):
        return [event for event in events if event["event"].get("type") == f"runtime.e2e.{kind}"]

    async def build(self):
        build = await self.request("GET", "/api/runtime-e2e/build")
        self.save("build.json", build)
        require(build["durable"] and build["instrumented"], "real durable runtime probes required")
        require(
            build["variant"] == self.args.variant, "deployed variant differs from requested variant"
        )
        require(build["wheel_sha256"] == self.expected["sha256"], "deployment wheel digest differs")
        require(
            build["installed_source_sha256"] == self.expected["installed_source_sha256"],
            "installed SDK modules differ from expected wheel",
        )
        self.build_metadata = build

    async def transport(self, name):
        session = self.session()
        body = self.body(background=name.startswith("background"), stream=name.endswith("stream"))
        response = await self.submit(body, session)
        state = await self.terminal(body["id"], session)
        require(state["session_sequence_number"] == 1, "first session admission is not 1")
        if body["background"]:
            require(response["id"] == body["id"], "accepted invocation ID differs")
            require(
                ("events_url" in response) == body["stream"],
                "background stream URL missing/unexpected",
            )
        elif not body["stream"]:
            require(
                response["output"] == state["response"],
                "foreground output differs from stored output",
            )
        else:
            require(
                any(event["event"] == "run.completed" for event in sse_events(response)),
                "foreground SSE has no terminal event",
            )
        public = await self.request("GET", f"/api/invocations/{body['id']}")
        require(public["output"] == state["response"], "headerless invocation GET differs")
        replay = sse_events(await self.request("GET", f"/api/invocations/{body['id']}/events"))
        require(replay[-1]["event"] == "run.completed", "headerless SSE replay incomplete")
        await self.events(session)
        self.current["state"] = state

    async def concurrent_fifo(self):
        session = self.session()
        bodies = [self.body(delay=8) for _ in range(6)]
        await complete_all(*(self.submit(body, session) for body in bodies))
        for _ in range(20):
            snapshot = await complete_all(*(self.state(body["id"]) for body in bodies))
            active = [row for row in snapshot if row["status"] == "ACTIVE"]
            queued = [row for row in snapshot if row["status"] == "QUEUED"]
            # These GETs are separate snapshots. Use committed events below to prove exclusivity.
            if active and queued:
                self.current["active_and_queued_snapshot"] = snapshot
                break
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("did not observe an active invocation with queued followers")
        states = await complete_all(*(self.terminal(body["id"], session) for body in bodies))
        events = await self.events(session)
        ordered = sorted(states, key=lambda row: row["session_sequence_number"])
        ids = [row["invocation_id"] for row in ordered]
        starts = self.markers(events, "start")
        ends = self.markers(events, "end")
        require(
            [row["session_sequence_number"] for row in ordered] == list(range(1, 7)),
            "admission sequence is not unique/consecutive",
        )
        require(
            [row["invocation_id"] for row in starts] == ids,
            "handler start order differs from store admission order",
        )
        require(
            [row["invocation_id"] for row in ends] == ids,
            "handler end order differs or an execution did not finish",
        )
        for first, second in zip(ids, ids[1:], strict=False):
            completed = next(
                row
                for row in events
                if row["invocation_id"] == first and row["event"]["type"] == "run.completed"
            )
            started = next(row for row in starts if row["invocation_id"] == second)
            require(
                completed["sequence_number"] < started["sequence_number"],
                "next handler started before preceding turn committed completion",
            )
        workers = sorted({row["event"]["boot_id"] for row in starts})
        self.current.update(states=ordered, events=events, worker_boot_ids=workers)
        if self.args.require_multi_worker:
            require(
                len(workers) >= 2,
                "FIFO work was not observed executing on at least two worker processes",
            )

    async def cross_session(self):
        sessions = [self.session(), self.session()]
        bodies = [self.body(delay=8), self.body(delay=8)]
        await complete_all(
            *(self.submit(body, session) for body, session in zip(bodies, sessions, strict=True))
        )
        for _ in range(20):
            states = await complete_all(*(self.state(body["id"]) for body in bodies))
            if all(state["status"] == "ACTIVE" for state in states):
                self.current["both_active_snapshot"] = states
                break
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("different sessions did not run concurrently")
        states = await complete_all(
            *(
                self.terminal(body["id"], session)
                for body, session in zip(bodies, sessions, strict=True)
            )
        )
        histories = await complete_all(*(self.events(session) for session in sessions))
        intervals = [
            (
                self.markers(events, "start")[0]["event"]["time"],
                self.markers(events, "end")[0]["event"]["time"],
            )
            for events in histories
        ]
        require(
            max(start for start, _ in intervals) < min(end for _, end in intervals),
            "persisted execution intervals did not overlap",
        )
        self.current.update(states=states, intervals=intervals)

    async def idempotency(self):
        session = self.session()
        body = self.body()
        await self.submit(body, session)
        before = await self.terminal(body["id"], session)
        await self.submit(body, session)
        await self.submit(body, self.session(), expected=409)
        await self.submit({**body, "input": {"messages": []}}, session, expected=409)
        after = await self.state(body["id"])
        require(before == after, "retry changed persisted invocation")
        events = await self.events(session)
        require(len(self.markers(events, "start")) == 1, "idempotent retry executed more than once")
        self.current["state"] = after

    async def header_contract(self):
        for extra_headers, extra_body in (
            ([], {}),
            ([(SESSION_HEADER, "")], {}),
            ([("Cookie", "__Host-databricks-app-router=legacy-cookie")], {}),
            ([], {"input": {"session_id": "nested-only"}}),
            ([(SESSION_HEADER, "one"), (SESSION_HEADER, "two")], {}),
            ([(SESSION_HEADER, "header-session")], {"session_id": "body-session"}),
        ):
            body = {**self.body(), **extra_body}
            await self.request(
                "POST", "/api/invocations", body, expected=422, headers=extra_headers
            )
            await self.request("GET", f"/api/runtime-e2e/invocations/{body['id']}", expected=404)
        session = self.session()
        body = self.body()
        body["input"]["session_id"] = "opaque-input-must-not-select-session"
        await self.submit(body, session)
        state = await self.terminal(body["id"], session)
        events = await self.events(session)
        require(len(self.markers(events, "start")) == 1, "context source-of-truth marker missing")
        require(state["request"]["input"] == body["input"], "opaque input changed")
        self.current["state"] = state

    async def event_replay(self):
        session = self.session()
        bodies = [self.body(), self.body()]
        for body in bodies:
            await self.submit(body, session)
            await self.terminal(body["id"], session)
        events = await self.events(session)
        sequences = [event["sequence_number"] for event in events]
        require(
            sequences == sorted(set(sequences)), "session replay cursor is not strictly increasing"
        )
        require(
            {event["invocation_id"] for event in events} == {body["id"] for body in bodies},
            "session replay did not contain both turns",
        )
        for cursor in (0, sequences[len(sequences) // 2], sequences[-1]):
            suffix = await self.events(session, after=cursor)
            require(
                suffix == [event for event in events if event["sequence_number"] > cursor],
                "session cursor did not return exact exclusive suffix",
            )
        for body in bodies:
            path = f"/api/invocations/{body['id']}/events"
            replay = sse_events(await self.request("GET", path))
            cursor = int(replay[len(replay) // 2]["id"])
            suffix = sse_events(await self.request("GET", f"{path}?after={cursor}"))
            require(
                suffix == [event for event in replay if int(event["id"]) > cursor],
                "invocation cursor did not return exact exclusive suffix",
            )
        self.current["events"] = events

    async def failure_queue(self):
        session = self.session()
        first, follower = self.body(delay=8), self.body()
        first["input"]["_runtime_e2e"]["fail"] = True
        await self.submit(first, session)
        await self.active(first["id"])
        await self.submit(follower, session)
        queued = await self.state(follower["id"])
        require(queued["status"] == "QUEUED", "follower did not wait for the failing turn")
        states = await complete_all(
            self.terminal(first["id"], session, expected_status="FAILED"),
            self.terminal(follower["id"], session),
        )
        events = await self.events(session)
        first_events = [event for event in events if event["invocation_id"] == first["id"]]
        messages = [event for event in first_events if event["event"].get("type") == "message"]
        require(MODEL_MARKER in json.dumps(messages), "failure occurred before real model output")
        require(
            len(self.markers(first_events, "error")) == 1,
            "intentional handler failure not recorded",
        )
        starts = self.markers(events, "start")
        require(
            [event["invocation_id"] for event in starts] == [first["id"], follower["id"]],
            "failure retried or follower did not execute once",
        )
        failed = next(event for event in first_events if event["event"]["type"] == "run.failed")
        require(
            failed["sequence_number"] < starts[1]["sequence_number"],
            "follower started before failure was committed",
        )
        require(
            [state["session_sequence_number"] for state in states] == [1, 2],
            "failure changed admission order",
        )
        self.current.update(queued_snapshot=queued, states=states, events=events)

    async def healthy_heartbeat(self):
        timings = self.build_metadata["timings"]
        heartbeat, stale, scan = (
            float(timings[key]) for key in ("heartbeat_seconds", "stale_seconds", "scan_seconds")
        )
        observation_seconds = stale + 2 * scan + 2 * heartbeat
        delay = observation_seconds + 4
        require(
            0 < heartbeat < stale and scan > 0 and delay <= 45,
            "probe timings do not permit bounded heartbeat observation",
        )
        session = self.session()
        body = self.body(delay=delay)
        await self.submit(body, session)
        first = await self.active(body["id"])
        started = time.monotonic()
        timeout_response = await self.request(
            "GET",
            f"/api/runtime-e2e/invocations/{body['id']}/wait?timeout_seconds=0.2",
            expected=408,
        )
        require(
            timeout_response == {"detail": "runtime wait timed out"},
            "timeout did not come from Runtime.wait",
        )
        samples = [{"elapsed_seconds": 0, "state": first}]
        while True:
            state = await self.state(body["id"])
            elapsed = time.monotonic() - started
            samples.append({"elapsed_seconds": round(elapsed, 3), "state": state})
            require(
                state["status"] == "ACTIVE" and state["attempt"] == 1,
                "wait timeout cancelled execution or healthy ownership was lost",
            )
            if elapsed >= observation_seconds:
                break
            await asyncio.sleep(min(1.0, observation_seconds - elapsed))
        state = await self.terminal(body["id"], session)
        waited = await self.request(
            "GET", f"/api/runtime-e2e/invocations/{body['id']}/wait?timeout_seconds=1"
        )
        require(
            waited["output"] == state["response"], "wait after completion differs from saved output"
        )
        events = await self.events(session)
        require(
            len(self.markers(events, "start")) == len(self.markers(events, "end")) == 1,
            "healthy invocation executed more than one attempt",
        )
        self.current.update(timings=timings, active_samples=samples, state=state, events=events)

    async def run_case(self, name, function):
        self.current = {"case": name, "sessions": [], "invocations": []}
        started = time.monotonic()
        try:
            await function()
            self.current["status"] = "PASSED"
        except Exception as exc:
            self.current.update(
                status="FAILED", error=str(sanitize(f"{type(exc).__name__}: {exc}", self.secrets))
            )
        self.current["duration_seconds"] = round(time.monotonic() - started, 3)
        self.results.append(self.current)
        self.save(f"cases/{name}.json", self.current)
        report(f"{self.current['status']}: {name}")
        return self.current["status"] == "PASSED"

    async def run(self):
        self.save("expected-wheel.json", self.expected)
        report(f"Evidence: {self.output}")
        if await self.run_case("build", self.build):
            for name in self.args.cases:
                if name in CASES[:4]:
                    await self.run_case(name, lambda name=name: self.transport(name))
                else:
                    await self.run_case(name, getattr(self, name.replace("-", "_")))
        passed = all(case["status"] == "PASSED" for case in self.results)
        self.save(
            "evidence.json",
            {
                "run_id": self.run_id,
                "app_url": self.args.app_url,
                "profile": self.args.profile,
                "workspace_host": self.workspace.config.host,
                "variant": self.args.variant,
                "expected_wheel": self.expected,
                "worker_boot_ids": sorted(self.boot_ids),
                "require_multi_worker": self.args.require_multi_worker,
                "passed": passed,
                "cases": self.results,
            },
        )
        self.save("artifact-manifest.json", dict(self.artifacts))
        return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--app-url", required=True)
    parser.add_argument("--expected-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("lg-api", "lg-ui", "oa-api", "oa-ui"))
    parser.add_argument(
        "--require-multi-worker",
        action="store_true",
        help="Require FIFO turns to execute across at least two worker boot IDs.",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help="Comma-separated case names; optional: failure-queue,healthy-heartbeat. Build verification always runs.",
    )
    args = parser.parse_args()
    args.cases = args.cases.split(",")
    available_cases = (*CASES, *OPTIONAL_CASES)
    if (
        not args.cases
        or set(args.cases) - set(available_cases)
        or len(args.cases) != len(set(args.cases))
    ):
        parser.error(f"--cases must be unique names from: {', '.join(available_cases)}")
    if args.require_multi_worker and "concurrent-fifo" not in args.cases:
        parser.error("--require-multi-worker requires the concurrent-fifo case")
    url = urlsplit(args.app_url)
    if (
        url.scheme != "https"
        or not (url.hostname or "").endswith(".databricksapps.com")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/")
    ):
        parser.error(
            "--app-url must be an HTTPS databricksapps.com base URL without credentials/query/path"
        )
    return asyncio.run(Runner(args).run())


if __name__ == "__main__":
    sys.exit(main())
