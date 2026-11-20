"""Measure fresh sandbox preparation and real managed web-search invocations.

Run with the normal sandbox CLI environment, requests and MLflow installed.
Outputs contain synthetic prompts, public search results, run IDs and trace spans.
The App must have zero warm spares and the exact --source revision deployed.
"""

# This standalone benchmark reports its progress to the terminal.
# ruff: noqa: T201

import argparse
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from databricks.sdk import WorkspaceClient

from databricks_agentkit.sandbox.bundle import build_bundle

TURNS = (
    (
        "first_search",
        "You must use a tool from the configured system.ai.web_search MCP service. "
        "Search official Databricks documentation for Model Context Protocol, then return "
        "the title and https URL of one result. Do not answer from memory. "
        "Remember my test codeword ORCHID-742 for this conversation.",
        "web_search",
    ),
    (
        "warm_search",
        "Use the configured web_search tool to search official Databricks documentation "
        "for Databricks Apps. Return the title and https URL of one result. "
        "Do not answer from memory.",
        "web_search",
    ),
    (
        "python_tool",
        "Call get_current_time once and tell me the result.",
        "get_current_time",
    ),
    ("history", "What was my test codeword? Reply with just the codeword.", None),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-url", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="DEFAULT")
    parser.add_argument("--model", default="system.ai.gpt-5-mini")
    parser.add_argument("--sessions", type=int, default=3)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.app_url.rstrip("/")
    workspace = WorkspaceClient(profile=args.profile)
    http = requests.Session()

    def request(method, path, **kwargs):
        response = http.request(
            method,
            base + path,
            headers=workspace.config.authenticate(),
            timeout=(15, 180),
            **kwargs,
        )
        response.raise_for_status()
        return response

    def call(method, path, **kwargs):
        return request(method, path, **kwargs).json()

    config = call("GET", "/api/config")
    assert config["sandbox_warm_pool_size"] == 0, "Disable warm spares for this benchmark"
    agent = next(a for a in call("GET", "/api/agents") if a["id"] == config["default_agent_id"])
    assert agent["source_hash"] == build_bundle(args.source).digest, "Wrong deployed source"
    assert "web_search" in config["workbench"]["tools"], "Web search is not configured"
    aid = agent["id"]
    evidence = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "app_url": base,
        "model": args.model,
        "config": config,
        "source": agent,
        "lock_sha256": hashlib.sha256((args.source / "uv.lock").read_bytes()).hexdigest(),
        "wheels": json.loads((args.source / "worker-build.json").read_text()),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": [],
    }

    def save():
        (args.output / "benchmark.json").write_text(json.dumps(evidence, indent=2) + "\n")

    def invoke(sid, label, prompt):
        started = time.perf_counter()
        created = call(
            "POST",
            "/api/runs",
            json={
                "agent_id": aid,
                "session_id": sid,
                "input": {"model": args.model, "message": prompt},
            },
        )
        result = {"id": created["id"], "label": label, "prompt": prompt, "first_text_s": None}
        last, updates = "", 0
        with request("GET", f"/api/runs/{created['id']}/events", stream=True) as response:
            for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                text = (event.get("stream") or {}).get("text", "")
                if text and text != last:
                    if result["first_text_s"] is None:
                        result["first_text_s"] = time.perf_counter() - started
                    updates += 1
                    last = text
                if event["status"] in {"succeeded", "failed", "interrupted"}:
                    break
        result.update(
            wall_s=time.perf_counter() - started,
            text_updates=updates,
            streamed_text=last,
            row=call("GET", f"/api/runs/{created['id']}"),
        )
        return result

    save()
    for index in range(args.sessions):
        sid = "websearch-" + uuid.uuid4().hex
        sample = {"session_id": sid, "turns": []}
        evidence["samples"].append(sample)
        save()
        print(f"Preparing session {index + 1}: {sid}", flush=True)
        try:
            started = time.perf_counter()
            row = call("POST", f"/api/agents/{aid}/sessions", json={"session_id": sid})
            deadline = time.monotonic() + 300
            while row["status"] in {"queued", "preparing"}:
                if time.monotonic() > deadline:
                    raise TimeoutError("Session preparation exceeded 300 seconds")
                time.sleep(0.5)
                row = call("GET", f"/api/agents/{aid}/sessions/{sid}")
            sample.update(preparation_client_s=time.perf_counter() - started, preparation=row)
            save()
            assert row["status"] == "ready", row
            prepared = row["result"]
            assert prepared["sandbox_created"] and not prepared["warm_pool_hit"]
            assert not prepared["worker"]["reused"]
            lifecycle = prepared["lifecycle"]
            print(
                f"Ready {lifecycle['preparation_ms'] / 1000:.3f}s; "
                f"handler {lifecycle['steps']['handler_prepare']['duration_ms'] / 1000:.3f}s",
                flush=True,
            )
            for label, prompt, expected_tool in TURNS:
                turn = invoke(sid, label, prompt)
                sample["turns"].append(turn)
                save()
                assert turn["row"]["status"] == "succeeded", turn["row"].get("error")
                result = turn["row"]["result"]
                answer = result["result"]
                assert turn["first_text_s"] is not None
                assert result["worker"]["reused"]
                assert result["worker"]["id"] == prepared["worker"]["id"]
                assert answer["session_backend"]["mode"] == "managed"
                if expected_tool:
                    assert expected_tool in answer["tool_calls"], answer
                    assert any(
                        event["name"] == expected_tool and event["status"] == "completed"
                        for event in answer["tool_events"]
                    ), answer
                if expected_tool == "web_search":
                    assert re.search(r"https://\S+", answer["reply"]), answer["reply"]
                if label == "history":
                    assert "ORCHID-742" in answer["reply"], answer["reply"]
                print(
                    f"{label}: first text {turn['first_text_s']:.3f}s; "
                    f"complete {turn['wall_s']:.3f}s; "
                    f"graph build {answer['worker_timings_ms']['agent_build'] / 1000:.3f}s",
                    flush=True,
                )
        except BaseException as error:
            sample["failure"] = repr(error)
            save()
            raise
        finally:
            started = time.perf_counter()
            try:
                sample["stop"] = call(
                    "POST", "/sessions/stop", json={"agent_id": aid, "session_id": sid}
                )
                sample["stop_client_s"] = time.perf_counter() - started
                print(f"Stopped in {sample['stop_client_s']:.3f}s", flush=True)
            except Exception as error:
                sample["stop_error"] = repr(error)
            save()
    assert all(s.get("stop", {}).get("status") == "stopped" for s in evidence["samples"])
    print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
