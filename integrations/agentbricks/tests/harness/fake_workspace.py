"""An in-process stand-in for a Databricks workspace, serving what the dev loop needs.

It answers config discovery, the current-user lookup, model serving and CLI telemetry. The model is
scripted: a prompt that names one of the request's tools makes it call that tool, a tool result is
answered verbatim, and anything else gets ``FAKE_MODEL_REPLY``, so a tool round-trips without a real
LLM. Every request is recorded in ``requests`` so a new endpoint dependency is visible when a test
fails.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# What the scripted model answers when the prompt names none of the request's tools.
FAKE_MODEL_REPLY = "AGENTBRICKS_FAKE_MODEL_OK"
_TOOL_ARGUMENT = "matrix"
TELEMETRY_PATH = "/telemetry-ext"


@dataclasses.dataclass(frozen=True)
class TelemetryUpload:
    body: dict[str, Any]
    authorization: str | None


class FakeWorkspaceServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.telemetry: list[TelemetryUpload] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _send(self, status: int, obj: dict[str, Any]) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                outer.requests.append(("GET", self.path))
                self._send(200, outer.get(self.path))

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(("POST", self.path))
                if self.path == TELEMETRY_PATH:
                    outer.telemetry.append(TelemetryUpload(body, self.headers.get("Authorization")))
                    self._send(200, {})
                elif self.path.rstrip("/").endswith("/chat/completions"):
                    outer.chat(self, body)
                else:
                    self._send(200, {})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def get(self, path: str) -> dict[str, Any]:
        if path == "/.well-known/databricks-config":
            return {"workspace_id": "1234567890", "host": self.url}
        if path.startswith("/api/2.0/preview/scim/v2/Me"):
            return {"id": "1", "userName": "fake@example.com"}
        return {}

    def chat(self, handler: BaseHTTPRequestHandler, request: dict[str, Any]) -> None:
        messages = request.get("messages") or []
        tools = [tool["function"] for tool in request.get("tools") or [] if "function" in tool]
        last = messages[-1] if messages else {}
        prompt = _latest_prompt(messages)
        called = _tool_named_in(prompt, tools)
        if last.get("role") == "tool":
            message: dict[str, Any] = {"role": "assistant", "content": str(last.get("content"))}
            finish = "stop"
        elif called is not None:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": called["name"],
                            "arguments": json.dumps(_arguments(called)),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": FAKE_MODEL_REPLY}, "stop"
        meta = {"id": "chatcmpl-fake", "created": 1, "model": request.get("model", "fake")}
        if not request.get("stream"):
            body = {
                **meta,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(body).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
            return
        delta = dict(message)
        if "tool_calls" in delta:
            delta["tool_calls"] = [{"index": 0, **call} for call in delta["tool_calls"]]
        chunks = [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        ]
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for chunk in chunks:
            payload = {**meta, "object": "chat.completion.chunk", **chunk}
            handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        handler.wfile.write(b"data: [DONE]\n\n")


def _latest_prompt(messages: list[dict[str, Any]]) -> str:
    prompts = [m for m in messages if m.get("role") == "user"]
    return str(prompts[-1].get("content")) if prompts else ""


def _tool_named_in(prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The offered tool the prompt names, so a request offering several picks the right one."""
    return next((tool for tool in tools if tool["name"] in prompt), None)


def _arguments(tool: dict[str, Any]) -> dict[str, str]:
    properties = (tool.get("parameters") or {}).get("properties") or {}
    return {name: _TOOL_ARGUMENT for name in properties}
