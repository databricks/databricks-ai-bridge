"""Live model -> sandbox -> model regression against a configured LangGraph agent."""

from __future__ import annotations

import json
import os
import pathlib
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENTBRICKS_SANDBOX_E2E") != "1",
    reason="Set RUN_AGENTBRICKS_SANDBOX_E2E=1 and AGENTBRICKS_E2E_AGENT_URL to run.",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("execution_error", [False, True], ids=["success", "execution-error"])
async def test_sandbox_result_reaches_next_model_turn(stream, execution_error):
    import httpx

    base_url = os.environ["AGENTBRICKS_E2E_AGENT_URL"].rstrip("/")
    marker = f"SANDBOX_ROUNDTRIP_{uuid.uuid4().hex}"
    code = f"raise ValueError({marker!r})" if execution_error else f"print({marker!r})"
    prompt = (
        "Call run_code exactly once with language='python' and this exact code:\n"
        f"{code}\n"
        "After the tool returns, report its output or execution error, including the exact "
        "SANDBOX_ROUNDTRIP marker. Do not retry, fix the code, or call any other tool."
    )
    invocation_id = str(uuid.uuid4())
    request = {
        "id": invocation_id,
        "session_id": invocation_id,
        "stream": stream,
        "input": {
            "model": os.environ.get("AGENTBRICKS_E2E_MODEL", "system.ai.claude-sonnet-4-5"),
            "messages": [{"role": "user", "content": prompt}],
        },
    }
    headers = {}
    if token := os.environ.get("AGENTBRICKS_E2E_APP_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    events = []
    response_body = None
    async with httpx.AsyncClient(timeout=180, headers=headers) as client:
        if stream:
            httpx_sse = pytest.importorskip("httpx_sse")
            async with httpx_sse.aconnect_sse(
                client, "POST", f"{base_url}/api/invocations", json=request
            ) as source:
                source.response.raise_for_status()
                async for event in source.aiter_sse():
                    events.append(event.json())
            completed = [event for event in events if event.get("type") == "run.completed"]
            assert not [event for event in events if event.get("type") == "run.failed"], events
            assert len(completed) == 1, events
            messages = [event["message"] for event in events if event.get("type") == "message"]
        else:
            response = await client.post(f"{base_url}/api/invocations", json=request)
            response.raise_for_status()
            response_body = response.json()
            assert response_body["status"] == "completed", response_body
            output = response_body["output"]
            assert output["status"] == "completed", output
            messages = output["output"]

    tool_messages = [message for message in messages if message.get("type") == "tool"]
    assert len(tool_messages) == 1, messages
    result = tool_messages[0]
    assert result["name"] == "run_code", result
    assert result["tool_call_id"], result
    assert marker in json.dumps(result["content"]), result
    execution = result["artifact"]["structured_content"]
    assert execution["outcome"] == ("execution_error" if execution_error else "succeeded"), result
    assert execution["language"] == "python", result
    assert execution["sandbox_id"], result
    if execution_error:
        assert f"ValueError: {marker}" in execution["output"], result
    else:
        assert execution["output"].strip() == marker, result
    for block in result["content"]:
        if isinstance(block, dict) and block.get("type") == "text":
            assert set(block) == {"type", "text"}, block
    final_message = messages[-1]
    assert final_message["type"] == "ai", messages
    assert final_message["content"], final_message
    assert marker in json.dumps(final_message["content"]), final_message
    if execution_error:
        assert "ValueError" in json.dumps(final_message["content"]), final_message
    calls = [
        call
        for message in messages
        if message.get("type") == "ai"
        for call in message.get("tool_calls", [])
    ]
    assert len(calls) == 1, calls
    assert calls[0]["name"] == "run_code", calls
    assert calls[0]["id"] == result["tool_call_id"], (calls, result)
    assert calls[0]["args"]["code"] == code, calls

    if output_dir := os.environ.get("AGENTBRICKS_E2E_OUTPUT"):
        directory = pathlib.Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        label = (
            f"{'execution-error' if execution_error else 'success'}-{'sse' if stream else 'json'}"
        )
        (directory / f"{label}.json").write_text(
            json.dumps(
                {
                    "case": label,
                    "url": f"{base_url}/api/invocations",
                    "request": request,
                    "response": response_body,
                    "events": events,
                    "tool_result": result,
                    "final_message": final_message,
                    "status": "passed",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
