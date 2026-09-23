"""Opt-in CLI → manifest → framework → workspace tests, with no service mocks."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import time
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from databricks_mason.cli.app import mason

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_MASON_GENIE_TESTS") != "1",
        reason="Set RUN_MASON_GENIE_TESTS=1 and MASON_GENIE_SPACE_ID to run live tests.",
    ),
]


def _save(name, value):
    directory = os.getenv("MASON_GENIE_EVIDENCE_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{name}.json").write_text(json.dumps(value, indent=2, default=str))


@pytest.fixture(params=["langgraph", "openai"])
def project(request, tmp_path, monkeypatch):
    framework = request.param
    source = tmp_path / framework
    runner = CliRunner()
    commands = [
        ["init", str(source), "--framework", framework, "--disable-chat-app"],
        ["tools", "add", "genie-one", "--source", str(source)],
        [
            "tools",
            "add",
            "genie-agent",
            os.environ["MASON_GENIE_SPACE_ID"],
            "--source",
            str(source),
        ],
    ]
    for command in commands:
        result = runner.invoke(mason, ["-o", "json", *command])
        assert result.exit_code == 0, result.output
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(source))
    _save(
        f"{framework}-manifest",
        {"commands": commands, "manifest": (source / "agent.toml").read_text()},
    )
    return framework


async def _native_call(framework, tool, arguments):
    if framework == "langgraph":
        return await tool.ainvoke(arguments)
    from agents.tool_context import ToolContext

    encoded = json.dumps(arguments)
    context = ToolContext(
        context=None, tool_name=tool.name, tool_call_id=uuid.uuid4().hex, tool_arguments=encoded
    )
    result = await tool.on_invoke_tool(context, encoded)
    return json.loads(result) if isinstance(result, str) else result


async def test_genie_agent_conversation_and_rows(project):
    framework = project
    adapter = importlib.import_module(f"databricks_mason.{framework}.genie")
    tools = {tool.name: tool for tool in adapter.genie_tools()}
    assert set(tools) == {"genie_agent_ask", "genie_agent_poll", "genie_agent_query_result"}
    question = os.getenv("MASON_GENIE_QUESTION", "How many rows are in samples.nyctaxi.trips?")
    question += " Use only this sample table and include a SQL result."
    started = time.monotonic()
    answer = await _native_call(
        framework, tools["genie_agent_ask"], {"question": question, "conversation_id": None}
    )
    deadline = started + 600
    while answer.get("timed_out") and time.monotonic() < deadline:
        answer = await _native_call(
            framework,
            tools["genie_agent_poll"],
            {
                "conversation_id": answer["conversation_id"],
                "message_id": answer["message_id"],
            },
        )
    _save(f"{framework}-agent-answer", answer)
    assert answer["status"] == "COMPLETED", answer
    attachment = next((item for item in answer["attachments"] if item.get("query")), None)
    assert attachment is not None, answer
    rows = await _native_call(
        framework,
        tools["genie_agent_query_result"],
        {
            "conversation_id": answer["conversation_id"],
            "message_id": answer["message_id"],
            "attachment_id": attachment["attachment_id"],
        },
    )
    _save(f"{framework}-agent-rows", rows)
    assert rows["status"]["state"] == "SUCCEEDED", rows
    assert rows["columns"] and rows["rows"], rows
    if expected := os.getenv("MASON_GENIE_EXPECTED_VALUE"):
        assert expected in {str(cell) for row in rows["rows"] for cell in row}
    followup = await _native_call(
        framework,
        tools["genie_agent_ask"],
        {
            "question": "What was the previous question in this conversation? Do not run another query.",
            "conversation_id": answer["conversation_id"],
        },
    )
    _save(f"{framework}-agent-followup", followup)
    assert followup["conversation_id"] == answer["conversation_id"]
    assert followup["message_id"] != answer["message_id"]
    assert followup["status"] == "COMPLETED", followup


def _mcp_payload(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True)
    structured = value.get("structuredContent") or value.get("structured_content")
    if structured:
        return structured
    for block in value.get("content", []):
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except json.JSONDecodeError:
                pass
    raise AssertionError(f"No structured MCP response: {value}")


async def test_genie_one_ask_poll_and_result(project):
    framework = project
    adapter = importlib.import_module(f"databricks_mason.{framework}.mcp")
    question = os.getenv("MASON_GENIE_QUESTION", "How many rows are in samples.nyctaxi.trips?")
    question += " Use only this sample table and include a SQL result."
    if framework == "langgraph":
        tools = {tool.name: tool for tool in await adapter.mcp_tools()}
        assert {"genie_ask", "genie_poll_response", "genie_get_query_result"} <= set(tools)

        async def call(name, arguments):
            message = await tools[name].ainvoke(
                {"type": "tool_call", "id": uuid.uuid4().hex, "name": name, "args": arguments}
            )
            artifact = message.artifact or {}
            return _mcp_payload({**artifact, "content": message.content})

        await _one_conversation(framework, call, question)
    else:
        servers = await adapter.mcp_servers()
        assert len(servers) == 1
        async with servers[0] as server:
            assert {"genie_ask", "genie_poll_response", "genie_get_query_result"} <= {
                tool.name for tool in await server.list_tools()
            }

            async def call(name, arguments):
                result = await server.call_tool(name, arguments)
                assert not result.isError, result
                return _mcp_payload(result)

            await _one_conversation(framework, call, question)


async def _one_conversation(framework, call, question):
    answer = await call("genie_ask", {"question": question})
    _save(f"{framework}-one-start", answer)
    deadline = time.monotonic() + 600
    while answer.get("status") == "in_progress" and time.monotonic() < deadline:
        await asyncio.sleep(5)
        answer = await call(
            "genie_poll_response",
            {
                "conversation_id": answer["conversation_id"],
                "response_id": answer["response_id"],
            },
        )
        _save(f"{framework}-one-answer", answer)
    assert answer["status"] == "completed", answer
    assert answer.get("deep_link") and answer.get("final_answer"), answer
    query = answer["query_items"][0]
    rows = await call(
        "genie_get_query_result",
        {
            "conversation_id": answer["conversation_id"],
            "response_id": answer["response_id"],
            "item_id": query["item_id"],
        },
    )
    _save(f"{framework}-one-rows", rows)
    assert rows.get("columns") and rows.get("rows"), rows
    if expected := os.getenv("MASON_GENIE_EXPECTED_VALUE"):
        assert expected in {str(cell) for row in rows["rows"] for cell in row}
