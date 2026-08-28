import asyncio
from contextlib import suppress

import pytest
from agent.mason.recovery import build_recovery_graph, recovery_config
from agent.tools import all_tools  # ty: ignore[unresolved-import]
from agent.tools.long_running import process_id
from langgraph.checkpoint.memory import InMemorySaver  # ty: ignore[unresolved-import]


def test_long_running_tools_are_registered():
    assert {f"tool_step_{step}" for step in range(1, 5)} <= {tool.name for tool in all_tools()}


def test_recovery_config_uses_same_actor_and_separate_thread(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
    first = recovery_config("session-1")
    second = recovery_config("session-1")

    assert first == second
    assert first["configurable"]["actor_id"] == "alice"
    assert first["configurable"]["thread_id"] != "session-1"


@pytest.mark.asyncio
async def test_default_sequence_runs_all_registered_tools(monkeypatch):
    monkeypatch.setenv("MASON_DEMO_TOOL_STEP_SECONDS", "0")
    graph = build_recovery_graph(InMemorySaver())
    result = await graph.ainvoke({"outputs": []}, config=recovery_config("complete-demo"))

    assert [output["tool"] for output in result["outputs"]] == [
        "tool_step_1",
        "tool_step_2",
        "tool_step_3",
        "tool_step_4",
    ]
    assert {output["instance_id"] for output in result["outputs"]} == {process_id()}


@pytest.mark.asyncio
async def test_resume_skips_checkpointed_steps_and_retries_incomplete_step():
    saver = InMemorySaver()
    calls: list[int] = []
    third_step_started = asyncio.Event()

    async def step_1():
        calls.append(1)
        return {"tool": "tool_step_1", "output": "one"}

    async def step_2():
        calls.append(2)
        return {"tool": "tool_step_2", "output": "two"}

    async def interrupted_step_3():
        calls.append(3)
        third_step_started.set()
        await asyncio.sleep(60)
        return {"tool": "tool_step_3", "output": "three"}

    async def resumed_step_3():
        calls.append(3)
        return {"tool": "tool_step_3", "output": "three"}

    async def step_4():
        calls.append(4)
        return {"tool": "tool_step_4", "output": "four"}

    config = recovery_config("crash-demo")
    first_graph = build_recovery_graph(
        saver,
        [
            ("tool_step_1", step_1),
            ("tool_step_2", step_2),
            ("tool_step_3", interrupted_step_3),
            ("tool_step_4", step_4),
        ],
    )
    run = asyncio.create_task(first_graph.ainvoke({"outputs": []}, config=config))
    await third_step_started.wait()
    run.cancel()
    with suppress(asyncio.CancelledError):
        await run

    checkpoint = await first_graph.aget_state(config)
    assert [output["tool"] for output in checkpoint.values["outputs"]] == [
        "tool_step_1",
        "tool_step_2",
    ]
    assert checkpoint.next == ("tool_step_3",)

    restarted_graph = build_recovery_graph(
        saver,
        [
            ("tool_step_1", step_1),
            ("tool_step_2", step_2),
            ("tool_step_3", resumed_step_3),
            ("tool_step_4", step_4),
        ],
    )
    result = await restarted_graph.ainvoke(None, config=config)

    assert [output["tool"] for output in result["outputs"]] == [
        "tool_step_1",
        "tool_step_2",
        "tool_step_3",
        "tool_step_4",
    ]
    assert calls == [1, 2, 3, 3, 4]
