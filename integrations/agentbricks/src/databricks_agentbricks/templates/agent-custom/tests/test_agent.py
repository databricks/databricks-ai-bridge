import pytest
from agent.agent import run_agent


@pytest.mark.asyncio
async def test_native_agent_runs_without_runtime_or_model():
    message = await run_agent([{"role": "user", "content": "Hello"}], memories=["Likes teal"])
    assert message == {
        "role": "assistant",
        "content": "Echo: Hello\nRelevant memories:\nLikes teal",
    }
