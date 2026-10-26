"""Replace this model-free example with your framework's async entrypoint."""


async def run_agent(messages: list[dict[str, str]]) -> dict[str, str]:
    return {"role": "assistant", "content": f"Echo: {messages[-1]['content']}"}
