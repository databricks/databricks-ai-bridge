"""Replace this deterministic example with your existing framework or harness."""

from collections.abc import Sequence


async def run_agent(
    messages: Sequence[dict[str, str]], *, memories: Sequence[str] = ()
) -> dict[str, str]:
    """Run native agent code; no Agent Bricks request/context types are required.

    ``messages`` contains the conversation so far, including this turn's input.
    ``memories`` contains relevant facts retrieved from the bound Memory Store.
    Replace the body with your framework's async entrypoint, preserving this return
    contract or adapting it in runtime/adapter.py. Add framework dependencies to pyproject.toml.
    """
    content = f"Echo: {messages[-1]['content']}"
    if memories:
        content += "\nRelevant memories:\n" + "\n".join(memories)
    return {"role": "assistant", "content": content}
