"""Agents SDK model for AI Gateway chat completions, with reasoning signature round-trip."""

from typing import Any


def chat_model(model: str) -> Any:
    """Return the Agents SDK model for ``model`` on the default OpenAI client.

    Pass the result to ``Agent(model=...)`` after ``set_default_openai_client``. Reasoning replies
    stream as text with reasoning as a separate item, and each reasoning signature is sent back on
    the next turn. Uses ``databricks-openai``'s ``DatabricksChatCompletionsModel`` when the release
    has it, and an equivalent model here otherwise. Without a default client, returns the name.
    """
    # The SDK has no public getter for the client set by set_default_openai_client.
    from agents.models import _openai_shared

    client = _openai_shared.get_default_openai_client()
    if client is None:
        return model
    try:
        # Not in databricks-openai 0.17.1 and earlier.
        from databricks_openai.agents import (
            DatabricksChatCompletionsModel,  # ty: ignore[unresolved-import]
        )
    except ImportError:
        from databricks_agentkit.openai._reasoning import ReasoningChatCompletionsModel

        return ReasoningChatCompletionsModel(model=model, openai_client=client)
    return DatabricksChatCompletionsModel(model=model, openai_client=client)
