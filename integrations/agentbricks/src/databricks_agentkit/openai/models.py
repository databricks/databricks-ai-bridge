"""Agents SDK model for AI Gateway chat completions, with reasoning signature round-trip."""

from typing import Any


def chat_model(model: str) -> Any:
    """Return the Agents SDK model for ``model`` on the default OpenAI client.

    Pass the result to ``Agent(model=...)`` after ``set_default_openai_client``. With a
    ``databricks-openai`` release that has ``DatabricksChatCompletionsModel``, reasoning replies
    stream as text with reasoning as a separate item, and each reasoning signature is sent back on
    the next turn. Older releases get the model name, which the SDK resolves as before.
    """
    try:
        # Not in databricks-openai 0.17.1 and earlier.
        from databricks_openai.agents import (
            DatabricksChatCompletionsModel,  # ty: ignore[unresolved-import]
        )
    except ImportError:
        return model
    # The SDK has no public getter for the client set by set_default_openai_client.
    from agents.models import _openai_shared

    client = _openai_shared.get_default_openai_client()
    if client is None:
        return model
    return DatabricksChatCompletionsModel(model=model, openai_client=client)
