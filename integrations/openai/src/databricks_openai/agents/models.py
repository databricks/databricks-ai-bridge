"""OpenAI Agents SDK model for Databricks AI Gateway chat completions."""

from contextvars import ContextVar
from typing import Any

from agents.models import openai_chatcompletions
from agents.models.chatcmpl_converter import Converter
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

_preserve_thinking_blocks: ContextVar[bool] = ContextVar(
    "databricks_preserve_thinking_blocks", default=False
)


class _ThinkingBlockConverter(Converter):
    """Converter that replays thinking blocks while a Databricks model builds its request.

    The SDK only asks for ``preserve_thinking_blocks`` from its LiteLLM and any-llm models. Other
    models are unaffected: the context variable is set only inside
    ``DatabricksChatCompletionsModel``.
    """

    @classmethod
    def items_to_messages(cls, *args: Any, **kwargs: Any) -> Any:
        if _preserve_thinking_blocks.get():
            kwargs["preserve_thinking_blocks"] = True
        return super().items_to_messages(*args, **kwargs)


openai_chatcompletions.Converter = _ThinkingBlockConverter  # ty: ignore[invalid-assignment]


class DatabricksChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat completions model that sends reasoning signatures back on the next turn.

    Claude models on the AI Gateway return reasoning with a signature that must be replayed on the
    following request, notably after a tool call. Use it with ``AsyncDatabricksOpenAI``, which
    turns any model's gateway reasoning content into the SDK's thinking blocks, and signed blocks
    back into reasoning content. Unsigned reasoning (Qwen, gpt-oss) is not sent back.

    Example:
        >>> from agents import Agent
        >>> from databricks_openai import AsyncDatabricksOpenAI
        >>> from databricks_openai.agents import DatabricksChatCompletionsModel
        >>> agent = Agent(
        ...     name="Agent",
        ...     model=DatabricksChatCompletionsModel(
        ...         model="system.ai.claude-opus-5-5",
        ...         openai_client=AsyncDatabricksOpenAI(use_ai_gateway=True),
        ...     ),
        ... )
    """

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        token = _preserve_thinking_blocks.set(True)
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            _preserve_thinking_blocks.reset(token)
