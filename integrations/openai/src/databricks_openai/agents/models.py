"""OpenAI Agents SDK model for Databricks AI Gateway chat completions."""

from contextvars import ContextVar
from typing import Any

from agents.models import openai_chatcompletions
from agents.models.chatcmpl_converter import Converter
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from databricks_openai.utils.clients import _is_claude_model

_preserve_thinking_blocks: ContextVar[bool] = ContextVar(
    "databricks_preserve_thinking_blocks", default=False
)


class _ThinkingBlockConverter(Converter):
    """Converter that replays Claude thinking blocks while a Databricks model builds its request.

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
    """Chat completions model that sends Claude reasoning signatures back on the next turn.

    Claude models on the AI Gateway return reasoning with a signature that must be replayed on the
    following request, notably after a tool call. Use it with ``AsyncDatabricksOpenAI``, which
    turns the gateway's reasoning content into the SDK's thinking blocks and back.

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
        token = _preserve_thinking_blocks.set(_is_claude_model(str(self.model)))
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            _preserve_thinking_blocks.reset(token)
