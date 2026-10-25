"""Reasoning list content for ``databricks-openai`` releases without ``DatabricksChatCompletionsModel``.

Reasoning models on the AI Gateway (Claude 5.x, Qwen, Gemini, gpt-oss) send ``content`` /
``delta.content`` as a list of ``reasoning`` and ``text`` parts, which the Agents SDK rejects
(``ResponseTextDeltaEvent.delta`` must be a string). This mirrors the newer release: the answer stays
as text, reasoning moves to the SDK's thinking blocks, and signed blocks go back as reasoning parts.
A reply that stops after unsigned reasoning, with no answer, is requested once more. On releases
that have the model, ``chat_model`` still wraps its client for that retry.
"""

from contextvars import ContextVar
from typing import Any, cast

from agents.models import openai_chatcompletions
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

_preserve_thinking_blocks: ContextVar[bool] = ContextVar(
    "agentkit_preserve_thinking_blocks", default=False
)


# Extend the installed converter, not the SDK's base: newer databricks-openai releases install their
# own (for DatabricksChatCompletionsModel), and replacing it would stop Claude signature replay.
class _ThinkingBlockConverter(openai_chatcompletions.Converter):
    """Converter that replays thinking blocks while ``ReasoningChatCompletionsModel`` builds a request."""

    @classmethod
    def items_to_messages(cls, *args: Any, **kwargs: Any) -> Any:
        if _preserve_thinking_blocks.get():
            kwargs["preserve_thinking_blocks"] = True
        return super().items_to_messages(*args, **kwargs)


openai_chatcompletions.Converter = _ThinkingBlockConverter  # ty: ignore[invalid-assignment]


def _is_reasoning_content(content: Any) -> bool:
    """Return True for list content made only of text and reasoning parts (not e.g. images)."""
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(part, str)
            or (isinstance(part, dict) and part.get("type") in ("text", "reasoning"))
            for part in content
        )
    )


def _split_message(message: Any) -> None:
    """Move list content's reasoning onto ``thinking_blocks`` and leave the answer as a string."""
    content = getattr(message, "content", None)
    if not isinstance(content, list) or not _is_reasoning_content(content):
        return
    texts = []
    thinking_blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif part.get("type") == "text":
            # Any text part is answer text, whatever extra keys it has (Gemini's thoughtSignature).
            texts.append(part.get("text") or "")
        else:
            for summary in part.get("summary") or []:
                if not isinstance(summary, dict):
                    continue
                block: dict[str, Any] = {"type": "thinking", "thinking": summary.get("text") or ""}
                if summary.get("signature"):
                    block["signature"] = summary["signature"]
                thinking_blocks.append(block)
    message.content = "".join(texts) if texts else None
    if thinking_blocks:
        message.thinking_blocks = thinking_blocks


def _split_response(response: Any) -> None:
    for choice in getattr(response, "choices", None) or []:
        _split_message(getattr(choice, "message", None) or getattr(choice, "delta", None))


def _restore_messages(messages: Any) -> None:
    """Send signed ``thinking_blocks`` back as reasoning parts; unsigned ones are dropped."""
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        parts: list[Any] = [
            {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": block.get("thinking") or "",
                        "signature": block["signature"],
                    }
                ],
            }
            for block in message.pop("thinking_blocks", None) or []
            if isinstance(block, dict)
            and block.get("type") == "thinking"
            and block.get("signature")
        ]
        if not parts:
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            parts.extend(content)
        message["content"] = parts


class _Reply:
    """Tracks whether a reply so far is only unsigned reasoning, with no answer or tool call."""

    def __init__(self) -> None:
        self.reasoning = False
        self.signed = False
        self.answered = False

    def add(self, response: Any) -> None:
        for choice in getattr(response, "choices", None) or []:
            message = getattr(choice, "message", None) or getattr(choice, "delta", None)
            content = getattr(message, "content", None)
            if (isinstance(content, str) and content.strip()) or getattr(
                message, "tool_calls", None
            ):
                self.answered = True
            for block in getattr(message, "thinking_blocks", None) or []:
                self.reasoning = True
                self.signed |= isinstance(block, dict) and bool(block.get("signature"))

    @property
    def unanswered(self) -> bool:
        # Signed reasoning can't be stitched to a retry's, so only unsigned replies are retried.
        return self.reasoning and not self.signed and not self.answered


class _SplitStream:
    """Async stream that splits each chunk's reasoning out of ``delta.content``.

    Some models (Qwen after a tool call) sometimes stop after reasoning, with no answer. The
    request is then sent once more and its chunks continue this stream.
    """

    def __init__(self, stream: Any, retry: Any = None):
        self._stream = stream
        self._retry = retry
        self._reply = _Reply()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                chunk = await self._stream.__anext__()
            except StopAsyncIteration:
                if self._retry is None or not self._reply.unanswered:
                    raise
                self._stream, self._retry = await self._retry(), None
                continue
            _split_response(chunk)
            self._reply.add(chunk)
            return chunk


class _Completions:
    def __init__(self, completions: Any):
        self._completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    async def create(self, **kwargs: Any) -> Any:
        _restore_messages(kwargs.get("messages"))
        response = await self._completions.create(**kwargs)
        if kwargs.get("stream"):
            return _SplitStream(response, retry=lambda: self._completions.create(**kwargs))
        _split_response(response)
        reply = _Reply()
        reply.add(response)
        if reply.unanswered:
            response = await self._completions.create(**kwargs)
            _split_response(response)
        return response


class _Client:
    """Client proxy whose ``chat.completions.create`` handles reasoning list content."""

    def __init__(self, client: Any):
        self._client = client
        self.chat = _Chat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def with_options(self, *args: Any, **kwargs: Any) -> "_Client":
        return _Client(self._client.with_options(*args, **kwargs))


class _Chat:
    def __init__(self, chat: Any):
        self._chat = chat
        self.completions = _Completions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class ReasoningChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat completions model that accepts reasoning list content from any model."""

    def __init__(self, *, model: str, openai_client: Any):
        super().__init__(model=model, openai_client=cast(Any, _Client(openai_client)))

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        token = _preserve_thinking_blocks.set(True)
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            _preserve_thinking_blocks.reset(token)
