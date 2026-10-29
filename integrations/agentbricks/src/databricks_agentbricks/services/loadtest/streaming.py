"""Observe SSE timings and optional successful response content."""

import json
import time


def _lines(response):
    pending = bytearray()
    after_cr = False
    for chunk in response.iter_content(chunk_size=1):
        for value in chunk:
            if value == 10 and after_cr:
                after_cr = False
                continue
            after_cr = value == 13
            if value in (10, 13):
                yield pending.decode("utf-8", errors="replace")
                pending.clear()
            else:
                pending.append(value)
    if pending:
        yield pending.decode("utf-8", errors="replace")


class StreamError(ValueError):
    """A content-free supported-protocol failure suitable for the final report."""


def output_tokens(event):
    """Return explicit provider usage; missing usage is unknown."""
    if not isinstance(event, dict):
        return None
    response = event.get("response")
    message = event.get("message")
    usage = event.get("usage")
    if not usage and isinstance(response, dict):
        usage = response.get("usage")
    if not usage and isinstance(message, dict):
        usage = message.get("usage_metadata")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens", usage.get("output_tokens"))
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class StreamMetrics:
    """Chat Completions, Responses, and runtime answer-text TTFT observations."""

    def __init__(self, *, capture_output=False):
        self.ttft_ms = None
        self.output_tokens = None
        self.supported = False
        self.completed = False
        self.trace_id = None
        self._text = [] if capture_output else None
        self._events = [] if capture_output else None
        self._message_tokens = {}
        self._runtime_tokens = 0

    @property
    def answer(self):
        if self._text is None:
            return None
        return "".join(self._text) if self._text else self._events

    def read(self, response, started, *, timeout=None):
        """Consume the complete stream; retain content only when explicitly requested."""
        data = []
        event_type = ""
        for line in _lines(response):
            if timeout is not None and time.perf_counter() - started >= timeout:
                raise TimeoutError("Request deadline exceeded")
            if not line:
                if data:
                    self.observe(
                        "\n".join(data), event_type, (time.perf_counter() - started) * 1000
                    )
                data, event_type = [], ""
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line.startswith("event:"):
                event_type = line[6:].strip()
        if data:
            self.observe("\n".join(data), event_type, (time.perf_counter() - started) * 1000)
        self.finish()

    def observe(self, data, event_type="", elapsed_ms=None):
        """Observe parsed or raw SSE data; shared with the excluded warm-up request."""
        if data == "[DONE]":
            self.completed = True
            return
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                if self.supported or event_type.startswith(("response.", "run.")):
                    raise StreamError("Invalid supported SSE event") from None
                if self._events is not None:
                    self._events.append(data)
                return
        if self._events is not None:
            self._events.append(data)
        if not isinstance(data, dict):
            return
        kind = data.get("type", event_type)
        if self._text is not None:
            self.trace_id = explicit_trace_id(data) or self.trace_id
        if data.get("error") or (
            isinstance(kind, str)
            and kind
            in {
                "error",
                "response.failed",
                "response.incomplete",
                "run.failed",
                "run.error",
                "run.cancelled",
            }
        ):
            raise StreamError("SSE response reported an error")
        text = None
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            self.supported = True
            choice = choices[0]
            if not isinstance(choice, dict):
                raise StreamError("Invalid chat SSE choice")
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise StreamError("Invalid chat SSE delta")
            text = delta.get("content")
            reason = choice.get("finish_reason")
            if reason in {"length", "content_filter"}:
                raise StreamError("SSE response was truncated")
            self.completed |= reason is not None
        elif isinstance(kind, str) and kind.startswith(("response.", "run.")):
            self.supported = True
            if kind == "response.output_text.delta":
                text = data.get("delta")
            self.completed |= kind in {"response.completed", "run.completed"}
        elif kind == "delta" and isinstance(data.get("content"), str):
            self.supported = True
            text = data["content"]
        if isinstance(text, str) and text and self.ttft_ms is None:
            self.ttft_ms = elapsed_ms
        if isinstance(text, str) and self._text is not None:
            self._text.append(text)
        tokens = output_tokens(data)
        if tokens is not None:
            if kind == "message" and isinstance(data.get("message"), dict):
                identifier = data["message"].get("id")
                previous = 0
                if isinstance(identifier, str) and identifier:
                    previous = self._message_tokens.get(identifier, 0)
                    self._message_tokens[identifier] = tokens
                self._runtime_tokens += tokens - previous
                self.output_tokens = self._runtime_tokens
            else:
                self.output_tokens = tokens

    def finish(self):
        """Require terminal completion only for protocols we recognize."""
        if self.supported and not self.completed:
            raise StreamError("Incomplete supported SSE response")


def explicit_trace_id(value):
    """Accept explicit IDs only; never guess one from answer text or request IDs."""
    if isinstance(value, dict):
        identifier = value.get("trace_id")
        if isinstance(identifier, str) and identifier and len(identifier) <= 256:
            return (
                identifier if all(char.isalnum() or char in "-_" for char in identifier) else None
            )
    return None
