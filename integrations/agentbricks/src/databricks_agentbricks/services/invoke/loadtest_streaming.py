"""Observe supported answer-text SSE timings without retaining agent content."""

import json
import time


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
    """Chat, Responses, and DurableAgentServer answer-text TTFT observations."""

    def __init__(self):
        self.ttft_ms = None
        self.output_tokens = None
        self.supported = False
        self.completed = False

    def read(self, response, started):
        """Consume the complete stream, retaining timings and usage only."""
        data = []
        event_type = ""
        # One-byte reads avoid Requests buffering short text events until EOF.
        for raw in response.iter_lines(chunk_size=1):
            line = raw.decode("utf-8", errors="replace")
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
        """Observe parsed or raw SSE data; shared with the excluded readiness probe."""
        if data == "[DONE]":
            self.completed = True
            return
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                if self.supported or event_type.startswith(("response.", "run.")):
                    raise StreamError("Invalid supported SSE event") from None
                return
        if not isinstance(data, dict):
            return
        kind = data.get("type", event_type)
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
            text = data.get("content")
        if isinstance(text, str) and text and self.ttft_ms is None:
            self.ttft_ms = elapsed_ms
        tokens = output_tokens(data)
        if tokens is not None:
            self.output_tokens = tokens

    def finish(self):
        """Require terminal completion only for protocols we recognize."""
        if self.supported and not self.completed:
            raise StreamError("Incomplete supported SSE response")
