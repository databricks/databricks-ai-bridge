"""Content-free SSE metrics for ordinary streaming HTTP responses."""

import pytest
from requests import Response

from databricks_agentbricks.services.loadtest.streaming import (
    StreamError,
    StreamMetrics,
    explicit_trace_id,
    output_tokens,
)


@pytest.mark.parametrize(
    "event",
    [
        {"response": "hello"},
        {"message": "hello"},
        {"usage": {"output_tokens": True}},
        {"usage": {"output_tokens": -1}},
        {"usage": {"output_tokens": "3"}},
    ],
)
def test_invalid_or_missing_token_usage_is_unknown(event):
    assert output_tokens(event) is None


def test_chat_sse_skips_role_metadata_and_preserves_zero_usage():
    metrics = StreamMetrics(capture_output=True)
    metrics.observe({"choices": [{"delta": {"role": "assistant"}}]}, elapsed_ms=10)
    metrics.observe({"choices": [{"delta": {"content": "answer"}}]}, elapsed_ms=80)
    metrics.observe({"choices": [{"delta": {"content": " tail"}}]}, elapsed_ms=90)
    metrics.observe({"choices": [{"delta": {}, "finish_reason": "stop"}]}, elapsed_ms=200)
    metrics.observe({"choices": [], "usage": {"completion_tokens": 0}})
    metrics.finish()
    assert metrics.ttft_ms == 80
    assert metrics.output_tokens == 0
    assert metrics.answer == "answer tail"


def test_responses_sse_preserves_answer_explicit_trace_and_usage():
    metrics = StreamMetrics(capture_output=True)
    metrics.observe({"type": "response.created"}, elapsed_ms=5)
    metrics.observe({"type": "response.output_text.delta", "delta": ""}, elapsed_ms=20)
    metrics.observe({"type": "response.output_text.delta", "delta": "answer"}, elapsed_ms=75)
    metrics.observe({"type": "response.output_text.delta", "delta": " tail"}, elapsed_ms=90)
    metrics.observe(
        {
            "type": "response.completed",
            "response": {"usage": {"output_tokens": 7}},
            "trace_id": "tr-explicit",
        }
    )
    metrics.finish()
    assert metrics.ttft_ms == 75
    assert metrics.output_tokens == 7
    assert metrics.answer == "answer tail"
    assert metrics.trace_id == "tr-explicit"


def test_runtime_sse_preserves_answer_explicit_trace_usage_and_completion():
    metrics = StreamMetrics(capture_output=True)
    metrics.observe({}, event_type="run.started", elapsed_ms=5)
    metrics.observe({"type": "delta", "content": ""}, elapsed_ms=20)
    metrics.observe({"type": "delta", "content": "answer"}, elapsed_ms=75)
    metrics.observe({"type": "delta", "content": " tail"}, elapsed_ms=90)
    metrics.observe({"type": "message", "message": {"usage_metadata": {"output_tokens": 7}}})
    metrics.observe({"trace_id": "tr-explicit"}, event_type="run.completed")
    metrics.finish()
    assert metrics.ttft_ms == 75
    assert metrics.output_tokens == 7
    assert metrics.answer == "answer tail"
    assert metrics.trace_id == "tr-explicit"


@pytest.mark.parametrize("identified", [False, True])
def test_runtime_usage_sums_distinct_model_messages(identified):
    metrics = StreamMetrics()
    for identifier, tokens in (("first", 7), ("second", 11)):
        message = {
            "usage_metadata": {"output_tokens": tokens},
            **({"id": identifier} if identified else {}),
        }
        metrics.observe({"type": "message", "message": message})
    assert metrics.output_tokens == 18
    if identified:
        metrics.observe(
            {"type": "message", "message": {"id": "first", "usage_metadata": {"output_tokens": 7}}}
        )
        assert metrics.output_tokens == 18
        metrics.observe(
            {
                "type": "message",
                "message": {"id": "second", "usage_metadata": {"output_tokens": 12}},
            }
        )
        assert metrics.output_tokens == 19


@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_provider_usage_snapshots_replace_rather_than_sum(protocol):
    metrics = StreamMetrics()
    for tokens in (7, 11, 11):
        metrics.observe(
            {"choices": [], "usage": {"completion_tokens": tokens}}
            if protocol == "chat"
            else {"type": "response.completed", "response": {"usage": {"output_tokens": tokens}}}
        )
    assert metrics.output_tokens == 11


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_real_response_bytes_preserve_named_multiline_events(newline):
    response = Response()
    response._content = newline.join(
        [
            b"event: run.started",
            b"data: {}",
            b"",
            b'data: {"type":"delta",',
            b'data: "content":"caf\xc3\xa9"}',
            b"",
            b"event: run.completed",
            b"data: {}",
            b"",
            b"",
        ]
    )
    response._content_consumed = True
    metrics = StreamMetrics(capture_output=True)
    metrics.read(response, 0)
    assert metrics.completed
    assert metrics.answer == "caf\u00e9"
    assert metrics.ttft_ms is not None


def test_generic_sse_preserves_content_but_does_not_invent_answer_timings():
    metrics = StreamMetrics(capture_output=True)
    metrics.observe({"text": "generic event"}, elapsed_ms=10)
    metrics.observe("plain-text-event", elapsed_ms=20)
    metrics.finish()
    assert metrics.ttft_ms is None
    assert metrics.output_tokens is None
    assert metrics.answer == [{"text": "generic event"}, "plain-text-event"]


@pytest.mark.parametrize(
    "event",
    [
        {"error": {"message": "private failure"}},
        {"type": "response.failed"},
        {"type": "response.incomplete"},
        {"type": "run.failed"},
        {"type": "run.error"},
        {"type": "run.cancelled"},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [{"delta": {}, "finish_reason": "content_filter"}]},
        {"choices": ["invalid"]},
        {"choices": [{"delta": "invalid"}]},
    ],
)
def test_supported_sse_errors_are_content_free(event):
    with pytest.raises(StreamError) as raised:
        StreamMetrics().observe(event)
    assert "private failure" not in str(raised.value)


@pytest.mark.parametrize("protocol", ["responses", "runtime"])
def test_supported_sse_requires_completion(protocol):
    metrics = StreamMetrics()
    metrics.observe(
        {"type": "response.output_text.delta", "delta": "partial"}
        if protocol == "responses"
        else {"type": "delta", "content": "partial"},
        elapsed_ms=50,
    )
    with pytest.raises(StreamError, match="Incomplete"):
        metrics.finish()


@pytest.mark.parametrize("event_type", ["run.started", "response.created"])
def test_malformed_named_supported_sse_is_rejected(event_type):
    with pytest.raises(StreamError, match="Invalid supported SSE event"):
        StreamMetrics().observe("private malformed event", event_type=event_type)


@pytest.mark.parametrize("value", [None, "", "spaces in id", "../escape", "x" * 257, 123])
def test_only_explicit_valid_trace_ids_are_captured(value):
    assert explicit_trace_id({"trace_id": value}) is None
    assert explicit_trace_id({"id": "tr-request-id", "output": "tr-answer-id"}) is None
