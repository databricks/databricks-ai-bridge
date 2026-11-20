"""Legacy chat display does not rewrite provider data or ordinary agent output."""

import copy
import json

import pytest

from databricks_agentkit.sandbox.display import chat_snapshot


def reasoning(text="", signature="test-signature"):
    return json.dumps(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text, "signature": signature}],
            }
        ]
    )


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_multiple_model_turns_and_partial_signed_block_keep_only_answer(status):
    first = reasoning("reasoning summary", "") + reasoning()
    second = reasoning("another summary")
    text = first + "Checking a tool. " + second + "The answer is 42."
    row = {"status": status, "stream": {"text": text}}
    before = copy.deepcopy(row)
    assert chat_snapshot(row)["display_text"] == "Checking a tool. The answer is 42."
    assert row == before
    # The transport may split a long signature across snapshots.
    for end in range(len('[{"type": "reasoning"'), len(second)):
        row["stream"]["text"] = first + "Checking a tool. " + second[:end]
        assert chat_snapshot(row)["display_text"] == "Checking a tool. "


def test_final_reply_overrides_stream_and_raw_result_is_preserved():
    raw = reasoning() + "**Answer**"
    row = {"result": {"result": {"reply": raw}}, "stream": {"text": "Tool preamble"}}
    before = copy.deepcopy(row)
    result = chat_snapshot(row)
    assert result["display_text"] == "**Answer**"
    assert result["result"]["result"]["reply"] == raw
    assert row == before


@pytest.mark.parametrize(
    "text",
    [
        "An ordinary answer",
        '{"count": 42}',
        '[{"type": "reasoning", "text": "user JSON without a signature"}]',
        "```json\n" + reasoning() + "\n```",
        "~~~json\n" + reasoning() + "\n~~~",
        "`" + reasoning() + "`",
        "```json\n" + reasoning()[:70],
    ],
)
def test_ordinary_json_and_code_are_not_provider_metadata(text):
    row = {"stream": {"text": text}, "status": "running"}
    assert chat_snapshot(row) == row


def test_json_result_is_not_replaced_by_legacy_stream():
    row = {"result": {"result": {"count": 42}}, "stream": {"text": reasoning()}}
    assert chat_snapshot(row) == row


def test_reasoning_only_is_an_explicit_empty_display_override():
    assert chat_snapshot({"stream": {"text": reasoning()}})["display_text"] == ""
