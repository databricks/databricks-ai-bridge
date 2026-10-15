"""Gemini tool-call metadata must survive LangChain history round trips."""

import json
from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import AIMessage, message_to_dict, messages_from_dict

from databricks_langchain.chat_models import (
    _convert_dict_to_message,
    _convert_dict_to_message_chunk,
    _convert_message_to_dict,
)


def _tool_call(call_id: str, arguments: str, signature: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "echo", "arguments": arguments},
        "thoughtSignature": signature,
    }


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("arguments", ['{"text":"one"}', "not-json"])
def test_gemini_signature_survives_tool_call_round_trip(streaming, checkpointed, arguments):
    raw: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [_tool_call("call-1", arguments, "opaque-one")],
    }
    original = deepcopy(raw)
    if streaming:
        first = deepcopy(raw)
        first["tool_calls"][0]["index"] = 0
        first["tool_calls"][0]["function"]["arguments"] = arguments[:4]
        last = {"tool_calls": [{"index": 0, "function": {"arguments": arguments[4:]}}]}
        message = _convert_dict_to_message_chunk(first, "assistant")
        message += _convert_dict_to_message_chunk(last, "assistant")
    else:
        message = _convert_dict_to_message(raw, None)
    if checkpointed:
        message = messages_from_dict(json.loads(json.dumps([message_to_dict(message)])))[0]

    restored = _convert_message_to_dict(message)

    assert restored["tool_calls"][0]["thoughtSignature"] == "opaque-one"
    restored_arguments = restored["tool_calls"][0]["function"]["arguments"]
    if arguments.endswith("}"):
        assert json.loads(restored_arguments) == json.loads(arguments)
    else:
        assert restored_arguments == arguments
    assert restored["tool_calls"][0]["id"] == "call-1"
    assert raw == original


def test_gemini_signatures_are_local_to_each_message_when_provider_reuses_ids():
    messages = [
        _convert_dict_to_message(
            {"role": "assistant", "tool_calls": [_tool_call("echo", arguments, signature)]},
            None,
        )
        for arguments, signature in [
            ('{"text":"one"}', "opaque-one"),
            ('{"text":"two"}', "opaque-two"),
        ]
    ]

    restored = [_convert_message_to_dict(message) for message in messages]

    assert [message["tool_calls"][0]["thoughtSignature"] for message in restored] == [
        "opaque-one",
        "opaque-two",
    ]
    assert [message["tool_calls"][0]["id"] for message in restored] == ["echo", "echo"]


def test_gemini_signature_follows_matching_call_id_without_overriding_parsed_arguments():
    raw_calls = [
        _tool_call("call-1", '{"text":"one"}', "opaque-one"),
        _tool_call("call-2", '{"text":"two"}', "opaque-two"),
    ]
    message = _convert_dict_to_message({"role": "assistant", "tool_calls": raw_calls}, None)
    assert isinstance(message, AIMessage)
    message.tool_calls.reverse()
    message.tool_calls[0]["args"] = {"text": "edited"}
    original = deepcopy(message)

    restored = _convert_message_to_dict(message)

    assert [call["thoughtSignature"] for call in restored["tool_calls"]] == [
        "opaque-two",
        "opaque-one",
    ]
    assert restored["tool_calls"][0]["function"]["arguments"] == '{"text": "edited"}'
    assert message == original


def test_raw_tool_call_fallback_preserves_only_supported_fields():
    raw = _tool_call("call-1", '{"text":"one"}', "opaque-one")
    raw["unsupported_metadata"] = "drop-this"
    message = AIMessage(content="")
    message.additional_kwargs["tool_calls"] = [raw]

    restored = _convert_message_to_dict(message)

    assert restored["tool_calls"] == [{k: v for k, v in raw.items() if k != "unsupported_metadata"}]
