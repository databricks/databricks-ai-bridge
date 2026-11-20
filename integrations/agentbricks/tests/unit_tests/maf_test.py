"""Native MAF execution against managed state with only external transports replaced."""

import pytest
from template_history_fixtures import SessionStoreTransport

pytest.importorskip("agent_framework")

from agent_framework import (
    Agent,
    AgentSession,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)

from databricks_agentkit.maf import sessions
from databricks_agentkit.maf.sessions import DatabricksHistoryProvider


class ReplyClient(BaseChatClient):
    """Deterministic model transport; Agent and its HistoryProvider execute normally."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.options = []

    def _inner_get_response(self, *, messages, stream=False, options, **kwargs):
        self.calls.append(list(messages))
        self.options.append(dict(options))
        response = ChatResponse(messages=[Message("assistant", ["hello"], message_id="answer")])
        if not stream:

            async def reply():
                return response

            return reply()

        async def updates():
            yield ChatResponseUpdate(
                role="assistant", contents=[Content.from_text("hello")], message_id="answer"
            )

        return ResponseStream(updates(), finalizer=lambda _: response)


@pytest.mark.asyncio
async def test_native_agent_restores_managed_history_after_new_provider(monkeypatch):
    transport = SessionStoreTransport()
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: "managed")
    first = Agent(
        client=ReplyClient(),
        context_providers=[DatabricksHistoryProvider("alice", client=transport)],
    )
    await first.run("first turn", session=AgentSession(session_id="conversation"))
    second_client = ReplyClient()
    second = Agent(
        client=second_client,
        context_providers=[DatabricksHistoryProvider("alice", client=transport)],
    )
    await second.run("second turn", session=AgentSession(session_id="conversation"))
    assert [message.text for message in second_client.calls[0]] == [
        "first turn",
        "hello",
        "second turn",
    ]
    assert len(transport.items["conversation"]) == 4
    provider = DatabricksHistoryProvider("alice", client=transport)
    native = Message(
        "assistant", [Content.from_function_call(call_id="call", name="sample", arguments={"n": 1})]
    )
    await provider.save_messages("conversation", [native])
    restored = await DatabricksHistoryProvider("alice", client=transport).get_messages(
        "conversation"
    )
    assert restored[-1].to_dict() == native.to_dict()
    with pytest.raises(ValueError, match="different actor"):
        await DatabricksHistoryProvider("bob", client=transport).get_messages("conversation")


@pytest.mark.asyncio
async def test_local_history_is_partitioned_by_actor_and_session(monkeypatch):
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: None)
    monkeypatch.setattr(sessions, "_local_messages", {})
    provider = DatabricksHistoryProvider("alice")
    await provider.save_messages("conversation", [Message("user", ["private"])])
    assert (await DatabricksHistoryProvider("alice").get_messages("conversation"))[
        0
    ].text == "private"
    assert await DatabricksHistoryProvider("bob").get_messages("conversation") == []
    assert await provider.get_messages("other") == []
    with pytest.raises(ValueError, match="session_id"):
        await provider.get_messages(None)


def test_tool_messages_keep_native_call_and_result_content():
    from databricks_agentkit.maf.messages import serialize_message

    call = serialize_message(
        Message(
            "assistant",
            [Content.from_function_call(call_id="call", name="sample", arguments='{"n":1}')],
        )
    )
    result = serialize_message(
        Message("tool", [Content.from_function_result(call_id="call", result="finished")])
    )
    assert call["tool_calls"] == [{"id": "call", "name": "sample", "args": {"n": 1}}]
    assert result == {"role": "tool", "content": "finished"}


def test_declared_user_tools_fail_before_client_construction(monkeypatch):
    from databricks_agentkit.maf import mcp
    from databricks_agentkit.runtime.tool_manifest import ToolManifestError, ToolRecord

    monkeypatch.setattr(
        mcp,
        "load_tools",
        lambda **_: (ToolRecord("private", "mcp", service="service", auth="user"),),
    )
    monkeypatch.setattr(
        mcp, "workspace_client", lambda: pytest.fail("auth rejected before client construction")
    )
    with pytest.raises(ToolManifestError, match="only App-authenticated"):
        mcp.mcp_tools()


def test_sandbox_policy_overrides_all_untrusted_metadata():
    from databricks_agentkit.maf.mcp import _ConfiguredMCPTool

    connection = _ConfiguredMCPTool(
        name="sandbox", url="https://example.test/mcp", protected_meta={"policy": "trusted"}
    )
    connection._tool_call_meta_by_name["execute"] = {"policy": "server-controlled"}
    _, meta = connection._prepare_call_kwargs(
        "execute", {"_meta": {"policy": "caller-controlled", "other": "kept"}}
    )
    assert meta == {"policy": "trusted", "other": "kept"}
