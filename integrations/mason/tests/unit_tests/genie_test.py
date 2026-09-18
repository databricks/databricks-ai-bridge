import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_mason.runtime import genie

SPACE = "a" * 32
CONVERSATION = "b" * 32
MESSAGE = "c" * 32
ATTACHMENT = "d" * 32


def _response(**values):
    return SimpleNamespace(as_dict=lambda: values)


@pytest.fixture
def client(monkeypatch):
    client = MagicMock()
    client.config.host = "https://example.databricks.com/"
    client.genie.start_conversation.return_value.response = SimpleNamespace(
        conversation_id=CONVERSATION, message_id=MESSAGE
    )
    client.genie.create_message.return_value.response = SimpleNamespace(message_id=MESSAGE)
    client.genie.get_message.return_value = _response(
        status="COMPLETED", attachments=[{"text": {"content": "42"}}]
    )
    monkeypatch.setattr(genie, "workspace_client", lambda: client)
    return client


@pytest.fixture
def deadline(monkeypatch):
    """Control deadline expiry at a chosen wait boundary, preserving real worker-thread calls."""
    clock = SimpleNamespace(now=0.0, expired_result=None, budgets=[])

    async def wait_for(awaitable, timeout):
        clock.budgets.append(timeout)
        if timeout <= 0:
            awaitable.close()
            raise asyncio.TimeoutError
        result = await awaitable
        if result is clock.expired_result:
            raise asyncio.TimeoutError
        return result

    async def sleep(delay):
        clock.now += delay

    monkeypatch.setattr(genie, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        genie,
        "asyncio",
        SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            create_task=asyncio.create_task,
            shield=asyncio.shield,
            to_thread=asyncio.to_thread,
            wait_for=wait_for,
            sleep=sleep,
            TimeoutError=asyncio.TimeoutError,
        ),
    )
    return clock


@pytest.mark.asyncio
async def test_ask_starts_and_returns_grounded_answer(client):
    answer = await genie.GenieAgent(SPACE).ask("How many?")
    client.genie.start_conversation.assert_called_once_with(SPACE, "How many?")
    client.genie.get_message.assert_called_once_with(SPACE, CONVERSATION, MESSAGE)
    assert answer["status"] == "COMPLETED"
    assert answer["attachments"][0]["text"]["content"] == "42"
    assert answer["conversation_id"] == CONVERSATION
    assert answer["message_id"] == MESSAGE
    assert CONVERSATION in answer["deep_link"]


@pytest.mark.asyncio
async def test_followup_uses_existing_conversation(client):
    answer = await genie.GenieAgent(SPACE).ask("And yesterday?", CONVERSATION)
    client.genie.create_message.assert_called_once_with(SPACE, CONVERSATION, "And yesterday?")
    client.genie.start_conversation.assert_not_called()
    assert answer["conversation_id"] == CONVERSATION


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"])
async def test_terminal_errors_preserved_without_retry_or_reexecution(client, status):
    client.genie.get_message.return_value = _response(status=status, error={"message": "detail"})
    answer = await genie.GenieAgent(SPACE).poll(CONVERSATION, MESSAGE)
    assert answer["status"] == status
    assert answer["error"]["message"] == "detail"
    client.genie.get_message.assert_called_once()
    client.genie.start_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_preserves_resume_ids(client, deadline):
    client.genie.get_message.return_value = _response(status="EXECUTING_QUERY")
    answer = await genie.GenieAgent(SPACE, wait_seconds=0.01).poll(CONVERSATION, MESSAGE)
    assert answer["timed_out"] is True
    assert answer["status"] == "EXECUTING_QUERY"
    assert answer["conversation_id"] == CONVERSATION
    assert answer["message_id"] == MESSAGE
    assert deadline.budgets == [0.01, 0.01]
    client.genie.get_message.assert_called_once()
    client.genie.create_message.assert_not_called()


@pytest.mark.asyncio
async def test_inflight_get_timeout_preserves_resume_ids(client, deadline):
    deadline.expired_result = client.genie.get_message.return_value
    answer = await genie.GenieAgent(SPACE, wait_seconds=0.01).poll(CONVERSATION, MESSAGE)
    assert answer["timed_out"] is True
    assert answer["status"] == "IN_PROGRESS"
    assert answer["conversation_id"] == CONVERSATION
    assert answer["message_id"] == MESSAGE
    assert deadline.budgets == [0.01, 0.01]
    client.genie.get_message.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id", [None, CONVERSATION])
async def test_slow_submission_is_indeterminate_without_invented_ids(
    client, deadline, conversation_id
):
    submit = (
        client.genie.start_conversation if conversation_id is None else client.genie.create_message
    )
    deadline.expired_result = submit.return_value
    answer = await genie.GenieAgent(SPACE, wait_seconds=0.01).ask("hello", conversation_id)

    assert answer["timed_out"] is True
    assert answer["status"] == "INDETERMINATE_SUBMISSION"
    assert answer["indeterminate_submission"] is True
    assert "do not resubmit automatically" in answer["warning"].lower()
    assert "message_id" not in answer
    assert answer.get("conversation_id") == conversation_id
    assert "deep_link" not in answer
    assert deadline.budgets == [0.01, 0.01]
    submit.assert_called_once()
    client.genie.get_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ask", "poll"])
async def test_slow_client_setup_is_bounded_without_submitting(
    client, monkeypatch, deadline, method
):
    setup_threads = []

    def setup():
        setup_threads.append(threading.get_ident())
        return client

    monkeypatch.setattr(genie, "workspace_client", setup)
    deadline.expired_result = client
    agent = genie.GenieAgent(SPACE, wait_seconds=0.01)
    answer = (
        await agent.ask("hello") if method == "ask" else await agent.poll(CONVERSATION, MESSAGE)
    )

    assert len(setup_threads) == 1
    assert setup_threads[0] != threading.get_ident()
    assert answer["timed_out"] is True
    assert deadline.budgets == [0.01]
    if method == "ask":
        assert answer["status"] == "NOT_SUBMITTED"
        assert answer["indeterminate_submission"] is False
        assert "conversation_id" not in answer
        assert "message_id" not in answer
    else:
        assert answer["conversation_id"] == CONVERSATION
        assert answer["message_id"] == MESSAGE
    client.genie.start_conversation.assert_not_called()
    client.genie.create_message.assert_not_called()
    client.genie.get_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id", [None, CONVERSATION])
async def test_setup_and_submission_are_deducted_from_poll_budget(
    client, monkeypatch, deadline, conversation_id
):
    def setup():
        deadline.now += 0.04
        return client

    submit = (
        client.genie.start_conversation if conversation_id is None else client.genie.create_message
    )
    operation = submit.return_value

    def submitted(*arguments):
        deadline.now += 0.04
        return operation

    monkeypatch.setattr(genie, "workspace_client", setup)
    submit.side_effect = submitted
    answer = await genie.GenieAgent(SPACE, wait_seconds=0.06).ask("hello", conversation_id)

    assert answer["timed_out"] is True
    assert answer["conversation_id"] == CONVERSATION
    assert answer["message_id"] == MESSAGE
    assert CONVERSATION in answer["deep_link"]
    assert not answer.get("indeterminate_submission")
    assert deadline.budgets == pytest.approx([0.06, 0.02])
    submit.assert_called_once()
    client.genie.get_message.assert_not_called()


@pytest.mark.asyncio
async def test_query_result_client_setup_runs_off_loop(client, monkeypatch):
    setup_threads = []

    def setup():
        setup_threads.append(threading.get_ident())
        return client

    monkeypatch.setattr(genie, "workspace_client", setup)
    client.genie.get_message_attachment_query_result.return_value = _response()
    await genie.GenieAgent(SPACE).query_result(CONVERSATION, MESSAGE, ATTACHMENT)
    assert len(setup_threads) == 1
    assert setup_threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_polling_yields_and_completes(client, monkeypatch):
    from unittest.mock import AsyncMock

    sleep = AsyncMock()
    monkeypatch.setattr(genie.asyncio, "sleep", sleep)
    client.genie.get_message.side_effect = [
        _response(status="EXECUTING_QUERY"),
        _response(status="COMPLETED"),
    ]
    answer = await genie.GenieAgent(SPACE).poll(CONVERSATION, MESSAGE)
    assert answer["status"] == "COMPLETED"
    sleep.assert_awaited_once()
    assert 1 <= sleep.call_args.args[0] <= 5


@pytest.mark.asyncio
async def test_query_result_preserves_schema_and_bounds_rows(client):
    columns = [{"name": "count", "type_name": "LONG"}]
    client.genie.get_message_attachment_query_result.return_value = _response(
        statement_response={
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": columns}, "total_row_count": 150},
            "result": {"data_array": [[str(index)] for index in range(150)]},
        }
    )
    answer = await genie.GenieAgent(SPACE).query_result(CONVERSATION, MESSAGE, ATTACHMENT)
    assert answer["columns"] == columns
    assert len(answer["rows"]) == 100
    assert answer["total_row_count"] == 150
    assert answer["truncated"] is True
    assert answer["attachment_id"] == ATTACHMENT
    client.genie.get_message_attachment_query_result.assert_called_once_with(
        SPACE, CONVERSATION, MESSAGE, ATTACHMENT
    )


@pytest.mark.asyncio
async def test_query_failure_does_not_look_like_empty_success(client):
    client.genie.get_message_attachment_query_result.return_value = _response(
        statement_response={"status": {"state": "FAILED", "error": {"message": "denied"}}}
    )
    answer = await genie.GenieAgent(SPACE).query_result(CONVERSATION, MESSAGE, ATTACHMENT)
    assert answer["status"]["state"] == "FAILED"
    assert answer["status"]["error"]["message"] == "denied"


@pytest.mark.asyncio
async def test_permission_errors_propagate_without_fallback(client):
    client.genie.start_conversation.side_effect = PermissionError("denied")
    with pytest.raises(PermissionError, match="denied"):
        await genie.GenieAgent(SPACE).ask("hello")
    client.genie.create_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["", "../other", "a/b", "x?query=1"])
async def test_reject_bad_conversation_before_api(client, identifier):
    with pytest.raises(ValueError, match="conversation_id"):
        await genie.GenieAgent(SPACE).ask("hello", identifier)
    client.genie.start_conversation.assert_not_called()
    client.genie.create_message.assert_not_called()


@pytest.mark.asyncio
async def test_blank_question_rejected_before_api(client):
    with pytest.raises(ValueError, match="question"):
        await genie.GenieAgent(SPACE).ask("   ")
    client.genie.start_conversation.assert_not_called()
