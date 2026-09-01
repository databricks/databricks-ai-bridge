"""Tests for typed session-store resources."""

import pytest
from resource_test_fixtures import (
    SESSION_ID,
    SESSION_STORE,
    STORE_ID,
    item_payload,
    resource_client,
    session_payload,
    session_store_payload,
)

from databricks_mason import Session, SessionItem, SessionStore


def test_create_list_and_get_stores() -> None:
    client, api = resource_client()
    api.create_session_store.return_value = session_store_payload()
    api.list_session_stores.side_effect = [
        {"session_stores": [session_store_payload(name="other")], "next_page_token": "p2"},
        {"session_stores": [session_store_payload()]},
    ]
    api.get_session_store.return_value = session_store_payload()

    created = client.session_stores.create(
        SESSION_STORE,
        description="Support history",
        metadata={"environment": "poc"},
    )
    stores = list(client.session_stores.list(page_size=1))
    fetched = client.session_stores.get(SESSION_STORE)

    assert isinstance(created, SessionStore)
    assert created.session_store_id == STORE_ID
    assert [store.name for store in stores] == ["other", SESSION_STORE]
    assert fetched.name == SESSION_STORE
    api.create_session_store.assert_called_once_with(
        SESSION_STORE,
        "Support history",
        {"environment": "poc"},
    )
    assert api.list_session_stores.call_args_list[1].kwargs == {
        "page_size": 1,
        "page_token": "p2",
    }
    api.get_session_store.assert_called_once_with(SESSION_STORE)


def test_update_requires_a_field() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    store = client.session_stores.get(SESSION_STORE)

    with pytest.raises(ValueError, match="at least one"):
        store.update()


def test_update_and_delete_store() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.update_session_store.return_value = session_store_payload(metadata={"environment": "prod"})
    store = client.session_stores.get(SESSION_STORE)

    updated = store.update(metadata={"environment": "prod"})
    updated.delete()

    assert updated.metadata == {"environment": "prod"}
    api.update_session_store.assert_called_once_with(
        SESSION_STORE,
        description=None,
        metadata={"environment": "prod"},
    )
    api.delete_session_store.assert_called_once_with(SESSION_STORE)


def test_create_get_and_list_sessions() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.create_session.return_value = session_payload()
    api.list_sessions.side_effect = [
        {"sessions": [session_payload()], "next_page_token": "p2"},
        {"sessions": [session_payload(session_id="case-789")]},
    ]
    api.get_session.return_value = session_payload()
    store = client.session_stores.get(SESSION_STORE)

    created = store.create_session(actor_id="customer-123", session_id=SESSION_ID)
    sessions = list(store.list_sessions(page_size=10))
    fetched = store.get_session(SESSION_ID)

    assert isinstance(created, Session)
    assert created.store_name == SESSION_STORE
    assert [session.session_id for session in sessions] == [SESSION_ID, "case-789"]
    assert fetched.session_id == SESSION_ID
    api.create_session.assert_called_once_with(
        SESSION_STORE,
        "customer-123",
        session_id=SESSION_ID,
        parent_session_id=None,
        metadata=None,
    )
    assert api.list_sessions.call_args_list[0].kwargs == {
        "filter": None,
        "order_by": "create_time desc",
        "page_size": 10,
        "page_token": None,
    }
    assert api.list_sessions.call_args_list[1].kwargs["page_token"] == "p2"
    api.get_session.assert_called_once_with(SESSION_ID, SESSION_STORE)


def test_update_and_delete_session() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.get_session.return_value = session_payload()
    api.update_session.return_value = session_payload(metadata={"status": "resolved"})
    session = client.session_stores.get(SESSION_STORE).get_session(SESSION_ID)

    updated = session.update(metadata={"status": "resolved"})
    updated.delete()

    assert updated.metadata == {"status": "resolved"}
    api.update_session.assert_called_once_with(
        SESSION_STORE,
        SESSION_ID,
        {"status": "resolved"},
    )
    api.delete_session.assert_called_once_with(SESSION_STORE, SESSION_ID)


def test_fork_unwraps_session_field() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.get_session.return_value = session_payload()
    api.fork_session.return_value = {
        "session": session_payload(session_id="fork-1", actor_id="agent")
    }
    session = client.session_stores.get(SESSION_STORE).get_session(SESSION_ID)

    forked = session.fork(actor_id="agent", up_to_item_id="item-1", session_id="fork-1")

    assert forked.session_id == "fork-1"
    api.fork_session.assert_called_once_with(
        SESSION_STORE,
        SESSION_ID,
        "agent",
        up_to_item_id="item-1",
        session_id="fork-1",
        metadata=None,
    )


def test_item_operations_and_pagination() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.get_session.return_value = session_payload()
    api.append_session_items.return_value = {"session_items": [item_payload()]}
    api.list_session_items.side_effect = [
        {"session_items": [item_payload()], "next_page_token": "p2"},
        {"session_items": [item_payload(item_id="item-2")]},
    ]
    api.pop_session_item.side_effect = [{"item": item_payload(item_id="item-2")}, {}]
    session = client.session_stores.get(SESSION_STORE).get_session(SESSION_ID)
    user_data = {"type": "message", "role": "user", "content": "Help"}

    appended = session.append_items([user_data])
    items = list(session.list_items(page_size=2, order_by="create_time asc"))
    popped = session.pop_item()
    empty = session.pop_item()
    session.clear_items()

    assert isinstance(appended[0], SessionItem)
    assert appended[0].data == item_payload()["data"]
    assert [item.item_id for item in items] == ["item-1", "item-2"]
    assert api.list_session_items.call_args_list[1].kwargs["page_token"] == "p2"
    assert popped is not None
    assert popped.item_id == "item-2"
    assert empty is None
    api.append_session_items.assert_called_once_with(SESSION_STORE, SESSION_ID, [user_data])
    api.clear_session_items.assert_called_once_with(SESSION_STORE, SESSION_ID)


def test_append_items_requires_an_item() -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.get_session.return_value = session_payload()
    session = client.session_stores.get(SESSION_STORE).get_session(SESSION_ID)

    with pytest.raises(ValueError, match="at least one item"):
        session.append_items([])


@pytest.mark.parametrize("page_size", [0, -1, 101])
def test_list_items_rejects_out_of_range_page_size(page_size: int) -> None:
    client, api = resource_client()
    api.get_session_store.return_value = session_store_payload()
    api.get_session.return_value = session_payload()
    session = client.session_stores.get(SESSION_STORE).get_session(SESSION_ID)

    with pytest.raises(ValueError, match="page_size"):
        list(session.list_items(page_size=page_size))
    api.list_session_items.assert_not_called()
