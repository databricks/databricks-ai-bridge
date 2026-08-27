"""Unit tests for the typed Mason SDK layer over AgentApiClient."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

from databricks_mason import (
    DatabricksAgentClient,
    ManagedMemoryStore,
    Session,
    SessionItem,
    SessionItemPage,
    SessionStore,
)
from databricks_mason.client import AgentApiClient

STORE_ID = "15402663-997b-4300-b695-46913ad90c9f"
MEM_STORE_NAME = f"memory-stores/{STORE_ID}"
ENTRY_ID = "absc-2edddvd"
ENTRY_NAME = f"{MEM_STORE_NAME}/entries/{ENTRY_ID}"
SESSION_STORE = "support-agent-sessions"
SESSION_ID = "case-456"


def mem_store_payload(*, name=MEM_STORE_NAME, display_name="coding_agent_memory", description="d"):
    # Memory stores serialize timestamps as epoch-millis created_at/updated_at.
    return {
        "name": name,
        "display_name": display_name,
        "workspace_id": 123,
        "owner_user_id": "456",
        "created_at": 1770000000000,
        "updated_at": 1770000600000,
        "description": description,
    }


def entry_payload(
    *, name=ENTRY_NAME, actor_id="alice", session_id="s1", path="/m/p.md", content="c"
):
    payload = {
        "name": name,
        "actor_id": actor_id,
        "path": path,
        "content": content,
        "description": "desc",
        "source_type": "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
        "create_time": "2026-08-14T01:02:03Z",
        "update_time": "2026-08-14T02:03:04.500Z",
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def session_store_payload(*, name=SESSION_STORE, description="Support history", metadata=None):
    return {
        "session_store_name": name,
        "session_store_id": STORE_ID,
        "creator_user_id": "123",
        "create_time": "2026-08-14T01:02:03Z",
        "update_time": "2026-08-14T02:03:04.500Z",
        "description": description,
        "metadata": metadata or {"environment": "poc"},
    }


def session_payload(*, session_id=SESSION_ID, actor_id="customer-123", metadata=None):
    return {
        "session_store_name": SESSION_STORE,
        "session_id": session_id,
        "actor_id": actor_id,
        "root_session_id": session_id,
        "metadata": metadata or {"channel": "chat"},
        "create_time": "2026-08-14T03:04:05Z",
        "update_time": "2026-08-14T04:05:06Z",
        "last_activity_time": "2026-08-14T05:06:07Z",
    }


def item_payload(*, item_id="item-1", data=None):
    return {
        "item_id": item_id,
        "data": data or {"type": "message", "role": "user", "content": "Help"},
        "create_time": "2026-08-14T06:07:08Z",
    }


def _sdk():
    api = mock.MagicMock(spec=AgentApiClient)
    return DatabricksAgentClient(api_client=api), api


class TestClientWiring:
    def test_constructs_agent_api_client_from_profile(self):
        with mock.patch("databricks_mason.sdk.AgentApiClient") as ctor:
            client = DatabricksAgentClient(profile="p")
        ctor.assert_called_once_with(profile="p")
        assert client.memory_store._api is ctor.return_value
        assert client.session_store._api is ctor.return_value

    def test_injected_api_client_is_shared(self):
        client, api = _sdk()
        assert client.memory_store._api is api
        assert client.session_store._api is api


class TestMemoryStore:
    def test_create_parses_epoch_millis_timestamps(self):
        client, api = _sdk()
        api.create_memory_store.return_value = mem_store_payload()

        store = client.memory_store.create(display_name="coding_agent_memory", description="d")

        assert isinstance(store, ManagedMemoryStore)
        assert store.store_id == STORE_ID
        assert store.create_time == datetime.fromtimestamp(1770000000, tz=timezone.utc)
        assert store.update_time == datetime.fromtimestamp(1770000600, tz=timezone.utc)
        api.create_memory_store.assert_called_once_with("coding_agent_memory", "d")

    def test_get_by_id(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()

        store = client.memory_store.get(store_id=STORE_ID)

        assert store.store_id == STORE_ID
        api.get_memory_store.assert_called_once_with(STORE_ID)

    def test_get_by_display_name_consumes_pages(self):
        client, api = _sdk()
        api.list_memory_stores.side_effect = [
            {
                "managed_memory_stores": [mem_store_payload(display_name="other")],
                "next_page_token": "p2",
            },
            {"managed_memory_stores": [mem_store_payload()]},
        ]

        store = client.memory_store.get(display_name="coding_agent_memory")

        assert store.store_id == STORE_ID
        assert api.list_memory_stores.call_count == 2

    def test_get_creates_when_absent(self):
        client, api = _sdk()
        api.list_memory_stores.return_value = {"managed_memory_stores": []}
        api.create_memory_store.return_value = mem_store_payload(description="new")

        store = client.memory_store.get(
            display_name="coding_agent_memory", create_if_not_exists=True, description="new"
        )

        assert store.description == "new"
        api.create_memory_store.assert_called_once_with("coding_agent_memory", "new")

    def test_get_validates_selectors(self):
        client, _ = _sdk()
        for kwargs in (
            {},
            {"store_id": STORE_ID, "display_name": "x"},
            {"store_id": STORE_ID, "create_if_not_exists": True},
        ):
            try:
                client.memory_store.get(**kwargs)
                raise AssertionError(f"expected ValueError for {kwargs}")
            except ValueError:
                pass

    def test_get_raises_keyerror_when_missing(self):
        client, api = _sdk()
        api.list_memory_stores.return_value = {"managed_memory_stores": []}
        try:
            client.memory_store.get(display_name="missing")
            raise AssertionError("expected KeyError")
        except KeyError:
            pass

    def test_bound_store_add(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        api.create_memory_entry.return_value = entry_payload()
        store = client.memory_store.get(store_id=STORE_ID)

        entry = store.add(
            actor_id="alice", session_id="s1", path="/m/p.md", content="c", description="desc"
        )

        assert entry.entry_id == ENTRY_ID
        api.create_memory_entry.assert_called_once_with(
            STORE_ID,
            "alice",
            "/m/p.md",
            content="c",
            description="desc",
            session_id="s1",
            source_type=None,
        )

    def test_list_entries_consumes_pages(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        api.list_memory_entries.side_effect = [
            {"managed_memory_entries": [entry_payload()], "next_page_token": "p2"},
            {"managed_memory_entries": [entry_payload(name=f"{MEM_STORE_NAME}/entries/second")]},
        ]
        store = client.memory_store.get(store_id=STORE_ID)

        entries = store.list(actor_id="alice", path_prefix="/m/")

        assert [e.entry_id for e in entries] == [ENTRY_ID, "second"]

    def test_search_parses_results_and_defaults_limit(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        api.search_memory_entries.return_value = {
            "results": [{"managed_memory_entry": entry_payload(), "score": 0.9}]
        }
        store = client.memory_store.get(store_id=STORE_ID)

        results = store.search(actor_id="alice", query="prefs")

        assert [r.entry_id for r in results] == [ENTRY_ID]
        api.search_memory_entries.assert_called_once_with(STORE_ID, "alice", "prefs", limit=100)

    def test_search_falls_back_to_alias(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        api.search_memory_entries.return_value = {"managed_memory_entries": [entry_payload()]}
        store = client.memory_store.get(store_id=STORE_ID)

        results = store.search(actor_id="alice", query="prefs", limit=10)

        assert [r.entry_id for r in results] == [ENTRY_ID]
        api.search_memory_entries.assert_called_once_with(STORE_ID, "alice", "prefs", limit=10)

    def test_search_rejects_out_of_range_limit(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        store = client.memory_store.get(store_id=STORE_ID)
        for limit in (0, 101):
            try:
                store.search(actor_id="alice", query="q", limit=limit)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass

    def test_append_creates_when_absent(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        api.list_memory_entries.return_value = {"managed_memory_entries": []}
        api.create_memory_entry.return_value = entry_payload(content="first")
        store = client.memory_store.get(store_id=STORE_ID)

        entry = store.append(actor_id="alice", session_id="s1", path="/m/p.md", content="first")

        assert entry.content == "first"
        api.create_memory_entry.assert_called_once()

    def test_append_read_modify_writes_existing(self):
        client, api = _sdk()
        api.get_memory_store.return_value = mem_store_payload()
        summary = entry_payload()
        summary.pop("content")
        api.list_memory_entries.return_value = {"managed_memory_entries": [summary]}
        api.get_memory_entry.return_value = entry_payload(content="first", session_id="s1")
        api.update_memory_entry.return_value = entry_payload(content="first\nsecond")
        store = client.memory_store.get(store_id=STORE_ID)

        entry = store.append(actor_id="alice", session_id="s1", path="/m/p.md", content="\nsecond")

        assert entry.content == "first\nsecond"
        api.update_memory_entry.assert_called_once_with(
            STORE_ID, ENTRY_ID, content="first\nsecond", description="desc"
        )


class TestSessionStore:
    def test_set_session_store_makes_no_call(self):
        client, api = _sdk()
        store = client.session_store.set_session_store(SESSION_STORE)
        assert isinstance(store, SessionStore)
        assert store.session_store_name == SESSION_STORE
        assert store.session_store_id is None
        api.get_session_store.assert_not_called()

    def test_set_session_store_requires_name(self):
        client, _ = _sdk()
        try:
            client.session_store.set_session_store("")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_create_list_get(self):
        client, api = _sdk()
        api.create_session_store.return_value = session_store_payload()
        api.list_session_stores.side_effect = [
            {"session_stores": [session_store_payload(name="other")], "next_page_token": "p2"},
            {"session_stores": [session_store_payload()]},
        ]
        api.get_session_store.return_value = session_store_payload()

        created = client.session_store.create(
            session_store_name=SESSION_STORE,
            description="Support history",
            metadata={"environment": "poc"},
        )
        stores = client.session_store.list(page_size=1)
        fetched = client.session_store.get(session_store_name=SESSION_STORE)

        assert isinstance(created, SessionStore)
        assert created.session_store_id == STORE_ID
        assert [s.session_store_name for s in stores] == ["other", SESSION_STORE]
        assert fetched.session_store_name == SESSION_STORE
        api.create_session_store.assert_called_once_with(
            SESSION_STORE, "Support history", {"environment": "poc"}
        )

    def test_update_requires_a_field(self):
        client, api = _sdk()
        store = client.session_store.set_session_store(SESSION_STORE)
        try:
            store.update()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_update_and_delete_store(self):
        client, api = _sdk()
        api.update_session_store.return_value = session_store_payload(
            metadata={"environment": "prod"}
        )
        store = client.session_store.set_session_store(SESSION_STORE)

        updated = store.update(metadata={"environment": "prod"})
        updated.delete()

        assert updated.metadata == {"environment": "prod"}
        api.update_session_store.assert_called_once_with(
            SESSION_STORE, description=None, metadata={"environment": "prod"}
        )
        api.delete_session_store.assert_called_once_with(SESSION_STORE)

    def test_direct_ops_require_one_store_scope(self):
        client, _ = _sdk()
        store = client.session_store.set_session_store(SESSION_STORE)
        try:
            client.session_store.get_session(session_id=SESSION_ID)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        try:
            client.session_store.get_session(
                store, session_store_name="other", session_id=SESSION_ID
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_create_session_and_list_defaults_create_time_desc(self):
        client, api = _sdk()
        api.create_session.return_value = session_payload()
        api.list_sessions.return_value = {"sessions": [session_payload()]}
        store = client.session_store.set_session_store(SESSION_STORE)

        created = store.create_session(actor_id="customer-123", session_id=SESSION_ID)
        sessions = store.list_sessions()

        assert isinstance(created, Session)
        assert [s.session_id for s in sessions] == [SESSION_ID]
        api.create_session.assert_called_once_with(
            SESSION_STORE,
            "customer-123",
            session_id=SESSION_ID,
            parent_session_id=None,
            metadata=None,
        )
        api.list_sessions.assert_called_once_with(
            SESSION_STORE, filter=None, order_by="create_time desc", page_size=None, page_token=None
        )

    def test_update_and_delete_session(self):
        client, api = _sdk()
        api.update_session.return_value = session_payload(metadata={"status": "resolved"})
        session = client.session_store._session_from_response(session_payload(), SESSION_STORE)

        updated = session.update(metadata={"status": "resolved"})
        updated.delete(force=True)

        assert updated.metadata == {"status": "resolved"}
        api.update_session.assert_called_once_with(
            SESSION_STORE, SESSION_ID, {"status": "resolved"}
        )
        api.delete_session.assert_called_once_with(SESSION_STORE, SESSION_ID, force=True)

    def test_fork_unwraps_session_field(self):
        client, api = _sdk()
        api.fork_session.return_value = {
            "session": session_payload(session_id="fork-1", actor_id="agent")
        }
        session = client.session_store._session_from_response(session_payload(), SESSION_STORE)

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

    def test_items_append_list_pop_clear(self):
        client, api = _sdk()
        api.append_session_items.return_value = {"session_items": [item_payload()]}
        api.list_session_items.return_value = {
            "session_items": [item_payload()],
            "next_page_token": "p2",
        }
        api.pop_session_item.side_effect = [{"item": item_payload(item_id="item-2")}, {}]
        session = client.session_store._session_from_response(session_payload(), SESSION_STORE)
        user_data = {"type": "message", "role": "user", "content": "Help"}

        appended = session.append([user_data])
        page = session.list_items(page_size=2, order_by="create_time asc")
        popped = session.pop()
        empty = session.pop()
        session.clear()

        assert isinstance(appended[0], SessionItem)
        assert appended[0].data == item_payload()["data"]
        assert isinstance(page, SessionItemPage)
        assert page.next_page_token == "p2"
        assert popped.item_id == "item-2"
        assert empty is None
        api.append_session_items.assert_called_once_with(SESSION_STORE, SESSION_ID, [user_data])
        api.clear_session_items.assert_called_once_with(SESSION_STORE, SESSION_ID)

    def test_append_requires_an_item(self):
        client, api = _sdk()
        session = client.session_store._session_from_response(session_payload(), SESSION_STORE)
        try:
            session.append([])
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_list_items_rejects_out_of_range_page_size(self):
        client, api = _sdk()
        session = client.session_store._session_from_response(session_payload(), SESSION_STORE)
        for page_size in (0, -1, 101):
            try:
                session.list_items(page_size=page_size)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
        api.list_session_items.assert_not_called()
