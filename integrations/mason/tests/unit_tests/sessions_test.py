"""Unit tests for `mason sessions items pop` and `sessions stores list` rendering."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason.sessions import items, stores


class _Client:
    def __init__(self, pop_result):
        self._pop_result = pop_result

    def pop_session_item(self, store, session_id):
        return self._pop_result


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def test_pop_shows_the_returned_item():
    client = _Client({"item": {"item_id": "it-1", "data": {"role": "user", "content": "hi"}}})
    result = CliRunner().invoke(
        items, ["pop", "--store", "s", "--session-id", "sid"], obj=_Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert "Popped last item" in result.output
    # The popped item's id and data are surfaced (previously hidden in text mode).
    assert "it-1" in result.output
    assert "role" in result.output and "hi" in result.output


def test_pop_empty_session_reports_empty():
    result = CliRunner().invoke(
        items, ["pop", "--store", "s", "--session-id", "sid"], obj=_Ctx(_Client({}))
    )
    assert result.exit_code == 0, result.output
    assert "already empty" in result.output


class _StoreListClient:
    def __init__(self, page):
        self._page = page
        self.calls = []

    def list_session_stores(self, page_size=None, page_token=None):
        self.calls.append((page_size, page_token))
        return self._page


def test_store_list_shows_resource_name_and_drops_creator():
    page = {
        "session_stores": [
            {
                "session_store_name": "my-sessions",
                "session_store_id": "sess-abc123",
                "creator_user_id": "user-99",
            }
        ]
    }
    result = CliRunner().invoke(stores, ["list"], obj=_Ctx(_StoreListClient(page)))
    assert result.exit_code == 0, result.output
    assert "my-sessions" in result.output  # resource name is the human-readable store name
    assert "RESOURCE NAME" in result.output.upper()
    assert "sess-abc123" not in result.output  # the opaque id is not shown
    assert result.output.upper().count("NAME") == 1  # single "Resource name" column, no duplicate
    assert "CREATOR" not in result.output.upper()  # creator column removed
    assert "user-99" not in result.output


def test_store_list_defaults_page_size_to_25():
    client = _StoreListClient({"session_stores": []})
    result = CliRunner().invoke(stores, ["list"], obj=_Ctx(client))
    assert result.exit_code == 0, result.output
    assert client.calls == [(25, None)]
