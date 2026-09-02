"""Unit tests for `mason sessions items pop` rendering."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason.sessions import items


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
