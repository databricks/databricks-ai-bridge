"""Unit tests for `mason memory entries create --content-file`."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason.memory import entries


class _Client:
    def __init__(self):
        self.calls = []

    def create_memory_entry(self, store, actor_id, path, content=None, description=None,
                            session_id=None, source_type=None):
        self.calls.append({"store": store, "path": path, "content": content})
        return {"name": f"memory-stores/{store}/entries/e", "path": path}


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def test_content_file_is_read_into_content(tmp_path):
    f = tmp_path / "content.md"
    f.write_text("line1\nline2 from file\n")
    client = _Client()
    result = CliRunner().invoke(
        entries,
        ["create", "--store", "s", "--actor-id", "a", "--path", "/p.md", "--content-file", str(f)],
        obj=_Ctx(client),
    )
    assert result.exit_code == 0, result.output
    assert client.calls[0]["content"] == "line1\nline2 from file\n"


def test_content_and_content_file_are_mutually_exclusive(tmp_path):
    f = tmp_path / "content.md"
    f.write_text("x")
    client = _Client()
    result = CliRunner().invoke(
        entries,
        [
            "create", "--store", "s", "--actor-id", "a", "--path", "/p.md",
            "--content", "inline", "--content-file", str(f),
        ],
        obj=_Ctx(client),
    )
    assert result.exit_code != 0
    assert "not both" in result.output
    assert client.calls == []  # never called the API


def test_missing_content_file_is_rejected_by_click():
    client = _Client()
    result = CliRunner().invoke(
        entries,
        ["create", "--store", "s", "--actor-id", "a", "--path", "/p.md",
         "--content-file", "/no/such/file.md"],
        obj=_Ctx(client),
    )
    assert result.exit_code != 0  # click validates existence
    assert client.calls == []
