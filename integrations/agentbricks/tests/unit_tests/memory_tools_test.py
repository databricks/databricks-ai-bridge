"""Tests for the framework memory tools (``remember`` / ``recall``) over the AgentKit SDK."""

import importlib
import json
from typing import Any, Optional

import pytest

from databricks_agentbricks.errors import AgentCliError

STORE = "agent-memory"


class FakeMemoryApi:
    """In-memory stand-in for the managed memory entries API, keyed like the server."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.search_response: Optional[dict[str, Any]] = None
        self.clients_built: list[object] = []
        # When set, List serves pages of this size regardless of the requested page_size.
        self.list_page_size: Optional[int] = None
        self._next_id = 1

    def seed(
        self,
        path: str,
        content: Optional[str],
        *,
        actor_id: str = "alice",
        session_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store an entry directly, bypassing the API; returns its resource name."""
        name = f"memory-stores/{STORE}/entries/e{self._next_id}"
        self._next_id += 1
        entry = {"name": name, "actor_id": actor_id, "path": path, "content": content}
        if session_id is not None:
            entry["session_id"] = session_id
        if description is not None:
            entry["description"] = description
        self.entries[name] = entry
        return name

    def create_memory_entry(
        self, store, actor_id, path, *, content=None, description=None, session_id=None, **_
    ):
        self.calls.append("create")
        for entry in self.entries.values():
            if (entry["actor_id"], entry.get("session_id"), entry["path"]) == (
                actor_id,
                session_id,
                path,
            ):
                raise AgentCliError("exists", error_code="ALREADY_EXISTS")
        name = self.seed(
            path, content, actor_id=actor_id, session_id=session_id, description=description
        )
        return dict(self.entries[name])

    def list_memory_entries(self, store, actor_id, *, path=None, page_token=None, **_):
        self.calls.append("list")
        matches = [
            dict(entry)
            for entry in self.entries.values()
            if entry["actor_id"] == actor_id and (path is None or entry["path"] == path)
        ]
        if self.list_page_size is None:
            return {"managed_memory_entries": matches}
        start = int(page_token or 0)
        end = start + self.list_page_size
        page: dict[str, Any] = {"managed_memory_entries": matches[start:end]}
        if end < len(matches):
            page["next_page_token"] = str(end)
        return page

    def delete_memory_entry(self, store, memory_id):
        self.calls.append("delete")
        del self.entries[f"memory-stores/{store}/entries/{memory_id}"]
        return {}

    def update_memory_entry(self, store, memory_id, *, content=None, description=None):
        self.calls.append("update")
        entry = self.entries[f"memory-stores/{store}/entries/{memory_id}"]
        if content is not None:
            entry["content"] = content
        if description is not None:
            entry["description"] = description
        return dict(entry)

    def search_memory_entries(self, store, actor_id, query, **kwargs):
        self.search_calls.append({"actor_id": actor_id, "query": query, **kwargs})
        if self.search_response is not None:
            return self.search_response
        return {
            "results": [
                {"managed_memory_entry": dict(entry), "score": 1.0}
                for entry in self.entries.values()
                if entry["actor_id"] == actor_id
            ]
        }

    def get_memory_store(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("memory tools must not fetch the store")


@pytest.fixture(params=["langgraph", "openai"])
def framework(request):
    pytest.importorskip("langchain_core.tools" if request.param == "langgraph" else "agents")
    return request.param


def _module(framework):
    return importlib.import_module(f"databricks_agentkit.{framework}.memory")


@pytest.fixture
def api(monkeypatch, framework):
    fake = FakeMemoryApi()

    def workspace_client():
        fake.clients_built.append(object())
        return fake.clients_built[-1]

    monkeypatch.setattr(_module(framework), "workspace_client", workspace_client)
    monkeypatch.setattr(
        "databricks_agentkit.client._AgentBricksApiClient",
        lambda workspace_client=None, **_: fake,
    )
    return fake


def _tools(framework, actor="alice", store: Optional[str] = STORE):
    return {tool.name: tool for tool in _module(framework).memory_tools(actor, store)}


def _schema(native_tool, framework):
    if framework == "langgraph":
        return native_tool.args_schema.model_json_schema()
    return native_tool.params_json_schema


async def _invoke(native_tool, arguments, framework) -> str:
    if framework == "langgraph":
        return await native_tool.ainvoke(arguments)
    from agents.tool_context import ToolContext

    payload = json.dumps(arguments)
    context = ToolContext(
        context=None,
        tool_name=native_tool.name,
        tool_call_id="test-call",
        tool_arguments=payload,
    )
    return await native_tool.on_invoke_tool(context, payload)


def test_unconfigured_store_returns_no_tools(framework, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)

    assert _tools(framework, store=None) == {}


def test_configured_store_returns_four_tools_with_stable_parameters(framework, api):
    tools = _tools(framework)

    assert set(tools) == {"remember", "recall", "forget", "list_memories"}
    assert set(_schema(tools["remember"], framework)["properties"]) == {"fact", "topic"}
    assert set(_schema(tools["recall"], framework)["properties"]) == {"query", "session_id"}
    assert set(_schema(tools["forget"], framework)["properties"]) == {
        "fact",
        "topic",
        "session_id",
    }
    assert set(_schema(tools["list_memories"], framework)["properties"]) == {"session_id"}


def test_remember_describes_topic_as_a_short_category(framework, api):
    remember = _tools(framework)["remember"]

    if framework == "langgraph":
        assert "category" in remember.description
    else:
        topic = _schema(remember, framework)["properties"]["topic"]
        assert "category" in topic["description"]


@pytest.mark.asyncio
async def test_facts_with_a_shared_prefix_append_to_one_topic_file(framework, api):
    remember = _tools(framework)["remember"]

    await _invoke(remember, {"fact": "User prefers email.", "topic": "preferences"}, framework)
    await _invoke(remember, {"fact": "User prefers dark mode.", "topic": "preferences"}, framework)

    (entry,) = api.entries.values()
    assert entry["path"] == "/preferences.md"
    assert entry["content"] == "User prefers email.\nUser prefers dark mode."
    assert entry["actor_id"] == "alice"
    assert "session_id" not in entry


@pytest.mark.asyncio
async def test_remembering_a_repeated_fact_is_a_no_op(framework, api):
    remember = _tools(framework)["remember"]

    await _invoke(remember, {"fact": "Timezone is PST.", "topic": "profile"}, framework)
    await _invoke(remember, {"fact": "  Timezone is PST.  ", "topic": "profile"}, framework)

    (entry,) = api.entries.values()
    assert entry["content"] == "Timezone is PST."
    assert "update" not in api.calls


@pytest.mark.asyncio
async def test_remember_files_the_fact_under_a_sanitized_topic(framework, api):
    remember = _tools(framework)["remember"]

    result = await _invoke(
        remember, {"fact": "Ships on Fridays.", "topic": "../Work/Q3"}, framework
    )

    (entry,) = api.entries.values()
    assert entry["path"] == "/work-q3.md"
    assert "/work-q3.md" in result


@pytest.mark.asyncio
async def test_blank_fact_is_not_written(framework, api):
    remember = _tools(framework)["remember"]

    result = await _invoke(remember, {"fact": "  \n ", "topic": "preferences"}, framework)

    assert result
    assert api.calls == []


@pytest.mark.asyncio
async def test_tools_build_the_store_once_without_fetching_it(framework, api):
    tools = _tools(framework)

    await _invoke(tools["remember"], {"fact": "One.", "topic": "a"}, framework)
    await _invoke(tools["remember"], {"fact": "Two.", "topic": "b"}, framework)
    await _invoke(tools["recall"], {"query": "one"}, framework)

    assert len(api.clients_built) == 1


@pytest.mark.asyncio
async def test_recall_requests_five_results_by_page_size(framework, api):
    await _invoke(_tools(framework)["recall"], {"query": "email"}, framework)

    (call,) = api.search_calls
    assert call["actor_id"] == "alice"
    assert call["query"] == "email"
    assert call["page_size"] == 5
    assert call.get("limit") is None
    assert call.get("session_id") is None


@pytest.mark.parametrize(("session_id", "sent"), [("s1", "s1"), (" s1 ", "s1"), ("", None)])
@pytest.mark.asyncio
async def test_recall_searches_one_session_when_given(framework, api, session_id, sent):
    await _invoke(
        _tools(framework)["recall"], {"query": "email", "session_id": session_id}, framework
    )

    (call,) = api.search_calls
    assert call.get("session_id") == sent


@pytest.mark.asyncio
async def test_recall_shows_each_hit_path_session_and_content(framework, api):
    api.search_response = {
        "results": [
            {
                "managed_memory_entry": {
                    "name": f"memory-stores/{STORE}/entries/e1",
                    "actor_id": "alice",
                    "path": "/preferences.md",
                    "content": "User prefers email.\nUser prefers dark mode.",
                },
                "score": 2.0,
            },
            {
                "managed_memory_entry": {
                    "name": f"memory-stores/{STORE}/entries/e2",
                    "actor_id": "alice",
                    "path": "/projects.md",
                    "session_id": "s1",
                    "content": "Migrating billing to Lakebase.",
                },
                "score": 1.0,
            },
        ]
    }

    result = await _invoke(_tools(framework)["recall"], {"query": "prefs"}, framework)

    assert result == (
        "/preferences.md:\n"
        "  User prefers email.\n"
        "  User prefers dark mode.\n"
        "/projects.md [session s1]:\n"
        "  Migrating billing to Lakebase."
    )


@pytest.mark.parametrize("session_id", [None, "s1"])
@pytest.mark.asyncio
async def test_recall_labels_an_entry_without_a_path_by_its_name(framework, api, session_id):
    name = f"memory-stores/{STORE}/entries/e1"
    api.search_response = {
        "results": [
            {
                "managed_memory_entry": {
                    "name": name,
                    "session_id": session_id,
                    "content": "User prefers email.",
                }
            }
        ]
    }

    result = await _invoke(_tools(framework)["recall"], {"query": "prefs"}, framework)

    label = f"{name} [session s1]" if session_id else name
    assert result == f"{label}:\n  User prefers email."


@pytest.mark.asyncio
async def test_recall_falls_back_to_the_deprecated_entries_field(framework, api):
    api.search_response = {
        "managed_memory_entries": [
            {
                "name": f"memory-stores/{STORE}/entries/e1",
                "actor_id": "alice",
                "path": "/preferences/User pre.md",
                "content": "User prefers email.",
            }
        ]
    }

    result = await _invoke(_tools(framework)["recall"], {"query": "prefs"}, framework)

    assert result == "/preferences/User pre.md:\n  User prefers email."


@pytest.mark.asyncio
async def test_recall_without_hits(framework, api):
    api.search_response = {"results": []}

    result = await _invoke(_tools(framework)["recall"], {"query": "prefs"}, framework)

    assert result == "No relevant memories."


@pytest.mark.parametrize(
    ("topic", "path"),
    [
        ("preferences", "/preferences.md"),
        ("", "/general.md"),
        ("   ", "/general.md"),
        ("..", "/general.md"),
        ("a/b", "/a-b.md"),
        ("Work Projects", "/work-projects.md"),
        ("--Team__Norms--", "/team__norms.md"),
        ("x" * 200, f"/{'x' * 64}.md"),
        (f"{'y' * 63} z", f"/{'y' * 63}.md"),
        ("Café Préférences", "/cafe-preferences.md"),
        ("日本語", "/general.md"),
    ],
)
def test_topic_path_sanitizes_model_supplied_topics(framework, topic: str, path: str) -> None:
    assert _module(framework)._topic_path(topic) == path


def _content(api, name: str) -> str:
    return api.entries[name]["content"]


@pytest.mark.asyncio
async def test_forget_removes_matching_lines_ignoring_case_and_whitespace(framework, api):
    name = api.seed(
        "/preferences.md",
        "User prefers email.\nTimezone is PST.\nuser  prefers EMAIL.",
        description="Preferences",
    )

    result = await _invoke(
        _tools(framework)["forget"],
        {"fact": "  User prefers   email. ", "topic": "Preferences"},
        framework,
    )

    assert result == "Forgot 2 facts in /preferences.md."
    assert _content(api, name) == "Timezone is PST."
    assert api.entries[name]["description"] == "Preferences"


@pytest.mark.asyncio
async def test_forget_keeps_the_layout_of_the_remaining_lines(framework, api):
    name = api.seed("/preferences.md", "# Preferences\n\n- Email\n- Dark mode")

    result = await _invoke(
        _tools(framework)["forget"], {"fact": "- Email", "topic": "preferences"}, framework
    )

    assert result == "Forgot 1 fact in /preferences.md."
    assert _content(api, name) == "# Preferences\n\n- Dark mode"


@pytest.mark.asyncio
async def test_forget_deletes_the_entry_when_its_last_fact_goes(framework, api):
    api.seed("/preferences.md", "User prefers email.\n\n")

    result = await _invoke(
        _tools(framework)["forget"],
        {"fact": "User prefers email.", "topic": "preferences"},
        framework,
    )

    assert result == "Forgot 1 fact in /preferences.md."
    assert api.entries == {}


@pytest.mark.asyncio
async def test_forget_without_a_matching_line_writes_nothing_and_shows_the_lines(framework, api):
    api.seed("/preferences.md", "User prefers email.\nTimezone is PST.")

    result = await _invoke(
        _tools(framework)["forget"], {"fact": "User prefers", "topic": "preferences"}, framework
    )

    assert result == "Not found in /preferences.md.\n  User prefers email.\n  Timezone is PST."
    assert "update" not in api.calls
    assert "delete" not in api.calls


@pytest.mark.asyncio
async def test_forget_from_a_missing_topic_file(framework, api):
    result = await _invoke(
        _tools(framework)["forget"], {"fact": "Anything.", "topic": "travel"}, framework
    )

    assert result == "No memories in /travel.md."
    assert api.calls == ["list"]


@pytest.mark.asyncio
async def test_forget_a_blank_fact_makes_no_call(framework, api):
    result = await _invoke(_tools(framework)["forget"], {"fact": " \n ", "topic": "a"}, framework)

    assert result
    assert api.calls == []


@pytest.mark.parametrize(
    "path",
    ["/preferences/User pre.md", "/Projects/Q3 Plan.md"],
    ids=["old-scheme-entry", "unsanitized-path"],
)
@pytest.mark.asyncio
async def test_forget_uses_a_path_verbatim(framework, api, path: str):
    api.seed(path, "Ships on Fridays.")

    result = await _invoke(
        _tools(framework)["forget"], {"fact": "ships on fridays.", "topic": path}, framework
    )

    assert result == f"Forgot 1 fact in {path}."
    assert api.entries == {}


@pytest.mark.parametrize("path", ["/", "//x.md", "/a//b.md", "/a/../b.md", "/a/./b.md", "/a/"])
@pytest.mark.asyncio
async def test_forget_rejects_an_invalid_path_without_a_call(framework, api, path: str):
    result = await _invoke(_tools(framework)["forget"], {"fact": "x", "topic": path}, framework)

    assert path in result
    assert api.calls == []


@pytest.mark.parametrize("arguments", [{}, {"session_id": None}, {"session_id": ""}])
@pytest.mark.asyncio
async def test_forget_without_a_session_edits_every_memory_at_the_path(framework, api, arguments):
    api.seed("/preferences.md", "User prefers email.", session_id="s1")
    shared = api.seed("/preferences.md", "User prefers email.\nTimezone is PST.")
    other = api.seed("/projects.md", "User prefers email.")

    result = await _invoke(
        _tools(framework)["forget"],
        {"fact": "User prefers email.", "topic": "preferences", **arguments},
        framework,
    )

    assert result == "Forgot 2 facts in /preferences.md."
    assert set(api.entries) == {shared, other}
    assert _content(api, shared) == "Timezone is PST."


@pytest.mark.asyncio
async def test_forget_with_a_session_edits_only_that_memory(framework, api):
    scoped = api.seed("/preferences.md", "User prefers email.\nTimezone is PST.", session_id="s1")
    shared = api.seed("/preferences.md", "User prefers email.")

    result = await _invoke(
        _tools(framework)["forget"],
        {"fact": "User prefers email.", "topic": "preferences", "session_id": "s1"},
        framework,
    )

    assert result == "Forgot 1 fact in /preferences.md [session s1]."
    assert _content(api, scoped) == "Timezone is PST."
    assert _content(api, shared) == "User prefers email."


@pytest.mark.asyncio
async def test_forget_with_an_unknown_session_writes_nothing(framework, api):
    api.seed("/preferences.md", "User prefers email.")

    result = await _invoke(
        _tools(framework)["forget"],
        {"fact": "User prefers email.", "topic": "preferences", "session_id": "s2"},
        framework,
    )

    assert result == "No memories in /preferences.md [session s2]."
    assert api.calls == ["list"]


@pytest.mark.asyncio
async def test_list_memories_shows_paths_sessions_fact_counts_and_descriptions(framework, api):
    api.seed("/projects.md", "Billing migration.\n\nQ3 launch.", description="Ongoing work")
    api.seed("/preferences.md", "User prefers email.")
    api.seed("/a/old.md", "")
    api.seed("/preferences.md", "Session-only fact.", session_id="s1")
    api.seed("/other-actor.md", "Someone else's fact.", actor_id="bob")

    result = await _invoke(_tools(framework)["list_memories"], {}, framework)

    assert result.splitlines() == [
        "/a/old.md (0 facts)",
        "/preferences.md (1 fact)",
        "/preferences.md [session s1] (1 fact)",
        "/projects.md (2 facts): Ongoing work",
    ]


@pytest.mark.parametrize("session_id", ["s1", " s1 "])
@pytest.mark.asyncio
async def test_list_memories_lists_one_session_when_given(framework, api, session_id):
    api.seed("/preferences.md", "User prefers email.")
    api.seed("/preferences.md", "Session-only fact.", session_id="s1")
    api.seed("/projects.md", "Other session fact.", session_id="s2")

    result = await _invoke(
        _tools(framework)["list_memories"], {"session_id": session_id}, framework
    )

    assert result == "/preferences.md [session s1] (1 fact)"


@pytest.mark.asyncio
async def test_list_memories_reads_every_page(framework, api):
    api.list_page_size = 2
    for index in range(5):
        api.seed(f"/topic-{index}.md", f"Fact {index}.")

    result = await _invoke(_tools(framework)["list_memories"], {}, framework)

    assert result.splitlines() == [f"/topic-{index}.md (1 fact)" for index in range(5)]
    assert api.calls.count("list") == 3


@pytest.mark.asyncio
async def test_list_memories_caps_the_output(framework, api):
    for index in range(103):
        api.seed(f"/topic-{index:03d}.md", "Fact.")

    lines = (await _invoke(_tools(framework)["list_memories"], {}, framework)).splitlines()

    assert len(lines) == 101
    assert lines[0] == "/topic-000.md (1 fact)"
    assert lines[99] == "/topic-099.md (1 fact)"
    assert lines[100] == "…and 3 more."


@pytest.mark.asyncio
async def test_list_memories_with_an_empty_store(framework, api):
    api.seed("/other-actor.md", "Someone else's fact.", actor_id="bob")

    result = await _invoke(_tools(framework)["list_memories"], {}, framework)

    assert result == "No memories yet."
