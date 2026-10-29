"""Load a generated agent template in-process and serve it over the real HTTP runtime."""

import importlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


def _is_template_module(name: str) -> bool:
    # Every template ships top-level `agent` and `runtime` packages, so one template's modules
    # must be gone before another's import, or the cached ones would be reused.
    return name in {"agent", "runtime"} or name.startswith(("agent.", "runtime."))


def import_template(monkeypatch, template: Path) -> tuple[ModuleType, ModuleType]:
    """Import ``template``'s ``runtime.adapter`` and ``agent.agent`` from a clean module cache."""
    monkeypatch.syspath_prepend(str(template))
    for name in list(sys.modules):
        if _is_template_module(name):
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("runtime.adapter"), importlib.import_module("agent.agent")


def forget_template() -> None:
    """Drop the imported template's modules, including ones monkeypatch didn't see imported."""
    for name in list(sys.modules):
        if _is_template_module(name):
            sys.modules.pop(name, None)


def scripted_model(first, after_tool):
    """A chat model that answers ``first`` until a tool result is in the conversation, then
    ``after_tool``; ``model.turns`` records the messages each turn received."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class ScriptedModel(FakeMessagesListChatModel):
        turns: list = []

        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            self.turns.append(messages)
            self.i = int(any(message.type == "tool" for message in messages))
            return self._generate(messages, stop, run_manager, **kwargs)

    return ScriptedModel(responses=[first, after_tool], turns=[])


def durable_app(adapter: ModuleType, **server_options):
    """The template's durable HTTP app, as its ``runtime.main`` wires it."""
    from databricks_agentkit import DurableAgentServer

    app = DurableAgentServer(**server_options)
    app.invoke(adapter.invoke)
    app.recover(adapter.recover)
    return app


@asynccontextmanager
async def runtime_client(app):
    """An HTTP client for ``app`` with its durable runtime started."""
    import httpx

    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://test"
        ) as client:
            yield client
    finally:
        await app._runtime.stop()
