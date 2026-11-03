"""Offline tests for the examples' shared helpers (no Databricks or Anthropic calls)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import _shared  # noqa: E402  # ty: ignore[unresolved-import] - examples/ on sys.path


def args_for(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _shared.add_args(parser)
    namespace = parser.parse_args([])
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def test_defaults_are_cli_auth_and_opus():
    args = args_for()
    assert args.auth == "cli"
    assert args.model == "claude-opus-5-5"
    assert args.budget_cents > 0


def test_missing_env_file_explains_how_to_create_it(tmp_path):
    with pytest.raises(SystemExit, match="Copy .env.anthropic.template"):
        _shared.load_settings(args_for(env_file=str(tmp_path / "nope")))


def test_placeholder_values_are_rejected(tmp_path):
    env = tmp_path / ".env.anthropic"
    env.write_text(
        "ANTHROPIC_API_KEY=sk-ant-api03-REPLACE_ME\nANTHROPIC_ENVIRONMENT_ID=env_1\n"
        "ANTHROPIC_ENVIRONMENT_KEY=sk-ant-oat01-x\n"
    )
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        _shared.load_settings(args_for(env_file=str(env)))


def test_settings_come_from_the_file_not_the_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stray-shell-key")
    env = tmp_path / ".env.anthropic"
    env.write_text(
        "ANTHROPIC_API_KEY=sk-ant-api03-real\nANTHROPIC_ENVIRONMENT_ID=env_1\n"
        "ANTHROPIC_ENVIRONMENT_KEY=sk-ant-oat01-x\n"
    )
    settings = _shared.load_settings(args_for(env_file=str(env)))
    assert settings["ANTHROPIC_API_KEY"] == "sk-ant-api03-real"
    client = _shared.anthropic_client(settings)
    assert str(client.base_url).startswith("https://api.anthropic.com")


def test_budget_shape():
    assert _shared.budget(150) == {
        "type": "limit",
        "max_list_cost": {"amount": "150", "currency": "USD"},
    }


def test_stream_turn_waits_through_requires_action_and_returns_last_message():
    events = [
        SimpleNamespace(
            type="agent.custom_tool_use", name="genie_ask_and_wait", input={"question": "q"}
        ),
        SimpleNamespace(
            type="session.status_idle", stop_reason=SimpleNamespace(type="requires_action")
        ),
        SimpleNamespace(
            type="agent.message", content=[SimpleNamespace(text="Total revenue is $66,471.")]
        ),
        SimpleNamespace(type="session.status_idle", stop_reason=SimpleNamespace(type="end_turn")),
    ]

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            async def gen():
                for event in events:
                    yield event

            return gen()

    async def stream(session_id):
        return Stream()

    api = SimpleNamespace(
        beta=SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(stream=stream)))
    )

    async def run():
        return await _shared.stream_turn(api, "sesn_1", asyncio.Event(), timeout_s=5)

    assert asyncio.run(run()) == "Total revenue is $66,471."
