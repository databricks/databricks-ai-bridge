"""Unit tests for the interactive `mason create` project generator."""

from __future__ import annotations

import json
import pathlib
from unittest import mock

import tomli
from click.testing import CliRunner

import databricks_mason.create as create_mod
from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError


class _Ctx:
    def __init__(self, output: str = "text", profile: str | None = None):
        self.output = output
        self.profile = profile


def _fake_scaffold(
    destination: pathlib.Path,
    *,
    framework: str,
    profile: str | None,
    chat_app_enabled: bool,
    repo: str | None = None,
    ref: str | None = None,
):
    destination.mkdir(parents=True)
    AgentProject.create(destination, framework=framework).write()
    return mock.Mock(
        framework=framework,
        template=f"agent-{framework}",
        directory=destination,
        chat_app_enabled=chat_app_enabled,
        env_profile=profile,
    )


def _fake_render(project: pathlib.Path, config) -> None:
    source = project / "agent" / "agent.py"
    source.parent.mkdir(parents=True)
    source.write_text(f"MODEL = {config.model!r}\n", encoding="utf-8")


def test_create_walks_interactive_flow_and_persists_config(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        mock.patch.object(create_mod, "scaffold_project", side_effect=_fake_scaffold) as scaffolded,
        mock.patch.object(create_mod, "render_project", side_effect=_fake_render),
    ):
        result = CliRunner().invoke(
            create_mod.create,
            [],
            obj=_Ctx(profile="workspace-profile"),
            input=(
                "support-agent\n"
                "support-agent\n"
                "openai\n"
                "databricks-claude-sonnet-4\n"
                "Answer support questions with concise steps.\n"
                "y\n"
                "y\n"
            ),
        )

    assert result.exit_code == 0, result.output
    assert "Create a Mason agent" in result.output
    assert "Create this project?" in result.output
    project = tmp_path / "support-agent"
    with (project / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest["agent"] == {
        "framework": "openai",
        "name": "support-agent",
        "model": "databricks-claude-sonnet-4",
        "instructions": "Answer support questions with concise steps.",
        "chat_app_enabled": True,
    }
    assert "databricks-claude-sonnet-4" in (project / "agent" / "agent.py").read_text()
    assert scaffolded.call_args.kwargs["profile"] == "workspace-profile"
    assert scaffolded.call_args.kwargs["chat_app_enabled"] is True
    assert scaffolded.call_args.kwargs["framework"] == "openai"


def test_create_non_interactive_json_is_scriptable(tmp_path: pathlib.Path):
    destination = tmp_path / "scripted-agent"
    with (
        mock.patch.object(create_mod, "scaffold_project", side_effect=_fake_scaffold),
        mock.patch.object(create_mod, "render_project", side_effect=_fake_render),
    ):
        result = CliRunner().invoke(
            create_mod.create,
            [
                str(destination),
                "--name",
                "scripted-agent",
                "--framework",
                "openai",
                "--model",
                "custom-endpoint",
                "--instructions",
                "Use tools when needed.",
                "--no-chat-app",
                "--no-interactive",
            ],
            obj=_Ctx(output="json"),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "name": "scripted-agent",
        "framework": "openai",
        "model": "custom-endpoint",
        "instructions": "Use tools when needed.",
        "directory": str(destination),
        "chat_app_enabled": False,
        "profile": None,
    }


def test_create_non_interactive_defaults_to_langgraph_with_chat_app(tmp_path: pathlib.Path):
    destination = tmp_path / "default-agent"
    with (
        mock.patch.object(create_mod, "scaffold_project", side_effect=_fake_scaffold) as scaffolded,
        mock.patch.object(create_mod, "render_project", side_effect=_fake_render),
    ):
        result = CliRunner().invoke(
            create_mod.create,
            [str(destination), "--no-interactive"],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert scaffolded.call_args.kwargs["framework"] == "langgraph"
    assert scaffolded.call_args.kwargs["chat_app_enabled"] is True


def test_create_rejects_interactive_json_before_scaffolding():
    with mock.patch.object(create_mod, "scaffold_project") as scaffolded:
        result = CliRunner().invoke(create_mod.create, [], obj=_Ctx(output="json"))

    assert result.exit_code != 0
    assert "--no-interactive" in result.output
    scaffolded.assert_not_called()


def test_create_confirmation_decline_writes_nothing(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(create_mod, "scaffold_project") as scaffolded:
        result = CliRunner().invoke(
            create_mod.create,
            [],
            obj=_Ctx(),
            input="declined\ndeclined\n\nmodel\ninstructions\nn\nn\n",
        )

    assert result.exit_code != 0
    assert not (tmp_path / "declined").exists()
    scaffolded.assert_not_called()


def test_create_renderer_failure_leaves_no_partial_destination(tmp_path: pathlib.Path):
    destination = tmp_path / "failed-agent"
    with (
        mock.patch.object(create_mod, "scaffold_project", side_effect=_fake_scaffold),
        mock.patch.object(
            create_mod,
            "render_project",
            side_effect=AgentCliError("render failed"),
        ),
    ):
        result = CliRunner().invoke(
            create_mod.create,
            [str(destination), "--no-interactive"],
            obj=_Ctx(),
        )

    assert result.exit_code != 0
    assert "render failed" in result.output
    assert not destination.exists()


def test_create_refuses_existing_destination(tmp_path: pathlib.Path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with mock.patch.object(create_mod, "scaffold_project") as scaffolded:
        result = CliRunner().invoke(
            create_mod.create,
            [str(destination), "--no-interactive"],
            obj=_Ctx(),
        )

    assert result.exit_code != 0
    assert "already exists" in result.output
    scaffolded.assert_not_called()
