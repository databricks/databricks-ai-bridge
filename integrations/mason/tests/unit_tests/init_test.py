"""Unit tests for `mason init`: template mapping, destination guard, scaffold flow.

The network-touching git clone (`_fetch_template`) is mocked; tests assert the command wires
framework -> template dir, refuses an existing destination, and reports the scaffolded path.
"""

from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest
import tomli
from click.testing import CliRunner

from databricks_mason import init as init_mod
from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import IntegrationRegistry, registry_relative_path


class _Ctx:
    """Stand-in for CliContext: init reads .output and .profile."""

    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


def test_framework_specs_have_repo_ref_path():
    for fw in ("openai", "langgraph"):
        spec = init_mod._TEMPLATES[fw]
        assert spec["repo"] and spec["ref"] and spec["path"]
    assert init_mod._TEMPLATES["openai"]["path"] == "integrations/mason/templates/agent-openai"
    assert (
        init_mod._TEMPLATES["langgraph"]["path"] == "integrations/mason/templates/agent-langgraph"
    )
    assert (
        init_mod._CHAT_APP_TEMPLATES["langgraph"]
        == "integrations/mason/templates/ui/agent-langgraph"
    )
    assert init_mod._CHAT_APP_TEMPLATES["openai"] == "integrations/mason/templates/ui/agent-openai"


def test_template_ref_pins_versioned_template_to_release_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(init_mod, "_installed_version", lambda _: "0.3.0")
    # A released CLI fetches each versioned template tagged for its own version.
    assert init_mod._template_ref("langgraph") == "databricks-mason-v0.3.0"
    assert init_mod._template_ref("openai") == "databricks-mason-v0.3.0"


@pytest.mark.parametrize("installed", ["0.1.0.dev0", "0.2.0+local"])
def test_template_ref_falls_back_to_main_for_unreleased_builds(
    installed: str, monkeypatch: pytest.MonkeyPatch
):
    # Dev/editable/local-version builds have no matching release tag, so fetch `main`.
    monkeypatch.setattr(init_mod, "_installed_version", lambda _: installed)
    assert init_mod._template_ref("langgraph") == "main"


def test_template_ref_falls_back_when_package_not_installed(monkeypatch: pytest.MonkeyPatch):
    def _raise(_):
        raise init_mod.PackageNotFoundError("databricks-mason")

    monkeypatch.setattr(init_mod, "_installed_version", _raise)
    assert init_mod._template_ref("langgraph") == "main"


def test_init_scaffolds_default_directory(tmp_path: pathlib.Path):
    dest = tmp_path / "agent-openai"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir(parents=True)
        (target / "app.yaml").write_text("command: []\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as fetched:
        result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    fetched.assert_called_once()
    # framework's repo + path passed through to the fetch
    assert fetched.call_args.args[0] == init_mod._TEMPLATES["openai"]["repo"]
    assert fetched.call_args.args[2] == "integrations/mason/templates/agent-openai"
    assert (dest / "app.yaml").exists()
    assert "agent-openai" in result.output


def test_init_defaults_to_langgraph_framework(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())  # no --framework
    assert result.exit_code == 0, result.output
    # omitting --framework scaffolds the langgraph template
    assert f.call_args.args[2] == init_mod._TEMPLATES["langgraph"]["path"]


def test_init_persists_selected_framework_and_template(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    with (dest / ".mason" / "project.toml").open("rb") as metadata_file:
        metadata = tomli.load(metadata_file)
    assert metadata == {
        "schema_version": 1,
        "framework": "langgraph",
        "template": "agent-langgraph",
    }


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_init_never_creates_agent_manifest(tmp_path: pathlib.Path, framework: str):
    dest = tmp_path / framework

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", framework, str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert not (dest / "agent.toml").exists()


def test_init_langgraph_creates_empty_code_registry(tmp_path: pathlib.Path):
    dest = tmp_path / "langgraph"

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert IntegrationRegistry.load(dest).integrations == []


def test_init_openai_creates_empty_code_registry(tmp_path: pathlib.Path):
    dest = tmp_path / "openai"

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "openai", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    relative_path = registry_relative_path("openai")
    assert IntegrationRegistry.load(dest, relative_path=relative_path).integrations == []
    assert relative_path == pathlib.Path("agent/databricks_tools.py")


def test_init_langgraph_does_not_vendor_runtime_plumbing(tmp_path: pathlib.Path):
    # Runtime plumbing now lives in the databricks_mason.runtime package (imported, not vendored),
    # so init must not write an agent/mason/ dir into the scaffold.
    dest = tmp_path / "langgraph"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        (target / "agent").mkdir(parents=True)
        (target / "agent" / "agent.py").write_text("USER_AGENT = True\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert (dest / "agent" / "agent.py").read_text() == "USER_AGENT = True\n"
    assert not (dest / "agent" / "mason").exists()


def test_init_openai_does_not_vendor_runtime_plumbing(tmp_path: pathlib.Path):
    dest = tmp_path / "openai"

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "openai", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert not (dest / "agent" / "mason").exists()


def test_init_langgraph_fetches_from_ai_bridge(tmp_path: pathlib.Path):
    dest = tmp_path / "lg"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir(parents=True)

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as fetched:
        result = CliRunner().invoke(
            init_mod.init, ["--framework", "langgraph", str(dest)], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    # langgraph pulls the nested template from the ai-bridge repo
    assert "databricks-ai-bridge" in fetched.call_args.args[0]
    assert fetched.call_args.args[2] == "integrations/mason/templates/agent-langgraph"


def test_init_repo_ref_override(tmp_path: pathlib.Path):
    dest = tmp_path / "ov"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            [
                "--framework",
                "langgraph",
                "--repo",
                "https://example.com/fork.git",
                "--ref",
                "wip",
                str(dest),
            ],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    assert f.call_args.args[0] == "https://example.com/fork.git"  # override wins
    assert f.call_args.args[1] == "wip"


def test_init_langgraph_includes_chat_app_by_default(tmp_path: pathlib.Path):
    dest = tmp_path / "chat"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert f.call_args.args[4] == ("integrations/mason/templates/ui/agent-langgraph",)
    assert "Chat app" in result.output


def test_init_disable_chat_app_omits_langgraph_overlay(tmp_path: pathlib.Path):
    dest = tmp_path / "api-only"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", "--disable-chat-app", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert f.call_args.args[4] == ()  # no chat-app overlay
    assert "Chat app" not in result.output


def test_init_enable_chat_app_flag_is_accepted_no_op(tmp_path: pathlib.Path):
    # Deprecated flag: kept so existing invocations don't break; chat app is on by default anyway.
    dest = tmp_path / "chat"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", "--enable-chat-app", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert f.call_args.args[4] == ("integrations/mason/templates/ui/agent-langgraph",)


def test_init_openai_includes_chat_app_by_default(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init, ["--framework", "openai", str(dest)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert f.call_args.args[4] == ("integrations/mason/templates/ui/agent-openai",)
    assert json.loads(result.output)["chat_app_enabled"] is True


def test_init_disable_chat_app_omits_openai_overlay(tmp_path: pathlib.Path):
    dest = tmp_path / "api-only"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "openai", "--disable-chat-app", str(dest)],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    assert f.call_args.args[4] == ()  # no chat-app overlay
    assert "Chat app" not in result.output


def test_init_json_output(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init, ["--framework", "langgraph", str(dest)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["framework"] == "langgraph"
    assert payload["template"] == "agent-langgraph"
    assert payload["directory"] == str(dest)
    assert payload["chat_app_enabled"] is True


def test_init_refuses_existing_destination(tmp_path: pathlib.Path):
    dest = tmp_path / "exists"
    dest.mkdir()
    with mock.patch.object(init_mod, "_fetch_template") as fetched:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code != 0
    # Rich may wrap the message across lines on a narrow terminal, so match whitespace-insensitively.
    assert "already exists" in " ".join(result.output.split())
    fetched.assert_not_called()


def test_init_rejects_unknown_framework(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        init_mod.init, ["--framework", "nope", str(tmp_path / "x")], obj=_Ctx()
    )
    assert result.exit_code != 0  # click.Choice rejects it


def test_write_env_seeds_profile_from_example(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text(
        "DATABRICKS_CONFIG_PROFILE=DEFAULT\n# MLFLOW_EXPERIMENT_ID=\n"
    )
    wrote = init_mod._write_env(tmp_path, "ml")
    assert wrote is True
    body = (tmp_path / ".env").read_text()
    assert "DATABRICKS_CONFIG_PROFILE=ml" in body
    assert "# MLFLOW_EXPERIMENT_ID=" in body  # rest of the example preserved


def test_write_env_never_clobbers_existing(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=keepme\n")
    assert init_mod._write_env(tmp_path, "ml") is False
    assert "keepme" in (tmp_path / ".env").read_text()


def test_init_profile_flag_writes_env(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir(parents=True)
        (target / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, ["--profile", "ml", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=ml" in (dest / ".env").read_text()


def test_init_uses_ctx_profile_when_flag_absent(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir(parents=True)
        (target / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx(profile="from-login"))
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-login" in (dest / ".env").read_text()


def test_init_no_profile_writes_no_env(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir(parents=True)
        (target / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert not (dest / ".env").exists()  # no profile -> scaffold-only, no .env


def test_fetch_template_missing_dir_raises(tmp_path: pathlib.Path):
    """When the sparse checkout yields no template dir, a clean AgentCliError is raised."""

    def fake_git(args, cwd=None):
        # simulate clone creating an empty repo dir, sparse-checkout adding nothing
        if args[0] == "clone":
            pathlib.Path(args[-1]).mkdir(parents=True, exist_ok=True)
        return mock.Mock(returncode=0)

    with mock.patch.object(init_mod, "_git", side_effect=fake_git):
        try:
            init_mod._fetch_template("repo", "main", "agent-missing", tmp_path / "out")
            raised = False
        except AgentCliError as e:
            raised = True
            assert "not found" in str(e)
    assert raised
