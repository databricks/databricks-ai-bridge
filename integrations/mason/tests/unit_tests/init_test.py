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


class _Ctx:
    """Stand-in for CliContext: init reads .output and .profile."""

    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


@pytest.fixture(autouse=True)
def _skip_generated_runtime_rewrite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(init_mod, "_configure_durable_runtime", lambda *_args: None)
    monkeypatch.setattr(init_mod, "_editable_template_source", lambda: None)


def test_framework_specs_have_repo_ref_path():
    for fw in ("openai", "langgraph"):
        spec = init_mod._TEMPLATES[fw]
        assert spec["repo"] and spec["ref"] and spec["path"]
        custom = init_mod._CUSTOM_SERVER_TEMPLATES[fw]
        assert custom["repo"] and custom["ref"] and custom["path"]
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


def test_installed_git_template_source_uses_recorded_commit(monkeypatch: pytest.MonkeyPatch):
    commit = "a" * 40
    direct_url = json.dumps(
        {
            "url": "https://github.com/example/databricks-ai-bridge",
            "vcs_info": {
                "vcs": "git",
                "commit_id": commit,
                "requested_revision": "feature",
            },
            "subdirectory": "integrations/mason",
        }
    )
    distribution = mock.Mock()
    distribution.read_text.return_value = direct_url
    monkeypatch.setattr(init_mod, "_distribution", lambda _: distribution)

    assert init_mod._installed_git_template_source() == (
        "https://github.com/example/databricks-ai-bridge",
        commit,
    )


@pytest.mark.parametrize("direct_url", [None, "not json", '{"url": "file:///tmp/mason"}'])
def test_installed_git_template_source_ignores_non_git_installs(
    direct_url: str | None, monkeypatch: pytest.MonkeyPatch
):
    distribution = mock.Mock()
    distribution.read_text.return_value = direct_url
    monkeypatch.setattr(init_mod, "_distribution", lambda _: distribution)

    assert init_mod._installed_git_template_source() is None


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


def test_init_removes_partial_destination_after_failure(tmp_path: pathlib.Path):
    dest = tmp_path / "partial"

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir()
        (target / "runtime.py").write_text("partial\n")
        raise AgentCliError("template validation failed")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())

    assert result.exit_code != 0
    assert not dest.exists()


def test_init_defaults_to_existing_langgraph_app(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert f.call_args.args[2] == init_mod._TEMPLATES["langgraph"]["path"]
    assert f.call_args.args[4] == ("integrations/mason/templates/ui/agent-langgraph",)
    with (dest / ".mason" / "project.toml").open("rb") as metadata_file:
        metadata = tomli.load(metadata_file)
    assert metadata == {
        "schema_version": 1,
        "framework": "langgraph",
        "template": "agent-langgraph",
    }
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest["durability"] == {"enabled": True}


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_init_no_durable_runtime_keeps_mason_server_without_binding(
    tmp_path: pathlib.Path,
    framework: str,
):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", framework, "--no-durable-runtime", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert f.call_args.args[2] == init_mod._TEMPLATES[framework]["path"]
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert "durability" not in manifest
    assert "Mason AgentApp" in result.output
    assert "Durable runtime" in result.output
    assert "disabled" in result.output


def test_init_help_hides_no_durable_runtime():
    result = CliRunner().invoke(init_mod.init, ["--help"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "--no-durable-runtime" not in result.output


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_init_custom_server_uses_minimal_template(
    tmp_path: pathlib.Path,
    framework: str,
):
    dest = tmp_path / "proj"
    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()) as f:
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", framework, "--server", "custom", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert f.call_args.args[2] == init_mod._CUSTOM_SERVER_TEMPLATES[framework]["path"]
    assert f.call_args.args[4] == ()
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert "durability" not in manifest
    with (dest / ".mason" / "project.toml").open("rb") as config_file:
        config = tomli.load(config_file)
    assert config["template"] == f"custom-agent-{framework}"
    assert "Custom FastAPI" in result.output
    assert "Chat app" not in result.output


def test_init_rejects_no_durable_runtime_for_custom_server(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        init_mod.init,
        ["--server", "custom", "--no-durable-runtime", str(tmp_path / "proj")],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "only applies to --server mason" in result.output


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


def test_init_creates_canonical_agent_manifest(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    with mock.patch.object(init_mod, "_fetch_template", side_effect=lambda *a: a[3].mkdir()):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "openai", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest == {
        "schema_version": 1,
        "agent": {"framework": "openai"},
        "durability": {"enabled": True},
    }


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

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir()
        (target / "pyproject.toml").write_text(
            '[project]\nname = "test"\ndependencies = ["databricks-mason[runtime]>=0.1"]\n'
        )
        return "a" * 40

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as f:
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
    with (dest / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomli.load(pyproject_file)
    assert pyproject["project"]["dependencies"] == ["databricks-mason[runtime]>=0.1"]
    assert pyproject["tool"]["uv"]["sources"]["databricks-mason"] == {
        "git": "https://example.com/fork.git",
        "rev": "a" * 40,
        "subdirectory": "integrations/mason",
    }


def test_init_uses_editable_checkout_templates_by_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "agent"
    repository = tmp_path / "bridge"
    commit = "a" * 40
    monkeypatch.setattr(
        init_mod,
        "_editable_template_source",
        lambda: (repository.as_uri(), commit),
    )

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir()
        return commit

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as fetched:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert fetched.call_args.args[:2] == (repository.as_uri(), commit)


def test_pin_mason_source_supports_runtime_extra(tmp_path: pathlib.Path):
    dest = tmp_path / "agent"
    dest.mkdir()
    (dest / "pyproject.toml").write_text(
        '[project]\nname = "test"\ndependencies = ["databricks-mason[runtime]>=0.1"]\n'
    )
    repo = tmp_path / "bridge"

    init_mod._pin_mason_source(dest, "langgraph", str(repo), "feature")

    with (dest / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomli.load(pyproject_file)
    assert pyproject["tool"]["uv"]["sources"]["databricks-mason"] == {
        "git": repo.resolve().as_uri(),
        "rev": "feature",
        "subdirectory": "integrations/mason",
    }


def test_pin_mason_source_supports_base_package(tmp_path: pathlib.Path):
    dest = tmp_path / "agent"
    dest.mkdir()
    (dest / "pyproject.toml").write_text(
        '[project]\nname = "test"\ndependencies = ["databricks-mason>=0.1"]\n'
    )
    repo = tmp_path / "bridge"

    init_mod._pin_mason_source(
        dest,
        "langgraph",
        str(repo),
        "feature",
        runtime_extra=False,
    )

    with (dest / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomli.load(pyproject_file)
    assert pyproject["tool"]["uv"]["sources"]["databricks-mason"] == {
        "git": repo.resolve().as_uri(),
        "rev": "feature",
        "subdirectory": "integrations/mason",
    }


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
    assert payload["server"] == "mason"
    assert payload["chat_app_enabled"] is True
    assert payload["durable_runtime"] is True


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


def test_init_uses_git_installed_templates_by_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "agent"
    repository = "https://github.com/example/databricks-ai-bridge"
    commit = "a" * 40
    monkeypatch.setattr(
        init_mod,
        "_installed_git_template_source",
        lambda: (repository, commit),
    )

    def fake_fetch(repo, ref, template_path, target, overlay_dirs=()):
        target.mkdir()
        (target / "pyproject.toml").write_text(
            '[project]\nname = "test"\ndependencies = ["databricks-mason[runtime]>=0.1"]\n'
        )
        return commit

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as fetched:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert fetched.call_args.args[:2] == (repository, commit)
    with (dest / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomli.load(pyproject_file)
    assert pyproject["tool"]["uv"]["sources"]["databricks-mason"]["rev"] == commit


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


def test_fetch_template_accepts_commit_sha(tmp_path: pathlib.Path):
    repository = tmp_path / "repository"
    template = repository / "templates" / "agent"
    template.mkdir(parents=True)
    init_mod._git(["init", str(repository)])
    init_mod._git(["config", "user.email", "test@databricks.com"], cwd=repository)
    init_mod._git(["config", "user.name", "Mason Test"], cwd=repository)
    (template / "value.txt").write_text("first\n")
    init_mod._git(["add", "."], cwd=repository)
    init_mod._git(["commit", "-m", "first"], cwd=repository)
    commit = (init_mod._git(["rev-parse", "HEAD"], cwd=repository).stdout or "").strip()
    (template / "value.txt").write_text("second\n")
    init_mod._git(["commit", "-am", "second"], cwd=repository)

    destination = tmp_path / "output"
    resolved = init_mod._fetch_template(
        repository.as_uri(),
        commit,
        "templates/agent",
        destination,
    )

    assert resolved == commit
    assert (destination / "value.txt").read_text() == "first\n"
