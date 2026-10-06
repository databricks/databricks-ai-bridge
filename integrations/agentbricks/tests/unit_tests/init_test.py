"""Unit tests for `agentbricks init`: template mapping, destination guard, scaffold flow.

Templates ship inside the package; `agentbricks init` copies them via `_copy_packaged_template`, which is
stubbed here so tests don't touch the real bundled templates.
"""

from __future__ import annotations

import json
import pathlib
import re
from unittest import mock

import pytest
import tomli
from click.testing import CliRunner

from databricks_agentbricks.cli import auth
from databricks_agentbricks.cli import init as init_mod
from databricks_agentbricks.cli.app import CliContext
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.types import AgentFramework, AgentServer


def _ctx(output: str = "text", profile=None) -> CliContext:
    return CliContext(profile, output)


def _default_store_token(manifest: dict, slug: str = "proj") -> str:
    """Validate the scaffold's default store names (`<slug>-<token>-<kind>`) and return the token.

    Default names carry a per-scaffold random token so fresh scaffolds don't collide, so they can't
    be compared literally. Check the shape and that both stores share the one token, then return it
    so callers can assert over the full manifest without mutating it.
    """
    mem = re.fullmatch(rf"{re.escape(slug)}-([a-z]{{6}})-memory", manifest["memory_store"]["name"])
    sess = re.fullmatch(
        rf"{re.escape(slug)}-([a-z]{{6}})-sessions", manifest["session_store"]["name"]
    )
    assert mem and sess, "default store names must be <slug>-<token>-<kind>"
    assert mem.group(1) == sess.group(1), "memory and session stores must share the scaffold token"
    return mem.group(1)


def _copy_writing(files: dict[str, str] | None = None):
    """A `_copy_packaged_template` stub that creates the destination and optional files in it."""

    def _copy(name, dest, overlay_names=()):
        dest.mkdir(parents=True, exist_ok=True)
        for rel, content in (files or {}).items():
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    return _copy


@pytest.fixture(autouse=True)
def _hermetic_install(monkeypatch: pytest.MonkeyPatch):
    # Copying the template just creates the destination (no real bundled files); individual tests
    # override this to write specific scaffold files.
    monkeypatch.setattr(init_mod, "_copy_packaged_template", _copy_writing())


def test_framework_templates_map_to_names():
    assert set(init_mod._TEMPLATES) == set(AgentFramework)
    langgraph = init_mod._TEMPLATES[AgentFramework.LANGGRAPH]
    assert (langgraph.agentbricks_server, langgraph.custom_server, langgraph.chat_app) == (
        "agent-langgraph",
        "custom-agent-langgraph",
        "ui/agent-langgraph",
    )
    openai = init_mod._TEMPLATES[AgentFramework.OPENAI]
    assert (openai.agentbricks_server, openai.custom_server, openai.chat_app) == (
        "agent-openai",
        "custom-agent-openai",
        "ui/agent-openai",
    )


def test_init_scaffolds_default_directory(tmp_path: pathlib.Path):
    dest = tmp_path / "agent-openai"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({"app.yaml": "command: []\n"}),
    ) as copied:
        result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_ctx())
    assert result.exit_code == 0, result.output
    # the packaged template name (not a repo path) is what init copies
    assert copied.call_args.args[0] == "agent-openai"
    assert (dest / "app.yaml").exists()
    assert "agent-openai" in result.output


def test_init_removes_partial_destination_after_failure(tmp_path: pathlib.Path):
    dest = tmp_path / "partial"

    def boom(name, dest, overlay_names=()):
        dest.mkdir()
        (dest / "runtime.py").write_text("partial\n")
        raise AgentCliError("template validation failed")

    with mock.patch.object(init_mod, "_copy_packaged_template", side_effect=boom):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_ctx())

    assert result.exit_code != 0
    assert not dest.exists()


def test_init_defaults_to_langgraph_with_chat_app(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing()
    ) as copied:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_ctx())
    assert result.exit_code == 0, result.output
    assert copied.call_args.args[0] == "agent-langgraph"
    assert copied.call_args.args[2] == ("ui/agent-langgraph",)  # chat-app overlay
    with (dest / ".agentbricks" / "project.toml").open("rb") as metadata_file:
        assert tomli.load(metadata_file) == {
            "schema_version": 1,
            "framework": "langgraph",
            "template": "agent-langgraph",
        }
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    token = _default_store_token(manifest)
    assert manifest == {
        "schema_version": 1,
        "agent": {"framework": "langgraph", "server": "agentbricks"},
        "memory_store": {"name": f"proj-{token}-memory"},
        "session_store": {"name": f"proj-{token}-sessions"},
        "tracing": {"experiment_name": f"/Shared/agentbricks_traces/proj-{token}"},
    }


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_init_custom_server_uses_minimal_template(tmp_path: pathlib.Path, framework: str):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing()
    ) as copied:
        result = CliRunner().invoke(
            init_mod.init, ["--framework", framework, "--server", "custom", str(dest)], obj=_ctx()
        )
    assert result.exit_code == 0, result.output
    assert copied.call_args.args[0] == f"custom-agent-{framework}"
    assert copied.call_args.args[2] == ()  # no chat-app overlay for the custom server
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest == {
        "schema_version": 1,
        "agent": {"framework": framework, "server": "custom"},
    }
    assert "memory_store" not in manifest
    assert "session_store" not in manifest
    with (dest / ".agentbricks" / "project.toml").open("rb") as config_file:
        assert tomli.load(config_file)["template"] == f"custom-agent-{framework}"
    assert "Custom FastAPI" in result.output
    assert "Chat app" not in result.output


def test_init_creates_canonical_agent_manifest(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_ctx())
    assert result.exit_code == 0, result.output
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    token = _default_store_token(manifest)
    assert manifest == {
        "schema_version": 1,
        "agent": {"framework": "openai", "server": "agentbricks"},
        "memory_store": {"name": f"proj-{token}-memory"},
        "session_store": {"name": f"proj-{token}-sessions"},
        "tracing": {"experiment_name": f"/Shared/agentbricks_traces/proj-{token}"},
    }


def test_init_langgraph_does_not_vendor_runtime_plumbing(tmp_path: pathlib.Path):
    # Runtime plumbing lives in databricks_agentkit.runtime (imported, not vendored), so init must not
    # write an agent/agentbricks/ dir into the scaffold.
    dest = tmp_path / "langgraph"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({"agent/agent.py": "USER_AGENT = True\n"}),
    ):
        result = CliRunner().invoke(
            init_mod.init, ["--framework", "langgraph", str(dest)], obj=_ctx()
        )
    assert result.exit_code == 0, result.output
    assert (dest / "agent" / "agent.py").read_text() == "USER_AGENT = True\n"
    assert not (dest / "agent" / "agentbricks").exists()


@pytest.mark.parametrize(
    ("args", "expected_overlay"),
    [
        (["--framework", "langgraph"], ("ui/agent-langgraph",)),
        (["--framework", "langgraph", "--disable-chat-app"], ()),
        (["--framework", "langgraph", "--enable-chat-app"], ("ui/agent-langgraph",)),
        (["--framework", "openai"], ("ui/agent-openai",)),
        (["--framework", "openai", "--disable-chat-app"], ()),
    ],
)
def test_init_chat_app_overlay(
    tmp_path: pathlib.Path, args: list[str], expected_overlay: tuple[str, ...]
):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing()
    ) as copied:
        result = CliRunner().invoke(init_mod.init, [*args, str(dest)], obj=_ctx())
    assert result.exit_code == 0, result.output
    assert copied.call_args.args[2] == expected_overlay
    assert ("Chat app" in result.output) == bool(expected_overlay)


def test_init_json_output(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = CliRunner().invoke(
        init_mod.init, ["--framework", "langgraph", str(dest)], obj=_ctx(output="json")
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["framework"] == "langgraph"
    assert payload["template"] == "agent-langgraph"
    assert payload["directory"] == str(dest)
    assert payload["server"] == "agentbricks"
    assert payload["chat_app_enabled"] is True
    # The scaffolded store names are reported (they carry a random token, so aren't inferable).
    assert re.fullmatch(r"proj-[a-z]{6}-memory", payload["memory_store"])
    assert re.fullmatch(r"proj-[a-z]{6}-sessions", payload["session_store"])


def test_init_refuses_existing_destination(tmp_path: pathlib.Path):
    dest = tmp_path / "exists"
    dest.mkdir()
    with mock.patch.object(init_mod, "_copy_packaged_template") as copied:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_ctx())
    assert result.exit_code != 0
    assert "already exists" in " ".join(result.output.split())
    copied.assert_not_called()


def test_init_rejects_unknown_framework(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        init_mod.init, ["--framework", "nope", str(tmp_path / "x")], obj=_ctx()
    )
    assert result.exit_code != 0  # click.Choice rejects it


def test_init_help_keeps_lowercase_selection_values():
    result = CliRunner().invoke(init_mod.init, ["--help"], obj=_ctx())
    assert result.exit_code == 0, result.output
    assert "[langgraph|openai]" in result.output
    assert "[agentbricks|custom]" in result.output
    assert "AgentFramework" not in result.output
    assert "AgentServer" not in result.output


@pytest.mark.parametrize("framework", list(AgentFramework))
@pytest.mark.parametrize("server", list(AgentServer))
def test_init_json_keeps_lowercase_framework_and_server(
    tmp_path: pathlib.Path, framework: AgentFramework, server: AgentServer
):
    result = CliRunner().invoke(
        init_mod.init,
        ["--framework", framework.value, "--server", server.value, str(tmp_path / "proj")],
        obj=_ctx(output="json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["framework"] == framework.value
    assert payload["server"] == server.value


_ENV_EXAMPLE = {".env.example": "DATABRICKS_CONFIG_PROFILE=DEFAULT\n# MLFLOW_EXPERIMENT_ID=\n"}


def _init(dest: pathlib.Path, *args: str, output: str = "text", profile=None, files=None):
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing(files or _ENV_EXAMPLE)
    ):
        return CliRunner().invoke(
            init_mod.init, [*args, str(dest)], obj=_ctx(output=output, profile=profile)
        )


@pytest.mark.parametrize("flag", ["--profile", "-p"])
def test_init_profile_flag_writes_env(tmp_path: pathlib.Path, flag: str):
    dest = tmp_path / "proj"
    result = _init(dest, flag, "ml")
    assert result.exit_code == 0, result.output
    body = (dest / ".env").read_text()
    assert "DATABRICKS_CONFIG_PROFILE=ml" in body
    assert "# MLFLOW_EXPERIMENT_ID=" in body  # rest of the example preserved


def test_init_flag_beats_global_profile(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = _init(dest, "--profile", "flag", profile="global")
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=flag" in (dest / ".env").read_text()


def test_init_uses_global_profile_when_flag_absent(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = _init(dest, profile="from-global")
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-global" in (dest / ".env").read_text()


def test_init_uses_env_var_profile_when_nothing_else_given(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")
    dest = tmp_path / "proj"
    result = _init(dest)
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-env-var" in (dest / ".env").read_text()


def test_init_without_any_profile_pins_default(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = _init(dest, files={})
    assert result.exit_code == 0, result.output
    assert (dest / ".env").read_text().startswith("DATABRICKS_CONFIG_PROFILE=DEFAULT")


def test_init_with_databricks_host_and_no_profile_writes_no_env(
    tmp_path: pathlib.Path, monkeypatch
):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    dest = tmp_path / "proj"

    result = _init(dest, output="json")

    assert result.exit_code == 0, result.output
    assert not (dest / ".env").exists()
    payload = json.loads(result.output)
    assert payload["env_profile"] is None
    assert payload["workspace_host"] == "https://env.example"
    assert payload["signed_in_user"] is None


def test_init_with_databricks_host_and_no_profile_text_summary(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")

    result = _init(tmp_path / "proj")

    out = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "Auth environment variables (DATABRICKS_HOST); no .env written" in out
    assert "Workspace https://env.example" in out
    assert "agentbricks profile login" not in out
    assert "agentbricks profile set <profile>" in out


def test_init_databricks_host_does_not_sign_in_when_interactive(
    tmp_path: pathlib.Path, monkeypatch, validated_profile
):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)

    result = _init(tmp_path / "proj")

    assert result.exit_code == 0, result.output
    validated_profile.assert_not_called()


def test_init_profile_flag_beats_databricks_host(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    dest = tmp_path / "proj"

    result = _init(dest, "--profile", "ml")

    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=ml" in (dest / ".env").read_text()


def test_init_env_var_profile_beats_databricks_host(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")
    dest = tmp_path / "proj"

    result = _init(dest)

    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-env-var" in (dest / ".env").read_text()


def test_init_text_summary_for_signed_out_run(tmp_path: pathlib.Path, write_databrickscfg):
    write_databrickscfg("[ml]\nhost = https://ml.example\n")
    result = _init(tmp_path / "proj", "--profile", "ml")
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "Profile (.env) ml" in out
    assert "Workspace https://ml.example" in out
    assert "Signed in as" not in out
    assert "agentbricks profile login ml" in out
    assert "agentbricks profile set <profile>" in out


def test_init_text_summary_without_configured_workspace(tmp_path: pathlib.Path):
    result = _init(tmp_path / "proj", "--profile", "ml")
    assert result.exit_code == 0, result.output
    assert "Workspace not configured yet" in " ".join(result.output.split())


def test_init_json_reports_profile_and_workspace(tmp_path: pathlib.Path, write_databrickscfg):
    write_databrickscfg("[ml]\nhost = https://ml.example\n")
    result = _init(tmp_path / "proj", "--profile", "ml", output="json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["env_profile"] == "ml"
    assert payload["workspace_host"] == "https://ml.example"
    assert payload["signed_in_user"] is None


def test_init_noninteractive_warns_about_unknown_profile(tmp_path: pathlib.Path):
    result = _init(tmp_path / "proj", "--profile", "ghost")
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "profile 'ghost', but it isn't in your Databricks config yet" in out


def test_init_noninteractive_json_has_no_warning_text(tmp_path: pathlib.Path):
    result = _init(tmp_path / "proj", "--profile", "ghost", output="json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["env_profile"] == "ghost"


def test_init_signs_in_when_interactive(tmp_path: pathlib.Path, monkeypatch, validated_profile):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    result = _init(tmp_path / "proj", "--profile", "ml", output="json")
    assert result.exit_code == 0, result.output
    validated_profile.assert_called_once_with("ml")
    assert json.loads(result.output)["signed_in_user"] == "me@example.com"


def test_init_text_summary_for_signed_in_run(
    tmp_path: pathlib.Path, monkeypatch, validated_profile
):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    result = _init(tmp_path / "proj", "--profile", "ml")
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "Signed in as me@example.com" in out
    assert "agentbricks profile login ml" not in out
    assert "agentbricks profile set <profile>" in out


def test_init_keeps_scaffold_when_interactive_sign_in_fails(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("denied")))
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=1)))
    dest = tmp_path / "proj"
    result = _init(dest, "--profile", "ml")
    assert result.exit_code == 0, result.output
    assert (dest / ".env").exists()
    assert "Sign-in to profile 'ml' failed" in " ".join(result.output.split())
    assert "Signed in as" not in result.output


def test_init_rerun_on_generated_project_changes_nothing(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    assert _init(dest, "--profile", "ml").exit_code == 0
    env_before = (dest / ".env").read_text()

    result = _init(dest, "--profile", "other")

    assert result.exit_code == 0, result.output
    assert (dest / ".env").read_text() == env_before
    assert "already an Agent Bricks project" in " ".join(result.output.split())
    assert "agentbricks profile set <profile>" in " ".join(result.output.split())


def test_init_rerun_on_generated_project_json(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    assert _init(dest).exit_code == 0

    result = _init(dest, output="json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"already_initialized": True, "directory": str(dest)}


def test_init_store_name_overrides(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = CliRunner().invoke(
        init_mod.init,
        [
            "--framework",
            "openai",
            "--memory-store",
            "mem-x",
            "--session-store",
            "sess-y",
            str(dest),
        ],
        obj=_ctx(),
    )
    assert result.exit_code == 0, result.output
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest["memory_store"] == {"name": "mem-x"}
    assert manifest["session_store"] == {"name": "sess-y"}


@pytest.mark.parametrize("chat_app", [True, False])
@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_existing_prepares_migration_without_changing_application(
    tmp_path: pathlib.Path, framework: str, chat_app: bool
):
    original = {
        "agent.py": b"# existing graph\n",
        "pyproject.toml": b"[project]\nname = 'existing'\n",
        "agent.toml": b"# existing bindings\n",
        "app.yaml": b"command: [existing-server]\n",
        ".env": b"DATABRICKS_CONFIG_PROFILE=keep\n",
        ".agentbricks/project.toml": b"# existing metadata\n",
        ".claude/skills/other/SKILL.md": b"# another skill\n",
    }
    for name, data in original.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    # Mirror the bundled templates: every framework scaffold ships a AGENTKIT_CONTRACT.md, and the
    # chat-app overlay adds a CHAT_APP.md.
    def _copy(name, dest, overlay_names=()):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "AGENTKIT_CONTRACT.md").write_text("# contract\n")
        if overlay_names:
            (dest / "CHAT_APP.md").write_text("# chat app\n")

    args = [
        "--existing",
        "--framework",
        framework,
        "--profile",
        "selected",
        "--memory-store",
        "chosen-memory",
        "--session-store",
        "chosen-session",
        str(tmp_path),
    ]
    if not chat_app:
        args.append("--disable-chat-app")
    with mock.patch.object(init_mod, "_copy_packaged_template", side_effect=_copy):
        result = CliRunner().invoke(init_mod.init, args, obj=_ctx(output="json"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    skill = tmp_path / "agent-bricks-migrate"
    assert pathlib.Path(payload["skill"]).is_file()
    assert pathlib.Path(payload["prompt_file"]).read_text().strip() == payload["prompt"]
    assert payload["mode"] == "existing"
    assert payload["framework"] == framework
    assert payload["chat_app_enabled"] is chat_app
    label = {"langgraph": "LangGraph", "openai": "OpenAI Agents SDK"}[framework]
    assert label in payload["prompt"]
    assert "agent-bricks-migrate/SKILL.md" in payload["prompt"]
    settings = json.loads((skill / "references/migration.json").read_text())
    assert settings["framework"] == framework
    assert settings["profile"] == "selected"
    assert settings["chat_app_enabled"] is chat_app
    reference = skill / "references/template"
    assert (reference / "AGENTKIT_CONTRACT.md").is_file()
    if chat_app:
        assert (reference / "CHAT_APP.md").is_file()
    with (reference / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest["memory_store"] == {"name": "chosen-memory"}
    assert manifest["session_store"] == {"name": "chosen-session"}
    assert manifest["agent"]["framework"] == framework
    assert manifest["agent"]["server"] == "agentbricks"

    # Every supported agent finds the one bundle through a pointer, rather than its own copy.
    pointers = [
        tmp_path / root / "skills/agent-bricks-migrate/SKILL.md" for root in (".claude", ".agent")
    ]
    assert payload["pointers"] == [str(pointer) for pointer in pointers]
    for pointer in pointers:
        body = pointer.read_text()
        assert "name: agent-bricks-migrate" in body
        assert "../../../agent-bricks-migrate/SKILL.md" in body
        link_target = body.split("](", 1)[1].split(")", 1)[0]
        assert (pointer.parent / link_target).resolve() == (skill / "SKILL.md").resolve()
        assert not (pointer.parent / "references").exists()

    for name, data in original.items():
        assert (tmp_path / name).read_bytes() == data


def test_existing_defaults_to_current_directory(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(init_mod.init, ["--existing"], obj=_ctx(profile="saved"))
    assert result.exit_code == 0, result.output
    settings = json.loads((tmp_path / "agent-bricks-migrate/references/migration.json").read_text())
    assert settings["profile"] == "saved"
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "agent.toml").exists()


@pytest.mark.parametrize(
    "conflict",
    [
        "bundle",
        "claude-skill",
        "agent-skill",
        "legacy-bundle",
        "legacy-claude-skill",
        "legacy-agent-skill",
        "file",
        "symlink",
    ],
)
def test_existing_refuses_migration_path_conflicts(tmp_path: pathlib.Path, conflict: str):
    claude = tmp_path / ".claude"
    migration_dir = (
        "agent-bricks-migrate" if conflict.startswith("legacy-") else "agent-bricks-migrate"
    )
    if conflict in ("bundle", "legacy-bundle"):
        bundle = tmp_path / migration_dir
        bundle.mkdir()
        (bundle / "SKILL.md").write_text("user instructions")
    elif conflict in ("claude-skill", "legacy-claude-skill"):
        skill = claude / "skills" / migration_dir
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("user instructions")
    elif conflict in ("agent-skill", "legacy-agent-skill"):
        skill = tmp_path / ".agent" / "skills" / migration_dir
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("user instructions")
    elif conflict == "file":
        claude.write_text("user file")
    else:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        claude.symlink_to(elsewhere, target_is_directory=True)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_ctx())
    assert result.exit_code != 0
    assert {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize(
    "args",
    [
        ["--existing", "--server", "custom"],
        ["--existing", "--framework", "openai", "--server", "custom"],
    ],
)
def test_existing_rejects_unsupported_modes(tmp_path: pathlib.Path, args: list[str]):
    result = CliRunner().invoke(init_mod.init, [*args, str(tmp_path)], obj=_ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_existing_missing_directory_is_rejected(tmp_path: pathlib.Path):
    dest = tmp_path / "missing"
    result = CliRunner().invoke(init_mod.init, ["--existing", str(dest)], obj=_ctx())
    assert result.exit_code != 0
    assert not dest.exists()


def test_existing_failed_copy_leaves_no_artifacts(tmp_path: pathlib.Path, monkeypatch):
    def failed_copy(name, dest, overlay_names=()):
        dest.mkdir(parents=True)
        (dest / "partial.txt").write_text("partial")
        raise AgentCliError("copy failed")

    monkeypatch.setattr(init_mod, "_copy_packaged_template", failed_copy)
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_existing_failed_pointer_removes_the_bundle(tmp_path: pathlib.Path, monkeypatch):
    real_mkdir = pathlib.Path.mkdir

    def failing_mkdir(self: pathlib.Path, *args, **kwargs):
        if ".claude" in self.parts:
            raise OSError("cannot create agent configuration directory")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []
