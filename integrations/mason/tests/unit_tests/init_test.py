"""Unit tests for `mason init`: template mapping, destination guard, scaffold flow.

Templates ship inside the package; `mason init` copies them via `_copy_packaged_template`, which is
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

from databricks_mason.cli import init as init_mod
from databricks_mason.errors import AgentCliError
from databricks_mason.project_types import AgentFramework, AgentServer


class _Ctx:
    """Stand-in for CliContext: init reads .output and .profile."""

    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


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
    assert (langgraph.mason_server, langgraph.custom_server, langgraph.chat_app) == (
        "agent-langgraph",
        "custom-agent-langgraph",
        "ui/agent-langgraph",
    )
    openai = init_mod._TEMPLATES[AgentFramework.OPENAI]
    assert (openai.mason_server, openai.custom_server, openai.chat_app) == (
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
        result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_Ctx())
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
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())

    assert result.exit_code != 0
    assert not dest.exists()


def test_init_defaults_to_langgraph_with_chat_app(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing()
    ) as copied:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert copied.call_args.args[0] == "agent-langgraph"
    assert copied.call_args.args[2] == ("ui/agent-langgraph",)  # chat-app overlay
    with (dest / ".mason" / "project.toml").open("rb") as metadata_file:
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
        "agent": {"framework": "langgraph", "server": "mason"},
        "memory_store": {"name": f"proj-{token}-memory"},
        "session_store": {"name": f"proj-{token}-sessions"},
    }


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_init_custom_server_uses_minimal_template(tmp_path: pathlib.Path, framework: str):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod, "_copy_packaged_template", side_effect=_copy_writing()
    ) as copied:
        result = CliRunner().invoke(
            init_mod.init, ["--framework", framework, "--server", "custom", str(dest)], obj=_Ctx()
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
    with (dest / ".mason" / "project.toml").open("rb") as config_file:
        assert tomli.load(config_file)["template"] == f"custom-agent-{framework}"
    assert "Custom FastAPI" in result.output
    assert "Chat app" not in result.output


def test_init_creates_canonical_agent_manifest(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    with (dest / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    token = _default_store_token(manifest)
    assert manifest == {
        "schema_version": 1,
        "agent": {"framework": "openai", "server": "mason"},
        "memory_store": {"name": f"proj-{token}-memory"},
        "session_store": {"name": f"proj-{token}-sessions"},
    }


def test_init_langgraph_does_not_vendor_runtime_plumbing(tmp_path: pathlib.Path):
    # Runtime plumbing lives in databricks_mason.runtime (imported, not vendored), so init must not
    # write an agent/mason/ dir into the scaffold.
    dest = tmp_path / "langgraph"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({"agent/agent.py": "USER_AGENT = True\n"}),
    ):
        result = CliRunner().invoke(
            init_mod.init, ["--framework", "langgraph", str(dest)], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert (dest / "agent" / "agent.py").read_text() == "USER_AGENT = True\n"
    assert not (dest / "agent" / "mason").exists()


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
        result = CliRunner().invoke(init_mod.init, [*args, str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert copied.call_args.args[2] == expected_overlay
    assert ("Chat app" in result.output) == bool(expected_overlay)


def test_init_json_output(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
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
    # The scaffolded store names are reported (they carry a random token, so aren't inferable).
    assert re.fullmatch(r"proj-[a-z]{6}-memory", payload["memory_store"])
    assert re.fullmatch(r"proj-[a-z]{6}-sessions", payload["session_store"])


def test_init_refuses_existing_destination(tmp_path: pathlib.Path):
    dest = tmp_path / "exists"
    dest.mkdir()
    with mock.patch.object(init_mod, "_copy_packaged_template") as copied:
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code != 0
    assert "already exists" in " ".join(result.output.split())
    copied.assert_not_called()


def test_init_rejects_unknown_framework(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        init_mod.init, ["--framework", "nope", str(tmp_path / "x")], obj=_Ctx()
    )
    assert result.exit_code != 0  # click.Choice rejects it


def test_init_help_keeps_lowercase_selection_values():
    result = CliRunner().invoke(init_mod.init, ["--help"], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "[langgraph|openai]" in result.output
    assert "[mason|custom]" in result.output
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
        obj=_Ctx(output="json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["framework"] == framework.value
    assert payload["server"] == server.value


def test_write_env_seeds_profile_from_example(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text(
        "DATABRICKS_CONFIG_PROFILE=DEFAULT\n# MLFLOW_EXPERIMENT_ID=\n"
    )
    assert init_mod._write_env(tmp_path, "ml") is True
    body = (tmp_path / ".env").read_text()
    assert "DATABRICKS_CONFIG_PROFILE=ml" in body
    assert "# MLFLOW_EXPERIMENT_ID=" in body  # rest of the example preserved


def test_write_env_never_clobbers_existing(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=keepme\n")
    assert init_mod._write_env(tmp_path, "ml") is False
    assert "keepme" in (tmp_path / ".env").read_text()


def test_init_profile_flag_writes_env(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({".env.example": "DATABRICKS_CONFIG_PROFILE=DEFAULT\n"}),
    ):
        result = CliRunner().invoke(init_mod.init, ["--profile", "ml", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=ml" in (dest / ".env").read_text()


def test_init_uses_ctx_profile_when_flag_absent(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({".env.example": "DATABRICKS_CONFIG_PROFILE=DEFAULT\n"}),
    ):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx(profile="from-login"))
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-login" in (dest / ".env").read_text()


def test_init_no_profile_writes_no_env(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"
    with mock.patch.object(
        init_mod,
        "_copy_packaged_template",
        side_effect=_copy_writing({".env.example": "DATABRICKS_CONFIG_PROFILE=DEFAULT\n"}),
    ):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert not (dest / ".env").exists()  # no profile -> scaffold-only, no .env


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
        obj=_Ctx(),
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
        ".mason/project.toml": b"# existing metadata\n",
        ".claude/skills/other/SKILL.md": b"# another skill\n",
    }
    for name, data in original.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    # Mirror the bundled templates: every framework scaffold ships a MASON_CONTRACT.md, and the
    # chat-app overlay adds a CHAT_APP.md.
    def _copy(name, dest, overlay_names=()):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "MASON_CONTRACT.md").write_text("# contract\n")
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
        result = CliRunner().invoke(init_mod.init, args, obj=_Ctx(output="json"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    skill = tmp_path / "mason-migrate"
    assert pathlib.Path(payload["skill"]).is_file()
    assert pathlib.Path(payload["prompt_file"]).read_text().strip() == payload["prompt"]
    assert payload["mode"] == "existing"
    assert payload["framework"] == framework
    assert payload["chat_app_enabled"] is chat_app
    label = {"langgraph": "LangGraph", "openai": "OpenAI Agents SDK"}[framework]
    assert label in payload["prompt"]
    settings = json.loads((skill / "references/migration.json").read_text())
    assert settings["framework"] == framework
    assert settings["profile"] == "selected"
    assert settings["chat_app_enabled"] is chat_app
    reference = skill / "references/template"
    assert (reference / "MASON_CONTRACT.md").is_file()
    if chat_app:
        assert (reference / "CHAT_APP.md").is_file()
    with (reference / "agent.toml").open("rb") as manifest_file:
        manifest = tomli.load(manifest_file)
    assert manifest["memory_store"] == {"name": "chosen-memory"}
    assert manifest["session_store"] == {"name": "chosen-session"}
    assert manifest["agent"]["framework"] == framework
    assert manifest["agent"]["server"] == "mason"

    # Every supported agent finds the one bundle through a pointer, rather than its own copy.
    pointers = [tmp_path / root / "skills/mason-migrate/SKILL.md" for root in (".claude", ".agent")]
    assert payload["pointers"] == [str(pointer) for pointer in pointers]
    for pointer in pointers:
        body = pointer.read_text()
        assert "name: mason-migrate" in body
        assert "../../../mason-migrate/SKILL.md" in body
        assert not (pointer.parent / "references").exists()

    for name, data in original.items():
        assert (tmp_path / name).read_bytes() == data


def test_existing_defaults_to_current_directory(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(init_mod.init, ["--existing"], obj=_Ctx(profile="saved"))
    assert result.exit_code == 0, result.output
    settings = json.loads((tmp_path / "mason-migrate/references/migration.json").read_text())
    assert settings["profile"] == "saved"
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "agent.toml").exists()


@pytest.mark.parametrize("conflict", ["bundle", "claude-skill", "agent-skill", "file", "symlink"])
def test_existing_refuses_migration_path_conflicts(tmp_path: pathlib.Path, conflict: str):
    claude = tmp_path / ".claude"
    if conflict == "bundle":
        bundle = tmp_path / "mason-migrate"
        bundle.mkdir()
        (bundle / "SKILL.md").write_text("user instructions")
    elif conflict == "claude-skill":
        skill = claude / "skills/mason-migrate"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("user instructions")
    elif conflict == "agent-skill":
        skill = tmp_path / ".agent/skills/mason-migrate"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("user instructions")
    elif conflict == "file":
        claude.write_text("user file")
    else:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        claude.symlink_to(elsewhere, target_is_directory=True)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_Ctx())
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
    result = CliRunner().invoke(init_mod.init, [*args, str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_existing_missing_directory_is_rejected(tmp_path: pathlib.Path):
    dest = tmp_path / "missing"
    result = CliRunner().invoke(init_mod.init, ["--existing", str(dest)], obj=_Ctx())
    assert result.exit_code != 0
    assert not dest.exists()


def test_existing_failed_copy_leaves_no_artifacts(tmp_path: pathlib.Path, monkeypatch):
    def failed_copy(name, dest, overlay_names=()):
        dest.mkdir(parents=True)
        (dest / "partial.txt").write_text("partial")
        raise AgentCliError("copy failed")

    monkeypatch.setattr(init_mod, "_copy_packaged_template", failed_copy)
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_existing_failed_pointer_removes_the_bundle(tmp_path: pathlib.Path, monkeypatch):
    real_mkdir = pathlib.Path.mkdir

    def failing_mkdir(self: pathlib.Path, *args, **kwargs):
        if ".claude" in self.parts:
            raise OSError("cannot create agent configuration directory")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
    result = CliRunner().invoke(init_mod.init, ["--existing", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []
