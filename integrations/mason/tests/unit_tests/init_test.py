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


def _pop_default_stores(manifest: dict, slug: str = "proj") -> dict:
    """Pop the scaffold's default store tables, asserting each is `<slug>-<token>-<kind>`.

    Default names carry a per-scaffold random token so fresh scaffolds don't collide, so they can't
    be compared literally. Check the shape and that both stores share the one token, then return the
    manifest without them for an exact-equality check on the rest.
    """
    mem = re.fullmatch(rf"{slug}-([a-z]{{6}})-memory", manifest.pop("memory_store")["name"])
    sess = re.fullmatch(rf"{slug}-([a-z]{{6}})-sessions", manifest.pop("session_store")["name"])
    assert mem and sess, "default store names must be <slug>-<token>-<kind>"
    assert mem.group(1) == sess.group(1), "memory and session stores must share the scaffold token"
    return manifest


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
    assert _pop_default_stores(manifest) == {
        "schema_version": 1,
        "agent": {"framework": "langgraph", "server": "mason"},
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
    assert _pop_default_stores(manifest) == {
        "schema_version": 1,
        "agent": {"framework": "openai", "server": "mason"},
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
