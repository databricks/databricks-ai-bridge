"""Unit tests for `mason init`: template mapping, destination guard, scaffold flow.

The network-touching git clone (`_fetch_template`) is mocked; tests assert the command wires
framework -> template dir, refuses an existing destination, and reports the scaffolded path.
"""

from __future__ import annotations

import ast
import json
import pathlib
from unittest import mock

import tomli
from click.testing import CliRunner

from databricks_mason import init as init_mod
from databricks_mason.errors import AgentCliError


class _Ctx:
    """Stand-in for CliContext: init reads .output and .profile."""

    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


def test_framework_specs_have_repo_ref_path():
    for fw in ("openai", "langgraph"):
        spec = init_mod._TEMPLATES[fw]
        assert spec["repo"] and spec["ref"] and spec["path"]
    assert init_mod._TEMPLATES["openai"]["path"] == "agent-openai-basic"
    assert (
        init_mod._TEMPLATES["langgraph"]["path"] == "integrations/mason/templates/agent-langgraph"
    )


def test_init_scaffolds_default_directory(tmp_path: pathlib.Path):
    dest = tmp_path / "agent-openai-basic"

    def fake_fetch(repo, ref, template_path, target):
        target.mkdir(parents=True)
        (target / "app.yaml").write_text("command: []\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch) as fetched:
        result = CliRunner().invoke(init_mod.init, ["--framework", "openai", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    fetched.assert_called_once()
    # framework's repo + path passed through to the fetch
    assert fetched.call_args.args[0] == init_mod._TEMPLATES["openai"]["repo"]
    assert fetched.call_args.args[2] == "agent-openai-basic"
    assert (dest / "app.yaml").exists()
    assert "agent-openai-basic" in result.output


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
    }


def test_init_installs_static_manifest_runtime_without_editing_langgraph_agent_code(
    tmp_path: pathlib.Path,
):
    dest = tmp_path / "langgraph"

    def fake_fetch(repo, ref, template_path, target):
        (target / "agent" / "mason").mkdir(parents=True)
        (target / "agent" / "agent.py").write_text("USER_AGENT = True\n")
        (target / "agent" / "mason" / "mcp_runtime.py").write_text("OLD = True\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(
            init_mod.init,
            ["--framework", "langgraph", str(dest)],
            obj=_Ctx(),
        )

    assert result.exit_code == 0, result.output
    assert (dest / "agent" / "agent.py").read_text() == "USER_AGENT = True\n"
    ast.parse((dest / "agent" / "mason" / "tool_manifest.py").read_text())
    runtime = (dest / "agent" / "mason" / "mcp_runtime.py").read_text()
    ast.parse(runtime)
    assert 'load_tools(expected_framework="langgraph")' in runtime


def test_init_openai_records_manifest_without_installing_runtime_adapter(tmp_path: pathlib.Path):
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

    def fake_fetch(repo, ref, template_path, target):
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

    def fake_fetch(repo, ref, template_path, target):
        target.mkdir(parents=True)
        (target / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, ["--profile", "ml", str(dest)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=ml" in (dest / ".env").read_text()


def test_init_uses_ctx_profile_when_flag_absent(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    def fake_fetch(repo, ref, template_path, target):
        target.mkdir(parents=True)
        (target / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")

    with mock.patch.object(init_mod, "_fetch_template", side_effect=fake_fetch):
        result = CliRunner().invoke(init_mod.init, [str(dest)], obj=_Ctx(profile="from-login"))
    assert result.exit_code == 0, result.output
    assert "DATABRICKS_CONFIG_PROFILE=from-login" in (dest / ".env").read_text()


def test_init_no_profile_writes_no_env(tmp_path: pathlib.Path):
    dest = tmp_path / "proj"

    def fake_fetch(repo, ref, template_path, target):
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
