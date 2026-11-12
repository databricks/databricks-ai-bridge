from __future__ import annotations

import pathlib

import pytest
import yaml

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.app_manifest import AppManifest


def test_parse_happy_path(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    manifest = AppManifest.parse(
        "command: ['uvicorn', 'app:app']\nenv:\n  - name: FOO\n    value: bar\n",
        source=source,
    )
    assert manifest.raw_env() == [{"name": "FOO", "value": "bar"}]


def test_parse_raises_on_invalid_yaml(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.parse("command: [unterminated", source=source)
    assert str(source) in str(exc_info.value)


def test_parse_raises_on_non_dict_top_level(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.parse("- just\n- a\n- list\n", source=source)
    message = str(exc_info.value)
    assert str(source) in message
    assert "top level must be an object" in message


def test_parse_raises_on_non_list_env(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.parse("command: []\nenv: not-a-list\n", source=source)
    message = str(exc_info.value)
    assert str(source) in message
    assert "env must be a list" in message


def test_parse_lenient_empty_for_non_dict():
    manifest = AppManifest.parse_lenient("- just\n- a\n- list\n")
    assert manifest.raw_env() == []


def test_parse_lenient_reads_dict():
    manifest = AppManifest.parse_lenient("command: []\nenv:\n  - name: FOO\n    value: bar\n")
    assert manifest.raw_env() == [{"name": "FOO", "value": "bar"}]


def test_require_deployable_names_the_resolved_dir_when_not_a_project(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.require_deployable(pathlib.Path("."))
    assert exc_info.value.message == (
        f"No Agent Bricks project in {tmp_path.resolve()} (no agent.toml or app.yaml)."
    )
    assert "--source <dir>" in (exc_info.value.hint or "")
    assert "agentbricks init" in (exc_info.value.hint or "")
    assert not (tmp_path / "app.yaml").exists()


def test_require_deployable_names_the_missing_app_yaml_of_a_project(tmp_path: pathlib.Path):
    (tmp_path / "agent.toml").write_text("")
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.require_deployable(tmp_path)
    assert exc_info.value.message == f"No app.yaml in {tmp_path.resolve()}."
    assert "agentbricks init" in (exc_info.value.hint or "")
    assert not (tmp_path / "app.yaml").exists()


@pytest.mark.parametrize(
    "text",
    [
        "command: [\"# TODO: set your run command, e.g. ['uvicorn', 'app:app']\"]\nenv: []\n",
        "command: REPLACE_WITH_YOUR_COMMAND\n",
        "command: [python, replace_with_entrypoint.py]\n",
    ],
)
def test_require_deployable_rejects_placeholder_commands(tmp_path: pathlib.Path, text: str):
    (tmp_path / "app.yaml").write_text(text)
    with pytest.raises(AgentCliError) as exc_info:
        AppManifest.require_deployable(tmp_path)
    assert exc_info.value.message == f"{tmp_path.resolve() / 'app.yaml'} has a placeholder command."


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("command: [unterminated", "Could not parse"),
        ("- just\n- a\n- list\n", "top level must be an object"),
        ("[]\n", "top level must be an object"),
        ("false\n", "top level must be an object"),
        ("0\n", "top level must be an object"),
        ("command: [x]\nenv: not-a-list\n", "env must be a list"),
    ],
)
def test_require_deployable_rejects_unparseable_app_yaml(
    tmp_path: pathlib.Path, text: str, match: str
):
    (tmp_path / "app.yaml").write_text(text)
    with pytest.raises(AgentCliError, match=match):
        AppManifest.require_deployable(tmp_path)


@pytest.mark.parametrize(
    "text", ["command: ['uvicorn', 'app:app']\n", "env: []\n", "", "# only a comment\n"]
)
def test_require_deployable_accepts_a_real_or_default_command(tmp_path: pathlib.Path, text: str):
    (tmp_path / "app.yaml").write_text(text)
    AppManifest.require_deployable(tmp_path)


def test_upsert_env_file_never_creates_app_yaml(tmp_path: pathlib.Path):
    with pytest.raises(AgentCliError, match="No app.yaml in"):
        AppManifest.upsert_env_file(tmp_path, {"FOO": "bar"})
    assert not (tmp_path / "app.yaml").exists()


def test_raw_env_empty_when_absent():
    manifest = AppManifest.parse_lenient("command: []\n")
    assert manifest.raw_env() == []


def test_raw_env_empty_when_non_list():
    manifest = AppManifest.parse_lenient("command: []\nenv: not-a-list\n")
    assert manifest.raw_env() == []


def test_raw_env_preserves_non_dict_entries():
    manifest = AppManifest.parse_lenient(
        "command: []\nenv:\n  - name: FOO\n    value: bar\n  - just-a-string\n"
    )
    assert manifest.raw_env() == [{"name": "FOO", "value": "bar"}, "just-a-string"]


def test_upsert_env_updates_existing_and_drops_value_from():
    manifest = AppManifest.parse_lenient(
        "command: []\nenv:\n  - name: FOO\n    valueFrom: some-secret\n"
    )
    manifest.upsert_env({"FOO": "new-value"})
    assert manifest.raw_env() == [{"name": "FOO", "value": "new-value"}]


def test_upsert_env_appends_new():
    manifest = AppManifest.parse_lenient("command: []\nenv: []\n")
    manifest.upsert_env({"FOO": "bar"})
    assert manifest.raw_env() == [{"name": "FOO", "value": "bar"}]


def test_upsert_env_drops_non_dict_entries():
    manifest = AppManifest.parse_lenient(
        "command: []\nenv:\n  - name: FOO\n    value: bar\n  - just-a-string\n"
    )
    manifest.upsert_env({})
    assert manifest.raw_env() == [{"name": "FOO", "value": "bar"}]


def test_upsert_env_drops_removals():
    manifest = AppManifest.parse_lenient(
        "command: []\nenv:\n  - name: FOO\n    value: bar\n  - name: BAZ\n    value: qux\n"
    )
    manifest.upsert_env({}, removals=["FOO"])
    assert manifest.raw_env() == [{"name": "BAZ", "value": "qux"}]


def test_upsert_env_removals_disjoint_from_updates():
    manifest = AppManifest.parse_lenient(
        "command: []\nenv:\n  - name: FOO\n    value: bar\n  - name: BAZ\n    value: old\n"
    )
    manifest.upsert_env({"BAZ": "new"}, removals=["FOO"])
    assert manifest.raw_env() == [{"name": "BAZ", "value": "new"}]


def test_set_env_replaces_and_preserves_non_dicts():
    manifest = AppManifest.parse_lenient("command: []\nenv:\n  - name: FOO\n    value: bar\n")
    manifest.set_env([{"name": "NEW", "value": "value"}, "raw-string"])
    assert manifest.raw_env() == [{"name": "NEW", "value": "value"}, "raw-string"]


def test_to_yaml_round_trips_and_preserves_key_order():
    manifest = AppManifest.parse_lenient("command: [uvicorn, app:app]\nenv: []\n")
    manifest.upsert_env({"FOO": "bar"})
    text = manifest.to_yaml()
    doc = yaml.safe_load(text)
    assert doc == {
        "command": ["uvicorn", "app:app"],
        "env": [{"name": "FOO", "value": "bar"}],
    }
    command_index = text.index("command")
    env_index = text.index("env")
    assert command_index < env_index
