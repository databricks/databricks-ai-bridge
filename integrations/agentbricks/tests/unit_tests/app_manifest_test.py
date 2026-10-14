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


def test_scaffold_has_placeholder_command_and_empty_env():
    manifest = AppManifest.scaffold()
    doc = yaml.safe_load(manifest.to_yaml())
    assert doc["command"] == ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"]
    assert doc["env"] == []


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
    manifest = AppManifest.scaffold()
    manifest.upsert_env({"FOO": "bar"})
    text = manifest.to_yaml()
    doc = yaml.safe_load(text)
    assert doc == {
        "command": ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"],
        "env": [{"name": "FOO", "value": "bar"}],
    }
    command_index = text.index("command")
    env_index = text.index("env")
    assert command_index < env_index
