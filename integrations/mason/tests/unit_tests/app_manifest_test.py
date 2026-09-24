"""Unit tests for AppManifest: parse/serialize contract for app.yaml shared by deploy and dev."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from databricks_mason.app_manifest import AppManifest
from databricks_mason.errors import AgentCliError

# --- parse (strict) ---


def test_parse_happy_path(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    text = yaml.safe_dump({"command": ["uvicorn", "app:app"], "env": [{"name": "X", "value": "1"}]})
    manifest = AppManifest.parse(text, source=source)
    assert manifest.raw_env() == [{"name": "X", "value": "1"}]


def test_parse_raises_on_invalid_yaml(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError, match="Could not parse"):
        AppManifest.parse("key: [\nbad yaml", source=source)


def test_parse_raises_on_non_dict_top_level(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError, match="top level must be an object"):
        AppManifest.parse("- item1\n- item2\n", source=source)


def test_parse_raises_on_non_list_env(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError, match="env must be a list"):
        AppManifest.parse("env:\n  KEY: value\n", source=source)


def test_parse_error_messages_include_source_path(tmp_path: pathlib.Path):
    source = tmp_path / "app.yaml"
    with pytest.raises(AgentCliError, match=str(source)):
        AppManifest.parse("- not a dict\n", source=source)


# --- parse_lenient ---


def test_parse_lenient_returns_empty_manifest_for_non_dict_doc(tmp_path: pathlib.Path):
    manifest = AppManifest.parse_lenient("- item\n- item2\n")
    assert manifest.raw_env() == []
    # to_yaml should still produce valid YAML (an empty dict)
    doc = yaml.safe_load(manifest.to_yaml())
    assert doc == {} or doc is None or doc == {}


def test_parse_lenient_reads_dict_doc():
    text = yaml.safe_dump({"command": ["x"], "env": [{"name": "A", "value": "b"}]})
    manifest = AppManifest.parse_lenient(text)
    assert manifest.raw_env() == [{"name": "A", "value": "b"}]


def test_parse_lenient_non_dict_becomes_empty_not_error():
    # A list document should not raise - it becomes an empty manifest
    manifest = AppManifest.parse_lenient("42\n")
    assert manifest.raw_env() == []


# --- scaffold ---


def test_scaffold_has_placeholder_command_and_empty_env():
    manifest = AppManifest.scaffold()
    doc = yaml.safe_load(manifest.to_yaml())
    assert doc["command"] == ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"]
    assert doc["env"] == []


def test_scaffold_placeholder_command_exact_string():
    manifest = AppManifest.scaffold()
    doc = yaml.safe_load(manifest.to_yaml())
    # Must match deploy.py's original placeholder exactly
    assert doc["command"][0] == "# TODO: set your run command, e.g. ['uvicorn', 'app:app']"


# --- raw_env ---


def test_raw_env_returns_empty_list_when_absent():
    manifest = AppManifest.parse_lenient("{}")
    assert manifest.raw_env() == []


def test_raw_env_returns_empty_list_when_not_a_list():
    manifest = AppManifest.parse_lenient("command: [x]\n")
    # No env key at all
    assert manifest.raw_env() == []


def test_raw_env_returns_empty_list_for_non_list_env_value():
    # parse_lenient doesn't validate env type
    manifest = AppManifest.parse_lenient("env:\n  KEY: value\n")
    assert manifest.raw_env() == []


def test_raw_env_returns_list_including_non_dict_entries():
    text = yaml.safe_dump(
        {"env": [{"name": "A", "value": "1"}, "raw-string-entry", {"name": "B", "value": "2"}]}
    )
    manifest = AppManifest.parse_lenient(text)
    result = manifest.raw_env()
    assert result == [{"name": "A", "value": "1"}, "raw-string-entry", {"name": "B", "value": "2"}]


# --- upsert_env ---


def test_upsert_env_updates_existing_value():
    text = yaml.safe_dump({"env": [{"name": "FOO", "value": "old"}]})
    manifest = AppManifest.parse_lenient(text)
    manifest.upsert_env({"FOO": "new"})
    env = manifest.raw_env()
    assert env == [{"name": "FOO", "value": "new"}]


def test_upsert_env_drops_value_from_on_existing_entry():
    text = yaml.safe_dump({"env": [{"name": "SECRET", "valueFrom": "secret-resource"}]})
    manifest = AppManifest.parse_lenient(text)
    manifest.upsert_env({"SECRET": "plain-value"})
    env = manifest.raw_env()
    assert len(env) == 1
    assert env[0]["value"] == "plain-value"
    assert "valueFrom" not in env[0]


def test_upsert_env_appends_new_entry():
    text = yaml.safe_dump({"env": [{"name": "A", "value": "1"}]})
    manifest = AppManifest.parse_lenient(text)
    manifest.upsert_env({"B": "2"})
    env = manifest.raw_env()
    assert {"name": "A", "value": "1"} in env
    assert {"name": "B", "value": "2"} in env


def test_upsert_env_drops_non_dict_entries():
    text = yaml.safe_dump({"env": [{"name": "A", "value": "1"}, "raw-string", 42]})
    manifest = AppManifest.parse_lenient(text)
    manifest.upsert_env({"B": "2"})
    env = manifest.raw_env()
    # non-dict entries are dropped
    assert all(isinstance(e, dict) for e in env)
    assert {"name": "A", "value": "1"} in env
    assert {"name": "B", "value": "2"} in env


def test_upsert_env_updates_multiple_entries():
    text = yaml.safe_dump(
        {"env": [{"name": "A", "value": "old-a"}, {"name": "B", "valueFrom": "ref"}]}
    )
    manifest = AppManifest.parse_lenient(text)
    manifest.upsert_env({"A": "new-a", "B": "new-b", "C": "c"})
    by_name = {e["name"]: e for e in manifest.raw_env()}
    assert by_name["A"]["value"] == "new-a"
    assert by_name["B"]["value"] == "new-b"
    assert "valueFrom" not in by_name["B"]
    assert by_name["C"]["value"] == "c"


# --- set_env ---


def test_set_env_replaces_entire_env_block():
    text = yaml.safe_dump({"command": ["x"], "env": [{"name": "OLD", "value": "1"}]})
    manifest = AppManifest.parse_lenient(text)
    new_entries = [{"name": "NEW", "value": "2"}, "non-dict-preserved"]
    manifest.set_env(new_entries)
    assert manifest.raw_env() == new_entries


def test_set_env_preserves_non_dict_entries():
    manifest = AppManifest.scaffold()
    entries = [{"name": "A", "value": "1"}, "string-entry", {"name": "B", "value": "2"}]
    manifest.set_env(entries)
    assert manifest.raw_env() == entries


# --- to_yaml ---


def test_to_yaml_round_trips():
    original = {"command": ["uvicorn", "app:app"], "env": [{"name": "X", "value": "1"}]}
    manifest = AppManifest.parse_lenient(yaml.safe_dump(original))
    result = yaml.safe_load(manifest.to_yaml())
    assert result == original


def test_to_yaml_preserves_key_order_command_before_env():
    # sort_keys=False must be used so key insertion order is preserved
    manifest = AppManifest.scaffold()
    manifest.upsert_env({"FOO": "bar"})
    output = manifest.to_yaml()
    # command must appear before env in the output
    assert output.index("command") < output.index("env")


def test_to_yaml_uses_sort_keys_false():
    # Verify that a key added before env stays before env (i.e. not alphabetically sorted)
    # "command" < "env" alphabetically anyway, but add a key that would sort after "env"
    # to confirm sort_keys=False behavior
    manifest = AppManifest.parse_lenient(
        yaml.safe_dump({"zebra": "last", "apple": "first", "env": []}, sort_keys=False)
    )
    output = manifest.to_yaml()
    # With sort_keys=False, order from the dict is preserved
    parsed = yaml.safe_load(output)
    assert list(parsed.keys()) == ["zebra", "apple", "env"]
