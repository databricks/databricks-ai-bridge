from unittest.mock import MagicMock

import pytest

from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.environments import EnvironmentStore, validate_name


def environment_store():
    control, sources = MagicMock(), MagicMock()
    control.namespace = "test"
    control.catalog.create_agent.return_value = {"id": "frozen-revision"}
    control.create_run.return_value = {"id": "run", "status": "queued"}
    store = EnvironmentStore(control, sources)
    store.get = lambda name: {"default_agent_id": "frozen-revision"}
    return store, control, sources


def test_every_turn_uses_published_revision_without_loading_mutable_source():
    store, control, sources = environment_store()
    sources.load.side_effect = FileNotFoundError("working source removed")
    for invocation in ["first", "second"]:
        assert store.invoke("demo", "session", invocation, {})["id"] == "run"
    assert [call.args[0] for call in control.create_run.call_args_list] == ["frozen-revision"] * 2
    sources.load.assert_not_called()
    control.registry.connect.assert_not_called()


def test_retry_uses_the_same_invocation_record_without_aliases():
    store, control, sources = environment_store()
    store.invoke("demo", "session", "call", {}, retry=True)
    control.create_run.assert_called_once_with("frozen-revision", "session", "call", {}, retry=True)
    sources.load.assert_not_called()
    control.registry.connect.assert_not_called()


def test_unknown_retry_does_not_load_source():
    store, control, sources = environment_store()
    control.create_run.side_effect = KeyError("Original invocation not found")
    with pytest.raises(KeyError, match="Original invocation"):
        store.invoke("demo", "session", "missing", {}, retry=True)
    sources.load.assert_not_called()
    control.catalog.create_agent.assert_not_called()


def test_publication_freezes_source_before_first_invocation():
    control, sources = MagicMock(), MagicMock()
    control.namespace = "test"
    control.catalog.get.side_effect = KeyError("missing")
    control.catalog.put.side_effect = lambda kind, name, row: row
    control.catalog.create_agent.return_value = {"id": "frozen"}
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"lock"})
    sources.load.return_value = bundle
    store = EnvironmentStore(control, sources)
    row = store.create("demo", "workspace", "/Users/me/source", "agent:invoke")
    assert row["default_agent_id"] == "frozen"
    sources.load.assert_called_once_with("workspace", "/Users/me/source")
    assert control.catalog.create_agent.call_args.args[-1] == bundle
    control.registry.connect.assert_not_called()


@pytest.mark.parametrize("name", ["../bad", "two words", "", "a" * 65])
def test_environment_names_cannot_be_paths(name):
    with pytest.raises(ValueError, match="Environment names"):
        validate_name(name)
