import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.workspace import ObjectType

from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.sources import SourceLoader, source_path, validate_entrypoint


@pytest.fixture
def source_files():
    return {
        "pyproject.toml": b"[project]\nname='test'",
        "uv.lock": b"version=1",
        "agent/main.py": b"# agent",
    }


def test_workspace_snapshot_matches_local_bundle_and_skips_environments(source_files, tmp_path):
    # Only the external Workspace API is mocked; the loader and archive builder run.
    workspace = MagicMock()
    root = "/Users/test/agent"
    workspace.workspace.get_status.return_value = SimpleNamespace(object_type=ObjectType.DIRECTORY)
    tree = {
        root: [
            SimpleNamespace(path=f"{root}/{name}", object_type=ObjectType.FILE, size=None)
            for name in ("pyproject.toml", "uv.lock", ".env")
        ]
        + [
            SimpleNamespace(path=f"{root}/{name}", object_type=ObjectType.DIRECTORY)
            for name in ("agent", ".venv")
        ],
        root + "/agent": [
            SimpleNamespace(path=root + "/agent/main.py", object_type=ObjectType.FILE, size=None)
        ],
    }
    workspace.workspace.list.side_effect = tree.__getitem__
    workspace.workspace.export.side_effect = lambda path, **kw: SimpleNamespace(
        content=base64.b64encode(source_files[path[len(root) + 1 :]]).decode()
    )
    bundle = SourceLoader(workspace, tmp_path).load("workspace", "/Workspace" + root)
    assert bundle == bundle_files(source_files)
    assert workspace.workspace.export.call_count == 3
    assert workspace.workspace.list.call_count == 2


def test_volume_source_is_downloaded_as_binary_and_streams_close(source_files, tmp_path):
    workspace = MagicMock()
    root = "/Volumes/catalog/schema/volume/agent"
    source_files = {k: v for k, v in source_files.items() if "/" not in k}
    source_files["payload.bin"] = bytes(range(256))
    workspace.files.list_directory_contents.return_value = [
        SimpleNamespace(path=root + "/" + name, is_directory=False, file_size=len(data))
        for name, data in source_files.items()
    ]
    streams = []

    def download(path):
        stream = io.BytesIO(source_files[path[len(root) + 1 :]])
        streams.append(stream)
        return SimpleNamespace(contents=stream)

    workspace.files.download.side_effect = download
    assert SourceLoader(workspace, tmp_path).load("volume", root) == bundle_files(source_files)
    assert all(stream.closed for stream in streams)


def test_loader_rejects_api_path_outside_selected_directory(tmp_path):
    workspace = MagicMock()
    workspace.files.list_directory_contents.return_value = [
        SimpleNamespace(path="/Volumes/c/s/v/other/private.py", is_directory=False)
    ]
    with pytest.raises(ValueError, match="outside"):
        SourceLoader(workspace, tmp_path).load("volume", "/Volumes/c/s/v/agent")
    workspace.files.download.assert_not_called()


def test_source_size_is_checked_before_download(tmp_path):
    workspace = MagicMock()
    workspace.files.list_directory_contents.return_value = [
        SimpleNamespace(path="/Volumes/c/s/v/huge", is_directory=False, file_size=30 * 1024 * 1024)
    ]
    with pytest.raises(ValueError, match="20 MiB"):
        SourceLoader(workspace, tmp_path).load("volume", "/Volumes/c/s/v")
    workspace.files.download.assert_not_called()


@pytest.mark.parametrize(
    "kind,path",
    [
        ("workspace", "/"),
        ("workspace", "/Workspace/"),
        ("workspace", "relative"),
        ("workspace", "/Users/a/../b"),
        ("volume", "/Volumes/c/s"),
        ("volume", "/Workspace/a"),
    ],
)
def test_rejects_invalid_source_locations(kind, path):
    with pytest.raises(ValueError):
        source_path(kind, path)


@pytest.mark.parametrize("value", ["agent.worker:run", "my_agent:invoke"])
def test_accepts_installed_module_entrypoints(value):
    assert validate_entrypoint(value) == value


@pytest.mark.parametrize("value", ["python worker.py", "a:b();evil", "../agent:run", "a:run.more"])
def test_rejects_commands_as_python_entrypoints(value):
    with pytest.raises(ValueError):
        validate_entrypoint(value)
