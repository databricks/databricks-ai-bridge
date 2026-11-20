import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import ObjectType

from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog


@pytest.fixture
def catalog():
    files, workspace = {}, MagicMock()

    def download(path, **kwargs):
        if path not in files:
            raise NotFound("missing")
        return io.BytesIO(files[path])

    def upload(path, contents, **kwargs):
        if path in files:
            raise ValueError("Already exists")
        files[path] = contents

    workspace.workspace.download.side_effect = download
    workspace.workspace.upload.side_effect = upload
    workspace.workspace.list.side_effect = lambda path: [
        SimpleNamespace(path=name, object_type=ObjectType.FILE)
        for name in files
        if name.rsplit("/", 1)[0] == path
    ]
    return WorkspaceCatalog(workspace, "/Users/me/deployment/catalog"), files, workspace


def publish(catalog):
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    return catalog.create_agent(
        "agent", "workspace", "/Users/me/source", "agent:invoke", bundle
    ), bundle


def test_revision_and_source_survive_app_restart_without_database(catalog):
    catalog, files, workspace = catalog
    revision, bundle = publish(catalog)
    restarted = WorkspaceCatalog(workspace, catalog.root)
    assert restarted.agent(revision["id"], with_bundle=True)["bundle_data"] == bundle.data
    assert restarted.list("revisions") == [revision]
    assert {name.rsplit(".", 1)[-1] for name in files} == {"gz", "json"}


def test_republishing_identical_source_is_idempotent(catalog):
    catalog, files, _ = catalog
    first, _ = publish(catalog)
    previous = dict(files)
    second, _ = publish(catalog)
    assert first == second and files == previous


def test_changed_archive_is_rejected_on_cold_read(catalog):
    catalog, files, workspace = catalog
    revision, _ = publish(catalog)
    files[revision["source_archive"]] = b"changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        WorkspaceCatalog(workspace, catalog.root).agent(revision["id"], with_bundle=True)


def test_warm_turns_reuse_verified_archive_without_workspace_reads(catalog):
    catalog, _, workspace = catalog
    revision, bundle = publish(catalog)
    assert catalog.bundle(revision["id"]) == bundle
    workspace.workspace.download.reset_mock()
    assert catalog.bundle(revision["id"]) == bundle
    workspace.workspace.download.assert_not_called()


def test_lost_publication_response_is_reconciled(catalog):
    catalog, files, workspace = catalog
    upload = workspace.workspace.upload.side_effect

    def lost_reply(path, data, **kwargs):
        upload(path, data, **kwargs)
        raise TimeoutError("response lost")

    workspace.workspace.upload.side_effect = lost_reply
    revision, bundle = publish(catalog)
    assert catalog.bundle(revision["id"]) == bundle
