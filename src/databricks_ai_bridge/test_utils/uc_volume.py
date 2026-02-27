import io
from typing import Generator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


VOLUME_NAME = "test_catalog.test_schema.test_volume"

SAMPLE_FILE_CONTENT = "This is the content of the test file."
SAMPLE_FILE_CONTENT_BYTES = SAMPLE_FILE_CONTENT.encode("utf-8")

def _make_dir_entry(name, path, is_directory, file_size=None):
    """Create a mock directory entry with a proper .name attribute."""
    entry = MagicMock()
    # MagicMock uses 'name' as a constructor param, so set it via PropertyMock
    type(entry).name = PropertyMock(return_value=name)
    entry.path = path
    entry.is_directory = is_directory
    entry.file_size = file_size
    return entry


SAMPLE_DIRECTORY_LISTING = [
    _make_dir_entry("report.txt", "/Volumes/cat/schema/vol/report.txt", False, 1234),
    _make_dir_entry("data.csv", "/Volumes/cat/schema/vol/data.csv", False, 5678),
    _make_dir_entry("subdir", "/Volumes/cat/schema/vol/subdir", True, None),
]


def _make_download_response(content_bytes: bytes = SAMPLE_FILE_CONTENT_BYTES) -> MagicMock:
    """Create a mock response for workspace_client.files.download()."""
    mock_resp = MagicMock()
    mock_resp.contents = io.BytesIO(content_bytes)
    return mock_resp


@pytest.fixture(autouse=True)
def mock_workspace_client() -> Generator:
    """Mock WorkspaceClient for UC Volume operations."""
    mock_client = MagicMock()

    # Mock files.download
    mock_client.files.download.return_value = _make_download_response()

    # Mock files.list_directory_contents
    mock_client.files.list_directory_contents.return_value = SAMPLE_DIRECTORY_LISTING

    # Mock files.upload (no return value needed)
    mock_client.files.upload.return_value = None

    # Mock files.create_directory (no return value needed)
    mock_client.files.create_directory.return_value = None

    with patch(
        "databricks.sdk.WorkspaceClient",
        return_value=mock_client,
    ):
        yield mock_client
