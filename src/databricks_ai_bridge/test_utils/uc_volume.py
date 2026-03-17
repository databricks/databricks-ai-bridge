import io
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

VOLUME_NAME = "test_catalog.test_schema.test_volume"

SAMPLE_FILE_CONTENT = "This is the content of the test file."
SAMPLE_FILE_CONTENT_BYTES = SAMPLE_FILE_CONTENT.encode("utf-8")


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

    with patch(
        "databricks.sdk.WorkspaceClient",
        return_value=mock_client,
    ):
        yield mock_client
