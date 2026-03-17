import logging
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_logger = logging.getLogger(__name__)


def _to_volume_path(volume_name: str, file_path: str = "") -> str:
    """
    Convert 'catalog.schema.volume' + 'path/to/file'
    to '/Volumes/catalog/schema/volume/path/to/file'.
    """
    parts = volume_name.split(".")
    if len(parts) != 3:
        raise ValueError(f"volume_name must be 'catalog.schema.volume', got '{volume_name}'")
    base = f"/Volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    if file_path:
        return f"{base}/{file_path.lstrip('/')}"
    return base


def read_volume_file(
    volume_name: str,
    file_path: str,
    *,
    encoding: str = "utf-8",
    as_bytes: bool = False,
    workspace_client: Optional["WorkspaceClient"] = None,
) -> Union[str, bytes]:
    """
    Read a file from a Unity Catalog Volume.

    Args:
        volume_name: Full volume name as 'catalog.schema.volume'.
        file_path: Relative path to the file within the volume.
        encoding: Text encoding (default: utf-8). Ignored if as_bytes=True.
        as_bytes: If True, return raw bytes instead of decoded string.
        workspace_client: Optional pre-configured WorkspaceClient.

    Returns:
        File contents as a string (default) or bytes (if as_bytes=True).
    """
    from databricks.sdk import WorkspaceClient

    client = workspace_client or WorkspaceClient()
    full_path = _to_volume_path(volume_name, file_path)
    resp = client.files.download(full_path)
    if resp.contents is None:
        raise ValueError(f"No content returned for '{full_path}'")
    content = resp.contents.read()
    if as_bytes:
        return content
    return content.decode(encoding)
