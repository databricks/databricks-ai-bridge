import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_logger = logging.getLogger(__name__)


@dataclass
class VolumeFileInfo:
    """Metadata about a file or directory in a UC Volume."""

    name: str
    path: str
    is_directory: bool
    file_size: Optional[int] = None


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
    content = resp.contents.read()
    if as_bytes:
        return content
    return content.decode(encoding)


def list_volume_files(
    volume_name: str,
    directory: str = "",
    *,
    workspace_client: Optional["WorkspaceClient"] = None,
) -> List[VolumeFileInfo]:
    """
    List files and directories in a Unity Catalog Volume.

    Args:
        volume_name: Full volume name as 'catalog.schema.volume'.
        directory: Relative directory path within the volume (empty for root).
        workspace_client: Optional pre-configured WorkspaceClient.

    Returns:
        List of VolumeFileInfo objects.
    """
    from databricks.sdk import WorkspaceClient

    client = workspace_client or WorkspaceClient()
    full_path = _to_volume_path(volume_name, directory)
    results = []
    for item in client.files.list_directory_contents(full_path):
        info = VolumeFileInfo(
            name=item.name,
            path=item.path,
            is_directory=item.is_directory,
            file_size=getattr(item, "file_size", None),
        )
        results.append(info)
    return results


def upload_volume_file(
    volume_name: str,
    file_path: str,
    data: Union[str, bytes, BinaryIO, Path],
    *,
    overwrite: bool = False,
    workspace_client: Optional["WorkspaceClient"] = None,
) -> str:
    """
    Upload a file to a Unity Catalog Volume.

    Args:
        volume_name: Full volume name as 'catalog.schema.volume'.
        file_path: Relative path for the file within the volume.
        data: File content as string, bytes, file-like object, or Path to a local file.
        overwrite: Whether to overwrite an existing file (default: False).
        workspace_client: Optional pre-configured WorkspaceClient.

    Returns:
        The full volume path where the file was uploaded.
    """
    from databricks.sdk import WorkspaceClient

    client = workspace_client or WorkspaceClient()
    full_path = _to_volume_path(volume_name, file_path)

    # Ensure parent directory exists
    parent_dir = full_path.rsplit("/", 1)[0]
    try:
        client.files.create_directory(parent_dir)
    except Exception:
        _logger.debug(f"Directory '{parent_dir}' may already exist, continuing.")

    if isinstance(data, str):
        binary_data = io.BytesIO(data.encode("utf-8"))
    elif isinstance(data, bytes):
        binary_data = io.BytesIO(data)
    elif isinstance(data, Path):
        with open(data, "rb") as f:
            binary_data = io.BytesIO(f.read())
    else:
        binary_data = data

    client.files.upload(full_path, binary_data, overwrite=overwrite)
    return full_path
