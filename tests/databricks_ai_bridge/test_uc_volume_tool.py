from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from databricks_ai_bridge.test_utils.uc_volume import mock_workspace_client  # noqa: F401
from databricks_ai_bridge.uc_volume_tool import UCVolumeToolMixin


VOLUME_NAME = "test_catalog.test_schema.test_volume"


class DummyUCVolumeTool(UCVolumeToolMixin):
    pass


class TestVolumeNameValidation:
    def test_valid_volume_name(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        assert tool.volume_name == VOLUME_NAME

    def test_invalid_volume_name_two_parts(self):
        with pytest.raises(ValidationError):
            DummyUCVolumeTool(volume_name="catalog.schema")

    def test_invalid_volume_name_one_part(self):
        with pytest.raises(ValidationError):
            DummyUCVolumeTool(volume_name="just_a_name")

    def test_invalid_volume_name_four_parts(self):
        with pytest.raises(ValidationError):
            DummyUCVolumeTool(volume_name="a.b.c.d")


class TestToolNameValidation:
    def test_valid_tool_name(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name="my_tool")
        assert tool.tool_name == "my_tool"

    def test_valid_tool_name_with_hyphens(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name="my-tool-123")
        assert tool.tool_name == "my-tool-123"

    def test_invalid_tool_name_special_chars(self):
        with pytest.raises(ValidationError):
            DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name="invalid@@@")

    def test_invalid_tool_name_dots(self):
        with pytest.raises(ValidationError):
            DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name="tool.name")

    def test_none_tool_name_is_valid(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name=None)
        assert tool.tool_name is None


class TestGetToolName:
    def test_default_tool_name_from_volume_name(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        assert tool._get_tool_name() == "test_catalog__test_schema__test_volume"

    def test_custom_tool_name(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME, tool_name="my_reader")
        assert tool._get_tool_name() == "my_reader"

    def test_long_tool_name_truncated(self):
        tool = DummyUCVolumeTool(
            volume_name="catalog.schema.really_really_really_long_volume_name_that_should_be_truncated_to_64_chars"
        )
        name = tool._get_tool_name()
        assert len(name) <= 64

    @pytest.mark.parametrize(
        "volume_name,expected",
        [
            ("cat.schema.vol", "cat__schema__vol"),
            ("my_cat.my_schema.my_vol", "my_cat__my_schema__my_vol"),
        ],
    )
    def test_name_derivation(self, volume_name, expected):
        tool = DummyUCVolumeTool(volume_name=volume_name)
        assert tool._get_tool_name() == expected


class TestGetDefaultToolDescription:
    def test_includes_volume_name(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        desc = tool._get_default_tool_description()
        assert VOLUME_NAME in desc
        assert "Reads files" in desc


class TestReadFile:
    def test_read_file_returns_content(self, mock_workspace_client):
        # The autouse fixture patches WorkspaceClient() globally
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        result = tool._read_file("reports/q4.txt")
        assert result == "This is the content of the test file."

    def test_read_file_empty_path_returns_error(self):
        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        result = tool._read_file("")
        assert "Error" in result

    def test_read_file_binary_returns_error(self, mock_workspace_client):
        import io

        # Make download return non-utf8 bytes
        mock_resp = MagicMock()
        mock_resp.contents = io.BytesIO(b"\x80\x81\x82\x83")
        mock_workspace_client.files.download.return_value = mock_resp

        tool = DummyUCVolumeTool(volume_name=VOLUME_NAME)
        result = tool._read_file("image.png")
        assert "binary file" in result
        assert "text-based files only" in result

    def test_read_file_calls_correct_path(self, mock_workspace_client):
        tool = DummyUCVolumeTool(volume_name="cat.schema.vol")
        tool._read_file("subfolder/file.txt")
        mock_workspace_client.files.download.assert_called_once_with(
            "/Volumes/cat/schema/vol/subfolder/file.txt"
        )
