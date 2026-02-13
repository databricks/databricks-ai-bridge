import io
import json
from typing import Any
from unittest.mock import MagicMock

import mlflow
import pytest
from databricks_ai_bridge.test_utils.uc_volume import (  # noqa: F401
    SAMPLE_FILE_CONTENT,
    VOLUME_NAME,
    mock_workspace_client,
)
from langchain_core.tools import BaseTool
from mlflow.entities import SpanType

from databricks_langchain import UCVolumeTool


def init_volume_tool(
    volume_name: str = VOLUME_NAME,
    tool_name: str | None = None,
    tool_description: str | None = None,
    **kwargs: Any,
) -> UCVolumeTool:
    kwargs.update(
        {
            "volume_name": volume_name,
            "tool_name": tool_name,
            "tool_description": tool_description,
        }
    )
    return UCVolumeTool(**kwargs)


class TestInit:
    def test_init_is_base_tool(self):
        tool = init_volume_tool()
        assert isinstance(tool, BaseTool)

    def test_init_with_custom_name_and_description(self):
        tool = init_volume_tool(tool_name="my_reader", tool_description="Reads docs")
        assert tool.name == "my_reader"
        assert tool.description == "Reads docs"

    def test_init_default_name_from_volume_name(self):
        tool = init_volume_tool()
        assert tool.name == VOLUME_NAME.replace(".", "__")

    def test_init_default_description(self):
        tool = init_volume_tool()
        assert VOLUME_NAME in tool.description
        assert "Reads files" in tool.description


class TestInvoke:
    def test_invoke_returns_file_content(self):
        tool = init_volume_tool()
        result = tool.invoke("reports/q4.txt")
        assert result == SAMPLE_FILE_CONTENT

    def test_invoke_with_dict_input(self):
        tool = init_volume_tool()
        result = tool.invoke({"file_path": "reports/q4.txt"})
        assert result == SAMPLE_FILE_CONTENT

    def test_invoke_binary_file_returns_error(self, mock_workspace_client):
        mock_resp = MagicMock()
        mock_resp.contents = io.BytesIO(b"\x80\x81\x82\x83")
        mock_workspace_client.files.download.return_value = mock_resp

        tool = init_volume_tool()
        result = tool.invoke("image.png")
        assert "binary file" in result

    def test_invoke_empty_path_returns_error(self):
        tool = init_volume_tool()
        result = tool.invoke({"file_path": ""})
        assert "Error" in result


class TestToolNameGeneration:
    def test_default_tool_name(self):
        tool = init_volume_tool(volume_name="cat.schema.vol")
        assert tool.name == "cat__schema__vol"

    @pytest.mark.parametrize("tool_name", [None, "valid_tool_name", "test_tool"])
    def test_valid_tool_names(self, tool_name):
        tool = init_volume_tool(tool_name=tool_name)
        assert tool.tool_name == tool_name
        if tool_name:
            assert tool.name == tool_name

    @pytest.mark.parametrize("tool_name", ["test.tool.name", "tool&name"])
    def test_invalid_tool_names(self, tool_name):
        with pytest.raises(ValueError):
            init_volume_tool(tool_name=tool_name)


class TestArgsSchema:
    def test_args_schema_has_file_path(self):
        tool = init_volume_tool()
        assert "file_path" in tool.args_schema.model_fields
        assert tool.args_schema.model_fields["file_path"].description is not None


class TestTracing:
    def test_tracing_with_default_name(self):
        tool = init_volume_tool()
        tool._run("reports/q4.txt")
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        spans = trace.search_spans(name=VOLUME_NAME, span_type=SpanType.TOOL)
        assert len(spans) == 1

    def test_tracing_with_custom_name(self):
        tool = init_volume_tool(tool_name="my_reader")
        tool._run("reports/q4.txt")
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        spans = trace.search_spans(name="my_reader", span_type=SpanType.TOOL)
        assert len(spans) == 1
