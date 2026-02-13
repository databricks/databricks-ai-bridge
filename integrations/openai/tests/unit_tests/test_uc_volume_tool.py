import io
import json
from typing import Any, Dict, Optional, cast
from unittest.mock import MagicMock

import mlflow
import pytest
from databricks_ai_bridge.test_utils.uc_volume import (  # noqa: F401
    SAMPLE_FILE_CONTENT,
    VOLUME_NAME,
    mock_workspace_client,
)
from mlflow.entities import SpanType
from pydantic import BaseModel

from databricks_openai import UCVolumeTool


def init_volume_tool(
    volume_name: str = VOLUME_NAME,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
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
    def test_init_is_base_model(self):
        tool = init_volume_tool()
        assert isinstance(tool, BaseModel)

    def test_init_creates_tool_spec(self):
        tool = init_volume_tool()
        assert tool.tool is not None
        assert tool.tool["type"] == "function"
        assert "function" in tool.tool

    def test_init_with_custom_name_and_description(self):
        tool = init_volume_tool(tool_name="my_reader", tool_description="Reads docs")
        assert tool.tool["function"]["name"] == "my_reader"
        assert tool.tool["function"]["description"] == "Reads docs"

    def test_init_default_name_from_volume_name(self):
        tool = init_volume_tool()
        assert tool.tool["function"]["name"] == VOLUME_NAME.replace(".", "__")

    def test_init_no_strict_mode(self):
        tool = init_volume_tool()
        assert "strict" not in tool.tool.get("function", {})

    def test_init_no_additional_properties(self):
        tool = init_volume_tool()
        assert "additionalProperties" not in tool.tool["function"]["parameters"]


class TestToolSchema:
    def test_tool_schema_has_file_path(self):
        tool = init_volume_tool()
        schema = cast(Dict[str, Any], tool.tool)
        properties = schema["function"]["parameters"]["properties"]
        assert "file_path" in properties
        assert "description" in properties["file_path"]

    def test_tool_schema_file_path_is_required(self):
        tool = init_volume_tool()
        schema = cast(Dict[str, Any], tool.tool)
        required = schema["function"]["parameters"].get("required", [])
        assert "file_path" in required


class TestExecute:
    def test_execute_returns_file_content(self):
        tool = init_volume_tool()
        result = tool.execute(file_path="reports/q4.txt")
        assert result == SAMPLE_FILE_CONTENT

    def test_execute_binary_file_returns_error(self, mock_workspace_client):
        mock_resp = MagicMock()
        mock_resp.contents = io.BytesIO(b"\x80\x81\x82\x83")
        mock_workspace_client.files.download.return_value = mock_resp

        tool = init_volume_tool()
        result = tool.execute(file_path="image.png")
        assert "binary file" in result

    def test_execute_empty_path_returns_error(self):
        tool = init_volume_tool()
        result = tool.execute(file_path="")
        assert "Error" in result

    def test_execute_calls_correct_volume_path(self, mock_workspace_client):
        tool = init_volume_tool(volume_name="cat.schema.vol")
        tool.execute(file_path="subfolder/file.txt")
        mock_workspace_client.files.download.assert_called_once_with(
            "/Volumes/cat/schema/vol/subfolder/file.txt"
        )


class TestToolNameGeneration:
    def test_default_tool_name(self):
        tool = init_volume_tool()
        assert tool.tool["function"]["name"] == VOLUME_NAME.replace(".", "__")

    @pytest.mark.parametrize(
        "volume_name",
        ["catalog.schema.really_really_really_long_volume_name_that_should_be_truncated_to_64_chars"],
    )
    def test_long_volume_name_truncated(self, volume_name):
        tool = init_volume_tool(volume_name=volume_name)
        assert len(tool.tool["function"]["name"]) <= 64

    @pytest.mark.parametrize("tool_name", [None, "valid_tool_name", "test_tool"])
    def test_valid_tool_names(self, tool_name):
        tool = init_volume_tool(tool_name=tool_name)
        assert tool.tool_name == tool_name

    @pytest.mark.parametrize("tool_name", ["test.tool.name", "tool&name"])
    def test_invalid_tool_names(self, tool_name):
        with pytest.raises(ValueError):
            init_volume_tool(tool_name=tool_name)


class TestTracing:
    def test_tracing_with_default_name(self):
        tool = init_volume_tool()
        tool.execute(file_path="reports/q4.txt")
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        spans = trace.search_spans(name=VOLUME_NAME, span_type=SpanType.TOOL)
        assert len(spans) == 1

    def test_tracing_with_custom_name(self):
        tool = init_volume_tool(tool_name="my_reader")
        tool.execute(file_path="reports/q4.txt")
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        spans = trace.search_spans(name="my_reader", span_type=SpanType.TOOL)
        assert len(spans) == 1

    def test_tracing_captures_inputs(self):
        tool = init_volume_tool()
        tool.execute(file_path="reports/q4.txt")
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        inputs = json.loads(
            trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanInputs"]
        )
        assert inputs["file_path"] == "reports/q4.txt"
