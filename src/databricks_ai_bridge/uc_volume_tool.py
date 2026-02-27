import logging
import re
from functools import wraps
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from databricks_ai_bridge.utils.uc_volume import read_volume_file

_logger = logging.getLogger(__name__)
DEFAULT_TOOL_DESCRIPTION = "A tool for reading files from a Databricks Unity Catalog Volume."


def uc_volume_tool_trace(func):
    """
    Decorator factory to trace UCVolumeTool with the tool name.
    Parallels vector_search_retriever_tool_trace.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        traced_func = mlflow.trace(
            name=self.tool_name or self.volume_name, span_type=SpanType.TOOL
        )(func)
        return traced_func(self, *args, **kwargs)

    return wrapper


class UCVolumeToolInput(BaseModel):
    """Input schema that the LLM sees and generates."""

    model_config = ConfigDict(extra="allow")
    file_path: str = Field(
        description=(
            "The path to the file to read, relative to the volume root. "
            "For example: 'reports/q4_summary.txt' or 'data/config.json'."
        )
    )


class UCVolumeToolMixin(BaseModel):
    """
    Mixin class for Databricks UC Volume tools.
    This class provides the common structure and interface that framework-specific
    implementations (LangChain, OpenAI) should follow.
    Parallels VectorSearchRetrieverToolMixin.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    volume_name: str = Field(
        ..., description="The full volume name: 'catalog.schema.volume'."
    )
    tool_name: Optional[str] = Field(None, description="The name of the tool.")
    tool_description: Optional[str] = Field(None, description="A description of the tool.")
    workspace_client: Optional[WorkspaceClient] = Field(
        None,
        description="Optional pre-configured WorkspaceClient for authentication.",
    )
    resources: Optional[list] = Field(
        None, description="Resources required to log a model that uses this tool."
    )

    @model_validator(mode="after")
    def _validate_volume_name(self):
        parts = self.volume_name.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"volume_name must be 'catalog.schema.volume', got '{self.volume_name}'"
            )
        return self

    @field_validator("tool_name")
    def validate_tool_name(cls, tool_name):
        if tool_name is not None:
            pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
            if not pattern.fullmatch(tool_name):
                raise ValueError("tool_name must match the pattern '^[a-zA-Z0-9_-]{1,64}$'")
        return tool_name

    def _get_tool_name(self) -> str:
        tool_name = self.tool_name or self.volume_name.replace(".", "__")
        if len(tool_name) > 64:
            _logger.warning(
                f"Tool name {tool_name} is too long, truncating to 64 characters {tool_name[-64:]}."
            )
            return tool_name[-64:]
        return tool_name

    def _get_default_tool_description(self) -> str:
        return (
            f"{DEFAULT_TOOL_DESCRIPTION} "
            f"Reads files from the Unity Catalog volume '{self.volume_name}'. "
            f"Provide the file path relative to the volume root."
        )

    def _read_file(self, file_path: str) -> str:
        """
        Core execution logic shared across frameworks.
        Reads a file from the volume and returns its text content.
        """
        from databricks.sdk import WorkspaceClient

        wc = self.workspace_client or WorkspaceClient()
        if not file_path:
            return "Error: file_path is required."
        try:
            content = read_volume_file(self.volume_name, file_path, workspace_client=wc)
        except UnicodeDecodeError:
            return (
                f"Cannot read '{file_path}': binary file. "
                f"This tool supports text-based files only."
            )
        return content
