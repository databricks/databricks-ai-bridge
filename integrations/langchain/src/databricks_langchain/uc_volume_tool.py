from typing import Type

from databricks_ai_bridge.uc_volume_tool import (
    UCVolumeToolInput,
    UCVolumeToolMixin,
    uc_volume_tool_trace,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class UCVolumeTool(BaseTool, UCVolumeToolMixin):
    """
    A LangChain tool for reading files from a Databricks Unity Catalog Volume.

    This class integrates with Databricks UC Volumes and provides a convenient interface
    for building a file reading tool for agents. Follows the same pattern as
    VectorSearchRetrieverTool.

    Example:

        .. code-block:: python

            from databricks_langchain import UCVolumeTool, ChatDatabricks

            vol_tool = UCVolumeTool(
                volume_name="catalog.schema.my_documents",
                tool_name="document_reader",
                tool_description="Reads files from the company documents volume.",
            )

            # Test locally
            vol_tool.invoke("reports/q4_summary.txt")

            # Bind to LLM
            llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
            llm_with_tools = llm.bind_tools([vol_tool])
            llm_with_tools.invoke("Read the Q4 summary from reports/q4_summary.txt")
    """

    # BaseTool requires 'name' and 'description' fields; populated in _validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = UCVolumeToolInput

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        self.name = self._get_tool_name()
        self.description = self.tool_description or self._get_default_tool_description()
        if not self.workspace_client:
            from databricks.sdk import WorkspaceClient

            self.workspace_client = WorkspaceClient()
        return self

    @uc_volume_tool_trace
    def _run(self, file_path: str, **kwargs) -> str:
        return self._read_file(file_path)
