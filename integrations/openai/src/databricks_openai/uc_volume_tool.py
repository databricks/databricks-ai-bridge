from typing import Any, Optional

from databricks_ai_bridge.uc_volume_tool import (
    UCVolumeToolInput,
    UCVolumeToolMixin,
    uc_volume_tool_trace,
)
from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import Field, model_validator


class UCVolumeTool(UCVolumeToolMixin):
    """
    An OpenAI-compatible tool for reading files from a Databricks Unity Catalog Volume.

    This class integrates with Databricks UC Volumes and provides a convenient interface
    for tool calling using the OpenAI SDK. Follows the same pattern as
    VectorSearchRetrieverTool.

    Example:
        Step 1: Call model with UCVolumeTool defined

        .. code-block:: python

            vol_tool = UCVolumeTool(volume_name="catalog.schema.my_documents")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Read the Q4 summary from reports/q4_summary.txt"},
            ]
            first_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=[vol_tool.tool]
            )

        Step 2: Execute function code – parse the model's response and handle function calls.

        .. code-block:: python

            tool_call = first_response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = vol_tool.execute(file_path=args["file_path"])

        Step 3: Supply model with results – so it can incorporate them into its final response.

        .. code-block:: python

            messages.append(first_response.choices[0].message)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
            second_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=[vol_tool.tool]
            )
    """

    tool: Optional[ChatCompletionToolParam] = Field(
        None, description="The tool input used in the OpenAI chat completion SDK"
    )

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        if not self.workspace_client:
            from databricks.sdk import WorkspaceClient

            self.workspace_client = WorkspaceClient()

        tool_name = self._get_tool_name()

        self.tool = pydantic_function_tool(
            UCVolumeToolInput,
            name=tool_name,
            description=self.tool_description or self._get_default_tool_description(),
        )
        # Remove strict mode for compatibility (same as VectorSearchRetrieverTool)
        if "function" in self.tool and "strict" in self.tool["function"]:
            del self.tool["function"]["strict"]
        if (
            "function" in self.tool
            and "parameters" in self.tool["function"]
            and "additionalProperties" in self.tool["function"]["parameters"]
        ):
            del self.tool["function"]["parameters"]["additionalProperties"]

        return self

    @uc_volume_tool_trace
    def execute(self, file_path: str, **kwargs: Any) -> str:
        """
        Execute the UCVolumeTool to read a file from the volume.

        Args:
            file_path: The path to the file relative to the volume root.

        Returns:
            The file contents as a string.
        """
        return self._read_file(file_path)
