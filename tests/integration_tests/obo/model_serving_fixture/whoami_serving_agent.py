"""
Minimal OBO whoami agent for Model Serving.

Calls the whoami() UC function via SQL Statement Execution API
using ModelServingUserCredentials to act as the invoking user.

This file gets logged as an MLflow model artifact via:
    mlflow.pyfunc.log_model(python_model="whoami_serving_agent.py", ...)
"""

import json
from typing import Any, Callable, Generator
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel

from databricks_ai_bridge import ModelServingUserCredentials

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-6"
SQL_WAREHOUSE_ID = ""  # Injected by deploy_serving_agent.py at log time


class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable


def create_whoami_tool(user_client: WorkspaceClient) -> ToolInfo:
    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_whoami(**kwargs) -> str:
        try:
            response = user_client.statement_execution.execute_statement(
                warehouse_id=SQL_WAREHOUSE_ID,
                statement="SELECT integration_testing.databricks_ai_bridge_mcp_test.whoami() as result",
                wait_timeout="30s",
            )
            if response.status.state == StatementState.SUCCEEDED:
                if response.result and response.result.data_array:
                    return str(response.result.data_array[0][0])
                return "No result returned"
            return f"Query failed with state: {response.status.state}"
        except Exception as e:
            return f"Error calling whoami: {e}"

    tool_spec = {
        "type": "function",
        "function": {
            "name": "whoami",
            "description": "Returns information about the current user",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    return ToolInfo(name="whoami", spec=tool_spec, exec_fn=execute_whoami)


class ToolCallingAgent(ResponsesAgent):
    def __init__(self, llm_endpoint: str, warehouse_id: str):
        self.llm_endpoint = llm_endpoint
        self.warehouse_id = warehouse_id
        self._tools_dict = None

    def get_tool_specs(self) -> list[dict]:
        if self._tools_dict is None:
            return []
        return [t.spec for t in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        return self._tools_dict[tool_name].exec_fn(**args)

    def call_llm(
        self, messages: list[dict[str, Any]], user_client: WorkspaceClient
    ) -> Generator[dict[str, Any], None, None]:
        client: OpenAI = user_client.serving_endpoints.get_open_ai_client()
        for chunk in client.chat.completions.create(
            model=self.llm_endpoint,
            messages=to_chat_completions_input(messages),
            tools=self.get_tool_specs(),
            stream=True,
        ):
            chunk_dict = chunk.to_dict()
            if len(chunk_dict.get("choices", [])) > 0:
                yield chunk_dict

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        try:
            args = json.loads(tool_call.get("arguments", "{}"))
        except Exception:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        user_client: WorkspaceClient,
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                return
            elif last_msg.get("type") == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages, user_client), aggregator=messages
                )
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        outputs = [
            event.item
            for event in self.predict_stream(request, user_client)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)

    def predict_stream(
        self, request: ResponsesAgentRequest, user_client: WorkspaceClient = None
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if user_client is None:
            user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        whoami_tool = create_whoami_tool(user_client)
        self._tools_dict = {whoami_tool.name: whoami_tool}
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        yield from self.call_and_run_tools(messages=messages, user_client=user_client)


AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, warehouse_id=SQL_WAREHOUSE_ID)
mlflow.models.set_model(AGENT)
