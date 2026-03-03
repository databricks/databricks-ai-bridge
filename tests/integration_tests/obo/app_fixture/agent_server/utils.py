import json
import logging
import os
from typing import AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import ResponsesAgentStreamEvent


def get_databricks_host(workspace_client: WorkspaceClient | None = None) -> Optional[str]:
    workspace_client = workspace_client or WorkspaceClient()
    try:
        return workspace_client.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


def get_user_workspace_client() -> WorkspaceClient:
    """Get a WorkspaceClient authenticated as the requesting user via OBO.

    Reads the x-forwarded-access-token header injected by the Databricks Apps
    proxy when user authorization scopes are configured on the app.
    Falls back to the app's default credentials if the header is absent.
    """
    headers = get_request_headers()
    token = headers.get("x-forwarded-access-token")
    if not token:
        logging.warning(
            "No x-forwarded-access-token header found. "
            "Ensure user authorization scopes are configured on the app. "
            "Available headers: %s",
            list(headers.keys()),
        )
        return WorkspaceClient()
    host = get_databricks_host()
    # Temporarily clear app SP credentials from env to avoid
    # "more than one authorization method" conflict in the SDK
    old_id = os.environ.pop("DATABRICKS_CLIENT_ID", None)
    old_secret = os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
    try:
        wc = WorkspaceClient(host=host, token=token)
    finally:
        if old_id is not None:
            os.environ["DATABRICKS_CLIENT_ID"] = old_id
        if old_secret is not None:
            os.environ["DATABRICKS_CLIENT_SECRET"] = old_secret
    return wc


async def process_agent_stream_events(
    async_stream: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    curr_item_id = str(uuid4())
    async for event in async_stream:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            if event_data["type"] == "response.output_item.added":
                curr_item_id = str(uuid4())
                event_data["item"]["id"] = curr_item_id
            elif (
                event_data.get("item") is not None
                and event_data["item"].get("id") is not None
            ):
                event_data["item"]["id"] = curr_item_id
            elif event_data.get("item_id") is not None:
                event_data["item_id"] = curr_item_id
            yield event_data
        elif (
            event.type == "run_item_stream_event"
            and event.item.type == "tool_call_output_item"
        ):
            output = event.item.to_input_item()
            if not isinstance(output.get("output"), str):
                try:
                    output["output"] = json.dumps(output.get("output"))
                except (TypeError, ValueError):
                    output["output"] = str(output.get("output"))
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=output,
            )
