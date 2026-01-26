import bisect
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

from databricks_mcp import DatabricksMCPClient

MAX_TOKENS_OF_DATA = 20000
MAX_ITERATIONS = 500  # for 250 s total
ITERATION_FREQUENCY = 0.5  # seconds

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "QUERY_RESULT_EXPIRED",
}


# Define a function to count tokens
def _count_tokens(text):
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


@dataclass
class GenieResponse:
    result: Union[str, pd.DataFrame]
    query: Optional[str] = ""
    description: Optional[str] = ""
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


@mlflow.trace(span_type="PARSER")
def _parse_query_result(
    resp, truncate_results, return_pandas: bool = False
) -> Union[str, pd.DataFrame]:
    output = resp["result"]
    if not output:
        return "EMPTY"

    columns = resp["manifest"]["schema"]["columns"]
    header = [str(col["name"]) for col in columns]
    rows = []

    for item in output["data_array"]:
        row = []
        values = item["values"]
        for column, value_obj in zip(columns, values):
            value = value_obj.get("string_value") if isinstance(value_obj, dict) else value_obj

            type_name = column["type_name"]
            if value is None:
                row.append(None)
                continue

            if type_name in ["INT", "LONG", "SHORT", "BYTE"]:
                row.append(int(value))
            elif type_name in ["FLOAT", "DOUBLE", "DECIMAL"]:
                row.append(float(value))
            elif type_name == "BOOLEAN":
                row.append(value.lower() == "true")
            elif type_name == "DATE":
                # first 10 characters represent the date
                row.append(datetime.strptime(value[:10], "%Y-%m-%d").date())
            elif type_name == "TIMESTAMP":
                # https://docs.databricks.com/aws/en/sql/language-manual/data-types/timestamp-type
                # first 19 characters represent the date and time to the second
                # doesn't account for possibility of +/- in first character
                stripped_value = value[:19]
                timestamp_formats = [
                    "%Y-%m-%dT%H:%M:%S",  # 2023-10-01T14:30:45
                    "%Y-%m-%d %H:%M:%S",  # 2023-10-01 14:30:45
                    "%Y-%m-%dT%H:%M",  # 2023-10-01T14:30
                    "%Y-%m-%d %H:%M",  # 2023-10-01 14:30
                    "%Y-%m-%dT%H",  # 2023-10-01T14
                    "%Y-%m-%d %H",  # 2023-10-01 14
                    "%Y-%m-%d",  # 2023-10-01
                ]

                parsed_timestamp = None
                for fmt in timestamp_formats:
                    try:
                        parsed_timestamp = datetime.strptime(stripped_value, fmt)
                        break
                    except ValueError:
                        continue

                if parsed_timestamp is None:
                    # Fallback: just parse the date part
                    parsed_timestamp = datetime.strptime(value[:10], "%Y-%m-%d")

                row.append(parsed_timestamp)
            elif type_name == "BINARY":
                row.append(bytes(value, "utf-8"))
            else:
                row.append(value)

        rows.append(row)

    dataframe = pd.DataFrame(rows, columns=header)
    if return_pandas:
        return dataframe

    if truncate_results:
        query_result = _truncate_result(dataframe)
    else:
        query_result = dataframe.to_markdown()

    return query_result.strip()


def _truncate_result(dataframe):
    query_result = dataframe.to_markdown()
    tokens_used = _count_tokens(query_result)

    # If the full result fits, return it
    if tokens_used <= MAX_TOKENS_OF_DATA:
        return query_result.strip()

    def is_too_big(n):
        return _count_tokens(dataframe.iloc[:n].to_markdown()) > MAX_TOKENS_OF_DATA

    # Use bisect_left to find the cutoff point of rows within the max token data limit in a O(log n) complexity
    # Passing True, as this is the target value we are looking for when _is_too_big returns
    cutoff = bisect.bisect_left(range(len(dataframe) + 1), True, key=is_too_big)

    # Slice to the found limit
    truncated_df = dataframe.iloc[:cutoff]

    # Edge case: Cannot return any rows because of tokens so return an empty string
    if len(truncated_df) == 0:
        return ""

    truncated_result = truncated_df.to_markdown()

    # Double-check edge case if we overshot by one
    if _count_tokens(truncated_result) > MAX_TOKENS_OF_DATA:
        truncated_result = truncated_df.iloc[:-1].to_markdown()
    return truncated_result


def _end_current_span(client, parent_trace_id, current_span, final_state, error=None):
    """Helper function to safely end a span with exception handling."""
    if current_span is None:
        return None

    try:
        attributes = {"final_state": final_state}
        if error is not None:
            attributes["error"] = error

        client.end_span(
            trace_id=parent_trace_id,
            span_id=current_span.span_id,
            attributes=attributes,
        )
    except Exception as e:
        logging.warning(f"Failed to end span for {final_state}: {e}")

    return None


def _parse_genie_mcp_response(
    mcp_result,
    truncate_results: bool,
    return_pandas: bool,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> GenieResponse:
    if not mcp_result.content or len(mcp_result.content) == 0:
        return GenieResponse(
            result="No content returned from Genie",
            conversation_id=conversation_id,
            message_id=message_id,
        )

    # Genie backend always returns 1 content block with JSON
    content_block = mcp_result.content[0]
    content_text = content_block.text if hasattr(content_block, "text") else "{}"

    try:
        genie_response = json.loads(content_text)
    except json.JSONDecodeError:
        return GenieResponse(
            result=f"Failed to parse response: {content_text}",
            conversation_id=conversation_id,
            message_id=message_id,
        )

    content = genie_response.get("content", {})
    conv_id = genie_response.get("conversationId", conversation_id)
    msg_id = genie_response.get("messageId", message_id)
    query_str = ""
    description = ""

    try:
        query_attachments = content.get("queryAttachments", [])
        text_attachments = content.get("textAttachments", [])

        if query_attachments:
            first_query = query_attachments[0]
            query_str = first_query.get("query", "")
            description = first_query.get("description", "")
            statement_response = first_query.get("statement_response")

            if statement_response and statement_response.get("status", {}).get("state") == "SUCCEEDED":
                result = _parse_query_result(statement_response, truncate_results, return_pandas)
            elif text_attachments:
                result = text_attachments[0]
            else:
                result = str(content)
        elif text_attachments:
            result = text_attachments[0]
        else:
            result = str(content)

    except (KeyError, TypeError, AttributeError):
        result = str(content)

    return GenieResponse(
        result=result,
        query=query_str,
        description=description,
        conversation_id=conv_id,
        message_id=msg_id,
    )


class Genie:
    def __init__(
        self,
        space_id,
        client: Optional["WorkspaceClient"] = None,
        truncate_results=False,
        return_pandas: bool = False,
    ):
        self.space_id = space_id
        workspace_client = client or WorkspaceClient()
        self.genie = workspace_client.genie
        self.description = self.genie.get_space(space_id).description

        server_url = f"{workspace_client.config.host}/api/2.0/mcp/genie/{space_id}"
        self._mcp_client = DatabricksMCPClient(server_url, workspace_client)

        tools = self._mcp_client.list_tools()
        if not tools:
            raise ValueError(f"No tools found in Genie MCP server for space {space_id}")

        query_tools = [tool for tool in tools if "query" in tool.name.lower()]
        poll_tools = [tool for tool in tools if "poll" in tool.name.lower()]

        self._query_tool_name = query_tools[0].name if query_tools else None
        self._poll_tool_name = poll_tools[0].name if poll_tools else None

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.truncate_results = truncate_results
        self.return_pandas = return_pandas

    @mlflow.trace()
    def start_conversation(self, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def create_message(self, conversation_id, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def poll_for_result(self, conversation_id, message_id):
        if not self._poll_tool_name:
            raise ValueError(
                f"Poll tool not available for Genie space {self.space_id}. "
                f"The MCP server must expose a poll tool to use poll_for_result()."
            )

        # Use MLflow client for manual span management to track status transitions
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_span(name="genie_timeline", span_type="CHAIN") as parent:
            parent_trace_id = parent.trace_id if parent else None
            parent_span_id = parent.span_id if parent else None

            # Track last status and current child span
            last_status = None
            current_span = None

            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1

                args = {"conversation_id": conversation_id, "message_id": message_id}
                mcp_result = self._mcp_client.call_tool(self._poll_tool_name, args)

                try:
                    if not mcp_result.content or len(mcp_result.content) == 0:
                        # End any active span before returning
                        _end_current_span(client, parent_trace_id, current_span, last_status)
                        return GenieResponse(
                            result="No content returned from Genie poll",
                            conversation_id=conversation_id,
                            message_id=message_id,
                        )

                    content_block = mcp_result.content[0]
                    content_text = content_block.text if hasattr(content_block, "text") else "{}"
                    genie_response = json.loads(content_text)
                    status = genie_response.get("status", "")
                except (json.JSONDecodeError, AttributeError, KeyError):
                    # End any active span before returning
                    _end_current_span(client, parent_trace_id, current_span, last_status)
                    return _parse_genie_mcp_response(
                        mcp_result,
                        self.truncate_results,
                        self.return_pandas,
                        conversation_id,
                        message_id,
                    )

                # On status change: end previous span, start new one
                if status != last_status:
                    # END previous span
                    current_span = _end_current_span(
                        client, parent_trace_id, current_span, last_status
                    )

                    # START new span for non-terminal states
                    if status not in TERMINAL_STATES:
                        try:
                            current_span = client.start_span(
                                name=status.lower(),
                                trace_id=parent_trace_id,
                                parent_id=parent_span_id,
                                span_type="CHAIN",
                                attributes={
                                    "state": status,
                                    "conversation_id": conversation_id,
                                    "message_id": message_id,
                                },
                            )
                        except Exception as e:
                            logging.warning(f"Failed to create span for {status}: {e}")
                            current_span = None

                    logging.debug(f"Status: {last_status} → {status}")
                    last_status = status

                # Check for terminal states
                if status in TERMINAL_STATES:
                    # End any active span before returning
                    _end_current_span(client, parent_trace_id, current_span, last_status)
                    return _parse_genie_mcp_response(
                        mcp_result,
                        self.truncate_results,
                        self.return_pandas,
                        conversation_id,
                        message_id,
                    )

                logging.debug(f"Polling: status={status}, iteration={iteration_count}")
                time.sleep(ITERATION_FREQUENCY)

            # Timeout - end any active span
            _end_current_span(client, parent_trace_id, current_span, last_status)
            return GenieResponse(
                result=f"Genie query timed out after {MAX_ITERATIONS} iterations of {ITERATION_FREQUENCY} seconds",
                conversation_id=conversation_id,
                message_id=message_id,
            )

    @mlflow.trace()
    def ask_question(self, question, conversation_id: Optional[str] = None):
        if not self._query_tool_name:
            raise ValueError(
                f"Query tool not available for Genie space {self.space_id}. "
                f"The MCP server must expose a query tool to use ask_question()."
            )

        args = {"query": question}
        if conversation_id:
            args["conversation_id"] = conversation_id

        mcp_result = self._mcp_client.call_tool(self._query_tool_name, args)
        return _parse_genie_mcp_response(
            mcp_result, self.truncate_results, self.return_pandas, conversation_id
        )
