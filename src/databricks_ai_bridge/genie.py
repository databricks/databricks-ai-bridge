import bisect
import json
import logging
import time
import warnings
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
        for column, value in zip(columns, item):
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
    """helper function to safely end a span with exception handling."""

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
    except mlflow.exceptions.MlflowTracingException as e:
        logging.warning(f"Failed to end span for {final_state}: {e}")

    return None


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

        # Initialize MCP client for communication with Genie backend
        server_url = f"{workspace_client.config.host}/api/2.0/mcp/genie/{space_id}"
        self._mcp_client = DatabricksMCPClient(server_url, workspace_client)
        self._tool_name = f"query_space_{space_id}"

        # Keep headers for deprecated REST methods (backwards compatibility)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        self.truncate_results = truncate_results
        self.return_pandas = return_pandas

    @mlflow.trace()
    def start_conversation(self, content):
        """Start a conversation with the Genie space.

        .. deprecated::
            This method is deprecated and will be removed in a future release.
            Use :meth:`ask_question` instead, which uses the MCP protocol.
        """
        warnings.warn(
            "start_conversation() is deprecated and will be removed in a future release. "
            "Use ask_question() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def create_message(self, conversation_id, content):
        """Create a message in an existing conversation.

        .. deprecated::
            This method is deprecated and will be removed in a future release.
            Use :meth:`ask_question` with a conversation_id parameter instead.
        """
        warnings.warn(
            "create_message() is deprecated and will be removed in a future release. "
            "Use ask_question(question, conversation_id=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def poll_for_result(self, conversation_id, message_id):
        """Poll for the result of a Genie query.

        .. deprecated::
            This method is deprecated and will be removed in a future release.
            Use :meth:`ask_question` instead, which handles polling automatically via MCP.
        """
        warnings.warn(
            "poll_for_result() is deprecated and will be removed in a future release. "
            "Use ask_question() instead, which handles polling automatically.",
            DeprecationWarning,
            stacklevel=2,
        )

        @mlflow.trace()
        def poll_query_results(
            attachment_id, query_str, description, conversation_id=conversation_id
        ):
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result",
                    headers=self.headers,
                )["statement_response"]
                state = resp["status"]["state"]
                returned_conversation_id = resp.get("conversation_id", None)
                if state == "SUCCEEDED":
                    result = _parse_query_result(resp, self.truncate_results, self.return_pandas)
                    return GenieResponse(result, query_str, description, returned_conversation_id)
                elif state in ["RUNNING", "PENDING"]:
                    logging.debug("Waiting for query result...")
                    time.sleep(ITERATION_FREQUENCY)
                else:
                    return GenieResponse(
                        f"No query result: {resp['state']}",
                        query_str,
                        description,
                        returned_conversation_id,
                    )
            return GenieResponse(
                f"Genie query for result timed out after {MAX_ITERATIONS} iterations of 5 seconds",
                query_str,
                description,
                conversation_id,
            )

        @mlflow.trace()
        def poll_result():
            iteration_count = 0

            # use MLflow client to get parent of any new spans we create from the current active span
            # (parenting keeps spans in the same trace)
            client = mlflow.tracking.MlflowClient()
            with mlflow.start_span(name="genie_timeline", span_type="CHAIN") as parent:
                parent_trace_id = parent.trace_id if parent else None
                parent_span_id = parent.span_id if parent else None

                # Track last status from API and the current child span
                last_status = None
                current_span = None

                while iteration_count < MAX_ITERATIONS:
                    iteration_count += 1
                    resp = self.genie._api.do(
                        "GET",
                        f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                        headers=self.headers,
                    )
                    returned_conversation_id = resp.get("conversation_id", None)

                    # get current status from API response
                    current_status = resp["status"]

                    # On status change: end previous span, start a new one (excluding terminal states)
                    if current_status != last_status:
                        # END previous span
                        current_span = _end_current_span(
                            client, parent_trace_id, current_span, last_status
                        )

                        # START new span for non-terminal states
                        if current_status not in TERMINAL_STATES:
                            # START new span
                            try:
                                current_span = client.start_span(
                                    name=current_status.lower(),
                                    trace_id=parent_trace_id,
                                    parent_id=parent_span_id,
                                    span_type="CHAIN",
                                    attributes={
                                        "state": current_status,
                                        "conversation_id": conversation_id,
                                        "message_id": message_id,
                                    },
                                )
                            except mlflow.exceptions.MlflowTracingException as e:
                                logging.warning(f"Failed to create span for {current_status}: {e}")
                                current_span = None

                        logging.debug(f"Status: {last_status} → {current_status}")
                        last_status = current_status

                    if current_status == "COMPLETED":
                        attachment = next((r for r in resp["attachments"] if "query" in r), None)
                        if attachment:
                            query_obj = attachment["query"]
                            description = query_obj.get("description", "")
                            query_str = query_obj.get("query", "")
                            attachment_id = attachment["attachment_id"]
                            return poll_query_results(
                                attachment_id,
                                query_str,
                                description,
                                returned_conversation_id,
                            )
                        if current_status == "COMPLETED":
                            text_content = next(r for r in resp["attachments"] if "text" in r)[
                                "text"
                            ]["content"]
                            return GenieResponse(
                                result=text_content,
                                conversation_id=returned_conversation_id,
                            )

                    elif current_status in {"CANCELLED", "QUERY_RESULT_EXPIRED"}:
                        return GenieResponse(result=f"Genie query {current_status.lower()}.")

                    elif current_status == "FAILED":
                        return GenieResponse(
                            result=f"Genie query failed with error: {resp.get('error', 'Unknown error')}"
                        )
                    # includes EXECUTING_QUERY, Genie can retry after this status
                    else:
                        logging.debug(f"Status: {current_status}")
                        time.sleep(ITERATION_FREQUENCY)  # faster poll rate

                # timeout path / end of while loop — close any open spans
                current_span = _end_current_span(
                    client,
                    parent_trace_id,
                    current_span,
                    last_status,
                )
                return GenieResponse(
                    f"Genie query timed out after {MAX_ITERATIONS} iterations of {ITERATION_FREQUENCY} seconds (total {MAX_ITERATIONS * ITERATION_FREQUENCY} seconds)",
                    conversation_id=conversation_id,
                )

        return poll_result()

    @mlflow.trace()
    def ask_question(self, question, conversation_id: Optional[str] = None):
        """Ask a question to the Genie space using MCP protocol.

        Args:
            question: The question to ask the Genie space
            conversation_id: Optional conversation ID to continue an existing conversation

        Returns:
            GenieResponse with result, query, description, and conversation_id
        """
        try:
            # Try using the new MCP approach
            args = {"query": question}
            if conversation_id:
                args["conversation_id"] = conversation_id

            mcp_result = self._mcp_client.call_tool(self._tool_name, args)

            if not mcp_result.content or len(mcp_result.content) == 0:
                return GenieResponse(
                    result="No content returned from Genie",
                    conversation_id=conversation_id,
                )

            # Genie backend always returns exactly 1 content block with JSON
            content_block = mcp_result.content[0]
            content_text = content_block.text if hasattr(content_block, "text") else "{}"

            try:
                genie_response = json.loads(content_text)
            except json.JSONDecodeError:
                return GenieResponse(
                    result=f"Failed to parse response: {content_text}",
                    conversation_id=conversation_id,
                )

            content = genie_response.get("content", "")
            conv_id = genie_response.get("conversationId", conversation_id)
            status = genie_response.get("status", "")

            try:
                content_data = json.loads(content)
                query_str = content_data.get("query", "")
                description = content_data.get("description", "")
                statement_response = content_data.get("statement_response")

                if (
                    statement_response
                    and statement_response.get("status", {}).get("state") == "SUCCEEDED"
                ):
                    result = _parse_query_result(
                        statement_response, self.truncate_results, self.return_pandas
                    )
                else:
                    result = content
                    query_str = ""
                    description = ""

            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                result = content
                query_str = ""
                description = ""

            return GenieResponse(
                result=result,
                query=query_str,
                description=description,
                conversation_id=conv_id,
            )

        except Exception as e:
            # Fall back to deprecated REST methods
            logging.warning(f"MCP call failed, falling back to REST API: {e}")

            if conversation_id:
                resp = self.create_message(conversation_id, question)
                return self.poll_for_result(conversation_id, resp["id"])
            else:
                resp = self.start_conversation(question)
                return self.poll_for_result(resp["conversation_id"], resp["id"])
