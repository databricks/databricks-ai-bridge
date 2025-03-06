import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import mlflow
import pandas as pd
import tiktoken
from databricks.sdk import WorkspaceClient

MAX_TOKENS_OF_DATA = 20000
MAX_ITERATIONS = 50


# Define a function to count tokens
def _count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


@dataclass
class GenieResponse:
    result: Union[str, pd.DataFrame]
    query: Optional[str] = ""
    description: Optional[str] = ""


@mlflow.trace(span_type="PARSER")
def _parse_query_result(resp) -> Union[str, pd.DataFrame]:
    columns = resp["manifest"]["schema"]["columns"]
    header = [str(col["name"]) for col in columns]
    rows = []
    output = resp["result"]
    if not output:
        return "EMPTY"

    for item in resp["result"]["data_typed_array"]:
        row = []
        for column, value in zip(columns, item["values"]):
            type_name = column["type_name"]
            str_value = value.get("str", None)
            if str_value is None:
                row.append(None)
                continue

            if type_name in ["INT", "LONG", "SHORT", "BYTE"]:
                row.append(int(str_value))
            elif type_name in ["FLOAT", "DOUBLE", "DECIMAL"]:
                row.append(float(str_value))
            elif type_name == "BOOLEAN":
                row.append(str_value.lower() == "true")
            elif type_name == "DATE" or type_name == "TIMESTAMP":
                row.append(datetime.strptime(str_value[:10], "%Y-%m-%d").date())
            elif type_name == "BINARY":
                row.append(bytes(str_value, "utf-8"))
            else:
                row.append(str_value)

        rows.append(row)

    query_result = pd.DataFrame(rows, columns=header).to_markdown()

    tokens_used = _count_tokens(query_result)
    while tokens_used > MAX_TOKENS_OF_DATA:
        rows.pop()
        query_result = pd.DataFrame(rows, columns=header).to_markdown()
        tokens_used = _count_tokens(query_result)

    return query_result.strip() if query_result else query_result


class Genie:
    def __init__(self, space_id, client: Optional["WorkspaceClient"] = None):
        self.space_id = space_id
        workspace_client = client or WorkspaceClient()
        self.genie = workspace_client.genie
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

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
        @mlflow.trace()
        def poll_query_results(query, description):
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/query-result",
                    headers=self.headers,
                )["statement_response"]
                state = resp["status"]["state"]
                if state == "SUCCEEDED":
                    result = _parse_query_result(resp)
                    return GenieResponse(result, query, description)
                elif state in ["RUNNING", "PENDING"]:
                    logging.debug("Waiting for query result...")
                    time.sleep(5)
                else:
                    return GenieResponse(f"No query result: {resp['state']}", query, description)
            return GenieResponse(
                f"Genie query for result timed out after {MAX_ITERATIONS} iterations of 5 seconds",
                query,
                description,
            )

        @mlflow.trace()
        def poll_result():
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                    headers=self.headers,
                )
                if resp["status"] in {"EXECUTING_QUERY", "COMPLETED"}:
                    query_attachment = next((r for r in resp["attachments"] if "query" in r), None)
                    if query_attachment:
                        query = query_attachment["query"]["query"]
                        description = query_attachment["query"].get("description", "")
                        return poll_query_results(query, description)
                    if resp["status"] == "COMPLETED":
                        text_content = next(r for r in resp["attachments"] if "text" in r)["text"][
                            "content"
                        ]
                        return GenieResponse(result=text_content)
                elif resp["status"] in {"CANCELLED", "QUERY_RESULT_EXPIRED"}:
                    return GenieResponse(result=f"Genie query {resp['status'].lower()}.")
                elif resp["status"] == "FAILED":
                    return GenieResponse(
                        result=f"Genie query failed with error: {resp.get('error', 'Unknown error')}"
                    )
                else:
                    logging.debug(f"Waiting...: {resp['status']}")
                    time.sleep(5)
            return GenieResponse(
                f"Genie query timed out after {MAX_ITERATIONS} iterations of 5 seconds",
                query,
                description,
            )

        return poll_result()

    @mlflow.trace()
    def ask_question(self, question):
        resp = self.start_conversation(question)
        return self.poll_for_result(resp["conversation_id"], resp["message_id"])
