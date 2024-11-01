import logging
import time
from datetime import datetime
from typing import Union

import pandas as pd
import tiktoken
from databricks.sdk import WorkspaceClient

MAX_TOKENS_OF_DATA = 20000  # max tokens of data in markdown format
MAX_ITERATIONS = 50  # max times to poll the API when polling for either result or the query results, each iteration is ~1 second, so max latency == 2 * MAX_ITERATIONS


# Define a function to count tokens
def _count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


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
            elif type_name == "DATE":
                row.append(datetime.strptime(str_value[:10], "%Y-%m-%d").date())
            elif type_name == "TIMESTAMP":
                row.append(datetime.strptime(str_value[:10], "%Y-%m-%d").date())
            elif type_name == "BINARY":
                row.append(bytes(str_value, "utf-8"))
            else:
                row.append(str_value)

        rows.append(row)

    query_result = pd.DataFrame(rows, columns=header).to_markdown()

    # trim down from the total rows until we get under the token limit
    tokens_used = _count_tokens(query_result)
    while tokens_used > MAX_TOKENS_OF_DATA:
        rows.pop()
        query_result = pd.DataFrame(rows, columns=header).to_markdown()
        tokens_used = _count_tokens(query_result)

    return query_result.strip() if query_result else query_result


class Genie:
    def __init__(self, space_id):
        self.space_id = space_id
        workspace_client = WorkspaceClient()
        self.genie = workspace_client.genie
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def start_conversation(self, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    def create_message(self, conversation_id, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    def poll_for_result(self, conversation_id, message_id):
        def poll_result():
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                    headers=self.headers,
                )
                if resp["status"] == "EXECUTING_QUERY":
                    query = next(r for r in resp["attachments"] if "query" in r)["query"]
                    description = query.get("description", "")
                    sql = query.get("query", "")
                    logging.debug(f"Description: {description}")
                    logging.debug(f"SQL: {sql}")
                    return poll_query_results()
                elif resp["status"] == "COMPLETED":
                    # Check if there is a query object in the attachments for the COMPLETED status
                    query_attachment = next((r for r in resp["attachments"] if "query" in r), None)
                    if query_attachment:
                        query = query_attachment["query"]
                        description = query.get("description", "")
                        sql = query.get("query", "")
                        logging.debug(f"Description: {description}")
                        logging.debug(f"SQL: {sql}")
                        return poll_query_results()
                    else:
                        # Handle the text object in the COMPLETED status
                        return next(r for r in resp["attachments"] if "text" in r)["text"][
                            "content"
                        ]
                elif resp["status"] == "FAILED":
                    logging.debug("Genie failed to execute the query")
                    return None
                elif resp["status"] == "CANCELLED":
                    logging.debug("Genie query cancelled")
                    return None
                elif resp["status"] == "QUERY_RESULT_EXPIRED":
                    logging.debug("Genie query result expired")
                    return None
                else:
                    logging.debug(f"Waiting...: {resp['status']}")
                    time.sleep(5)

        def poll_query_results():
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
                    return _parse_query_result(resp)
                elif state == "RUNNING" or state == "PENDING":
                    logging.debug("Waiting for query result...")
                    time.sleep(5)
                else:
                    logging.debug(f"No query result: {resp['state']}")
                    return None

        return poll_result()

    def ask_question(self, question):
        resp = self.start_conversation(question)
        # TODO (prithvi): return the query and the result
        return self.poll_for_result(resp["conversation_id"], resp["message_id"])
