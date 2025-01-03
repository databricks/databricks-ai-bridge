from typing import Any, Dict, List, Optional, Type, Tuple

from openai import OpenAI
from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageToolCall
from openai.types.chat import ChatCompletion

from pydantic import BaseModel, Field, PrivateAttr, model_validator, create_model

from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolMixin, VectorSearchRetrieverToolInput
from databricks_ai_bridge.utils.vector_search import IndexDetails, parse_vector_search_response, validate_and_get_text_column, validate_and_get_return_columns
from databricks.vector_search.client import VectorSearchIndex
import json

class VectorSearchRetrieverTool(VectorSearchRetrieverToolMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for tool calling using the OpenAI SDK.

    Example:
        dbvs_tool = VectorSearchRetrieverTool("index_name")
        tools = [dbvs_tool.tool, ...]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=initial_messages,
            tools=tools,
        )
        retriever_call_message = dbvs_tool.execute_retriever_calls(response)

        ### If needed, execute potential remaining tool calls here ###
        remaining_tool_call_messages = execute_remaining_tool_calls(response)

        final_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=initial_messages + retriever_call_message + remaining_tool_call_messages,
            tools=tools,
        )
        final_response.choices[0].message.content
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
                    "Required for direct-access index or delta-sync index with "
                    "self-managed embeddings. Used for direct access indexes or "
                    "delta-sync indexes with self-managed embeddings",
    )

    tool: ChatCompletionToolParam = Field(
        None, description="The tool input used in the OpenAI chat completion SDK"
    )
    _index: VectorSearchIndex = PrivateAttr()
    _index_details: IndexDetails = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        from databricks.vector_search.client import VectorSearchClient  # import here so we can mock in tests

        self._index = VectorSearchClient().get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)
        self.text_column = validate_and_get_text_column(self.text_column, self._index_details)
        self.columns = validate_and_get_return_columns(self.columns or [], self.text_column, self._index_details)

        # OpenAI tool names must match the pattern '^[a-zA-Z0-9_-]+$'."
        # The '.' from the index name are not allowed
        def rewrite_index_name(index_name: str):
            return index_name.split(".")[-1]

        self.tool = pydantic_function_tool(
            VectorSearchRetrieverToolInput,
            name=self.tool_name or rewrite_index_name(self.index_name),
            description=self.tool_description or self._get_default_tool_description(self._index_details)
        )
        return self

    def execute_retriever_calls(self,
                               response: ChatCompletion,
                               choice_index: int = 0,
                               embedding_model_name: str = None,
                               openai_client: OpenAI = None) -> List[Dict[str, Any]]:
        """
        Execute the VectorSearchIndex tool calls from the ChatCompletions response that correspond to the
        self.tool VectorSearchRetrieverToolInput and attach the retrieved documents into toll call messages.

        Args:
            response: The chat completion response object returned by the OpenAI API.
            choice_index: The index of the choice to process. Defaults to 0. Note that multiple
                choices are not supported yet.
            embedding_model_name: The name of the embedding model to use for embedding the query text.
                                  Required for direct access indexes or delta-sync indexes with self-managed embeddings.
            openai_client: The OpenAI client object used to generate embeddings for retrieval queries. If not provided,
                           the default OpenAI client in the current environment will be used.

        Returns:
            A list of messages containing the assistant message and the retriever call results
            that correspond to the self.tool VectorSearchRetrieverToolInput.
        """

        def get_query_text_vector(tool_call: ChatCompletionMessageToolCall) -> Tuple[Optional[str], Optional[List[float]]]:
            query = json.loads(tool_call.function.arguments)["query"]
            if self._index_details.is_databricks_managed_embeddings():
                if embedding_model_name:
                    raise ValueError(
                        f"The index '{self._index_details.name}' uses Databricks-managed embeddings. "
                        "Do not pass the `embedding_model_name` parameter when executing retriever calls."
                    )
                return query, None

            # For non-Databricks-managed embeddings
            from openai import OpenAI
            oai_client = openai_client or OpenAI()
            if not oai_client.api_key:
                raise ValueError("OpenAI API key is required to generate embeddings for retrieval queries.")
            if not embedding_model_name:
                raise ValueError("The embedding model name is required for non-Databricks-managed "
                                 "embeddings Vector Search indexes in order to generate embeddings for retrieval queries.")

            text = query if self.query_type and self.query_type.upper() == "HYBRID" else None
            vector = oai_client.embeddings.create(
                input=query,
                model=embedding_model_name
            )['data'][0]['embedding']
            if (index_embedding_dimension := self._index_details.embedding_vector_column.get("embedding_dimension")) and \
                    len(vector) != index_embedding_dimension:
                raise ValueError(
                    f"Expected embedding dimension {index_embedding_dimension} but got {len(vector)}"
                )
            return text, vector

        def is_tool_call_for_index(tool_call: ChatCompletionMessageToolCall) -> bool:
            tool_call_arguments: Set[str] = set(json.loads(tool_call.function.arguments).keys())
            vs_index_arguments: Set[str] = set(self.tool["function"]["parameters"]["properties"].keys())
            return tool_call.function.name == self.tool["function"]["name"] and \
                tool_call_arguments == vs_index_arguments

        if type(response) is not ChatCompletion:
            raise ValueError("response must be an instance of ChatCompletion")
        message = response.choices[choice_index].message
        llm_tool_calls = message.tool_calls
        function_calls = []
        if llm_tool_calls:
            for llm_tool_call in llm_tool_calls:
                # Only process tool calls that correspond to the self.tool VectorSearchRetrieverToolInput
                if not is_tool_call_for_index(llm_tool_call):
                    continue

                query_text, query_vector = get_query_text_vector(llm_tool_call)
                search_resp = self._index.similarity_search(
                    columns=self.columns,
                    query_text=query_text,
                    query_vector=query_vector,
                    filters=self.filters,
                    num_results=self.num_results,
                    query_type=self.query_type,
                )
                docs_with_score: List[Tuple[Dict, float]] = \
                    parse_vector_search_response(
                        search_resp,
                        self._index_details,
                        self.text_column,
                        ignore_cols=[],
                        document_class=dict
                    )

                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps({"content": docs_with_score}),
                    "tool_call_id": llm_tool_call.id,
                }
                function_calls.append(function_call_result_message)
        assistant_message = message.to_dict()
        return [assistant_message, *function_calls]
