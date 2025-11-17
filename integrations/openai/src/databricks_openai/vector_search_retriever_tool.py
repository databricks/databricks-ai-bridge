import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

from databricks.vector_search.client import VectorSearchIndex
from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    RetrieverSchema,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolMixin,
    vector_search_retriever_tool_trace,
)
from openai import OpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import Field, PrivateAttr, model_validator

_logger = logging.getLogger(__name__)


class VectorSearchRetrieverTool(VectorSearchRetrieverToolMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for tool calling using the OpenAI SDK.

    Example:
        Step 1: Call model with VectorSearchRetrieverTool defined

        .. code-block:: python

            dbvs_tool = VectorSearchRetrieverTool(index_name="catalog.schema.my_index_name")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Using the Databricks documentation, answer what is Spark?",
                },
            ]
            first_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=[dbvs_tool.tool]
            )

        Step 2: Execute function code – parse the model's response and handle function calls.

        .. code-block:: python

            tool_call = first_response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = dbvs_tool.execute(
                query=args["query"], filters=args.get("filters", None)
            )  # For self-managed embeddings, optionally pass in openai_client=client

        Step 3: Supply model with results – so it can incorporate them into its final response.

        .. code-block:: python

            messages.append(first_response.choices[0].message)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )
            second_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tools
            )

    **Note**: Any additional keyword arguments passed to the constructor will be passed along to
    `databricks.vector_search.client.VectorSearchIndex.similarity_search` when executing the tool. `See
    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
    to see the full set of supported keyword arguments,
    e.g. `score_threshold`. Also, see documentation for
    :class:`~databricks_ai_bridge.vector_search_retriever_tool.VectorSearchRetrieverToolMixin` for additional supported constructor
    arguments not listed below, including `query_type` and `num_results`.

    WorkspaceClient instances with auth types PAT, OAuth-M2M (client ID and client secret), or model serving credential strategy will be used to instantiate the underlying VectorSearchClient.
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding_model_name: Optional[str] = Field(
        None,
        description="The name of the embedding model to use for embedding the query text."
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )

    tool: ChatCompletionToolParam = Field(
        None, description="The tool input used in the OpenAI chat completion SDK"
    )
    _index: VectorSearchIndex = PrivateAttr()
    _index_details: IndexDetails = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        from databricks.vector_search.client import (
            VectorSearchClient,  # import here so we can mock in tests
        )
        from databricks.vector_search.utils import CredentialStrategy

        splits = self.index_name.split(".")
        if len(splits) != 3:
            raise ValueError(
                f"Index name {self.index_name} is not in the expected format 'catalog.schema.index'."
            )
        client_args = {
            "disable_notice": True,
        }
        if self.workspace_client is not None:
            config = self.workspace_client.config
            if config.auth_type == "model_serving_user_credentials":
                client_args.setdefault(
                    "credential_strategy", CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
                )
            elif config.auth_type == "pat":
                client_args.setdefault("personal_access_token", config.token)
            elif config.auth_type == "oauth-m2m":
                client_args.setdefault("workspace_url", config.host)
                client_args.setdefault("service_principal_client_id", config.client_id)
                client_args.setdefault("service_principal_client_secret", config.client_secret)
        self._index = VectorSearchClient(**client_args).get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)
        self.text_column = validate_and_get_text_column(self.text_column, self._index_details)
        self.columns = validate_and_get_return_columns(
            self.columns or [],
            self.text_column,
            self._index_details,
            self.doc_uri,
            self.primary_key,
        )
        self._retriever_schema = RetrieverSchema(
            text_column=self.text_column,
            doc_uri=self.doc_uri,
            primary_key=self.primary_key,
            other_columns=self.columns,
        )

        if (
            not self._index_details.is_databricks_managed_embeddings()
            and not self.embedding_model_name
        ):
            raise ValueError(
                "The embedding model name is required for non-Databricks-managed "
                "embeddings Vector Search indexes in order to generate embeddings for retrieval queries."
            )

        tool_name = self._get_tool_name()

        # Create tool input model based on dynamic_filter setting
        if self.dynamic_filter:
            tool_input_class = self._create_enhanced_input_model()
        else:
            tool_input_class = self._create_basic_input_model()

        self.tool = pydantic_function_tool(
            tool_input_class,
            name=tool_name,
            description=self.tool_description
            or self._get_default_tool_description(self._index_details),
        )
        # We need to remove strict: True from the tool in order to support arbitrary filters
        if "function" in self.tool and "strict" in self.tool["function"]:
            del self.tool["function"]["strict"]
        # We need to remove additionalProperties from the tool in order to support arbitrary kwargs
        if (
            "function" in self.tool
            and "parameters" in self.tool["function"]
            and "additionalProperties" in self.tool["function"]["parameters"]
        ):
            del self.tool["function"]["parameters"]["additionalProperties"]

        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.errors.platform import ResourceDoesNotExist

            if self.workspace_client is not None:
                self.workspace_client.serving_endpoints.get(self.embedding_model_name)
            else:
                WorkspaceClient().serving_endpoints.get(self.embedding_model_name)
            self.resources = self._get_resources(
                self.index_name, self.embedding_model_name, self._index_details
            )
        except ResourceDoesNotExist:
            self.resources = self._get_resources(self.index_name, None, self._index_details)

        return self

    @vector_search_retriever_tool_trace
    def execute(
        self,
        query: str,
        filters: Optional[List[FilterItem]] = None,
        openai_client: OpenAI = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Execute the VectorSearchIndex tool calls from the ChatCompletions response that correspond to the
        self.tool VectorSearchRetrieverToolInput and attach the retrieved documents into tool call messages.

        Args:
            query: The query text to use for the retrieval.
            openai_client: The OpenAI client object used to generate embeddings for retrieval queries. If not provided,
                           the default OpenAI client in the current environment will be used.

        Returns:
            A list of documents
        """

        if self._index_details.is_databricks_managed_embeddings():
            query_text, query_vector = query, None
        else:  # For non-Databricks-managed embeddings
            from openai import OpenAI

            oai_client = openai_client or OpenAI()
            if not oai_client.api_key:
                raise ValueError(
                    "OpenAI API key is required to generate embeddings for retrieval queries."
                )

            query_text = query if self.query_type and self.query_type.upper() == "HYBRID" else None
            query_vector = (
                oai_client.embeddings.create(input=query, model=self.embedding_model_name)
                .data[0]
                .embedding
            )
            if (
                index_embedding_dimension := self._index_details.embedding_vector_column.get(
                    "embedding_dimension"
                )
            ) and len(query_vector) != index_embedding_dimension:
                raise ValueError(
                    f"Expected embedding dimension {index_embedding_dimension} but got {len(query_vector)}"
                )

        # Since LLM can generate either a dict or FilterItem, convert to dict always
        filters_dict = {dict(item)["key"]: dict(item)["value"] for item in (filters or [])}
        combined_filters = {**filters_dict, **(self.filters or {})}

        signature = inspect.signature(self._index.similarity_search)
        kwargs = {**kwargs, **(self.model_extra or {})}
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

        # Allow kwargs to override the default values upon invocation
        num_results = kwargs.pop("num_results", self.num_results)
        query_type = kwargs.pop("query_type", self.query_type)
        reranker = kwargs.pop("reranker", self.reranker)

        kwargs.update(
            {
                "query_text": query_text,
                "query_vector": query_vector,
                "columns": self.columns,
                "filters": combined_filters,
                "num_results": num_results,
                "query_type": query_type,
                "reranker": reranker,
            }
        )
        search_resp = self._index.similarity_search(**kwargs)
        docs_with_score: List[Tuple[Dict, float]] = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=self._retriever_schema,
            document_class=dict,
            include_score=self.include_score,
        )
        return [doc for doc, _ in docs_with_score]
