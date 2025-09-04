import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import dspy
from databricks.sdk import WorkspaceClient
from dspy.primitives.prediction import Prediction

logger = logging.getLogger(__name__)


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "type": self.type,
        }


class DatabricksRM(dspy.Retrieve):
    """
    A retriever module that uses a Databricks Mosaic AI Vector Search Index to return the top-k
    embeddings for a given query.

    Examples:
        Below is a code snippet that shows how to set up a Databricks Vector Search Index
        and configure a DatabricksRM DSPy retriever module to query the index.

        (example adapted from "Databricks: How to create and query a Vector Search Index:
        https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

        ```python
        from databricks.vector_search.client import VectorSearchClient
        from databricks.sdk import WorkspaceClient

        # Create a Databricks workspace client
        w = WorkspaceClient()

        # Create a Databricks Vector Search Endpoint
        client = VectorSearchClient()
        client.create_endpoint(name="your_vector_search_endpoint_name", endpoint_type="STANDARD")

        # Create a Databricks Direct Access Vector Search Index
        index = client.create_direct_access_index(
            endpoint_name="your_vector_search_endpoint_name",
            index_name="your_index_name",
            primary_key="id",
            embedding_dimension=1024,
            embedding_vector_column="text_vector",
            schema={
                "id": "int",
                "field2": "str",
                "field3": "float",
                "text_vector": "array<float>",
            },
        )

        # Create a DatabricksRM retriever module to query the Databricks Direct Access Vector
        # Search Index
        retriever = DatabricksRM(
            databricks_index_name="your_index_name",
            docs_id_column_name="id",
            text_column_name="field2",
            k=3,
            workspace_client=w,
        )
        ```

        Below is a code snippet that shows how to query the Databricks Direct Access Vector
        Search Index using the DatabricksRM retriever module:

        ```python
        retrieved_results = DatabricksRM(query="Example query text"))
        ```
    """

    def __init__(
        self,
        databricks_index_name: str,
        databricks_endpoint: str | None = None,
        databricks_token: str | None = None,
        databricks_client_id: str | None = None,
        databricks_client_secret: str | None = None,
        columns: list[str] | None = None,
        filters_json: str | None = None,
        k: int = 3,
        docs_id_column_name: str = "id",
        docs_uri_column_name: str | None = None,
        text_column_name: str = "text",
        use_with_databricks_agent_framework: bool = False,
        workspace_client: WorkspaceClient | None = None,
    ):
        """
        Args:
            databricks_index_name (str): The name of the Databricks Vector Search Index to query.
            databricks_endpoint (Optional[str]): The URL of the Databricks Workspace containing
                the Vector Search Index. Defaults to the value of the ``DATABRICKS_HOST``
                environment variable. If unspecified, the Databricks SDK is used to identify the
                endpoint based on the current environment.
            databricks_token (Optional[str]): The Databricks Workspace authentication token to use
                when querying the Vector Search Index. Defaults to the value of the
                ``DATABRICKS_TOKEN`` environment variable. If unspecified, the Databricks SDK is
                used to identify the token based on the current environment.
            databricks_client_id (str): Databricks service principal id. If not specified,
                the token is resolved from the current environment (DATABRICKS_CLIENT_ID).
            databricks_client_secret (str): Databricks service principal secret. If not specified,
                the endpoint is resolved from the current environment (DATABRICKS_CLIENT_SECRET).
            columns (Optional[list[str]]): Extra column names to include in response,
                in addition to the document id and text columns specified by
                ``docs_id_column_name`` and ``text_column_name``.
            filters_json (Optional[str]): A JSON string specifying additional query filters.
                Example filters: ``{"id <": 5}`` selects records that have an ``id`` column value
                less than 5, and ``{"id >=": 5, "id <": 10}`` selects records that have an ``id``
                column value greater than or equal to 5 and less than 10.
            k (int): The number of documents to retrieve.
            docs_id_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document IDs.
            docs_uri_column_name (Optional[str]): The name of the column in the Databricks Vector
                Search Index containing document URI.
            text_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document text to retrieve.
            use_with_databricks_agent_framework (bool): Whether to use the `DatabricksRM` in a way
                that is compatible with the Databricks Mosaic Agent Framework.
            workspace_client (Optional[WorkspaceClient]): The workspace client to use. If not
                provided, a new one will be created with default credentials from the environment.
        """
        super().__init__(k=k)
        self.databricks_token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
        self.databricks_endpoint = databricks_endpoint or os.environ.get("DATABRICKS_HOST")
        self.databricks_client_id = databricks_client_id or os.environ.get("DATABRICKS_CLIENT_ID")
        self.databricks_client_secret = databricks_client_secret or os.environ.get(
            "DATABRICKS_CLIENT_SECRET"
        )
        self.databricks_index_name = databricks_index_name
        self.columns = list({docs_id_column_name, text_column_name, *(columns or [])})
        self.filters_json = filters_json
        self.k = k
        self.docs_id_column_name = docs_id_column_name
        self.docs_uri_column_name = docs_uri_column_name
        self.text_column_name = text_column_name
        self.use_with_databricks_agent_framework = use_with_databricks_agent_framework
        if self.use_with_databricks_agent_framework:
            try:
                import mlflow

                mlflow.models.set_retriever_schema(
                    primary_key="doc_id",
                    text_column="page_content",
                    doc_uri="doc_uri",
                )
            except ImportError:
                raise ImportError(
                    "To use the `DatabricksRM` retriever module with the Databricks "
                    "Mosaic Agent Framework, you must install the mlflow Python "
                    "library. Please install mlflow via `pip install mlflow`."
                ) from None

        # Use provided workspace client or create one based on credentials
        if workspace_client:
            self.workspace_client = workspace_client
        elif databricks_client_secret and databricks_client_id:
            # Use client ID and secret for authentication if they are provided
            self.workspace_client = WorkspaceClient(
                client_id=databricks_client_id,
                client_secret=databricks_client_secret,
            )
            logger.info(
                "Creating Databricks workspace client using service principal authentication."
            )
        elif databricks_token and databricks_endpoint:
            # token-based authentication
            self.workspace_client = WorkspaceClient(
                host=databricks_endpoint,
                token=databricks_token,
            )
            logger.info("Creating Databricks workspace client using token authentication.")
        else:
            # fallback to default authentication, i.e., using `~/.databrickscfg` file.
            self.workspace_client = WorkspaceClient()
            logger.info(
                "Creating Databricks workspace client using credentials from `~/.databrickscfg` file."
            )

    def _extract_doc_ids(self, item: dict[str, Any]) -> str:
        """Extracts the document id from a search result

        Args:
            item: dict[str, Any]: a record from the search results.
        Returns:
            str: document id.
        """
        if self.docs_id_column_name == "metadata":
            docs_dict = json.loads(item["metadata"])
            return docs_dict["document_id"]
        return item[self.docs_id_column_name]

    def _get_extra_columns(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extracts search result column values, excluding the "text" and not "id" columns

        Args:
            item: dict[str, Any]: a record from the search results.
        Returns:
            dict[str, Any]: Search result column values, excluding the "text", "id" and "uri" columns.
        """
        extra_columns = {
            k: v
            for k, v in item.items()
            if k not in [self.docs_id_column_name, self.text_column_name, self.docs_uri_column_name]
        }
        if self.docs_id_column_name == "metadata":
            extra_columns = {
                **extra_columns,
                **{
                    "metadata": {
                        k: v for k, v in json.loads(item["metadata"]).items() if k != "document_id"
                    }
                },
            }
        return extra_columns

    def forward(
        self,
        query: str | list[float],
        query_type: str = "ANN",
        filters_json: str | None = None,
    ) -> dspy.Prediction | list[dict[str, Any]]:
        """
        Retrieve documents from a Databricks Mosaic AI Vector Search Index that are relevant to the
        specified query.

        Args:
            query (Union[str, list[float]]): The query text or numeric query vector for which to
                retrieve relevant documents.
            query_type (str): The type of search query to perform against the Databricks Vector
                Search Index. Must be either 'ANN' (approximate nearest neighbor) or 'HYBRID'
                (hybrid search).
            filters_json (Optional[str]): A JSON string specifying additional query filters.
                Example filters: ``{"id <": 5}`` selects records that have an ``id`` column value
                less than 5, and ``{"id >=": 5, "id <": 10}`` selects records that have an ``id``
                column value greater than or equal to 5 and less than 10. If specified, this
                parameter overrides the `filters_json` parameter passed to the constructor.

        Returns:
            A list of dictionaries when ``use_with_databricks_agent_framework`` is ``True``,
            or a ``dspy.Prediction`` object when ``use_with_databricks_agent_framework`` is
            ``False``.
        """
        if query_type not in ["ANN", "HYBRID"]:
            raise ValueError(f"Invalid query_type: {query_type}. Must be one of 'ANN' or 'HYBRID'.")

        if isinstance(query, str):
            query_text = query
            query_vector = None
        elif isinstance(query, list):
            query_text = None
            query_vector = query
        else:
            raise ValueError("Query must be a string or a list of floats.")

        results = self._query_vector_search_index(
            index_name=self.databricks_index_name,
            k=self.k,
            columns=self.columns,
            query_type=query_type,
            query_text=query_text,
            query_vector=query_vector,
            filters_json=filters_json or self.filters_json,
        )

        # Checking if defined columns are present in the index columns
        col_names = [column["name"] for column in results["manifest"]["columns"]]

        if self.docs_id_column_name not in col_names:
            raise ValueError(
                f"docs_id_column_name: '{self.docs_id_column_name}' is not in the index "
                f"columns: \n {col_names}"
            )

        if self.text_column_name not in col_names:
            raise ValueError(
                f"text_column_name: '{self.text_column_name}' is not in the index "
                "columns: \n {col_names}"
            )

        # Extracting the results
        items = []
        if "data_array" in results["result"]:
            for _, data_row in enumerate(results["result"]["data_array"]):
                item = {}
                for col_name, val in zip(col_names, data_row, strict=False):
                    item[col_name] = val
                items.append(item)

        # Sorting results by score in descending order
        sorted_docs = sorted(items, key=lambda x: x["score"], reverse=True)[: self.k]

        if self.use_with_databricks_agent_framework:
            return [
                Document(
                    page_content=doc[self.text_column_name],
                    metadata={
                        "doc_id": self._extract_doc_ids(doc),
                        "doc_uri": doc[self.docs_uri_column_name]
                        if self.docs_uri_column_name
                        else None,
                    }
                    | self._get_extra_columns(doc),
                    type="Document",
                ).to_dict()
                for doc in sorted_docs
            ]
        else:
            # Returning the prediction
            return Prediction(
                docs=[doc[self.text_column_name] for doc in sorted_docs],
                doc_ids=[self._extract_doc_ids(doc) for doc in sorted_docs],
                doc_uris=[doc[self.docs_uri_column_name] for doc in sorted_docs]
                if self.docs_uri_column_name
                else None,
                extra_columns=[self._get_extra_columns(item) for item in sorted_docs],
            )

    def _query_vector_search_index(
        self,
        index_name: str,
        k: int,
        columns: list[str],
        query_type: str,
        query_text: str | None,
        query_vector: list[float] | None,
        filters_json: str | None,
    ) -> dict[str, Any]:
        """
        Query a Databricks Vector Search Index via the Databricks SDK.

        Args:
            index_name (str): Name of the Databricks vector search index to query
            k (int): Number of relevant documents to retrieve.
            columns (list[str]): Column names to include in response.
            query_text (Optional[str]): Text query for which to find relevant documents. Exactly
                one of query_text or query_vector must be specified.
            query_vector (Optional[list[float]]): Numeric query vector for which to find relevant
                documents. Exactly one of query_text or query_vector must be specified.
            filters_json (Optional[str]): JSON string representing additional query filters.

        Returns:
            dict[str, Any]: Parsed JSON response from the Databricks Vector Search Index query.
        """
        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")

        return self.workspace_client.vector_search_indexes.query_index(
            index_name=index_name,
            query_type=query_type,
            query_text=query_text,
            query_vector=query_vector,
            columns=columns,
            filters_json=filters_json,
            num_results=k,
        ).as_dict()
