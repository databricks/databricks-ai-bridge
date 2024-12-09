from databricks_langchain.vector_search import DatabricksVectorSearch
class VectorSearchRetrieverTool():
    def __init__(self, *args, **kwargs):
        vector_store = DatabricksVectorSearch(
            endpoint=endpoint_name,
            index_name=index_name,
        )
        return vector_store.as_retriever().as_tool()
