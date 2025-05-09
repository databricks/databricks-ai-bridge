import pytest
import json
from unittest.mock import MagicMock, Mock, patch

from databricks.vector_search.client import VectorSearchIndex
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DIRECT_ACCESS_INDEX,
    ENDPOINT_NAME,
    INDEX_DETAILS,
    INPUT_TEXTS,
    EXAMPLE_SEARCH_RESPONSE,
    mock_vs_client,  # noqa: F401
)

# Import the modules you want to test
from databricks_ai_bridge import (
    IndexDetails, 
    IndexType, 
    RetrieverSchema,
    get_metadata, 
    parse_vector_search_response,
    validate_and_get_text_column,
    validate_and_get_return_columns
)

class TestIndexDetailsContract:
    """Contract tests for IndexDetails class."""
    
    def setup_method(self):
        """Create a mock index with consistent test data"""
        mock_index = MagicMock(spec=VectorSearchIndex)
        mock_index.describe.return_value = INDEX_DETAILS[DELTA_SYNC_INDEX]
        mock_index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
        self.index_details = IndexDetails(mock_index)

    def test_basic_properties(self):
        """Test that basic properties are accessible and return expected values."""
        assert self.index_details.name == DELTA_SYNC_INDEX
        assert self.index_details.primary_key == "id"
        assert self.index_details.is_direct_access_index() is False
        assert self.index_details.is_delta_sync_index() is True
    
    def test_embedding_columns(self):
        """Test that embedding column information is accessible."""
        vector_column = self.index_details.embedding_vector_column
        source_column = self.index_details.embedding_source_column
        
        assert vector_column.get("name") == "embedding"
        assert vector_column.get("dimension") == 768
        assert source_column.get("name") == "text"
    
    def test_databricks_managed_embeddings_detection(self):
        """Test detection of databricks managed embeddings."""
        assert self.index_details.is_databricks_managed_embeddings() is False
        assert self.delta_index_details.is_databricks_managed_embeddings() is True


class TestVectorSearchContract:
    """Contract tests for vector search functionality."""
    
    def setup_method(self):
        # Test data for vector search
        self.columns = ["id", "text", "doc_uri", "score"]
        self.search_response = EXAMPLE_SEARCH_RESPONSE
        self.retriever_schema = RetrieverSchema(
            text_column="text",
            doc_uri="doc_uri",
            primary_key="id"
        )
        
        # Setup mock index details
        mock_index = MagicMock(spec=VectorSearchIndex)
        mock_index.describe.return_value = INDEX_DETAILS[DELTA_SYNC_INDEX]
        mock_index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
        self.index_details = IndexDetails(mock_index)
    
    def test_get_metadata(self):
        """Test that metadata is correctly extracted from search results."""
        columns = ["id", "text", "doc_uri", "score"]
        result = ["doc1", "This is a test document", "uri1", 0.95]
        ignore_cols = ["text"]
        
        metadata = get_metadata(columns, result, self.retriever_schema, ignore_cols)
        
        assert "chunk_id" in metadata
        assert metadata["chunk_id"] == "doc1"
        assert "doc_uri" in metadata
        assert metadata["doc_uri"] == "uri1"
        assert "text" not in metadata  # Should be ignored
    
    def test_parse_vector_search_response(self):
        """Test that search responses are correctly parsed."""
        docs_with_score = parse_vector_search_response(
            self.search_response, 
            self.retriever_schema
        )
        
        assert len(docs_with_score) == 2
        doc1, score1 = docs_with_score[0]
        
        assert doc1["page_content"] == "This is a test document"
        assert doc1["metadata"]["chunk_id"] == "doc1"
        assert doc1["metadata"]["doc_uri"] == "uri1"
        assert score1 == 0.95
    
    def test_validate_text_column(self):
        """Test text column validation."""
        # For regular direct access index
        text_col = validate_and_get_text_column("text", self.index_details)
        assert text_col == "text"
        
        # Should raise error when text_column is None for non-managed embeddings
        with pytest.raises(ValueError):
            validate_and_get_text_column(None, self.index_details)
        
        # Create a mock for managed embeddings
        managed_mock = Mock()
        managed_mock.describe.return_value = {
            "name": "managed_index",
            "primary_key": "id",
            "index_type": IndexType.DELTA_SYNC.value,
            "delta_sync_index_spec": {
                "embedding_source_columns": [{"name": "text"}],
                "embedding_vector_columns": [{"name": "embedding"}]
            }
        }
        managed_index = IndexDetails(managed_mock)
        
        # For managed embeddings, should return source column
        text_col = validate_and_get_text_column(None, managed_index)
        assert text_col == "text"
        
        # Should raise error if text_column is provided but doesn't match
        with pytest.raises(ValueError):
            validate_and_get_text_column("wrong_column", managed_index)
    
    def test_validate_return_columns(self):
        """Test column validation for return columns."""
        columns = ["text"]
        validated = validate_and_get_return_columns(
            columns, 
            "text", 
            self.index_details, 
            doc_uri="doc_uri"
        )
        
        # Should add required columns
        assert "id" in validated  # primary key
        assert "text" in validated  # text column
        assert "doc_uri" in validated  # doc_uri
        
        # Should raise error for non-existent columns
        with pytest.raises(ValueError):
            validate_and_get_return_columns(
                ["non_existent_column"], 
                "text", 
                self.index_details
            )