import inspect
import pytest
from unittest.mock import MagicMock

# Import the modules that databricks-langchain depends on
from databricks_ai_bridge import (
    IndexDetails, 
    IndexType, 
    RetrieverSchema,
    get_metadata, 
    parse_vector_search_response,
    validate_and_get_text_column,
    validate_and_get_return_columns
)

from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    DELTA_SYNC_INDEX,
    INDEX_DETAILS,
    EXAMPLE_SEARCH_RESPONSE,
)

class TestApiSignatures:
    """Verify that the public API signatures remain compatible with databricks-langchain."""
    
    def test_index_details_signatures(self):
        """Test IndexDetails class signature."""
        # Verify constructor signature
        init_sig = inspect.signature(IndexDetails.__init__)
        assert list(init_sig.parameters.keys())[1] == "index", "Constructor must accept 'index' parameter"
        
        properties_to_check = [
            "name", "schema", "primary_key", "index_spec", 
            "embedding_vector_column", "embedding_source_column"
        ]
        
        mock_index = MagicMock(spec=VectorSearchIndex)
        mock_index.describe.return_value = INDEX_DETAILS[DELTA_SYNC_INDEX]
        mock_index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
        index_details = IndexDetails(mock_index)

        for prop_name in properties_to_check:
            assert hasattr(index_details, prop_name), f"Missing property: {prop_name}"
    
    def test_function_signatures(self):
        """
        Test function signatures for backward compatibility.
        Ensures:
        1. Required parameters haven't been added (all new params must have defaults)
        2. Parameter order remains the same for the base parameters
        3. No parameters have been removed
        """
        # Define expected minimal signatures with required parameter counts
        functions_to_check = {
            "get_metadata": {
                "required_params": ["columns", "result", "retriever_schema", "ignore_cols"],
                "optional_params": []  # Parameters with default values
            },
            "parse_vector_search_response": {
                "required_params": ["search_resp", "retriever_schema"],
                "optional_params": ["ignore_cols", "document_class"]
            },
            "validate_and_get_text_column": {
                "required_params": ["text_column", "index_details"],
                "optional_params": []
            },
            "validate_and_get_return_columns": {
                "required_params": ["columns", "text_column", "index_details"],
                "optional_params": ["doc_uri", "primary_key"]
            }
        }
        
        for func_name, expected in functions_to_check.items():
            func = eval(func_name)  # Get the function object
            sig = inspect.signature(func)
            
            # Get required parameters (those without default values)
            required_params = [
                name for name, param in sig.parameters.items() 
                if param.default is inspect.Parameter.empty
            ]
            
            # Check if we've added any new required parameters
            required_count = len(expected["required_params"])
            assert len(required_params) <= required_count, \
                f"Function {func_name} has {len(required_params)} required parameters, but should have at most {required_count}. " \
                f"New parameters must have default values."
            
            # Check that original required parameters are still in the same order
            for i, param_name in enumerate(expected["required_params"]):
                if i < len(required_params):  # In case the function now has fewer required params
                    assert required_params[i] == param_name, \
                        f"Function {func_name}: parameter {i+1} should be '{param_name}', got '{required_params[i]}'"
            
            # Make sure all original parameters (required and optional) still exist
            all_original_params = expected["required_params"] + expected["optional_params"]
            current_params = list(sig.parameters.keys())
            
            for param in all_original_params:
                assert param in current_params, \
                    f"Function {func_name}: parameter '{param}' has been removed"
