# Add Dynamic Filter Support to VectorSearchRetrieverTool

## Summary

This PR adds opt-in support for LLM-generated filter parameters in `VectorSearchRetrieverTool`, enabling LLMs to dynamically construct filters based on natural language queries. This feature is controlled by a new `dynamic_filter` parameter (default: `False`) that exposes filter parameters in the tool schema when enabled.

**Key Feature**: The filter parameter description includes guidance for LLMs to use a **fallback strategy** - searching WITHOUT filters first to get broad results, then optionally adding filters to narrow down. This approach helps avoid zero results due to incorrect filter values while maintaining filtering flexibility.

## Changes

### Core Changes

**1. New `dynamic_filter` Parameter**
- Added `dynamic_filter: bool` field to `VectorSearchRetrieverToolMixin` (default: `False`)
- When `True`, exposes filter parameters in the tool schema for LLM-generated filters
- When `False` (default), maintains backward-compatible behavior with no filter parameters exposed

**2. Mutual Exclusivity Validation**
- Added `@model_validator` to ensure `dynamic_filter=True` and predefined `filters` cannot be used together
- Prevents ambiguous filter configuration by enforcing one approach or the other
- Clear error message guides users to the correct usage pattern

**3. Enhanced Filter Parameter Descriptions with Fallback Strategy**
- Extracts column metadata from Unity Catalog (`workspace_client.tables.get()`)
- Includes available columns with types in the filter parameter description
- Example: `"Available columns for filtering: product_category (STRING), product_sub_category (STRING)..."`
- **NEW**: Includes guidance to search WITHOUT filters first: *"IMPORTANT: If unsure about filter values, try searching WITHOUT filters first to get broad results, then optionally add filters to narrow down if needed. This ensures you don't miss relevant results due to incorrect filter values."*
- Provides comprehensive operator documentation and examples

### Integration Updates

**OpenAI Integration** (`integrations/openai/src/databricks_openai/vector_search_retriever_tool.py`)
- Conditionally creates `EnhancedVectorSearchRetrieverToolInput` (with optional filters) or `BasicVectorSearchRetrieverToolInput` (without filters) based on `dynamic_filter` setting
- Filter parameter is marked as `Optional[List[FilterItem]]` with `default=None`
- Inlines column metadata extraction during tool schema generation
- Fixed bug: Originally tried to use `index.describe()["columns"]` which doesn't exist; now uses Unity Catalog tables API

**LangChain Integration** (`integrations/langchain/src/databricks_langchain/vector_search_retriever_tool.py`)
- Similar conditional `args_schema` creation based on `dynamic_filter` setting
- Filter parameter is marked as optional (`Optional[List[FilterItem]]` with `default=None`)
- Maintains compatibility with LangChain's tool invocation patterns


### Tests

**New Test Coverage:**
- `test_cannot_use_both_dynamic_filter_and_predefined_filters` - Validates mutual exclusivity
- `test_predefined_filters_work_without_dynamic_filter` - Ensures predefined filters work without dynamic mode
- `test_enhanced_filter_description_with_column_metadata` - Verifies column info is included
- `test_enhanced_filter_description_without_column_metadata` - Handles missing column info gracefully
- `test_filter_item_serialization` - Tests FilterItem schema

**Test Results:**
- ✅ OpenAI Integration: 48 tests passing
- ✅ LangChain Integration: 37 tests passing

## Usage

### Basic Usage (OpenAI)

```python
from databricks_openai import VectorSearchRetrieverTool

# Enable dynamic filters
tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    dynamic_filter=True  # Exposes optional filter parameters to LLM
)

# LLM receives guidance to try without filters first
# Then can optionally generate filters like:
# {"query": "wireless headphones", "filters": [{"key": "price <", "value": 100}]}
result = tool.execute(
    query="wireless headphones",
    filters=[{"key": "price <", "value": 100}]  # Optional!
)
```

### Recommended Pattern (Fallback Strategy)

```python
# Step 1: Search WITHOUT filters first (broad search)
broad_results = tool.execute(
    query="wireless headphones",
    openai_client=client
)

# Step 2: Examine results to understand available filter values
categories = extract_categories_from_results(broad_results)

# Step 3: If needed, narrow with accurate filter values
filtered_results = tool.execute(
    query="wireless headphones",
    filters=[{"key": "category", "value": categories[0]}],
    openai_client=client
)
```

### Basic Usage (LangChain)

```python
from databricks_langchain import VectorSearchRetrieverTool

tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    dynamic_filter=True
)

# Use with LangChain agents - filter parameter is optional
result = tool.invoke({
    "query": "wireless headphones",
    "filters": [{"key": "price <", "value": 100}]  # Optional!
})
```

## Tradeoffs

### ✅ Benefits

1. **Increased Flexibility**: LLMs can dynamically construct filters based on user queries without requiring predefined filter logic
2. **Natural Language Queries**: Users can express filtering requirements in natural language (e.g., "Find products under $100") and the LLM translates them to filters
3. **Rich Filter Operations**: Supports complex operators (NOT, <, >=, LIKE, OR) that LLMs can apply intelligently
4. **Column Metadata**: Provides column names and types to guide LLM filter generation
5. **Backward Compatible**: Default `dynamic_filter=False` maintains existing behavior
6. **Fallback Strategy Guidance**: Built-in guidance helps LLMs avoid zero-result scenarios
7. **Optional Filters**: Filters are truly optional, enabling LLMs to choose when to apply them

### ⚠️ Tradeoffs & Limitations

1. **LLM Hallucination Risk**:
   - LLMs may generate filters with **non-existent column values**
   - Example: Filtering for `product_category="Data Engineering"` when actual values are `["Appliances", "Books", "Sports Equipment"]`
   - Result: Zero results returned, potentially confusing user experience
   - **Mitigation**: Fallback strategy guidance encourages LLMs to search without filters first

2. **No Value Validation**:
   - Column metadata includes names and types but **not possible values**
   - LLMs must "guess" valid values based on query context
   - No mechanism to constrain LLM to only valid enum values

3. **Unpredictable Behavior**:
   - Filter generation depends on LLM reasoning capabilities
   - May produce overly restrictive filters (zero results) or insufficiently restrictive filters (too many results)
   - Different LLMs may generate different filters for the same query

4. **Debugging Complexity**:
   - When searches return no results, unclear if it's due to poor query match or invalid filter values
   - Requires inspecting generated filters to diagnose issues

### 🎯 Recommendations

**When to use `dynamic_filter=True`:**
- Column values are discoverable from context (in retrieved documents)
- Filter requirements are simple and commonly understood (e.g., date ranges, numeric comparisons)
- Acceptable to have some queries return zero results due to filter mismatches
- Users can iteratively refine queries based on results
- LLM can follow the fallback strategy (search without filters first)

**When to use predefined `filters`:**
- Column values are constrained enums (product categories, status values, etc.)
- Filter logic is deterministic and known in advance
- Zero tolerance for LLM hallucination in filter values
- Consistent, predictable behavior is required

**Best Practices:**

1. **Leverage the Fallback Strategy**:
   - The tool description guides LLMs to search WITHOUT filters first
   - This provides broad results and reveals actual column values
   - Then LLMs can apply filters more accurately based on observed data
   - Example shown in `demo_filter_example.py` Example 3

2. **Hybrid Approach**:
   - Use predefined filters for enum columns
   - Allow dynamic filters for numeric/date ranges
   - Validate/suggest filter values before invoking the tool

3. **Result Inspection**:
   - Have LLMs examine initial results to understand available filter values
   - Use discovered values for more accurate filtering

## Implementation Details

### Fallback Strategy Mechanism

The fallback strategy is implemented through **tool description guidance** rather than execution-time logic:

1. **Filter Parameter Description** includes: *"IMPORTANT: If unsure about filter values, try searching WITHOUT filters first..."*
2. **Filters are Optional**: Marked as `Optional[List[FilterItem]]` with `default=None`
3. **LLM Follows Guidance**: When the LLM sees this description, it learns to:
   - First invoke the tool without filters to get broad results
   - Examine the results to understand available filter values
   - Optionally invoke again with accurate filters to narrow results

This approach:
- ✅ Leverages LLM's ability to follow instructions in tool descriptions
- ✅ Doesn't require shipping complex merge/fallback logic
- ✅ Simple to implement (text-based guidance)
- ✅ Backward compatible
- ✅ Educates LLMs on best practices without changing execution

### Inspiration

This fallback strategy is inspired by work from Databricks Knowledge Assistants team (Cindy Wang et al.), who found that searching with AND without filters and merging results significantly improves filter accuracy. Our simplified approach achieves similar benefits by guiding the LLM through tool descriptions.

## Future Improvements

Potential enhancements to address remaining limitations:

1. **Column Value Discovery**: Query index for distinct values in categorical columns and include in tool description
2. **Filter Validation**: Add optional runtime validation of filter values against known valid values
3. **Automatic Merge Logic**: Implement automatic search with/without filters and merge results (like KA internal implementation)
4. **Filter Feedback Loop**: Return filter statistics (e.g., "0 results with filter X") to help LLM adjust
5. **Hybrid Mode**: Allow both predefined filters (for enums) and dynamic filters (for ranges) simultaneously

---

## Validation

### Testing in Practice

When tested with LangChain `AgentExecutor`, we observe that **the LLM intelligently decides when to generate filters** based on the user prompt and the guidance in the tool description. The fallback strategy works as intended - LLMs learn from the IMPORTANT guidance to search without filters first when unsure about filter values, then retry with filters if appropriate.

### Demo Output: Fallback Strategy in Action

#### Filter Parameter Description (What the LLM Sees)

```
"Optional filters to refine vector search results as an array of key-value pairs.
IMPORTANT: If unsure about filter values, try searching WITHOUT filters first to get
broad results, then optionally add filters to narrow down if needed. This ensures you
don't miss relevant results due to incorrect filter values.

Available columns for filtering:
  product_category (STRING), product_sub_category (STRING), product_name (STRING),
  product_doc (STRING), product_id (STRING), indexed_doc (STRING)

Supports the following operators:
- Inclusion: [{"key": "column", "value": value}] or [{"key": "column", "value": [value1, value2]}]
- Exclusion: [{"key": "column NOT", "value": value}]
- Comparisons: [{"key": "column <", "value": value}], [{"key": "column >=", "value": value}]
- Pattern match: [{"key": "column LIKE", "value": "word"}]
- OR logic: [{"key": "column1 OR column2", "value": [value1, value2]}]
..."
```

#### Observed LLM Behavior

When tested with LangChain `AgentExecutor`, the LLM demonstrates intelligent filter usage:

**Example Query**: "Find documentation about Data Engineering products"

**LLM Actions**:
1. **First attempt**: Tries with a filter based on the query:
   ```python
   {'query': 'Data Engineering products',
    'filters': [{'key': 'product_category', 'value': 'Data Engineering'}]}
   ```
   Result: Empty (0 results) - the category doesn't exist in the index

2. **Second attempt**: Following the IMPORTANT guidance, automatically retries WITHOUT filters:
   ```python
   {'query': 'Data Engineering'}
   ```
   Result: Success! Returns relevant results from actual categories (Software, Computers, etc.)

**Key Observation**: The LLM learns from the guidance to:
- Try with filters when the user query suggests specific filter criteria
- Automatically fall back to searching without filters when the first attempt fails
- Get broader, more relevant results instead of returning zero results

This demonstrates that **guidance-based fallback works in practice** - LLMs follow the instructions in the tool description without requiring execution-time merge logic!
