"""
This file contains the integration test for ChatDatabricks class.

We run the integration tests nightly by the trusted CI/CD system defined in
a private repository, in order to securely run the tests. With this design,
integration test is not intended to be run manually by OSS contributors.
If you want to update the ChatDatabricks implementation and you think that you
need to update the corresponding integration test, please contact to the
maintainers of the repository to verify the changes.
"""

import os
from typing import Annotated

import pytest
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from databricks_langchain.chat_models import ChatDatabricks

_FOUNDATION_MODELS = [
    "databricks-claude-3-7-sonnet",
    "databricks-meta-llama-3-3-70b-instruct",
]

# Endpoint constants for easier maintenance
RESPONSES_AGENT_ENDPOINT_WITH_ANNOTATIONS = "agents_ml-bbqiu-annotationsv2"
RESPONSES_AGENT_ENDPOINT_WITH_TOOL_CALLING = "agents_ml-bbqiu-resp-fmapi"
CHAT_AGENT_ENDPOINT_WITH_TOOL_CALLING = "agents_smurching-default-test_external_monitor_cuj"
DATABRICKS_CLI_PROFILE = "dogfood"

_RESPONSES_API_ENDPOINTS = [
    RESPONSES_AGENT_ENDPOINT_WITH_ANNOTATIONS,
    RESPONSES_AGENT_ENDPOINT_WITH_TOOL_CALLING,
]


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_invoke(model):
    chat = ChatDatabricks(model=model, temperature=0, max_tokens=10, stop=["Java"])

    response = chat.invoke("How to learn Java? Start the response by 'To learn Java,'")
    assert isinstance(response, AIMessage)
    assert response.content == "To learn "
    assert 20 <= response.response_metadata["prompt_tokens"] <= 30
    assert 1 <= response.response_metadata["completion_tokens"] <= 10
    expected_total = (
        response.response_metadata["prompt_tokens"]
        + response.response_metadata["completion_tokens"]
    )
    assert response.response_metadata["total_tokens"] == expected_total

    response = chat.invoke("How to learn Python? Start the response by 'To learn Python,'")
    assert response.content.startswith("To learn Python,")
    assert len(response.content.split(" ")) <= 15  # Give some margin for tokenization difference

    # Call with a system message
    response = chat.invoke(
        [
            ("system", "You are helpful programming tutor."),
            ("user", "How to learn Python? Start the response by 'To learn Python,'"),
        ]
    )
    assert response.content.startswith("To learn Python,")

    # Call with message history
    response = chat.invoke(
        [
            SystemMessage(content="You are helpful sports coach."),
            HumanMessage(content="How to swim better?"),
            AIMessage(content="You need more and more practice.", id="12345"),
            HumanMessage(content="No, I need more tips."),
        ]
    )
    assert response.content is not None


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_invoke_multiple_completions(model):
    chat = ChatDatabricks(
        model=model,
        temperature=0.5,
        n=3,
        max_tokens=10,
    )
    response = chat.invoke("How to learn Python?")
    assert isinstance(response, AIMessage)


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_stream(model):
    class FakeCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.chunk_counts = 0

        def on_llm_new_token(self, *args, **kwargs):
            self.chunk_counts += 1

    callback = FakeCallbackHandler()

    chat = ChatDatabricks(
        model=model,
        temperature=0,
        stop=["Python"],
        max_tokens=100,
    )

    chunks = list(chat.stream("How to learn Python?", config={"callbacks": [callback]}))
    assert len(chunks) > 0
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert all("Python" not in chunk.content for chunk in chunks)
    assert callback.chunk_counts == len(chunks)

    last_chunk = chunks[-1]
    assert last_chunk.response_metadata["finish_reason"] == "stop"


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_stream_with_usage(model):
    class FakeCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.chunk_counts = 0

        def on_llm_new_token(self, *args, **kwargs):
            self.chunk_counts += 1

    callback = FakeCallbackHandler()

    chat = ChatDatabricks(
        model=model,
        temperature=0,
        stop=["Python"],
        max_tokens=100,
        stream_usage=True,
    )

    chunks = list(chat.stream("How to learn Python?", config={"callbacks": [callback]}))
    assert len(chunks) > 0
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert all("Python" not in chunk.content for chunk in chunks)
    assert callback.chunk_counts == len(chunks)

    last_chunk = chunks[-1]
    assert last_chunk.response_metadata["finish_reason"] == "stop"
    assert last_chunk.usage_metadata is not None
    assert last_chunk.usage_metadata["input_tokens"] > 0
    assert last_chunk.usage_metadata["output_tokens"] > 0
    assert last_chunk.usage_metadata["total_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
async def test_chat_databricks_ainvoke(model):
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=10,
    )

    response = await chat.ainvoke("How to learn Python? Start the response by 'To learn Python,'")
    assert isinstance(response, AIMessage)
    assert response.content.startswith("To learn Python,")


@pytest.mark.asyncio
@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
async def test_chat_databricks_astream(model):
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=10,
    )
    chunk_count = 0
    async for chunk in chat.astream("How to learn Python?"):
        assert isinstance(chunk, AIMessageChunk)
        chunk_count += 1
    assert chunk_count > 0


@pytest.mark.asyncio
@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
async def test_chat_databricks_abatch(model):
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=10,
    )

    responses = await chat.abatch(
        [
            "How to learn Python?",
            "How to learn Java?",
            "How to learn C++?",
        ]
    )
    assert len(responses) == 3
    assert all(isinstance(response, AIMessage) for response in responses)


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
@pytest.mark.parametrize("tool_choice", [None, "auto", "required", "any", "none"])
def test_chat_databricks_tool_calls(model, tool_choice):
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=100,
    )

    class GetWeather(BaseModel):
        """Get the current weather in a given location"""

        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

    llm_with_tools = chat.bind_tools([GetWeather], tool_choice=tool_choice)
    question = "Which is the current weather in Los Angeles, CA?"

    response = llm_with_tools.invoke(question)
    if tool_choice == "none":
        assert response.tool_calls == []
        return

    # Models should make at least one tool call when tool_choice is not "none"
    assert (
        len(response.tool_calls) >= 1
    ), f"Expected at least 1 tool call, got {len(response.tool_calls)}"

    # The first tool call should be for GetWeather
    first_call = response.tool_calls[0]
    assert first_call["name"] == "GetWeather", f"Expected GetWeather tool, got {first_call['name']}"
    assert "location" in first_call["args"], f"Expected location in args, got {first_call['args']}"
    assert first_call["type"] == "tool_call"
    assert first_call["id"] is not None

    tool_msg = ToolMessage(
        "Sunny, 72°F",
        tool_call_id=response.additional_kwargs["tool_calls"][0]["id"],
    )
    response = llm_with_tools.invoke(
        [
            HumanMessage(question),
            response,
            tool_msg,
            HumanMessage("What about New York, NY?"),
        ]
    )
    # Should call GetWeather tool for the followup question
    assert (
        len(response.tool_calls) >= 1
    ), f"Expected at least 1 tool call, got {len(response.tool_calls)}"
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "GetWeather", f"Expected GetWeather tool, got {tool_call['name']}"
    assert "location" in tool_call["args"], f"Expected location in args, got {tool_call['args']}"
    assert tool_call["type"] == "tool_call"
    assert tool_call["id"] is not None


# Pydantic-based schema
class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str = Field(description="The answer to the user question.")
    justification: str = Field(description="The justification for the answer.")


# Raw JSON schema
JSON_SCHEMA = {
    "title": "AnswerWithJustification",
    "description": "An answer to the user question along with justification.",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user question.",
        },
        "justification": {
            "type": "string",
            "description": "The justification for the answer.",
        },
    },
    "required": ["answer", "justification"],
}


@pytest.mark.parametrize("schema", [AnswerWithJustification, JSON_SCHEMA, None])
@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
@pytest.mark.parametrize("method", ["function_calling", "json_mode"])
def test_chat_databricks_with_structured_output(model, schema, method):
    llm = ChatDatabricks(model=model)

    if schema is None and method == "function_calling":
        pytest.skip("Cannot use function_calling without schema")

    structured_llm = llm.with_structured_output(schema, method=method)

    if method == "function_calling":
        prompt = "What day comes two days after Monday?"
    else:
        prompt = (
            "What day comes two days after Monday? Return in JSON format with key "
            "'answer' for the answer and 'justification' for the justification."
        )

    response = structured_llm.invoke(prompt)

    if schema == AnswerWithJustification:
        assert response.answer == "Wednesday"
        assert response.justification is not None
    else:
        assert response["answer"] == "Wednesday"
        assert response["justification"] is not None

    # Invoke with raw output
    structured_llm = llm.with_structured_output(schema, method=method, include_raw=True)
    response_with_raw = structured_llm.invoke(prompt)
    assert isinstance(response_with_raw["raw"], AIMessage)


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_runnable_sequence(model):
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=100,
    )

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    chain = prompt | chat | StrOutputParser()

    response = chain.invoke({"topic": "chicken"})
    assert "chicken" in response


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_agent_executor(model):
    model = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=100,
    )
    tools = [add, multiply]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": "What is (10 + 5) * 3?"})
    assert "45" in response["output"]


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_langgraph(model):
    model = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=100,
    )
    tools = [add, multiply]

    app = create_react_agent(model, tools)
    response = app.invoke({"messages": [("human", "What is (10 + 5) * 3?")]})
    assert "45" in response["messages"][-1].content


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_langgraph_with_memory(model):
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [add, multiply]
    llm = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=100,
    )
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=MemorySaver())

    response = graph.invoke(
        {"messages": [("user", "What is (10 + 5) * 3?")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert "45" in response["messages"][-1].content

    response = graph.invoke(
        {"messages": [("user", "Subtract 5 from it")]},
        config={"configurable": {"thread_id": "1"}},
    )

    # Interestingly, the agent sometimes mistakes the subtraction for addition:(
    # In such case, the agent asks for a retry so we need one more step.
    if "Let me try again." in response["messages"][-1].content:
        response = graph.invoke(
            {"messages": [("user", "Ok, try again")]},
            config={"configurable": {"thread_id": "1"}},
        )

    assert "40" in response["messages"][-1].content


@pytest.mark.st_endpoints
@pytest.mark.parametrize("endpoint", _RESPONSES_API_ENDPOINTS)
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_chat_databricks_responses_api_invoke(endpoint):
    """Test ChatDatabricks with responses API."""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=endpoint,
        workspace_client=workspace_client,
        use_responses_api=True,
        temperature=0,
        max_tokens=500,
    )

    response = chat.invoke("What is the 100th fibonacci number?")
    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.st_endpoints
@pytest.mark.parametrize("endpoint", _RESPONSES_API_ENDPOINTS)
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_chat_databricks_responses_api_stream(endpoint):
    """Test ChatDatabricks streaming with responses API."""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=endpoint,
        workspace_client=workspace_client,
        use_responses_api=True,
        temperature=0,
        max_tokens=500,
    )

    chunks = list(chat.stream("What is the 100th fibonacci number?"))
    assert len(chunks) > 0

    # Responses API can return both AIMessageChunk and ToolMessageChunk
    from langchain_core.messages import BaseMessageChunk

    assert all(isinstance(chunk, BaseMessageChunk) for chunk in chunks)

    # Combine all AI message chunks to get text content
    ai_chunks = [chunk for chunk in chunks if isinstance(chunk, AIMessageChunk)]
    text_content = []
    for chunk in ai_chunks:
        if chunk.content:
            for content_item in chunk.content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_content.append(content_item.get("text", ""))
                elif isinstance(content_item, str):
                    text_content.append(content_item)

    full_text = "".join(text_content)
    assert len(full_text) > 0


@pytest.mark.st_endpoints
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_chat_databricks_chatagent_invoke():
    """Test ChatDatabricks with ChatAgent endpoint."""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=CHAT_AGENT_ENDPOINT_WITH_TOOL_CALLING,
        workspace_client=workspace_client,
        temperature=0,
        max_tokens=500,
    )

    response = chat.invoke("What is the 100th fibonacci number?")
    assert isinstance(response, AIMessage)
    assert response.content is not None

    # ChatAgent should use tool calls for complex computations like fibonacci
    # The response content is a list containing message objects including tool calls
    has_tool_calls = False
    python_tool_used = False

    if isinstance(response.content, list):
        # Check for tool calls in the message sequence
        for item in response.content:
            if isinstance(item, dict):
                # Check for tool_calls in assistant messages
                if item.get("tool_calls"):
                    has_tool_calls = True
                    for tool_call in item["tool_calls"]:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        if "python" in tool_name.lower() and "exec" in tool_name.lower():
                            python_tool_used = True
                # Check for tool role messages (responses from tools)
                elif item.get("role") == "tool":
                    has_tool_calls = True
                # Check for function_call type content blocks
                elif item.get("type") == "function_call":
                    has_tool_calls = True
                    if (
                        "python" in item.get("name", "").lower()
                        and "exec" in item.get("name", "").lower()
                    ):
                        python_tool_used = True

    assert has_tool_calls, f"Expected ChatAgent to use tool calls for fibonacci computation. Content: {response.content}"
    assert python_tool_used, f"Expected ChatAgent to use python execution tool for fibonacci computation. Content: {response.content}"


@pytest.mark.st_endpoints
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_chat_databricks_chatagent_stream():
    """Test ChatDatabricks streaming with ChatAgent endpoint."""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=CHAT_AGENT_ENDPOINT_WITH_TOOL_CALLING,
        workspace_client=workspace_client,
        temperature=0,
        max_tokens=500,
    )

    chunks = list(chat.stream("What is the 100th fibonacci number?"))
    assert len(chunks) > 0

    # ChatAgent streaming can include both AIMessageChunk and ToolMessageChunk
    from langchain_core.messages import BaseMessageChunk

    assert all(isinstance(chunk, BaseMessageChunk) for chunk in chunks)

    # For streaming ChatAgent, just verify we get meaningful content
    # Tool call detection in streaming is more complex and may vary
    total_content = ""
    for chunk in chunks:
        if isinstance(chunk.content, str):
            total_content += chunk.content
        elif isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict) and item.get("text"):
                    total_content += item["text"]

    # Verify we get a meaningful response (should contain the fibonacci result or computation)
    assert len(total_content) > 0, "Expected non-empty content from streaming ChatAgent"


@pytest.mark.st_endpoints
@pytest.mark.parametrize("endpoint", _RESPONSES_API_ENDPOINTS)
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_responses_api_extra_body_custom_inputs(endpoint):
    """Test that extra_body parameter can pass custom_inputs to Responses API endpoint"""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=endpoint,
        workspace_client=workspace_client,
        use_responses_api=True,
        temperature=0,
        max_tokens=500,
        extra_params={
            "extra_body": {
                "custom_inputs": {"test_key": "test_value", "user_preference": "concise"}
            }
        },
    )

    response = chat.invoke("What is the 100th fibonacci number?")

    assert isinstance(response, AIMessage)
    assert response.content
    # Test passes if the endpoint accepts the extra_body without error


@pytest.mark.st_endpoints
@pytest.mark.skipif(
    os.environ.get("RUN_ST_ENDPOINT_TESTS", "").lower() != "true",
    reason="Single tenant endpoint tests require special endpoint access. Set RUN_ST_ENDPOINT_TESTS=true to run.",
)
def test_chatagent_extra_body_custom_inputs():
    """Test that extra_body parameter works with ChatAgent endpoints"""
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
    chat = ChatDatabricks(
        model=CHAT_AGENT_ENDPOINT_WITH_TOOL_CALLING,
        workspace_client=workspace_client,
        temperature=0,
        max_tokens=50,
        extra_params={
            "extra_body": {"custom_inputs": {"test_mode": "integration", "response_style": "brief"}}
        },
    )

    response = chat.invoke("Hello! How are you?")

    assert isinstance(response, AIMessage)
    assert response.content
    # Test passes if the endpoint accepts the extra_body without error


@pytest.mark.foundation_models
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
def test_chat_databricks_utf8_encoding(model):
    """Test that ChatDatabricks properly handles UTF-8 encoding."""
    chat = ChatDatabricks(
        model=model,
        temperature=0,
        max_tokens=200,
    )
    messages = [
        SystemMessage(content="Du er en hjælpsom assistent der kan dansk."),
        HumanMessage(content="Sig blåbær på dansk, med små bogstaver."),
    ]

    # Test invoke with UTF-8 characters
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert "blåbær" in response.content

    # Test with streaming as well to ensure chunks handle UTF-8
    stream_chunks = list(chat.stream(messages))
    assert len(stream_chunks) > 0

    # Combine all chunks to verify content
    full_content = ""
    for chunk in stream_chunks:
        if hasattr(chunk, "content") and chunk.content:
            full_content += chunk.content
    assert "blåbær" in full_content.lower()
