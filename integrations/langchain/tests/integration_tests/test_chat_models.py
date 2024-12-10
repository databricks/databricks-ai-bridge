"""
This file contains the integration test for ChatDatabricks class.

We run the integration tests nightly by the trusted CI/CD system defined in
a private repository, in order to securely run the tests. With this design,
integration test is not intended to be run manually by OSS contributors.
If you want to update the ChatDatabricks implementation and you think that you
need to update the corresponding integration test, please contact to the
maintainers of the repository to verify the changes.
"""

from typing import Annotated
from unittest import mock

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

from langchain_databricks.chat_models import ChatDatabricks

_TEST_ENDPOINT = "databricks-meta-llama-3-70b-instruct"


def test_chat_databricks_invoke():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT, temperature=0, max_tokens=10, stop=["Java"]
    )

    response = chat.invoke("How to learn Java? Start the response by 'To learn Java,'")
    assert isinstance(response, AIMessage)
    assert response.content == "To learn "
    assert response.response_metadata["prompt_tokens"] == 24
    assert response.response_metadata["completion_tokens"] == 3
    assert response.response_metadata["total_tokens"] == 27

    response = chat.invoke(
        "How to learn Python? Start the response by 'To learn Python,'"
    )
    assert response.content.startswith("To learn Python,")
    assert (
        len(response.content.split(" ")) <= 15
    )  # Give some margin for tokenization difference

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


def test_chat_databricks_invoke_multiple_completions():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
        temperature=0.5,
        n=3,
        max_tokens=10,
    )
    response = chat.invoke("How to learn Python?")
    assert isinstance(response, AIMessage)


def test_chat_databricks_stream():
    class FakeCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.chunk_counts = 0

        def on_llm_new_token(self, *args, **kwargs):
            self.chunk_counts += 1

    callback = FakeCallbackHandler()

    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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


def test_chat_databricks_stream_with_usage():
    class FakeCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.chunk_counts = 0

        def on_llm_new_token(self, *args, **kwargs):
            self.chunk_counts += 1

    callback = FakeCallbackHandler()

    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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
async def test_chat_databricks_ainvoke():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
        temperature=0,
        max_tokens=10,
    )

    response = await chat.ainvoke(
        "How to learn Python? Start the response by 'To learn Python,'"
    )
    assert isinstance(response, AIMessage)
    assert response.content.startswith("To learn Python,")


async def test_chat_databricks_astream():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
        temperature=0,
        max_tokens=10,
    )
    chunk_count = 0
    async for chunk in chat.astream("How to learn Python?"):
        assert isinstance(chunk, AIMessageChunk)
        chunk_count += 1
    assert chunk_count > 0


@pytest.mark.asyncio
async def test_chat_databricks_abatch():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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


@pytest.mark.parametrize("tool_choice", [None, "auto", "required", "any", "none"])
def test_chat_databricks_tool_calls(tool_choice):
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
        temperature=0,
        max_tokens=100,
    )

    class GetWeather(BaseModel):
        """Get the current weather in a given location"""

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )

    llm_with_tools = chat.bind_tools([GetWeather], tool_choice=tool_choice)
    question = "Which is the current weather in Los Angeles, CA?"
    response = llm_with_tools.invoke(question)

    if tool_choice == "none":
        assert response.tool_calls == []
        return

    assert response.tool_calls == [
        {
            "name": "GetWeather",
            "args": {"location": "Los Angeles, CA"},
            "id": mock.ANY,
            "type": "tool_call",
        }
    ]

    tool_msg = ToolMessage(
        "GetWeather",
        tool_call_id=response.additional_kwargs["tool_calls"][0]["id"],
    )
    response = llm_with_tools.invoke(
        [
            HumanMessage(question),
            response,
            tool_msg,
            HumanMessage("What about San Francisco, CA?"),
        ]
    )

    assert response.tool_calls == [
        {
            "name": "GetWeather",
            "args": {"location": "San Francisco, CA"},
            "id": mock.ANY,
            "type": "tool_call",
        }
    ]


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
@pytest.mark.parametrize("method", ["function_calling", "json_mode"])
def test_chat_databricks_with_structured_output(schema, method):
    llm = ChatDatabricks(endpoint=_TEST_ENDPOINT)

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


def test_chat_databricks_runnable_sequence():
    chat = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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


def test_chat_databricks_agent_executor():
    model = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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


def test_chat_databricks_langgraph():
    model = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
        temperature=0,
        max_tokens=100,
    )
    tools = [add, multiply]

    app = create_react_agent(model, tools)
    response = app.invoke({"messages": [("human", "What is (10 + 5) * 3?")]})
    assert "45" in response["messages"][-1].content


def test_chat_databricks_langgraph_with_memory():
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [add, multiply]
    llm = ChatDatabricks(
        endpoint=_TEST_ENDPOINT,
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
