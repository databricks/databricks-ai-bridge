from typing import Callable, Optional

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import Genie


@mlflow.trace()
def _concat_messages_array(messages):
    concatenated_message = "\n".join(
        [
            f"{message.get('role', message.get('name', 'unknown'))}: {message.get('content', '')}"
            if isinstance(message, dict)
            else f"{getattr(message, 'role', getattr(message, 'name', 'unknown'))}: {getattr(message, 'content', '')}"
            for message in messages
        ]
    )
    return concatenated_message


@mlflow.trace()
def _query_genie_as_agent(
    input,
    genie: Genie,
    genie_agent_name,
    include_context: bool = False,
    message_processor: Optional[Callable] = None,
):
    """
    Query Genie as an agent.

    Args:
        input: Input dictionary containing messages and optionally a conversation_id
        genie: Genie instance
        genie_agent_name: Name of the agent
        include_context: Whether to include query reasoning and SQL in the response
        message_processor: Optional function to process messages before querying.
                            It should accept an input of messages and return a query string.
                            If not included, this function will do its own message processing.

    Returns:
        A dictionary containing the messages and conversation_id.
        If include_context is True, the dictionary will also contain the query_reasoning and query_sql fields.
        If Genie returned a dataframe because it was told to do returns in Pandas format, the dictionary will also contain the dataframe field.
    """
    from langchain_core.messages import AIMessage

    messages = input.get("messages", [])
    # Get conversation_id from input state if it exists
    conversation_id = input.get("conversation_id", None)

    # Apply message processor if provided
    if message_processor:
        query = message_processor(messages)
    else:
        query = f"I will provide you a chat history, where your name is {genie_agent_name}. Please help with the described information in the chat history.\n"
        # Concatenate messages to form the chat history
        query += _concat_messages_array(messages)

    # Send the message and wait for a response, passing conversation_id if available
    genie_response = genie.ask_question(query, conversation_id=conversation_id)

    query_reasoning = genie_response.description or ""
    query_sql = genie_response.query or ""
    query_result = genie_response.result if genie_response.result is not None else ""
    query_conversation_id = genie_response.conversation_id or ""

    # Create a list of AIMessage to return
    messages = []

    if include_context:
        messages.append(AIMessage(content=query_reasoning, name="query_reasoning"))
        messages.append(AIMessage(content=query_sql, name="query_sql"))

    # Handle DataFrame vs string results
    if isinstance(query_result, pd.DataFrame):  # if we asked for Pandas return
        # Convert to markdown for message display
        query_result_content = query_result.to_markdown(index=False)
        messages.append(AIMessage(content=query_result_content, name="query_result"))

        # Return with DataFrame included
        return {
            "messages": messages,
            "conversation_id": query_conversation_id,
            "dataframe": query_result,  # Include raw DataFrame if Genie returned dataframe
        }
    else:
        # String result - just add to messages
        messages.append(AIMessage(content=query_result, name="query_result"))

        # Return without DataFrame field
        return {"messages": messages, "conversation_id": query_conversation_id}


@mlflow.trace(span_type="AGENT")
def GenieAgent(
    genie_space_id,
    genie_agent_name: str = "Genie",
    description: str = "",
    include_context: bool = False,
    message_processor: Optional[Callable] = None,
    client: Optional["WorkspaceClient"] = None,
    return_pandas: bool = False,
):
    """Create a genie agent that can be used to query the API. If a description is not provided, the description of the genie space will be used.

    Args:
        genie_space_id: The ID of the genie space to use
        genie_agent_name: Name for the agent (default: "Genie")
        description: Custom description for the agent
        include_context: Whether to include query reasoning and SQL in the response
        message_processor: Optional function to process messages before querying. It should accept a list of either dict
                            or LangChain Message objects and return a query string. If not provided, the agent will
                            use the chat history to form the query.
        client: Optional WorkspaceClient instance
        return_pandas: Whether to return results as pandas DataFrames (if False, returns markdown strings)


    Examples:
        # Basic usage
        agent = GenieAgent("space-123")

        # Only pass the last message
        def last_message_processor(messages):
            if not messages:
                return ""
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                return last_msg.get("content", "")
            else:
                return last_msg.content

        agent = GenieAgent("space-123", message_processor=last_message_processor)

        # Custom message filtering
        def custom_processor(messages):
            # Only process messages containing "data"
            filtered = [msg for msg in messages if "data" in (
                msg.get("content", "") if isinstance(msg, dict) else msg.content
            )]
            return "\\n".join([
                msg.get("content", "") if isinstance(msg, dict) else msg.content
                for msg in filtered
            ])

        agent = GenieAgent("space-123", message_processor=custom_processor)
    """
    if not genie_space_id:
        raise ValueError("genie_space_id is required to create a GenieAgent")

    from functools import partial

    from langchain_core.runnables import RunnableLambda

    genie = Genie(
        genie_space_id,
        client=client,
        return_pandas=return_pandas,
    )

    # Create a partial function with the genie_space_id pre-filled
    partial_genie_agent = partial(
        _query_genie_as_agent,
        genie=genie,
        genie_agent_name=genie_agent_name,
        include_context=include_context,
        message_processor=message_processor,
    )

    runnable = RunnableLambda(partial_genie_agent)
    runnable.name = genie_agent_name
    runnable.description = description or genie.description
    return runnable
