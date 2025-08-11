from typing import Callable, Optional

import mlflow
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
    from langchain_core.messages import AIMessage

    messages = input.get("messages", [])

    # Apply message processor if provided
    if message_processor:
        query = message_processor(messages)
    else:
        query = f"I will provide you a chat history, where your name is {genie_agent_name}. Please help with the described information in the chat history.\n"
        # Concatenate messages to form the chat history
        query += _concat_messages_array(messages)

    # Send the message and wait for a response
    genie_response = genie.ask_question(query)

    query_reasoning = genie_response.description or ""
    query_sql = genie_response.query or ""
    query_result = genie_response.result or ""

    # Create a list of AIMessage to return
    messages = []

    if include_context:
        messages.append(AIMessage(content=query_reasoning, name="query_reasoning"))
        messages.append(AIMessage(content=query_sql, name="query_sql"))
    messages.append(AIMessage(content=query_result, name="query_result"))

    return {"messages": messages}


@mlflow.trace(span_type="AGENT")
def GenieAgent(
    genie_space_id,
    genie_agent_name: str = "Genie",
    description: str = "",
    include_context: bool = False,
    message_processor: Optional[Callable] = None,
    client: Optional["WorkspaceClient"] = None,
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

    genie = Genie(genie_space_id, client=client)

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
