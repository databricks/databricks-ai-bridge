from typing import Optional

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
    input, genie_space_id, genie_agent_name, client: Optional[WorkspaceClient] = None
):
    from langchain_core.messages import AIMessage

    genie = Genie(genie_space_id, client=client)

    message = f"I will provide you a chat history, where your name is {genie_agent_name}. Please help with the described information in the chat history.\n"

    # Concatenate messages to form the chat history
    message += _concat_messages_array(input.get("messages"))

    # Send the message and wait for a response
    genie_response = genie.ask_question(message)

    if query_result := genie_response.result:
        return {"messages": [AIMessage(content=query_result)]}
    else:
        return {"messages": [AIMessage(content="")]}


@mlflow.trace(span_type="AGENT")
def GenieAgent(
    genie_space_id,
    genie_agent_name: str = "Genie",
    description: str = "",
    client: Optional["WorkspaceClient"] = None,
):
    """Create a genie agent that can be used to query the API"""
    if not genie_space_id:
        raise ValueError("genie_space_id is required to create a GenieAgent")

    from functools import partial

    from langchain_core.runnables import RunnableLambda

    # Create a partial function with the genie_space_id pre-filled
    partial_genie_agent = partial(
        _query_genie_as_agent,
        genie_space_id=genie_space_id,
        genie_agent_name=genie_agent_name,
        client=client,
    )

    # Use the partial function in the RunnableLambda
    return RunnableLambda(partial_genie_agent)
