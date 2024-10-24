from databricks_ai_bridge.genie import Genie


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


def _query_genie_as_agent(input, genie_space_id, genie_agent_name):
    from langchain_core.messages import AIMessage

    genie = Genie(genie_space_id)

    message = f"I will provide you a chat history, where your name is {genie_agent_name}. Please help with the described information in the chat history.\n"

    # Concatenate messages to form the chat history
    message += _concat_messages_array(input.get("messages"))

    # Send the message and wait for a response
    genie_response = genie.ask_question(message)

    if genie_response:
        return {"messages": [AIMessage(content=genie_response)]}
    else:
        return {"messages": [AIMessage(content="")]}


def GenieAgent(genie_space_id, genie_agent_name="Genie", description=""):
    """Create a genie agent that can be used to query the API"""
    from functools import partial

    from langchain_core.runnables import RunnableLambda

    # Create a partial function with the genie_space_id pre-filled
    partial_genie_agent = partial(
        _query_genie_as_agent, genie_space_id=genie_space_id, genie_agent_name=genie_agent_name
    )

    # Use the partial function in the RunnableLambda
    return RunnableLambda(partial_genie_agent)
