import uuid
from typing import Tuple, Type

from pydantic import BaseModel, Field, Optional

from databricks_ai_bridge.genie import Genie, GenieResult


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


def GenieTool(genie_space_id: str, genie_agent_name: str, description):
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks.manager import CallbackManagerForToolRun

    genie = Genie(genie_space_id)

    class GenieToolInput(BaseModel):
        question: str = Field(description="question to ask the agent")
        summarized_chat_history: str = Field(
            description="summarized chat history to provide the agent context of what may have been talked about. "
                        "Say 'No history' if there is no history to provide.")

    class GenieQuestionToolWithTrace(BaseTool):
        name: str = genie_agent_name
        description: str = description
        args_schema: Type[BaseModel] = GenieToolInput
        response_format: str  = "content_and_artifact"

        def _run(
                self,
                question: str,
                summarized_chat_history: str,
                run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> Tuple[str, GenieResult]:
            message = (f"I will provide you a chat history, where your name is {genie_agent_name}. "
                       f"Please answer the following question: {question} with the following chat history "
                       f"for context: {summarized_chat_history}.\n")
            response = genie.ask_question(message, with_details=True)
            return response.response, response

    tool_with_details = GenieQuestionToolWithTrace()

    class GenieQuestionToolCall(BaseTool):
        name: str = genie_agent_name
        description: str = description
        args_schema: Type[BaseModel] = GenieToolInput

        def _run(
                self,
                question: str,
                summarized_chat_history: str,
                run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> Tuple[str, GenieResult]:
            tool_result = tool_with_details.invoke({
                "name": "GenieQuestionToolWithDetails",
                "args": {"question": question, "summarized_chat_history": summarized_chat_history},
                "id": str(uuid.uuid4()),
                "type": "tool_call"
            })
            return tool_result.content

    return GenieQuestionToolCall()


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
