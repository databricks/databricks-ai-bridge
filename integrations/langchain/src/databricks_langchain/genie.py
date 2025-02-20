import mlflow
from langchain_core.messages import AIMessage, BaseMessage

from databricks_ai_bridge.genie import Genie

from langchain_core.runnables import RunnableLambda

from typing import Dict, Any


class GenieAgent(RunnableLambda):
    def __init__(self, genie_space_id,
                 genie_agent_name: str = "Genie",
                 description: str = "",
                 return_metadata: bool = False):
        self.genie_space_id = genie_space_id
        self.genie_agent_name = genie_agent_name
        self.description = description
        self.return_metadata = return_metadata
        self.genie = Genie(genie_space_id)
        super().__init__(self._query_genie_as_agent, name=genie_agent_name)

    @mlflow.trace()
    def _concat_messages_array(self, messages):

        data = []

        for message in messages:
            if isinstance(message, dict):
                data.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
            elif isinstance(message, BaseMessage):
                data.append(f"{message.type}: {message.content}")
            else:
                data.append(f"{getattr(message, 'role', getattr(message, 'name', 'unknown'))}: {getattr(message, 'content', '')}")

        concatenated_message = "\n".join([e for e in data if e])

        return concatenated_message

    @mlflow.trace()
    def _query_genie_as_agent(self, state: Dict[str, Any]):
        message = (f"I will provide you a chat history, where your name is {self.genie_agent_name}. "
                   f"Please help with the described information in the chat history.\n")

        # Concatenate messages to form the chat history
        message += self._concat_messages_array(state.get("messages"))

        # Send the message and wait for a response
        genie_response = self.genie.ask_question(message)

        content = ""
        metadata = None

        if genie_response.result:
            content = genie_response.result
            metadata = genie_response

        if self.return_metadata:
            return {"messages": [AIMessage(content=content)], "metadata": metadata}

        return {"messages": [AIMessage(content=content)]}

