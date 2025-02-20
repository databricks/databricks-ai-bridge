import mlflow
from langchain_core.messages import AIMessage, BaseMessage

from databricks_ai_bridge.genie import Genie

from langchain_core.runnables import RunnableLambda

from typing import Dict, Any


class GenieAgent(RunnableLambda):
    """
    A class that implements an agent to send user questions to Genie Space in Databricks through the Genie API.

    This class implements an agent that uses the GenieAPI to send user questions to Genie Space in Databricks.
    If return_metadata is False, the agent's response will be a dictionary containing a single key, 'messages',
    which holds the result of the SQL query executed by the Genie Space.
    If `return_metadata` is set to True, the agent's response will be a dictionary containing two keys: `messages`
    and `metadata`. The `messages` key will contain only one element, similar to the previous case.
    The `metadata` key will include the `GenieResponse` from the API, which will consist of the result of the SQL query,
    the SQL query itself, and a brief description of what the query is doing.

    Attributes:
        genie_space_id (str): The ID of the Genie space created in Databricks will be called by the Genie API.
        description (str): Description of the Genie space created in Databricks that will be accessed by the GenieAPI.
        genie_agent_name (str): The name of the genie agent that will be displayed in the trace.
        return_metadata (bool): Whether to return the GenieResponse generated by the GenieAPI when the agent is called.
        genie (Genie): The Genie API class.

    Methods:
        invoke(state): Returns a dictionary with two possible keys: "messages" and "metadata," which contain the results
        of the query executed by Genie Space and the associated metadata.

    Examples:
        >>> genie_agent = GenieAgent("01ef92421857143785bb9e765454520f")
        >>> genie_agent.invoke({"messages": [{"role": "user", "content": "What is the average total invoice across the different customers?"}]})
        {'messages': [AIMessage(content='|    |   average_total_invoice |\n|---:|------------------------:|\n|  0 |                 195.648 |',
            additional_kwargs={}, response_metadata={})]}
        >>> genie_agent = GenieAgent("01ef92421857143785bb9e765454520f", return_metadata=True)
        >>> genie_agent.invoke({"messages": [{"role": "user", "content": "What is the average total invoice across the different customers?"}]})
        {'messages': [AIMessage(content='|    |   avg_total_invoice |\n|---:|--------------------:|\n|  0 |             195.648 |',
            additional_kwargs={}, response_metadata={})],
        'metadata': GenieResponse(result='|    |   avg_total_invoice |\n|---:|--------------------:|\n|  0 |             195.648 |',
        query='SELECT AVG(`total_invoice`) AS avg_total_invoice FROM `finance`.`external_customers`.`invoices`',
        description='This query calculates the average total invoice amount from all customer invoices, providing insight into overall billing trends.')}
    """
    def __init__(self, genie_space_id: str,
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

