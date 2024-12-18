import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.llm import (
    ChatAgentMessage,
    ChatAgentParams,
    ChatAgentResponse,
    FunctionToolCallArguments,
    ToolCall,
)
from pydantic import BaseModel


def add_agent_messages(left: List[Dict], right: List[Dict]):
    # assign missing ids
    for m in left:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())
    for m in right:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())

    # merge
    left_idx_by_id = {m.get("id"): i for i, m in enumerate(left)}
    merged = left.copy()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.get("id"))) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


# We create the ChatAgentState that we will pass around
class ChatAgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[list, add_agent_messages]
    custom_outputs: Dict[str, Any]


@dataclass
class SystemMessage(ChatAgentMessage):
    role: Literal["system"] = field(default="system")


def parse_tool_calls(id, tool_calls: list[Dict[str, Any]]) -> Dict:
    return ChatAgentMessage(
        role="assistant",
        content="",
        id=id,
        tool_calls=[
            ToolCall(
                id=tool_call.get("id"),
                function=FunctionToolCallArguments(
                    arguments=json.dumps(tool_call.get("args", {})), name=tool_call.get("name")
                ),
            )
            for tool_call in tool_calls
        ],
        # attachments = ...
    ).to_dict()


def parse_tool_result(tool_msg: ToolMessage, attachments=None) -> Dict:
    return ChatAgentMessage(
        role="tool",
        id=tool_msg.tool_call_id,
        content=tool_msg.content,
        name=tool_msg.name,
        tool_call_id=tool_msg.tool_call_id,
        attachments=attachments,
    ).to_dict()


def parse_message(msg, key: str = None, attachments: Dict = None) -> Dict:
    """Parse different message types into their string representations"""
    # tool call result
    if isinstance(msg, ToolMessage):
        return parse_tool_result(msg, attachments=attachments)
    # tool call
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        return parse_tool_calls(msg.id, msg.tool_calls)
    elif isinstance(msg, AIMessage):
        args = {
            "role": "assistant",
            "id": msg.id,
            "content": msg.content,
            # "attachments": ...
        }
        if key:
            args["name"] = key
        return ChatAgentMessage(**args).to_dict()
    elif isinstance(msg, HumanMessage):
        return ChatAgentMessage(
            role="user",
            id=msg.id,
            content=msg.content,
            # attachments = ...
        ).to_dict()
    else:
        logging.warning(f"Unexpected message type: {type(msg), str(msg)}")


class ChatAgentToolNode(ToolNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        result = super().invoke(input, config, **kwargs)
        messages = []
        custom_outputs = None
        for m in result["messages"]:
            try:
                return_obj = json.loads(m.content)
                if "custom_outputs" in return_obj:
                    custom_outputs = return_obj["custom_outputs"]
                messages.append(parse_message(m, attachments=return_obj.get("attachments")))
            except Exception:
                messages.append(parse_message(m))
        return {"messages": messages, "custom_outputs": custom_outputs}

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        store: BaseStore,
    ) -> ToolCall:
        tool_call["name"] = tool_call["function"]["name"]
        tool_call["args"] = json.loads(tool_call["function"]["arguments"])
        temp = super()._inject_tool_args(tool_call, input, store)
        return temp

    def _parse_input(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        store: BaseStore,
    ) -> Tuple[List[ToolCall], Literal["list", "dict"]]:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            output_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        tool_calls = [self._inject_tool_args(call, input, store) for call in message["tool_calls"]]
        return tool_calls, output_type


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent):
        self.agent = agent

    def _convert_messages_to_dict(self, messages: List[ChatAgentMessage]):
        return [m.to_dict() for m in messages]

    def predict(self, messages: List[ChatAgentMessage], params: Optional[ChatAgentParams] = None):
        response = ChatAgentResponse(messages=[])
        for event in self.agent.stream(
            {"messages": self._convert_messages_to_dict(messages)}, stream_mode="updates"
        ):
            for node in event:
                response.messages.extend(event[node]["messages"])
                if "custom_outputs" in event[node]:
                    response.custom_outputs = event[node]["custom_outputs"]
        return response.to_dict()

    def predict_stream(
        self, messages: List[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ):
        for event in self.agent.stream(
            {"messages": self._convert_messages_to_dict(messages)}, stream_mode="updates"
        ):
            for node in event:
                yield ChatAgentResponse(**event[node]).to_dict()