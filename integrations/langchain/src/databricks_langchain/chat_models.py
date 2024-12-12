### langchain/chat_models.py ###

from typing import Iterator, List, Dict, Any, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, AIMessageChunk, BaseMessage, ChatResult, ChatGeneration, ChatGenerationChunk
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser, PydanticToolsParser, make_invalid_tool_call, parse_tool_call
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from databricks_langchain.utils import get_deployment_client
from .base_chat_models import BaseChatDatabricks


class ChatDatabricks(BaseChatDatabricks, BaseChatModel):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        data = self._prepare_inputs([_convert_message_to_dict(msg) for msg in messages], stop, **kwargs)
        resp = self.client.predict(endpoint=self.endpoint, inputs=data)
        return self._convert_response_to_chat_result(resp)

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, *, stream_usage: Optional[bool] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        data = self._prepare_inputs([_convert_message_to_dict(msg) for msg in messages], stop, **kwargs)
        first_chunk_role = None
        for chunk in self.client.predict_stream(endpoint=self.endpoint, inputs=data):
            if chunk["choices"]:
                choice = chunk["choices"][0]
                chunk_delta = choice["delta"]
                if first_chunk_role is None:
                    first_chunk_role = chunk_delta.get("role")

                usage = chunk.get("usage") if stream_usage else None
                chunk_message = _convert_dict_to_message_chunk(chunk_delta, first_chunk_role, usage=usage)
                generation_info = {
                    "finish_reason": choice.get("finish_reason", ""),
                    "logprobs": choice.get("logprobs", {}),
                }

                yield ChatGenerationChunk(
                    message=chunk_message, generation_info=generation_info or None
                )

    @property
    def _llm_type(self) -> str:
        return "chat-databricks"


### Conversion Functions ###

def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    message_dict = {"content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", **message_dict}
    elif isinstance(message, BaseMessage):
        return {"role": message.role, **message_dict}
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def _convert_dict_to_message(_dict: Dict[str, Any]) -> BaseMessage:
    role = _dict["role"]
    content = _dict.get("content", "")
    if role == "assistant":
        return AIMessage(content=content)
    else:
        return BaseMessage(content=content, role=role)


def _convert_dict_to_message_chunk(
    _dict: Dict[str, Any], default_role: str = "assistant", usage: Optional[Dict[str, Any]] = None
) -> AIMessageChunk:
    content = _dict.get("content", "")
    return AIMessageChunk(content=content, usage_metadata=usage)
