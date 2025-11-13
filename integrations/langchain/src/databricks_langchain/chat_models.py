"""Databricks chat models."""

import json
import logging
import warnings
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field

from databricks_langchain.utils import get_openai_client

logger = logging.getLogger(__name__)


class ChatDatabricks(BaseChatModel):
    """Databricks chat model integration.

    **Instantiate**:

        .. code-block:: python

            from databricks_langchain import ChatDatabricks

            # Using default authentication (e.g., from environment variables)
            llm = ChatDatabricks(
                model="databricks-claude-3-7-sonnet",
                temperature=0,
                max_tokens=500,
                timeout=30.0,  # Timeout in seconds
                max_retries=3,  # Maximum number of retries
            )

            # Using a WorkspaceClient instance for custom authentication
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient(host="...", token="...")
            llm = ChatDatabricks(
                model="databricks-claude-3-7-sonnet",
                workspace_client=workspace_client,
            )

    For Responses API endpoints like a ResponsesAgent, set ``use_responses_api=True``:
        .. code-block:: python
            llm = ChatDatabricks(
                model="my-responses-agent-endpoint",
                use_responses_api=True,
            )

    **Invoke**:

        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={"prompt_tokens": 32, "completion_tokens": 9, "total_tokens": 41},
                id="run-64eebbdd-88a8-4a25-b508-21e9a5f146c5-0",
            )

    **Stream**:

        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            content='J' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content="'" id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ad' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ore' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content=' la' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content=' programm' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ation' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='.' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='' response_metadata={'finish_reason': 'stop'} id='run-609b8f47-e580-4691-9ee4-e2109f53155e'

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content="J'adore la programmation.",
                response_metadata={"finish_reason": "stop"},
                id="run-4cef851f-6223-424f-ad26-4a54e5852aa5",
            )

        To get token usage returned when streaming, pass the ``stream_usage`` kwarg:

        .. code-block:: python

            stream = llm.stream(messages, stream_usage=True)
            next(stream).usage_metadata

        .. code-block:: python

            {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}

        Alternatively, setting ``stream_usage`` when instantiating the model can be
        useful when incorporating ``ChatDatabricks`` into LCEL chains-- or when using
        methods like ``.with_structured_output``, which generate chains under the
        hood.

        .. code-block:: python

            llm = ChatDatabricks(model="databricks-claude-3-7-sonnet", stream_usage=True)
            structured_llm = llm.with_structured_output(...)

    **Async**:

        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in llm.astream(messages)

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={"prompt_tokens": 32, "completion_tokens": 9, "total_tokens": 41},
                id="run-e4bb043e-772b-4e1d-9f98-77ccc00c0271-0",
            )

    **Tool calling**:

        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    "name": "GetWeather",
                    "args": {"location": "Los Angeles, CA"},
                    "id": "call_ea0a6004-8e64-4ae8-a192-a40e295bfa24",
                    "type": "tool_call",
                }
            ]

        To use tool calls, your model endpoint must support ``tools`` parameter. See [Function calling on Databricks](https://python.langchain.com/docs/integrations/chat/databricks/#function-calling-on-databricks) for more information.

    **Custom inputs and outputs**:

        Pass additional context or configuration to the model using the ``custom_inputs`` keyword argument.
        Access model-specific metadata from the ``custom_outputs`` attribute of the response.

        .. code-block:: python

            # Pass custom inputs to the model
            messages = [("human", "Hello, how are you?")]
            custom_inputs = {"user_id": "12345", "session_id": "abc123"}

            response = llm.invoke(messages, custom_inputs=custom_inputs)
            print(response.content)

            # Access custom outputs from the response (if provided by the model)
            if hasattr(response, "custom_outputs"):
                print(f"Custom outputs: {response.custom_outputs}")

        .. code-block:: python

            # Custom inputs also work with streaming
            for chunk in llm.stream(messages, custom_inputs=custom_inputs):
                print(chunk.content, end="")

                # Check for custom outputs in chunks
                if hasattr(chunk, "custom_outputs"):
                    print(f"Custom outputs: {chunk.custom_outputs}")

    """  # noqa: E501

    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(alias="endpoint")
    """Name of Databricks Model Serving endpoint to query."""
    target_uri: Optional[str] = None
    """The target MLflow deployment URI to use. Deprecated: use workspace_client instead."""
    workspace_client: Optional[Any] = Field(default=None, exclude=True)
    """Optional WorkspaceClient instance to use for authentication. If not provided, uses default authentication."""
    temperature: Optional[float] = None
    """Sampling temperature. Higher values make the model more creative."""
    n: int = 1
    """The number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """List of strings to stop generation at."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    extra_params: Optional[Dict[str, Any]] = None
    """Whether to include usage metadata in streaming output. If True, additional
    message chunks will be generated during the stream including usage metadata.
    """
    stream_usage: bool = True
    """Any extra parameters to pass to the endpoint."""
    use_responses_api: bool = False
    """Whether to use the Responses API to format inputs and outputs."""
    timeout: Optional[float] = None
    """Timeout in seconds for the HTTP request. If None, uses the default timeout."""
    max_retries: Optional[int] = None
    """Maximum number of retries for failed requests. If None, uses the default retry count."""
    client: Optional[object] = Field(default=None, exclude=True)  #: :meta private:

    @property
    def endpoint(self) -> str:
        warnings.warn(
            "The `endpoint` attribute is deprecated and will be removed in a future version. "
            "Use `model` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model

    @endpoint.setter
    def endpoint(self, value: str) -> None:
        warnings.warn(
            "The `endpoint` attribute is deprecated and will be removed in a future version. "
            "Use `model` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = value

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Handle deprecated target_uri parameter
        if self.target_uri:
            warnings.warn(
                "The 'target_uri' parameter is deprecated and will be removed in a future version. "
                "Use 'workspace_client' parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.workspace_client:
                raise ValueError(
                    "Cannot specify both 'workspace_client' and 'target_uri'. "
                    "Please use 'workspace_client' only."
                )

        # Always use OpenAI client (supports both chat completions and responses API)
        # Prepare kwargs for the SDK call
        openai_kwargs = {}
        if self.timeout is not None:
            openai_kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            openai_kwargs["max_retries"] = self.max_retries

        self.client = get_openai_client(workspace_client=self.workspace_client, **openai_kwargs)

        self.use_responses_api = kwargs.get("use_responses_api", False)
        self.extra_params = self.extra_params or {}

    @property
    def _default_params(self) -> Dict[str, Any]:
        exclude_if_none = {
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "extra_params": self.extra_params,
        }

        params = {
            "model": self.model,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
        }
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        custom_inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        data = self._prepare_inputs(messages, stop, custom_inputs=custom_inputs, **kwargs)

        if self.use_responses_api:
            # Use OpenAI client with responses API
            resp = self.client.responses.create(**data)  # type: ignore
            return self._convert_responses_api_response_to_chat_result(resp)
        else:
            # Use OpenAI client with chat completions API
            resp = self.client.chat.completions.create(**data)  # type: ignore
            return self._convert_response_to_chat_result(resp)

    def _prepare_inputs(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
        *,
        custom_inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            **self.extra_params,  # type: ignore
            **kwargs,
        }
        if custom_inputs is not None:
            data["extra_body"] = {
                "custom_inputs": custom_inputs,
            }

        # Format data based on whether we're using responses API or chat completions
        if self.use_responses_api:
            # Responses API expects "input" parameter and has different supported parameters
            data["input"] = _convert_lc_messages_to_responses_api(messages)
            # Responses API only supports temperature, not max_tokens, stop, or n
            if self.temperature is not None:
                data["temperature"] = self.temperature
        else:
            # Chat completions API expects "messages" parameter
            data["messages"] = [_convert_message_to_dict(msg) for msg in messages]

            if self.temperature is not None:
                data["temperature"] = self.temperature
            if stop := self.stop or stop:
                data["stop"] = stop
            if self.max_tokens is not None:
                data["max_tokens"] = self.max_tokens
            if self.n != 1:
                data["n"] = self.n

        return data

    def _convert_responses_api_response_to_chat_result(self, response: Any) -> ChatResult:
        """
        A Responses API response has an array of messages, but a ChatResult can only have a single message.
        To accomodate this, we combine the messages into a single message, following LangChain convention.
        """
        # Handle error response
        if response.error:
            raise ValueError(response.error)
        # Combine all content and tool calls from output items
        content_blocks = []
        tool_calls = []
        invalid_tool_calls = []

        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        annotations = []
                        if content.annotations:
                            # Convert annotation objects to dictionaries
                            annotations = [
                                ann.model_dump() if hasattr(ann, "model_dump") else ann
                                for ann in content.annotations
                            ]

                        content_blocks.append(
                            {
                                "type": "text",
                                "text": content.text,
                                "annotations": annotations,
                                "id": getattr(content, "id", None),
                            }
                        )
                    elif content.type == "refusal":
                        content_blocks.append(
                            {
                                "type": "refusal",
                                "refusal": content.refusal,
                                "id": getattr(content, "id", None),
                            }
                        )
            elif item.type == "function_call":
                # Convert to dict for content_blocks to maintain backward compatibility
                item_dict = {
                    "type": "function_call",
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id,
                }
                content_blocks.append(item_dict)

                try:
                    args = json.loads(item.arguments, strict=False)
                    error = None
                except json.JSONDecodeError as e:
                    error = str(e)
                    args = item.arguments
                if error is None:
                    tool_calls.append(
                        {
                            "type": "tool_call",
                            "name": item.name,
                            "args": args,
                            "id": item.call_id,
                        }
                    )
                else:
                    invalid_tool_calls.append(
                        {
                            "type": "invalid_tool_call",
                            "name": item.name,
                            "args": args,
                            "id": item.call_id,
                            "error": error,
                        }
                    )
            elif item.type == "function_call_output":
                content_blocks.append(
                    {
                        "role": "tool",
                        "content": item.output,
                        "tool_call_id": item.call_id,
                    }
                )
            elif item.type in (
                "reasoning",
                "web_search_call",
                "file_search_call",
                "computer_call",
                "code_interpreter_call",
                "mcp_call",
                "mcp_list_tools",
                "mcp_approval_request",
                "image_generation_call",
            ):
                # For these special types, convert to dict if possible
                if hasattr(item, "model_dump"):
                    content_blocks.append(item.model_dump())
                else:
                    content_blocks.append(item)

        # Create AI message with combined content and tool calls
        message = AIMessage(
            content=content_blocks,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            id=response.id,
        )
        if hasattr(response, "custom_outputs"):
            message.custom_outputs = response.custom_outputs
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_chatagent_response_to_chat_result(self, response: Any) -> ChatResult:
        """
        A ChatAgent response has an array of messages, but a ChatResult can only have a single message.
        To accomodate this, we combine the messages into a single message, following LangChain convention.

        ex: https://github.com/langchain-ai/langchain/blob/2d3020f6cd9d3bf94738f2b6732b68acc55d9cce/libs/partners/openai/langchain_openai/chat_models/base.py#L3739
        """
        # Since we only enter this when response.messages exists, we can directly access it
        messages = response.messages
        message = AIMessage(content=messages, id=getattr(response, "id", None))
        if hasattr(response, "custom_outputs"):
            message.custom_outputs = response.custom_outputs
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_response_to_chat_result(self, response: Any) -> ChatResult:
        # Check if this is a ChatAgent response (has messages but no choices)
        if (
            hasattr(response, "choices")
            and response.choices is None
            and hasattr(response, "messages")
        ):
            return self._convert_chatagent_response_to_chat_result(response)

        generations = []
        for choice in response.choices:
            message_dict = choice.message.model_dump(exclude_unset=True)
            # Ensure content is not None
            if message_dict.get("content") is None:
                message_dict["content"] = ""

            generation_info = {}
            if choice.finish_reason:
                generation_info["finish_reason"] = choice.finish_reason

            generations.append(
                ChatGeneration(
                    message=_convert_dict_to_message(message_dict),
                    generation_info=generation_info,
                )
            )

        llm_output = {}
        if response.usage:
            llm_output["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Add individual token counts for backwards compatibility with tests
            llm_output["prompt_tokens"] = response.usage.prompt_tokens
            llm_output["completion_tokens"] = response.usage.completion_tokens
            llm_output["total_tokens"] = response.usage.total_tokens
        if response.model:
            llm_output["model"] = response.model
            llm_output["model_name"] = response.model

        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage

        data = self._prepare_inputs(
            messages, stop, stream=True, custom_inputs=custom_inputs, **kwargs
        )

        usage_chunk_emitted = False
        final_usage = None

        if self.use_responses_api:
            prev_chunk = None
            stream = self.client.responses.create(**data)  # type: ignore
            for chunk in stream:
                chunk_message = _convert_responses_api_chunk_to_lc_chunk(chunk, prev_chunk)
                prev_chunk = chunk
                if chunk_message:
                    yield ChatGenerationChunk(message=chunk_message)
                # Check for usage in the chunk if available
                if stream_usage and hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", None)
                    output_tokens = getattr(chunk.usage, "completion_tokens", None)
                    if input_tokens is not None and output_tokens is not None:
                        final_usage = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
            # Emit special usage chunk at end of stream
            if stream_usage and final_usage and not usage_chunk_emitted:
                # Usage chunk is an AIMessageChunk with empty content and usage_metadata
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", usage_metadata=UsageMetadata(**final_usage))
                )
                usage_chunk_emitted = True
        else:
            first_chunk_role = None
            stream = self.client.chat.completions.create(**data)  # type: ignore
            for chunk in stream:
                # Handle ChatAgent chunks that don't have choices but have delta
                if hasattr(chunk, "choices") and chunk.choices is None and hasattr(chunk, "delta"):
                    delta = chunk.delta
                    chunk_delta_dict = {
                        "role": delta.get("role"),
                        "content": delta.get("content", ""),
                    }
                    if hasattr(chunk, "custom_outputs"):
                        chunk_delta_dict["custom_outputs"] = chunk.custom_outputs
                    chunk_message = _convert_dict_to_message_chunk(
                        chunk_delta_dict, first_chunk_role
                    )
                    generation_chunk = ChatGenerationChunk(message=chunk_message)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                        )
                    yield generation_chunk
                elif chunk.choices:
                    choice = chunk.choices[0]
                    chunk_delta = choice.delta
                    if first_chunk_role is None:
                        first_chunk_role = chunk_delta.role

                    usage = None
                    # Collect usage info only if both are present
                    if stream_usage and chunk.usage:
                        input_tokens = getattr(chunk.usage, "prompt_tokens", None)
                        output_tokens = getattr(chunk.usage, "completion_tokens", None)
                        if input_tokens is not None and output_tokens is not None:
                            usage = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                            }
                            final_usage = usage  # store for usage chunk at end
                    # Use model_dump instead of manual dict reconstruction
                    chunk_delta_dict = chunk_delta.model_dump(exclude_unset=True)
                    chunk_message = _convert_dict_to_message_chunk(
                        chunk_delta_dict, first_chunk_role, usage=usage
                    )
                    generation_info = {}
                    if choice.finish_reason:
                        generation_info["finish_reason"] = choice.finish_reason
                    if choice.logprobs:
                        generation_info["logprobs"] = choice.logprobs

                    generation_chunk = ChatGenerationChunk(
                        message=chunk_message, generation_info=generation_info or None
                    )

                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=generation_info.get("logprobs"),
                        )
                    yield generation_chunk

            # Emit special usage chunk at end of stream
            if stream_usage and final_usage and not usage_chunk_emitted:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", usage_metadata=UsageMetadata(**final_usage))
                )
                usage_chunk_emitted = True

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call. Options are:

                - name of the tool (str): Calls corresponding tool.
                - **"auto"**: Automatically selects a tool (including no tool).
                - **"none"**: Model does not generate any tool calls and instead must generate a standard assistant message.
                - **"required"**: The model picks the most relevant tool in tools and must generate a tool call or a dictionary of the form:

                    .. code-block:: json

                        {
                            "type": "function",
                            "function": {
                                "name": "<<tool_name>>"
                            }
                        }

            **kwargs: Any additional parameters to pass to the
                `Runnable <langchain-core:Runnable>`__ constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required", "any"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API,
                # but supported by other models in Langchain.
                # Ref: https://github.com/langchain-ai/langchain/blob/202d7f6c4a2ca8c7e5949d935bcf0ba9b0c23fb0/libs/partners/openai/langchain_openai/chat_models/base.py#L1098C1-L1101C45
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"] for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"] for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type]] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then OpenAI's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A `Runnable <langchain-core:Runnable>`__ that takes any ChatModel input and returns as output:

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]

        **Examples**:

        Function-calling, Pydantic schema (method="function_calling", include_raw=False)

            .. code-block:: python

                from databricks_langchain import ChatDatabricks
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatDatabricks(model="databricks-claude-3-7-sonnet")
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Function-calling, Pydantic schema (method="function_calling", include_raw=True):

            .. code-block:: python

                from databricks_langchain import ChatDatabricks
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatDatabricks(model="databricks-claude-3-7-sonnet")
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Function-calling, dict schema (method="function_calling", include_raw=False):

            .. code-block:: python

                from databricks_langchain import ChatDatabricks
                from langchain_core.utils.function_calling import convert_to_openai_tool
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatDatabricks(model="databricks-claude-3-7-sonnet")
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        JSON mode, Pydantic schema (method="json_mode", include_raw=True):

            .. code-block::

                from databricks_langchain import ChatDatabricks
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatDatabricks(model="databricks-claude-3-7-sonnet")
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'."
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{    "answer": "They are both the same weight.",    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." }'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        JSON mode, no schema (schema=None, method="json_mode", include_raw=True):

            .. code-block::

                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'."
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{    "answer": "They are both the same weight.",    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." }'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }


        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. Received None."
                )
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools([schema], tool_choice=tool_name)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'json_schema'. Received None."
                )
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "schema": (
                        schema.model_json_schema() if is_pydantic_schema else schema  # type: ignore[union-attr]
                    ),
                },
            }
            llm = self.bind(response_format=response_format)
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )

        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_mode' or 'json_schema'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model FOR THE CALLBACKS."""
        return {
            **self._default_params,
            **super()._get_invocation_params(stop=stop, **kwargs),
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-databricks"


### Conversion function to convert Pydantic models to dictionaries and vice versa. ###


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict = {"content": message.content}

    # NB: We don't propagate 'name' field from input message to the endpoint because
    #  FMAPI doesn't support it. We should update the endpoints to be compatible with
    #  OpenAI and then we can uncomment the following code.
    # if (name := message.name or message.additional_kwargs.get("name")) is not None:
    #     message_dict["name"] = name

    if isinstance(message, ChatMessage):
        return {"role": message.role, **message_dict}
    elif isinstance(message, HumanMessage):
        return {"role": "user", **message_dict}
    elif isinstance(message, AIMessage):
        if tool_calls := _get_tool_calls_from_ai_message(message):
            message_dict["tool_calls"] = tool_calls  # type: ignore[assignment]
            # If tool calls present, content null value should be None not empty string.
            message_dict["content"] = message_dict["content"] or None  # type: ignore[assignment]
        return {"role": "assistant", **message_dict}
    elif isinstance(message, SystemMessage):
        return {"role": "system", **message_dict}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            **message_dict,
        }
    elif isinstance(message, FunctionMessage) or "function_call" in message.additional_kwargs:
        raise ValueError(
            "Function messages are not supported by Databricks. Please"
            " create a feature request at https://github.com/mlflow/mlflow/issues."
        )
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")


def _convert_lc_messages_to_responses_api(messages: List[BaseMessage]) -> dict:
    """
    Convert a LangChain message to a Responses API message.
    """
    # TODO: add multimodal support
    input_items = []
    for lc_msg in messages:
        cc_msg = _convert_message_to_dict(lc_msg)
        # "name" parameter unsupported
        if "name" in cc_msg:
            cc_msg.pop("name")
        role = cc_msg["role"]
        if role == "assistant":
            if isinstance(cc_msg.get("content"), list):
                for block in cc_msg["content"]:
                    if isinstance(block, dict) and (block_type := block.get("type")):
                        if block_type in ("text", "output_text"):
                            input_items.append(
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": block["text"],
                                            "annotations": block.get("annotations") or [],
                                        }
                                    ],
                                    "role": "assistant",
                                    "id": lc_msg.id,
                                }
                            )
                        elif block_type == "refusal":
                            input_items.append(
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "refusal",
                                            "refusal": block["refusal"],
                                        }
                                    ],
                                    "role": "assistant",
                                    "id": lc_msg.id,
                                }
                            )
                        elif block_type in (
                            "reasoning",
                            "web_search_call",
                            "file_search_call",
                            "function_call",
                            "computer_call",
                            "code_interpreter_call",
                            "mcp_call",
                            "mcp_list_tools",
                            "mcp_approval_request",
                        ):
                            input_items.append(block | {"id": lc_msg.id})
            elif isinstance(cc_msg.get("content"), str):
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "id": lc_msg.id,
                        "content": [{"type": "output_text", "text": cc_msg["content"]}],
                    }
                )

            if tool_calls := cc_msg.get("tool_calls"):
                input_items.extend(
                    [
                        {
                            "type": "function_call",
                            "id": lc_msg.id,
                            "call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        }
                        for tool_call in tool_calls
                    ]
                )
        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": cc_msg["tool_call_id"],
                    "output": cc_msg["content"],
                }
            )
        elif role in ("user", "system", "developer"):
            input_items.append(cc_msg)
        else:
            pass
    return input_items


def _get_tool_calls_from_ai_message(message: AIMessage) -> List[Dict]:
    tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        for tc in message.tool_calls
    ]

    invalid_tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": tc["args"],
            },
        }
        for tc in message.invalid_tool_calls
    ]

    if tool_calls or invalid_tool_calls:
        return tool_calls + invalid_tool_calls

    # Get tool calls from additional kwargs if present.
    return [
        {
            k: v
            for k, v in tool_call.items()  # type: ignore[union-attr]
            if k in {"id", "type", "function"}
        }
        for tool_call in message.additional_kwargs.get("tool_calls", [])
    ]


def _convert_dict_to_message(_dict: Dict) -> BaseMessage:
    role = _dict["role"]
    content = _dict.get("content") or ""
    if not isinstance(content, str):
        # for non-string content, serialize it into a string to maintain compatibility with downstream consumers
        # for example, output parsers expect a string
        content = json.dumps(content)

    lc_message = None
    if role == "user":
        lc_message = HumanMessage(content=content)
    elif role == "system":
        lc_message = SystemMessage(content=content)
    elif role == "tool":
        lc_message = ToolMessage(
            content=content, tool_call_id=_dict.get("tool_call_id"), id=_dict.get("id")
        )
    elif role == "assistant":
        additional_kwargs: Dict = {}
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
        lc_message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            id=_dict.get("id"),
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    else:
        lc_message = ChatMessage(content=content, role=role)

    if _dict.get("custom_outputs"):
        lc_message.custom_outputs = _dict.get("custom_outputs")
    return lc_message


def _convert_dict_to_message_chunk(
    _dict: Mapping[str, Any],
    default_role: str,
    usage: Optional[Dict[str, Any]] = None,
) -> BaseMessageChunk:
    role = _dict.get("role", default_role)
    content = _dict.get("content") or ""
    if not isinstance(content, str):
        # for non-string content, serialize it into a string to maintain compatibility with downstream consumers
        # for example, output parsers expect a string
        content = json.dumps(content)

    lc_chunk = None
    if role == "user":
        lc_chunk = HumanMessageChunk(content=content)
    elif role == "system":
        lc_chunk = SystemMessageChunk(content=content)
    elif role == "tool":
        lc_chunk = ToolMessageChunk(
            content=content, tool_call_id=_dict.get("tool_call_id", ""), id=_dict.get("id")
        )
    elif role == "assistant" or role is None:
        # If role is None (common in streaming), default to assistant
        additional_kwargs: Dict = {}
        tool_call_chunks = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            try:
                tool_call_chunks = [
                    tool_call_chunk(
                        name=tc["function"].get("name"),
                        args=tc["function"].get("arguments"),
                        id=tc.get("id"),
                        index=tc["index"],
                    )
                    for tc in raw_tool_calls
                ]
            except KeyError:
                pass
        usage_metadata = UsageMetadata(**usage) if usage else None  # type: ignore
        lc_chunk = AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=_dict.get("id"),
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
        )
    else:
        lc_chunk = ChatMessageChunk(content=content, role=role)

    if _dict.get("custom_outputs"):
        lc_chunk.custom_outputs = _dict.get("custom_outputs")
    return lc_chunk


def _convert_responses_api_chunk_to_lc_chunk(
    chunk: Any, previous_chunk: Optional[Any] = None
) -> Optional[BaseMessageChunk]:
    # Handle OpenAI responses API chunks
    content = []
    tool_call_chunks = []
    id = None
    lc_chunk = None

    if chunk.type == "response.output_text.delta":
        id = getattr(chunk, "item_id", None)
        content.append(
            {
                "type": "text",
                "text": chunk.delta,
            }
        )
    elif chunk.type == "response.output_item.done":
        item = chunk.item
        if item.type == "function_call_output":
            lc_chunk = ToolMessageChunk(
                content=item.output,
                tool_call_id=item.call_id,
            )
        elif item.type == "function_call":
            id = item.call_id
            tool_call_chunks.append(
                tool_call_chunk(
                    name=item.name,
                    args=item.arguments,
                    id=item.call_id,
                )
            )
        elif item.type == "message":
            id = item.id
            # skip text outputs that have already been streamed, but keep the annotations
            prev_type = previous_chunk.type if previous_chunk else None
            prev_item_id = getattr(previous_chunk, "item_id", None) if previous_chunk else None
            skip_duplicate_text = (
                previous_chunk and prev_type == "response.output_text.delta" and id == prev_item_id
            )
            for content_item in item.content:
                if content_item.type == "output_text":
                    if skip_duplicate_text:
                        if content_item.annotations:
                            # Convert annotation objects to dictionaries
                            annotations = [
                                ann.model_dump() if hasattr(ann, "model_dump") else ann
                                for ann in content_item.annotations
                            ]
                            content.append({"annotations": annotations})
                    else:
                        annotations = []
                        if content_item.annotations:
                            # Convert annotation objects to dictionaries
                            annotations = [
                                ann.model_dump() if hasattr(ann, "model_dump") else ann
                                for ann in content_item.annotations
                            ]

                        content.append(
                            {
                                "type": "text",
                                "text": content_item.text,
                                "annotations": annotations,
                            }
                        )
                elif content_item.type == "refusal":
                    content.append(
                        {
                            "type": "refusal",
                            "refusal": content_item.refusal,
                        }
                    )
        elif item.type in (
            "web_search_call",
            "file_search_call",
            "computer_call",
            "code_interpreter_call",
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "image_generation_call",
            "reasoning",
        ):
            # Convert item to dictionary for LangChain compatibility
            if hasattr(item, "model_dump"):
                content.append(item.model_dump())
            else:
                content.append(item)
    elif chunk.type == "response.created":
        id = chunk.response.id if chunk.response else None
        lc_chunk = AIMessageChunk(content="", id=id)
    elif chunk.type == "response.completed":
        # This indicates the response is done
        return None
    elif chunk.type == "error":
        raise ValueError(chunk.error)
    else:
        # Return None for unknown chunk types
        return None

    # Only return a chunk if we have content or tool calls
    if content or tool_call_chunks:
        lc_chunk = AIMessageChunk(
            content=content,
            tool_call_chunks=tool_call_chunks,
            id=id,
        )

    if hasattr(chunk, "custom_outputs"):
        lc_chunk.custom_outputs = chunk.custom_outputs
    return lc_chunk
