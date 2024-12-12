### base_chat_models.py ###

from typing import List, Dict, Any, Optional, Union
from databricks_ai_bridge.utils import get_deployment_client


class BaseChatDatabricks:
    endpoint: str
    target_uri: str = "databricks"
    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    stream_usage: bool = False
    client: Optional[Any] = None

    def __init__(self, **kwargs: Any):
        self.endpoint = kwargs.get("endpoint", "")
        self.target_uri = kwargs.get("target_uri", "databricks")
        self.temperature = kwargs.get("temperature", 0.0)
        self.n = kwargs.get("n", 1)
        self.stop = kwargs.get("stop", None)
        self.max_tokens = kwargs.get("max_tokens", None)
        self.extra_params = kwargs.get("extra_params", {})
        self.stream_usage = kwargs.get("stream_usage", False)
        self.client = get_deployment_client(self.target_uri)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "target_uri": self.target_uri,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "extra_params": self.extra_params,
        }

    def _prepare_inputs(self, messages: List[Dict[str, Any]], stop: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        data = {
            "messages": messages,
            "temperature": self.temperature,
            "n": self.n,
            **self.extra_params,
            **kwargs,
        }
        if stop := self.stop or stop:
            data["stop"] = stop
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        return data

    def _convert_response_to_chat_result(self, response: Dict[str, Any]) -> Dict[str, Any]:
        generations = [
            {
                "message": choice["message"],
                "generation_info": choice.get("usage", {}),
            }
            for choice in response["choices"]
        ]
        usage = response.get("usage", {})
        return {"generations": generations, "llm_output": usage}

    def _stream(self, messages: List[Dict[str, Any]], stop: Optional[List[str]] = None, **kwargs: Any):
        data = self._prepare_inputs(messages, stop, **kwargs)
        for chunk in self.client.predict_stream(endpoint=self.endpoint, inputs=data):
            if chunk["choices"]:
                choice = chunk["choices"][0]
                yield {
                    "message": choice["delta"],
                    "generation_info": {"finish_reason": choice.get("finish_reason", "")},
                }
