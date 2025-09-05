from dspy.streaming.messages import StreamResponse
from dspy.streaming.streaming_listener import StreamListener
from litellm import ModelResponseStream

from databricks_dspy.adapters.types.citation import DatabricksCitations


class DatabricksStreamListener(StreamListener):
    def receive(self, chunk: ModelResponseStream):
        # Handle anthropic citations. see https://docs.litellm.ai/docs/providers/anthropic#beta-citations-api
        try:
            if self._is_citation_type():
                if chunk_citation := chunk.choices[0].delta.provider_specific_fields.get("citation", None):
                    return StreamResponse(
                        self.predict_name,
                        self.signature_field_name,
                        DatabricksCitations.from_dict_list([chunk_citation]),
                        is_last_chunk=False,
                    )
        except Exception:
            pass
    
        super().receive(chunk)

    def _is_citation_type(self) -> bool:
        """Check if the signature field is a citations field."""
        from dspy.predict import Predict
        return isinstance(self.predict, Predict) and getattr(self.predict.signature.output_fields.get(self.signature_field_name, None), "annotation", None) == Citations
