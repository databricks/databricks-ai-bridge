from unittest import mock

import dspy
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import databricks_dspy


@pytest.mark.anyio
async def test_streaming_with_citations():
    class AnswerWithSources(dspy.Signature):
        """Answer questions using provided documents with citations."""
        documents: list[databricks_dspy.DatabricksDocument] = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: databricks_dspy.DatabricksCitations = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(AnswerWithSources)

        def forward(self, documents, question, **kwargs):
            return self.predict(documents=documents, question=question, **kwargs)

    async def citation_stream(*args, **kwargs):
        # Stream chunks with citation data in provider_specific_fields
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="Water"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" boils"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" at"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" 100°C"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="."))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content='[{"type": "char_location", "cited_text": "Water boils at 100°C", "document_index": 0, "document_title": "Physics Facts", "start_char_index": 0, "end_char_index": 19}]'))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(
            content="",
            provider_specific_fields={
                "citation": {
                    "type": "char_location",
                    "cited_text": "Water boils at 100°C",
                    "document_index": 0,
                    "document_title": "Physics Facts",
                    "start_char_index": 0,
                    "end_char_index": 19
                }
            }
        ))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    # Mock the final response choice to include provider_specific_fields with citations
    with mock.patch("litellm.acompletion", return_value=citation_stream()):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="citations"),
            ],
        )

        # Create test documents
        docs = [dspy.Document(data="Water boils at 100°C at standard pressure.", title="Physics Facts")]

        with dspy.context(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022", cache=False)):
            output = program(documents=docs, question="What temperature does water boil?")
            citation_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse) and value.signature_field_name == "citations":
                    citation_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final_prediction = value

            # Test that we received citation chunks from streaming
            assert len(citation_chunks) > 0
            citation_chunk = citation_chunks[0]
            assert isinstance(citation_chunk.chunk, databricks_dspy.DatabricksCitations)
            assert len(citation_chunk.chunk) == 1
            assert citation_chunk.chunk[0].cited_text == "Water boils at 100°C"
            assert citation_chunk.chunk[0].document_title == "Physics Facts"

            # Test that prediction contains the expected fields
            assert final_prediction is not None
            assert hasattr(final_prediction, "answer")
            assert hasattr(final_prediction, "citations")
