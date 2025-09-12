from unittest import mock

import dspy
import pydantic
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import databricks_dspy


def test_citation_validate_input():
    citation = databricks_dspy.DatabricksCitations.Citation(
        cited_text="The Earth orbits the Sun.",
        document_index=0,
        start_char_index=0,
        end_char_index=23,
        supported_text="The Earth orbits the Sun.",
    )
    assert citation.cited_text == "The Earth orbits the Sun."
    assert citation.document_index == 0
    assert citation.start_char_index == 0
    assert citation.end_char_index == 23
    assert citation.type == "char_location"
    assert citation.supported_text == "The Earth orbits the Sun."

    with pytest.raises(pydantic.ValidationError):
        databricks_dspy.DatabricksCitations.Citation(cited_text="text")


def test_citations_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        citations: databricks_dspy.DatabricksCitations

    citation = databricks_dspy.DatabricksCitations.Citation(
        cited_text="Hello, world!",
        document_index=0,
        start_char_index=0,
        end_char_index=13,
        supported_text="Hello, world!",
    )
    citations = databricks_dspy.DatabricksCitations(citations=[citation])
    wrapper = Wrapper(citations=citations)
    assert wrapper.citations.citations[0].cited_text == "Hello, world!"


def test_citation_with_all_fields():
    citation = databricks_dspy.DatabricksCitations.Citation(
        cited_text="Water boils at 100°C.",
        document_index=1,
        document_title="Physics Facts",
        start_char_index=10,
        end_char_index=31,
        supported_text="Water boils at 100°C.",
    )
    assert citation.cited_text == "Water boils at 100°C."
    assert citation.document_index == 1
    assert citation.document_title == "Physics Facts"
    assert citation.start_char_index == 10
    assert citation.end_char_index == 31
    assert citation.supported_text == "Water boils at 100°C."


def test_citation_format():
    citation = databricks_dspy.DatabricksCitations.Citation(
        cited_text="The sky is blue.",
        document_index=0,
        document_title="Weather Guide",
        start_char_index=5,
        end_char_index=21,
        supported_text="The sky is blue.",
    )

    formatted = citation.format()

    assert formatted["type"] == "char_location"
    assert formatted["cited_text"] == "The sky is blue."
    assert formatted["document_index"] == 0
    assert formatted["document_title"] == "Weather Guide"
    assert formatted["start_char_index"] == 5
    assert formatted["end_char_index"] == 21
    assert formatted["supported_text"] == "The sky is blue."


def test_citations_format():
    citations = databricks_dspy.DatabricksCitations(
        citations=[
            databricks_dspy.DatabricksCitations.Citation(
                cited_text="First citation",
                document_index=0,
                start_char_index=0,
                end_char_index=14,
                supported_text="First citation",
            ),
            databricks_dspy.DatabricksCitations.Citation(
                cited_text="Second citation",
                document_index=1,
                document_title="Source",
                start_char_index=20,
                end_char_index=35,
                supported_text="Second citation",
            ),
        ]
    )

    formatted = citations.format()

    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert formatted[0]["cited_text"] == "First citation"
    assert formatted[1]["cited_text"] == "Second citation"
    assert formatted[1]["document_title"] == "Source"


def test_citations_from_dict_list():
    citations_data = [
        {
            "cited_text": "The sky is blue",
            "document_index": 0,
            "document_title": "Weather Guide",
            "start_char_index": 0,
            "end_char_index": 15,
            "supported_text": "The sky was blue yesterday.",
        }
    ]

    citations = databricks_dspy.DatabricksCitations.from_dict_list(citations_data)

    assert len(citations.citations) == 1
    assert citations.citations[0].cited_text == "The sky is blue"
    assert citations.citations[0].document_title == "Weather Guide"


@pytest.mark.asyncio
@pytest.mark.skipif(
    dspy.__version__ <= "3.0.3",
    reason="Streaming with custom types is not supported in dspy < 3.0.3",
)
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
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" answer"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="Water"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" boils"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" at"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" 100°C"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="."))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="\n\n"))]
        )
        yield ModelResponseStream(
            model="claude",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content="",
                        provider_specific_fields={
                            "citation": {
                                "type": "char_location",
                                "cited_text": "Water boils at 100°C",
                                "document_index": 0,
                                "document_title": "Physics Facts",
                                "start_char_index": 0,
                                "end_char_index": 19,
                            }
                        },
                    )
                )
            ],
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="\n\n"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" completed"))]
        )
        yield ModelResponseStream(
            model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))]
        )

    # Mock the final response choice to include provider_specific_fields with citations
    with mock.patch("litellm.acompletion", return_value=citation_stream()):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="citations"),
            ],
        )

        # Create test documents
        docs = [
            databricks_dspy.DatabricksDocument(
                data="Water boils at 100°C at standard pressure.", title="Physics Facts"
            )
        ]

        with dspy.context(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022", cache=False)):
            output = program(documents=docs, question="What temperature does water boil?")
            citation_chunks = []
            final_prediction = None
            async for value in output:
                if (
                    isinstance(value, dspy.streaming.StreamResponse)
                    and value.signature_field_name == "citations"
                ):
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
