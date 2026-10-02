"""Evaluation validates datasets and grades final assistant answers, not transport success."""

import json

import pytest

from databricks_agentbricks import evaluation
from databricks_agentbricks.cli.endpoint_transport import EndpointResponse
from databricks_agentbricks.cli.init import _copy_packaged_template


def answer(value="hello", *, status="completed"):
    return {
        "status": "completed",
        "output": {
            "status": status,
            "output": [
                {"role": "tool", "content": "tool result must not pass a case"},
                {"role": "assistant", "content": value},
            ],
        },
    }


def test_extracts_only_final_assistant_text():
    assert evaluation.response_text(answer()) == "hello"
    assert (
        evaluation.response_text(
            answer(
                [
                    {"type": "text", "text": "visible"},
                    {"type": "reasoning", "text": "opaque"},
                    {"type": "output_text", "text": "answer"},
                ]
            )
        )
        == "visible\nanswer"
    )
    with pytest.raises(ValueError, match="approval"):
        evaluation.response_text(answer(status="interrupted"))
    with pytest.raises(ValueError, match="successfully"):
        evaluation.response_text({"status": "failed", "error": "boom"})


def test_langgraph_wire_format_extracts_ai_messages_and_excludes_tools():
    response = {
        "status": "completed",
        "output": {
            "status": "completed",
            "output": [
                {"type": "human", "content": "What is 2 + 2?"},
                {"type": "tool", "content": "incorrect tool result"},
                {"type": "ai", "content": "4"},
            ],
        },
    }
    assert evaluation.response_text(response) == "4"


def test_real_request_contract_uses_unique_sessions_and_keeps_project_model(monkeypatch):
    requests = []

    def send(self, request):
        requests.append(request)
        return EndpointResponse(request.url, 200, {}, answer(), 0.1)

    monkeypatch.setattr(evaluation.HttpSession, "send", send)
    for _ in range(2):
        assert (
            evaluation.invoke_case("http://localhost:8000", "hello", authorization=None, timeout=5)[
                "answer"
            ]
            == "hello"
        )
    assert requests[0].body["session_id"] != requests[1].body["session_id"]
    assert requests[0].body["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert requests[0].headers["X-Routing-Key"] == requests[0].body["session_id"]


def test_http_error_record_does_not_log_sensitive_response_body(monkeypatch):
    monkeypatch.setattr(
        evaluation.HttpSession,
        "send",
        lambda _, r: EndpointResponse(r.url, 401, {}, "credential-containing-auth-page", 0.1),
    )
    output = evaluation.invoke_case(
        "http://localhost:8000", "hello", authorization="Bearer secret", timeout=5
    )
    assert "HTTP 401" in output["error"]
    assert "secret" not in str(output)
    assert "credential" not in str(output)


@pytest.mark.parametrize(
    "case",
    [{}, {"inputs": {"query": " "}}, {"inputs": {"query": "x"}, "expectations": {"contains": "x"}}],
)
def test_invalid_dataset_fails_before_invocation(tmp_path, case):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="Invalid evaluation case"):
        evaluation.load_cases(path)


def test_app_requires_explicit_profile():
    with pytest.raises(SystemExit) as error:
        evaluation.main(["--app", "test"])
    assert error.value.code == 2


@pytest.mark.parametrize("framework", ["openai", "langgraph"])
def test_generated_evaluator_does_not_require_unreleased_runtime_module(tmp_path, framework):
    destination = tmp_path / framework
    _copy_packaged_template(f"agent-{framework}", destination)
    source = (destination / "evals/run.py").read_text()
    assert "from databricks_agentbricks.evaluation" not in source
    assert "def run_evaluation(" in source
    assert len(evaluation.load_cases(destination / "evals/cases.jsonl")) == 2
