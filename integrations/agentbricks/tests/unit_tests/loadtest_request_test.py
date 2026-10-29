"""Identity lifetimes and ordinary HTTP call semantics."""

import json
import time
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from databricks_agentbricks.services.loadtest.request import materialize_body, perform_request
from databricks_agentbricks.services.loadtest.streaming import StreamError


def test_request_identity_changes_per_invocation_and_user_identity_stays_stable():
    template = json.dumps(
        {"request": "{{request_id}}", "legacy": "{{uuid}}", "user": "{{user_id}}"}
    )
    first_body = materialize_body(template, "virtual-user")
    second_body = materialize_body(template, "virtual-user")
    assert first_body is not None
    assert second_body is not None
    first = json.loads(first_body)
    second = json.loads(second_body)
    assert UUID(first["request"])
    assert first["request"] == first["legacy"]
    assert first["request"] != second["request"]
    assert first["user"] == second["user"] == "virtual-user"
    assert "{{request_id}}" in template
    assert materialize_body(None, "virtual-user") is None
    assert materialize_body("null", "virtual-user") == "null"


def _request():
    return {"url": "http://localhost/run", "method": "POST", "timeout": 3}


def test_accepted_response_is_one_http_call_without_following_urls():
    session = MagicMock()
    response = session.request.return_value.__enter__.return_value
    response.status_code = 202
    response.headers = {"Content-Type": "application/json", "X-MLflow-Trace-ID": "tr-header"}
    response.content = b'{"events_url":"https://example.invalid/events","output":"accepted"}'
    result = perform_request(session, _request(), None, time.perf_counter(), capture_output=True)
    assert result["outputs"] == "accepted"
    assert result["trace_id"] == "tr-header"
    assert session.request.call_count == 1
    assert session.request.call_args.kwargs["data"] is None
    assert session.request.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("status", [302, 400, 429, 503])
def test_failed_http_status_is_content_free(status):
    session = MagicMock()
    response = session.request.return_value.__enter__.return_value
    response.status_code = status
    response.content = b"private error response"
    with pytest.raises(StreamError, match=f"HTTP {status}") as raised:
        perform_request(session, _request(), "null", time.perf_counter(), capture_output=True)
    assert "private" not in str(raised.value)
    assert session.request.call_count == 1
