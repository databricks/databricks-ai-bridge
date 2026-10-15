from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import runtime.main as runtime


def test_foreground_invocation(monkeypatch) -> None:
    invoke = AsyncMock(return_value={"output": [{"role": "assistant", "content": "hello"}]})
    monkeypatch.setattr(runtime, "invoke", invoke)

    response = TestClient(runtime.app).post(
        "/invocations",
        json={"input": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"] == "hello"
    invoke.assert_awaited_once_with([{"role": "user", "content": "hi"}])


def test_custom_server_rejects_runtime_transport_fields() -> None:
    response = TestClient(runtime.app).post(
        "/invocations",
        json={"input": [], "background": True},
    )

    assert response.status_code == 422
    assert "/api/invocations" not in runtime.app.openapi()["paths"]
