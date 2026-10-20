import pytest
from fastapi.testclient import TestClient
from runtime.main import app


def test_custom_server_runs_native_agent():
    with TestClient(app) as client:
        response = client.post("/invocations", json={"input": [{"role": "user", "content": "Hi"}]})
    assert response.status_code == 200
    assert response.json()["output"] == [{"role": "assistant", "content": "Echo: Hi"}]


@pytest.mark.parametrize("body", [{}, {"input": []}, {"input": [{"role": "user", "content": " "}]}])
def test_custom_server_validates_input(body):
    with TestClient(app) as client:
        assert client.post("/invocations", json=body).status_code == 422
