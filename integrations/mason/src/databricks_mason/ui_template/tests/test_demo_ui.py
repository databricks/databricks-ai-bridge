from fastapi import FastAPI
from fastapi.testclient import TestClient
from runtime.ui import install_ui


def test_demo_ui_routes(monkeypatch):
    monkeypatch.delenv("MASON_DEMO_CRASH_ENABLED", raising=False)
    app = FastAPI()
    install_ui(app)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/ui-assets/app.js").status_code == 200

    config = client.get("/api/demo/config").json()
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["crash"]["enabled"] is False

    crash = client.post("/api/demo/crash")
    assert crash.status_code == 403
