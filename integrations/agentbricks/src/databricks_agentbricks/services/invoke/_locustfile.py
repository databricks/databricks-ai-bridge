"""Locust task loaded by ``python -m locust``, after gevent is initialized."""

import json
import sys
from urllib.parse import urlsplit
from uuid import uuid4

from locust import HttpUser, task

_CONFIG = json.load(sys.stdin)
_REQUEST = _CONFIG["request"]
_BODY = json.dumps(_REQUEST["body"]) if _REQUEST["body_set"] else None


class EndpointUser(HttpUser):
    host = _REQUEST["url"]

    def on_start(self):
        self.client.headers.update(_REQUEST["headers"])
        if _CONFIG["per_user_routing"]:
            self.client.headers["X-Routing-Key"] = str(uuid4())

    @task
    def invoke(self):
        # A placeholder lets durable agents receive a fresh invocation ID without
        # imposing an agent-specific request schema on this HTTP command.
        body = _BODY.replace("{{uuid}}", str(uuid4())) if _BODY is not None else None
        with self.client.request(
            _REQUEST["method"],
            _REQUEST["url"],
            name=urlsplit(_REQUEST["url"]).path or "/",
            data=body,
            timeout=_REQUEST["timeout"],
            allow_redirects=False,
            catch_response=True,
        ) as response:
            if response.status_code and not 200 <= response.status_code < 300:
                response.failure(f"HTTP {response.status_code}")
