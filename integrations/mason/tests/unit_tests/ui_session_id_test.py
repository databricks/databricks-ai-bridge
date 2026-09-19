"""Browser-session ID generation works on local network origins without secure-context APIs."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("framework", ["agent-langgraph", "agent-openai"])
def test_session_id_falls_back_when_random_uuid_is_unavailable(framework: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js")
    app_js = (
        pathlib.Path(__file__).parents[2]
        / "src"
        / "databricks_mason"
        / "templates"
        / "ui"
        / framework
        / "ui"
        / "app.js"
    ).read_text()
    function = re.search(r"function newSessionId\(\) \{.*?\n\}", app_js, re.DOTALL)
    assert function, "newSessionId function not found"
    harness = f"""
      const vm = require("node:vm");
      const context = {{
        crypto: {{ getRandomValues(bytes) {{ bytes.fill(0xab); return bytes; }} }},
      }};
      vm.createContext(context);
      vm.runInContext({json.dumps(function.group(0) + "; result = newSessionId();")}, context);
      process.stdout.write(context.result);
    """

    result = subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)

    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", result.stdout
    )
