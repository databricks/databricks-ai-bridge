"""Check the chat UI can mint a session id where crypto.randomUUID is unavailable."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI_ROOT = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
TEMPLATES = ["agent-openai", "agent-langgraph"]


def _new_session_id_source(template):
    source = (UI_ROOT / template / "ui/app.js").read_text()
    match = re.search(r"^function newSessionId\(\) \{.*?^\}", source, re.M | re.S)
    assert match, "Missing newSessionId in app.js"
    return match.group()


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("secure_context", [True, False])
def test_new_session_id_is_a_v4_uuid(template, secure_context):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    # Insecure contexts (plain http on a non-localhost origin such as 0.0.0.0) expose
    # getRandomValues but not randomUUID.
    crypto_shim = (
        "const crypto = globalThis.crypto;"
        if secure_context
        else "const crypto = { getRandomValues: (array) => globalThis.crypto.getRandomValues(array) };"
    )
    program = (
        "import assert from 'node:assert/strict';\n"
        + crypto_shim
        + "\n"
        + _new_session_id_source(template)
        + r"""
const v4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const ids = new Set(Array.from({ length: 200 }, () => newSessionId()));
assert.equal(ids.size, 200);
for (const id of ids) assert.match(id, v4);
"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
