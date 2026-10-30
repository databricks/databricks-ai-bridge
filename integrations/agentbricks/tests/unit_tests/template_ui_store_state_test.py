"""Check the chat UI's Memory/Session connection copy in both shipped templates.

`agentbricks dev` strips the store env vars, so a store that agent.toml declares is not connected
locally. The cards must say "Connects on deploy" then, and suggest `bind` only when nothing is
declared. These tests run the shipped helper functions from app.js under Node.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI_ROOT = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
TEMPLATES = ["agent-openai", "agent-langgraph"]

_HARNESS = r"""
const config = (memory, session, deployed) => ({ deployed, memory, session });
const memory = (extra) => ({ enabled: false, declared: null, actor: "alice", store: null, ...extra });
const session = (extra) => ({ managed: false, declared: null, actor: "alice", ...extra });
const out = {};
for (const [name, cfg] of Object.entries(CASES)) {
  const m = memoryConnection(cfg);
  const s = sessionConnection(cfg);
  out[name] = { label: memoryModeLabel(cfg, m), orb: memoryCapabilityState(m), memory: m, session: s };
}
console.log(JSON.stringify(out));
"""

_CASES = r"""
const CASES = {
  connected: config(memory({ enabled: true, declared: "mem", store: "memory-stores/mem" }), session({ managed: true, declared: "ses" }), false),
  declaredLocal: config(memory({ declared: "mem" }), session({ declared: "ses" }), false),
  declaredDeployed: config(memory({ declared: "mem" }), session({ declared: "ses" }), true),
  none: config(memory(), session(), false),
  noneDeployed: config(memory(), session(), true),
};
"""


def _run(template: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (UI_ROOT / template / "ui/app.js").read_text()
    match = re.search(
        r"^// ---- Store connection state.*?(?=^async function loadConfig)", source, re.M | re.S
    )
    assert match, "Missing store connection helpers in app.js"
    # The `config`/`memory`/`session` builders must exist before CASES is built.
    helpers, rest = _HARNESS.split("const out", 1)
    program = match.group() + helpers + _CASES + "const out" + rest
    result = subprocess.run(
        [node, "--input-type=module", "-e", program], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("template", TEMPLATES)
def test_memory_card_label_has_four_states(template):
    cases = _run(template)
    assert cases["connected"]["label"] == "Connected · actor alice"
    assert cases["declaredLocal"]["label"] == "Connects on deploy"
    assert cases["declaredDeployed"]["label"] == "Not connected"
    assert cases["none"]["label"] == "Not connected"
    assert cases["noneDeployed"]["label"] == "Not connected"


@pytest.mark.parametrize("template", TEMPLATES)
def test_memory_orb_is_degraded_only_while_waiting_for_deploy(template):
    cases = _run(template)
    orbs = {name: case["orb"] for name, case in cases.items()}
    assert orbs == {
        "connected": True,
        "declaredLocal": "degraded",
        "declaredDeployed": False,
        "none": False,
        "noneDeployed": False,
    }


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("kind", ["memory", "session"])
def test_connected_store_has_no_hint(template, kind):
    connection = _run(template)["connected"][kind]
    assert connection == {"phase": "connected", "detail": None, "command": None}


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize(
    ("kind", "bind", "env"),
    [
        ("memory", "agentbricks memory bind <store-name>", "AGENT_MEMORY_STORE"),
        ("session", "agentbricks sessions bind <store-name>", "AGENT_SESSION_STORE"),
    ],
)
def test_bind_hint_only_when_nothing_is_declared(template, kind, bind, env):
    cases = _run(template)
    for name in ("none", "noneDeployed"):
        assert cases[name][kind] == {"phase": "unbound", "detail": None, "command": bind}

    local = cases["declaredLocal"][kind]
    assert local["phase"] == "on-deploy"
    assert local["command"] is None
    assert "bind" not in local["detail"]
    assert "off by default in `agentbricks dev`" in local["detail"]
    assert "connects after `agentbricks deploy`" in local["detail"]
    assert env in local["detail"]
    name = "mem" if kind == "memory" else "ses"
    assert f'"{name}"' in local["detail"]

    deployed = cases["declaredDeployed"][kind]
    assert deployed["phase"] == "redeploy"
    assert deployed["command"] == "agentbricks deploy"
    assert "bind" not in deployed["detail"]
    assert f'"{name}"' in deployed["detail"]


@pytest.mark.parametrize("template", TEMPLATES)
def test_store_copy_is_rendered_with_text_content_only(template):
    source = (UI_ROOT / template / "ui/app.js").read_text()
    assert 'elements.memoryMode.setAttribute("title", memoryConnectionState.detail)' in source
    assert "elements.memoryMode.textContent = memoryModeLabel(" in source
    assert "memoryMessage(\n      connection.detail ||" in source
    assert "innerHTML = memoryConnectionState" not in source
