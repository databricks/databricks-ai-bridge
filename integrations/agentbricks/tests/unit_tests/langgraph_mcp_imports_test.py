"""Unconfigured MCP support must not initialize the optional protocol stack."""

import subprocess
import sys
from pathlib import Path


def test_empty_manifest_and_template_do_not_load_mcp(tmp_path):
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
    )
    template = (
        Path(__file__).parents[2]
        / "src/databricks_agentbricks/templates/agent-langgraph/agent/mcps.py"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, os, runpy, sys
os.environ['AGENTBRICKS_PROJECT_ROOT'] = sys.argv[1]
from databricks_agentkit.langgraph import mcp_tools
assert asyncio.run(mcp_tools()) == []
template = runpy.run_path(sys.argv[2])
assert template['build_mcp_servers']() == []
for name in ('mcp', 'langchain_mcp_adapters', 'databricks_mcp',
             'databricks_langchain.multi_server_mcp_client'):
    assert name not in sys.modules, name
""",
            str(tmp_path),
            str(template),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
