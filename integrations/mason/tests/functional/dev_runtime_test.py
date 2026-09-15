"""Cross-process regression test for ``mason dev`` local durability selection."""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import sys

import yaml


def test_mason_dev_uses_in_memory_durability_across_cli_processes(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "agent"
    project.mkdir()
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\n\n'
        "[durability]\nenabled = true\n\n[tracing]\ndisabled = true\n"
    )
    probe = project / "probe.py"
    probe.write_text(
        "import os\n"
        "from databricks_mason import AgentApp\n"
        "from databricks_mason.agent_project import AgentProject\n"
        "enabled = AgentProject.load().durability_enabled\n"
        "app = AgentApp(durable_runtime=enabled)\n"
        "assert enabled is True\n"
        "assert os.environ['DATABRICKS_MASON_RUNTIME_LOCAL'] == 'true'\n"
        "assert 'DATABRICKS_MASON_RUNTIME_ENDPOINT' not in os.environ\n"
        "assert type(app._runtime.durability_store).__name__ == 'InMemoryDurabilityStore'\n"
        "print('local durable runtime: InMemoryDurabilityStore')\n"
    )
    original_manifest = {
        "command": [sys.executable, str(probe)],
        "env": [
            {"name": "PRESERVED", "value": "yes"},
            {"name": "PIP_INDEX_URL", "value": "https://deploy-only.invalid/simple"},
        ],
    }
    app_yaml = project / "app.yaml"
    app_yaml.write_text(yaml.safe_dump(original_manifest, sort_keys=False))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_databricks = bin_dir / "databricks"
    fake_databricks.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, subprocess, sys, yaml\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['apps', 'run-local'], args\n"
        "manifest = pathlib.Path(args[args.index('--entry-point') + 1])\n"
        "assert manifest.name == 'app.masondev.yaml'\n"
        "assert not manifest.is_absolute()\n"
        "doc = yaml.safe_load(manifest.read_text())\n"
        "env = os.environ.copy()\n"
        "env.update({item['name']: item['value'] for item in doc.get('env', [])})\n"
        "assert env['PRESERVED'] == 'yes'\n"
        "assert env['DATABRICKS_MASON_RUNTIME_LOCAL'] == 'true'\n"
        "assert env.get('PIP_INDEX_URL') != 'https://deploy-only.invalid/simple'\n"
        "raise SystemExit(subprocess.run(doc['command'], cwd=manifest.parent, env=env).returncode)\n"
    )
    fake_databricks.chmod(fake_databricks.stat().st_mode | stat.S_IXUSR)

    env = {key: value for key, value in os.environ.items() if not key.startswith("DATABRICKS_")}
    env["PATH"] = os.pathsep.join((str(bin_dir), env["PATH"]))
    config_file = tmp_path / "empty-databrickscfg"
    config_file.touch()
    env["DATABRICKS_CONFIG_FILE"] = str(config_file)
    mason = pathlib.Path(sys.executable).with_name("mason")
    assert mason.is_file()
    result = subprocess.run(
        [str(mason), "dev", "--source", str(project), "--no-prepare-environment"],
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "local durable runtime: InMemoryDurabilityStore" in result.stdout
    assert yaml.safe_load(app_yaml.read_text()) == original_manifest
    assert not (project / "app.masondev.yaml").exists()
