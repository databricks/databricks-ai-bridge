"""The published project's package index must survive sandbox bootstrap."""

import json
import shutil
import subprocess
import sys
import zipfile

import pytest

from databricks_agentkit.sandbox.bootstrap import SCRIPT
from databricks_agentkit.sandbox.bundle import bundle_files


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires uv")
@pytest.mark.parametrize("config_file", ["pyproject.toml", "uv.toml"])
def test_locked_install_uses_project_index_despite_sandbox_defaults(
    tmp_path, monkeypatch, config_file
):
    # A real, local package index makes a wrong index fail without external services.
    index = tmp_path / "index" / "demo-dependency"
    index.mkdir(parents=True)
    wheel = index / "demo_dependency-1.0-py3-none-any.whl"
    info = "demo_dependency-1.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo_dependency.py", "VALUE = 42\n")
        archive.writestr(
            f"{info}/METADATA", "Metadata-Version: 2.1\nName: demo-dependency\nVersion: 1.0\n"
        )
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    (index / "index.html").write_text(f'<a href="{wheel.name}">{wheel.name}</a>')
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="test-agent"\nversion="1"\nrequires-python=">=3.12"\n'
        'dependencies=["demo-dependency==1.0"]\n'
    )
    prefix = "tool.uv." if config_file == "pyproject.toml" else ""
    with (project / config_file).open("a") as config:
        config.write(f'[[{prefix}index]]\nurl="{index.parent.as_uri()}"\ndefault=true\n')
    monkeypatch.setenv("UV_OFFLINE", "1")
    for key in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL"):
        monkeypatch.delenv(key, raising=False)
    subprocess.run(["uv", "lock", "--python", "3.12"], cwd=project, check=True, capture_output=True)
    files = {path.name: path.read_bytes() for path in project.iterdir() if path.is_file()}
    bundle = bundle_files(files)
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "source.tar.gz").write_bytes(bundle.data)
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://image-default.invalid/simple")
    monkeypatch.setenv("UV_INDEX_URL", "https://image-default.invalid/simple")
    monkeypatch.setenv("UV_INDEX", "https://image-extra.invalid/simple")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://image-extra.invalid/simple")
    monkeypatch.setattr(sys, "argv", ["bootstrap", str(root), bundle.digest, "agent:invoke"])
    run = subprocess.run

    def run_phase(command, **kwargs):
        if command[1] == "sync":
            # Exercise the real locked install; project wheel building is unrelated.
            return run(command, **kwargs)
        if command[1] == "build":
            (root / "wheels").mkdir()
            (root / "wheels/agent.whl").write_bytes(b"unused")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run_phase)
    exec(SCRIPT, {})
    assert (root / "source/uv.lock").read_bytes() == files["uv.lock"]
    ready = json.loads((root / "ready.json").read_text())
    run(
        [ready["python"], "-c", "import demo_dependency; assert demo_dependency.VALUE == 42"],
        check=True,
        capture_output=True,
    )
