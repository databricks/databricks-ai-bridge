"""Packaging smoke test: build the shipped wheel, install ONLY it, and drive `mason`.

The other functional tests run against an *editable* install — `uv run` installs the project
editable, and `cli_smoke_test` uses whatever `mason` sits next to the test interpreter. An editable
install exposes the whole source tree, so it can't catch a *packaging* regression: a module,
subpackage, or template that imports fine from source but is missing from the built distribution
(a stale `[tool.hatch...]` package/artifact glob, a subpackage with no `__init__`, template files
not declared as package data). This test closes that gap. It builds the wheel that would ship,
installs it — and nothing else — into an isolated venv, then runs the console entrypoint. A file
that isn't in the wheel makes it fail here even though every other tier stays green.

Base install only (no extras, no dev group): it also proves the CLI's declared base dependencies
are enough to load every command and scaffold a project, the way a fresh `pip install
databricks-mason` gets it.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

_MASON_PKG = pathlib.Path(__file__).resolve().parents[2]  # integrations/mason


@pytest.fixture(scope="session")
def wheel_mason(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build the wheel, install just it into a fresh venv, and return that venv's `mason` binary."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("requires uv on PATH")

    def _run(*args: str) -> None:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, (
            f"`{' '.join(args)}` exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    dist = tmp_path_factory.mktemp("dist")
    _run(uv, "build", "--wheel", "--out-dir", str(dist), str(_MASON_PKG))
    wheels = list(dist.glob("databricks_mason-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"

    venv = tmp_path_factory.mktemp("venv")
    _run(uv, "venv", str(venv))
    venv_python = venv / "bin" / "python"
    # Install the wheel and nothing else — no editable project, no dev/test group, no extras.
    _run(uv, "pip", "install", "--python", str(venv_python), str(wheels[0]))

    mason = venv / "bin" / "mason"
    assert mason.is_file(), "the wheel did not install a `mason` console script"
    return mason


def _mason(mason: pathlib.Path, *args: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    empty_cfg = tmp_path / "empty.databrickscfg"
    empty_cfg.write_text("")
    # env= replaces the environment wholesale so no ambient DATABRICKS_* / profile leaks in.
    env = {
        "PATH": f"{mason.parent}:/usr/bin:/bin",
        "HOME": str(home),
        "DATABRICKS_CONFIG_FILE": str(empty_cfg),
    }
    result = subprocess.run(
        [str(mason), *args], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (
        f"`mason {' '.join(args)}` exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def test_wheel_help_loads_every_command(wheel_mason: pathlib.Path, tmp_path: pathlib.Path) -> None:
    # `--help` builds the whole command tree, so it imports every command module as shipped: a
    # subpackage or module missing from the wheel fails here.
    output = _mason(wheel_mason, "--help", tmp_path=tmp_path).stdout
    assert "Usage" in output
    for command in ("init", "deploy", "dev", "tools", "memory", "sessions", "tracing", "endpoint"):
        assert command in output, f"`{command}` missing from --help — a command module didn't ship"


def test_wheel_init_scaffolds_from_packaged_templates(
    wheel_mason: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # `mason init` copies templates that ship as package data; if they're not in the wheel, the
    # scaffold is missing files even though it works from a source checkout.
    dest = tmp_path / "agent"
    _mason(wheel_mason, "init", "--framework", "langgraph", str(dest), tmp_path=tmp_path)
    for expected in ("pyproject.toml", "app.yaml", "agent.toml"):
        assert (dest / expected).is_file(), f"scaffold missing {expected} — template not packaged"
