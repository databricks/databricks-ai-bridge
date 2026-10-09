"""Offline adoption, discovery, and scaffold coverage for bundled workflow skills."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from databricks_agentbricks import skills as skill_mod
from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.runtime.app import _InvocationRequest


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        entry.relative_to(directory).as_posix(): entry.read_bytes()
        for entry in directory.rglob("*")
        if entry.is_file()
    }


def test_bundled_skill_has_portable_metadata_and_valid_references():
    manifest = skill_mod.bundled_workflow_skill()
    content = manifest.read_text()
    _, frontmatter, body = content.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == manifest.parent.name == "agent-bricks-workflow"
    assert 0 < len(metadata["description"]) <= 1024
    assert set(metadata) == {"name", "description"}
    references = re.findall(r"\]\((references/[^)]+)\)", body)
    assert references
    for relative in references:
        assert (manifest.parent / relative).is_file()


@pytest.mark.parametrize("actor_location", ["input", "top_level"])
def test_documented_memory_payload_obeys_runtime_actor_placement(actor_location):
    reference = skill_mod.bundled_workflow_skill().parent / "references/deployment.md"
    example = re.search(r"```json\n(.*?)\n```", reference.read_text(), re.DOTALL)
    assert example is not None
    payload = json.loads(example.group(1))
    if actor_location == "top_level":
        payload["actor"] = payload["input"].pop("actor")
        with pytest.raises(ValidationError) as raised:
            _InvocationRequest.model_validate(payload)
        assert raised.value.errors()[0]["type"] == "extra_forbidden"
        assert raised.value.errors()[0]["loc"] == ("actor",)
    else:
        request = _InvocationRequest.model_validate(payload)
        assert request.input == payload["input"]
        assert request.session_id == payload["session_id"]


def test_install_has_one_bundle_and_resolvable_compatibility_pointers(tmp_path):
    unrelated = tmp_path / ".claude/skills/other/SKILL.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("user-owned")
    manifests = skill_mod.install_workflow_skill(tmp_path)
    assert manifests == tuple(
        tmp_path / root / "skills/agent-bricks-workflow/SKILL.md"
        for root in (".agents", ".claude", ".agent")
    )
    assert _snapshot(manifests[0].parent) == _snapshot(skill_mod.bundled_workflow_skill().parent)
    for pointer in manifests[1:]:
        _, frontmatter, body = pointer.read_text().split("---", 2)
        assert yaml.safe_load(frontmatter)["name"] == "agent-bricks-workflow"
        link = re.search(r"\]\(([^)]+)\)", body)
        assert link is not None
        target = link.group(1)
        assert (pointer.parent / target).resolve() == manifests[0].resolve()
        assert not (pointer.parent / "references").exists()
    assert unrelated.read_text() == "user-owned"


def test_identical_installation_is_not_rewritten(tmp_path):
    manifests = skill_mod.install_workflow_skill(tmp_path)
    before = {manifest: manifest.stat().st_mtime_ns for manifest in manifests}
    assert skill_mod.install_workflow_skill(tmp_path) == manifests
    assert {manifest: manifest.stat().st_mtime_ns for manifest in manifests} == before


@pytest.mark.parametrize("root", [".agents", ".claude", ".agent"])
def test_conflicting_skills_are_preserved_without_partial_installation(tmp_path, root):
    conflict = tmp_path / root / "skills/agent-bricks-workflow/SKILL.md"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("user-owned")
    before = _snapshot(tmp_path)
    with pytest.raises(AgentCliError, match="already exist"):
        skill_mod.install_workflow_skill(tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "relative", [".agents", ".claude/skills", ".agent/skills/agent-bricks-workflow"]
)
def test_install_rejects_symlinked_discovery_directories(tmp_path, relative):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(AgentCliError, match="Cannot write skills"):
        skill_mod.install_workflow_skill(project)
    assert not list(outside.iterdir())
    assert not list(project.rglob("SKILL.md"))


def test_install_rolls_back_only_created_directories(tmp_path, monkeypatch):
    original = tmp_path / ".claude/skills/other/SKILL.md"
    original.parent.mkdir(parents=True)
    original.write_text("keep")
    copytree = skill_mod.shutil.copytree

    def fail_pointer(source, destination, *args, **kwargs):
        if Path(destination) == tmp_path / ".claude/skills/agent-bricks-workflow":
            raise OSError("disk full")
        return copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(skill_mod.shutil, "copytree", fail_pointer)
    with pytest.raises(OSError, match="disk full"):
        skill_mod.install_workflow_skill(tmp_path)
    assert _snapshot(tmp_path) == {".claude/skills/other/SKILL.md": b"keep"}
    assert not (tmp_path / ".agents").exists()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("server", ["agentbricks", "custom"])
def test_real_scaffolds_install_workflow_skills(tmp_path, framework, server):
    destination = tmp_path / "agent"
    result = CliRunner().invoke(
        agentbricks,
        ["-o", "json", "init", "--framework", framework, "--server", server, str(destination)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    manifest = Path(payload["workflow_skill"])
    assert manifest.is_file()
    assert len(payload["skill_pointers"]) == 2
    for pointer in payload["skill_pointers"]:
        assert Path(pointer).is_file()
    if server == "agentbricks":
        instructions = destination / "AGENTS.md"
        link = re.search(r"\]\((\.agents/[^)]+)\)", instructions.read_text())
        assert link is not None
        assert link.group(1) == (manifest.relative_to(destination).as_posix())
