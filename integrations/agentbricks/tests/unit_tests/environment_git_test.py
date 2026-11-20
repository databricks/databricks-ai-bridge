import subprocess

import pytest

from databricks_agentbricks.environment_source import git_project_files, repository_url
from databricks_agentbricks.errors import AgentCliError


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    source = tmp_path / "nested" / "agent"
    source.mkdir(parents=True)
    (source / "pyproject.toml").write_text('[project]\nname="committed"\n')
    (source / "uv.lock").write_text("version=1\n")
    (source / "agent.py").write_text('VALUE = "version-one"\n')
    (source / ".env").write_text("PRIVATE=excluded")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        "Version one",
    )
    git("tag", "version-one")
    return source, git


def commit(git):
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        "Update",
    )


def test_git_export_pins_a_tag_subdirectory_and_ignores_dirty_files(repository):
    source, git = repository
    original = git("rev-parse", "version-one")
    (source / "agent.py").write_text('VALUE = "version-two"\n')
    commit(git)
    head = git("rev-parse", "HEAD")
    (source / "agent.py").write_text("dirty")
    (source / "untracked.py").write_text("untracked")
    git("remote", "add", "origin", "https://user:secret@example.com/team/repo.git?token=private")
    files, provenance = git_project_files(source, "version-one")
    assert files == {
        "pyproject.toml": b'[project]\nname="committed"\n',
        "uv.lock": b"version=1\n",
        "agent.py": b'VALUE = "version-one"\n',
    }
    assert provenance["commit"] == original
    assert provenance["subdirectory"] == "nested/agent"
    assert provenance["repository_url"] == "https://example.com/team/repo"
    assert git("rev-parse", "HEAD") == head
    assert (source / "agent.py").read_text() == "dirty"


def test_lockfile_must_exist_in_commit_even_when_present_in_working_tree(repository):
    source, git = repository
    git("rm", "nested/agent/uv.lock")
    commit(git)
    (source / "uv.lock").write_text("version=1\n")
    with pytest.raises(AgentCliError, match="Git revision must include uv.lock"):
        git_project_files(source, "HEAD")


def test_committed_symlink_is_rejected(repository):
    source, git = repository
    (source / "linked.py").symlink_to("agent.py")
    commit(git)
    with pytest.raises(AgentCliError, match="symlinks and submodules"):
        git_project_files(source, "HEAD")


def test_git_source_size_limit_applies_before_upload(repository, monkeypatch):
    from databricks_agentbricks import environment_source

    monkeypatch.setattr(environment_source, "MAX_BYTES", 1)
    with pytest.raises(AgentCliError, match="20 MiB"):
        git_project_files(repository[0], "HEAD")


@pytest.mark.parametrize(
    "remote,expected",
    [
        ("git@github.com:team/repo.git", "https://github.com/team/repo"),
        ("ssh://git@example.com/team/repo.git", "https://example.com/team/repo"),
        ("/local/repo", None),
        ("file:///local/repo", None),
    ],
)
def test_repository_provenance_contains_no_transport_identity(remote, expected):
    assert repository_url(remote) == expected
