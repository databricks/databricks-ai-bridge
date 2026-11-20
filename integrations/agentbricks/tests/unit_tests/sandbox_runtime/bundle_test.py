import io
import tarfile

import pytest

from databricks_agentkit.sandbox.bundle import build_bundle


@pytest.fixture
def source(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"\n')
    (tmp_path / "uv.lock").write_text("version=1\n")
    (tmp_path / "agent.py").write_text("print('hello')\n")
    return tmp_path


def names(bundle):
    with tarfile.open(fileobj=io.BytesIO(bundle.data)) as archive:
        return archive.getnames()


def test_bundle_excludes_environment_credentials_and_build_outputs(source):
    for directory in (".venv", ".git", "__pycache__", "dist"):
        (source / directory).mkdir()
        (source / directory / "private.py").write_text("do not copy")
    for filename in (".env", "credentials.key", "identity.pem"):
        (source / filename).write_text("do not copy")
    assert names(build_bundle(source)) == ["agent.py", "pyproject.toml", "uv.lock"]


def test_hash_tracks_contents_and_ignores_mtime(source):
    import os

    first = build_bundle(source)
    os.utime(source / "agent.py", (1, 1))
    assert build_bundle(source).digest == first.digest
    (source / "agent.py").write_text("print('changed')\n")
    assert build_bundle(source).digest != first.digest


def test_requires_lockfile(source):
    (source / "uv.lock").unlink()
    with pytest.raises(ValueError, match="uv.lock"):
        build_bundle(source)


def test_rejects_symlink_to_files_outside_source(source, tmp_path_factory):
    secret = tmp_path_factory.mktemp("outside") / "credentials"
    secret.write_text("do not copy")
    (source / "agent.py").unlink()
    (source / "agent.py").symlink_to(secret)
    with pytest.raises(ValueError, match="symlinks"):
        build_bundle(source)
