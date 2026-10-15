"""Unit tests for the project `.env` reader and DATABRICKS_CONFIG_PROFILE helpers."""

from __future__ import annotations

import pathlib

from databricks_agentbricks.projects.env_file import read_env, update_env_profile, write_env


def test_read_env_parses_pairs_and_export(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("# comment\n\nA=1\nexport B=2\nC='three'\n")

    assert read_env(tmp_path / ".env") == {"A": "1", "B": "2", "C": "three"}


def test_write_env_seeds_from_example(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\nOTHER=1\n")

    assert write_env(tmp_path, "ml") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\nOTHER=1\n"


def test_write_env_never_clobbers_existing(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=keepme\n")

    assert write_env(tmp_path, "ml") is False

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=keepme\n"


def test_update_env_profile_replaces_in_place_keeping_other_lines(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("A=1\nDATABRICKS_CONFIG_PROFILE=old\n# note\nB=2\n")

    assert update_env_profile(tmp_path, "new") is True

    assert (tmp_path / ".env").read_text() == "A=1\nDATABRICKS_CONFIG_PROFILE=new\n# note\nB=2\n"


# The writer matches what the reader honors (`export`, spacing, indentation), and drops later
# duplicates so the reader can't pick up a stale value.
def test_update_env_profile_rewrites_export_and_drops_duplicates(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text(
        "export DATABRICKS_CONFIG_PROFILE=old\nOTHER=1\n  DATABRICKS_CONFIG_PROFILE = stale\n"
    )

    assert update_env_profile(tmp_path, "new") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=new\nOTHER=1\n"


def test_update_env_profile_adds_missing_line(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("A=1\n")

    assert update_env_profile(tmp_path, "new") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=new\nA=1\n"


def test_update_env_profile_same_value_leaves_file_untouched(tmp_path: pathlib.Path):
    body = b"A=1\nDATABRICKS_CONFIG_PROFILE=ml\n"
    (tmp_path / ".env").write_bytes(body)

    assert update_env_profile(tmp_path, "ml") is False

    assert (tmp_path / ".env").read_bytes() == body
