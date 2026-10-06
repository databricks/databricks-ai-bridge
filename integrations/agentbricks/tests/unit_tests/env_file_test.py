"""Unit tests for the project `.env` DATABRICKS_CONFIG_PROFILE helpers."""

from __future__ import annotations

import pathlib

from databricks_agentbricks.projects.env_file import update_env_profile, write_env


def test_write_env_seeds_profile_from_example(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text(
        "DATABRICKS_CONFIG_PROFILE=DEFAULT\n# MLFLOW_EXPERIMENT_ID=\n"
    )

    assert write_env(tmp_path, "ml") is True

    assert (tmp_path / ".env").read_text() == (
        "DATABRICKS_CONFIG_PROFILE=ml\n# MLFLOW_EXPERIMENT_ID=\n"
    )


def test_write_env_without_example_contains_only_the_profile(tmp_path: pathlib.Path):
    assert write_env(tmp_path, "ml") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\n"


def test_write_env_inserts_profile_when_example_lacks_it(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text("OTHER=1\n")

    write_env(tmp_path, "ml")

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\nOTHER=1\n"


def test_write_env_never_clobbers_existing(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=keepme\n")

    assert write_env(tmp_path, "ml") is False

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=keepme\n"


def test_update_env_profile_replaces_only_the_profile_line(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("A=1\nDATABRICKS_CONFIG_PROFILE=old\n# note\nB=2\n")

    assert update_env_profile(tmp_path, "new") is True

    assert (tmp_path / ".env").read_text() == "A=1\nDATABRICKS_CONFIG_PROFILE=new\n# note\nB=2\n"


def test_update_env_profile_same_value_is_unchanged(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("A=1\nDATABRICKS_CONFIG_PROFILE=same\n")

    assert update_env_profile(tmp_path, "same") is False


def test_update_env_profile_adds_missing_line(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("A=1\n")

    assert update_env_profile(tmp_path, "ml") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\nA=1\n"


def test_update_env_profile_seeds_env_when_missing(tmp_path: pathlib.Path):
    (tmp_path / ".env.example").write_text("KEEP=1\n")

    assert update_env_profile(tmp_path, "ml") is True

    assert (tmp_path / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\nKEEP=1\n"
