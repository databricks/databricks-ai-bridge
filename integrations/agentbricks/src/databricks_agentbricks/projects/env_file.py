"""Read-modify-write of a project's local `.env` DATABRICKS_CONFIG_PROFILE entry."""

from __future__ import annotations

import pathlib


def _with_profile_line(lines: list[str], profile: str) -> list[str]:
    """`lines` with the DATABRICKS_CONFIG_PROFILE entry set to `profile` (inserted when absent)."""
    updated, replaced = [], False
    for line in lines:
        if line.startswith("DATABRICKS_CONFIG_PROFILE="):
            updated.append(f"DATABRICKS_CONFIG_PROFILE={profile}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"DATABRICKS_CONFIG_PROFILE={profile}")
    return updated


def write_env(dest: pathlib.Path, profile: str) -> bool:
    """Seed a local `.env` from `.env.example` with DATABRICKS_CONFIG_PROFILE=<profile>.

    Returns True if a `.env` was written. Skips if `.env` already exists (never clobbers). The
    template reads DATABRICKS_CONFIG_PROFILE for local model auth, so this makes the scaffolded
    project runnable with `agentbricks dev` without a manual `cp .env.example .env` step.
    """
    env_path = dest / ".env"
    if env_path.exists():
        return False
    example = dest / ".env.example"
    base = example.read_text().splitlines() if example.exists() else []
    env_path.write_text("\n".join(_with_profile_line(base, profile)) + "\n")
    return True


def update_env_profile(dest: pathlib.Path, profile: str) -> bool:
    """Point a project's `.env` at a different profile, preserving every other line.

    Returns True if the file changed (same profile -> False). A project without a `.env` gets one
    seeded from `.env.example`, matching what a fresh scaffold produces.
    """
    env_path = dest / ".env"
    if not env_path.exists():
        return write_env(dest, profile)
    before = env_path.read_text()
    after = "\n".join(_with_profile_line(before.splitlines(), profile)) + "\n"
    if after == before:
        return False
    env_path.write_text(after)
    return True
