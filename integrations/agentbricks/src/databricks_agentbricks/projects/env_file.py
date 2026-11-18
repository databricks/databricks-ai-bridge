"""Reading and read-modify-writing of a project's local `.env` profile entry."""

from __future__ import annotations

import io
import os
import pathlib
import tempfile

from dotenv import dotenv_values
from dotenv.parser import parse_stream

_BOM = "\ufeff"


def _read_text(path: pathlib.Path) -> str:
    # surrogateescape keeps bytes that aren't UTF-8 (e.g. in a secret) intact through a rewrite.
    return path.read_text(encoding="utf-8", errors="surrogateescape")


def _write_text_atomically(path: pathlib.Path, text: str) -> None:
    """Replace `path` in one step so an interrupted write can't truncate a `.env` with secrets."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def read_env(path: pathlib.Path) -> dict[str, str]:
    """Read a project `.env` the way the generated agent's `load_dotenv` does."""
    try:
        text = _read_text(path)
    except OSError:
        return {}
    return _parse_env(text)


def _parse_env(text: str, *, interpolate: bool = True) -> dict[str, str]:
    # Same parser and default interpolation as the agent's `load_dotenv`, so both resolve one value.
    values = dotenv_values(stream=io.StringIO(text.removeprefix(_BOM)), interpolate=interpolate)
    return {key: value for key, value in values.items() if value is not None}


def _with_profile_binding(text: str, profile: str) -> str:
    """Rewrite parsed profile bindings while preserving all unrelated dotenv text verbatim."""
    bom = _BOM if text.startswith(_BOM) else ""
    body = text.removeprefix(_BOM)
    updated: list[str] = []
    replaced = False
    for binding in parse_stream(io.StringIO(body)):
        original = binding.original.string
        if not binding.error and binding.key == "DATABRICKS_CONFIG_PROFILE":
            if not replaced:
                ending = (
                    "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
                )
                updated.append(f"DATABRICKS_CONFIG_PROFILE={profile}{ending}")
                replaced = True
            continue
        updated.append(original)
    if not replaced:
        ending = "\r\n" if "\r\n" in body else "\n"
        updated.insert(0, f"DATABRICKS_CONFIG_PROFILE={profile}{ending}")
    return bom + "".join(updated)


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
    base = _read_text(example) if example.exists() else ""
    updated = _with_profile_binding(base, profile)
    _write_text_atomically(env_path, updated if updated.endswith("\n") else updated + "\n")
    return True


def update_env_profile(dest: pathlib.Path, profile: str) -> bool:
    """Point a project's `.env` at a different profile, preserving every other line.

    Returns True if the file changed (same profile -> False). A project without a `.env` gets one
    seeded from `.env.example`, matching what a fresh scaffold produces.
    """
    env_path = dest / ".env"
    if not env_path.exists():
        return write_env(dest, profile)
    before = _read_text(env_path)
    after = _with_profile_binding(before, profile)
    if after == before:
        return False
    _write_text_atomically(env_path, after)
    return True
