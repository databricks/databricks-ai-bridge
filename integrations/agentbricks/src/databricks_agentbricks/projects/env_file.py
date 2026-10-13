"""Reading and read-modify-writing of a project's local `.env` profile entry."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile

# Same shape the reader accepts, so a line the CLI honors is a line it can rewrite.
_PROFILE_LINE = re.compile(r"\s*(export\s+)?DATABRICKS_CONFIG_PROFILE\s*=")


_BOM = "\ufeff"


def _read_text(path: pathlib.Path) -> str:
    # surrogateescape keeps bytes that aren't UTF-8 (e.g. in a secret) intact through a rewrite.
    return path.read_text(encoding="utf-8", errors="surrogateescape")


def _lines(text: str) -> list[str]:
    # Split on newlines only: str.splitlines() also breaks on control characters inside values.
    lines = [line.removesuffix("\r") for line in text.removeprefix(_BOM).split("\n")]
    return lines[:-1] if lines and lines[-1] == "" else lines


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
    """Minimal KEY=VALUE reader for a project `.env` (python-dotenv is not a CLI dependency)."""
    values: dict[str, str] = {}
    try:
        text = _read_text(path)
    except OSError:
        return values
    for line in _lines(text):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end != -1:
                # A quoted value ends at its matching close quote: a `#` inside is literal, and
                # anything after (an inline comment) is dropped.
                value = value[1:end]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        else:
            # Match dotenv for unquoted values: a `#` that follows whitespace starts a comment.
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if key:
            values[key] = value
    return values


def _with_profile_line(lines: list[str], profile: str) -> list[str]:
    """`lines` with the DATABRICKS_CONFIG_PROFILE entry set to `profile` (inserted when absent)."""
    updated, replaced = [], False
    for line in lines:
        if _PROFILE_LINE.match(line):
            if replaced:
                # Drop later duplicates so the reader can't pick up an old value.
                continue
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
    base = _lines(_read_text(example)) if example.exists() else []
    _write_text_atomically(env_path, "\n".join(_with_profile_line(base, profile)) + "\n")
    return True


def update_env_profile(dest: pathlib.Path, profile: str) -> bool:
    """Point a project's `.env` at a different profile, preserving every other line.

    Returns True if the file changed (same profile -> False). A project without a `.env` gets one
    seeded from `.env.example`, matching what a fresh scaffold produces.
    """
    env_path = dest / ".env"
    if not env_path.exists():
        return write_env(dest, profile)
    if read_env(env_path).get("DATABRICKS_CONFIG_PROFILE") == profile:
        # Compare the parsed profile, not the file text: a lone CRLF or trailing-newline
        # difference must not count as a change.
        return False
    before = _read_text(env_path)
    bom = _BOM if before.startswith(_BOM) else ""
    lines = _with_profile_line(_lines(before), profile)
    _write_text_atomically(env_path, bom + "\n".join(lines) + "\n")
    return True
