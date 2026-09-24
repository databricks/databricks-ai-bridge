"""Value object that centralizes the ``app.yaml`` parse/serialize contract shared by
``mason deploy`` and ``mason dev``.

Each caller keeps its own env policy:
- ``deploy`` uses ``parse_lenient`` (a non-dict document silently becomes an empty manifest)
  and ``upsert_env`` (drops non-dict entries, upserts by name, strips ``valueFrom``).
- ``dev`` uses ``parse`` (raises ``AgentCliError`` on invalid YAML, non-dict top level, or
  non-list env), reads env via ``raw_env`` (preserves non-dict entries), filters it with its
  own list comprehensions, then calls ``set_env`` with the result.

This split preserves two intentional divergences between the commands:
1. Parse strictness - deploy is lenient, dev is strict.
2. Non-dict env entry handling - deploy drops them during upsert, dev preserves them through
   its own filtering.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from databricks_mason.errors import AgentCliError

_SCAFFOLD_COMMAND = ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"]


class AppManifest:
    """A parsed ``app.yaml`` document that can be mutated and serialized back to YAML.

    Construct via ``parse`` (strict), ``parse_lenient`` (tolerant), or ``scaffold``
    (brand-new placeholder). Mutate via ``upsert_env`` or ``set_env``. Serialize via
    ``to_yaml``.
    """

    def __init__(self, doc: dict) -> None:
        self._doc = doc

    @classmethod
    def parse(cls, text: str, *, source: pathlib.Path) -> "AppManifest":
        """Strict parse (dev): raise AgentCliError on invalid YAML, non-dict top level, or
        non-list env."""
        try:
            doc = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise AgentCliError(f"Could not parse {source}: {exc}") from exc
        if not isinstance(doc, dict):
            raise AgentCliError(f"Invalid {source}: top level must be an object.")
        env = doc.get("env")
        if env is not None and not isinstance(env, list):
            raise AgentCliError(f"Invalid {source}: env must be a list.")
        return cls(doc)

    @classmethod
    def parse_lenient(cls, text: str) -> "AppManifest":
        """Tolerant parse (deploy): a non-dict document becomes an empty manifest."""
        loaded = yaml.safe_load(text)
        return cls(loaded if isinstance(loaded, dict) else {})

    @classmethod
    def scaffold(cls) -> "AppManifest":
        """A fresh manifest with a placeholder command and empty env (deploy's missing-file path)."""
        return cls({"command": _SCAFFOLD_COMMAND, "env": []})

    def raw_env(self) -> list:
        """The ``env`` block as a list (or [] when absent / not a list), entries un-normalized."""
        raw = self._doc.get("env")
        return raw if isinstance(raw, list) else []

    def upsert_env(self, updates: dict[str, str]) -> None:
        """deploy's env upsert: normalize to dict entries (dropping non-dicts), set value /
        drop valueFrom for an existing name, append for a new name."""
        env: list[dict[str, Any]] = [e for e in self.raw_env() if isinstance(e, dict)]
        by_name = {e.get("name"): e for e in env}
        for name, value in updates.items():
            if name in by_name:
                by_name[name]["value"] = value
                by_name[name].pop("valueFrom", None)
            else:
                env.append({"name": name, "value": value})
        self._doc["env"] = env

    def set_env(self, entries: list) -> None:
        """Replace the env block wholesale (dev's write path after its own filtering)."""
        self._doc["env"] = entries

    def to_yaml(self) -> str:
        """Serialize the manifest back to YAML with key order preserved."""
        return yaml.safe_dump(self._doc, sort_keys=False)
