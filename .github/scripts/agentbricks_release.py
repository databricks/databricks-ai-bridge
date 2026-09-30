#!/usr/bin/env python3
"""Plan an Agent Bricks release and stamp its package metadata."""

import argparse
import json
import re
from pathlib import Path


VERSION_PATTERN = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.dev(?P<dev>0|[1-9][0-9]*))?"
)
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
PROJECT_VERSION_PATTERN = re.compile(
    r'(?m)^(?P<prefix>version\s*=\s*")[^"]*(?P<suffix>"[^\n]*)$'
)
TEMPLATE_DEPENDENCY_PATTERN = re.compile(
    r'(?P<prefix>"databricks-agentbricks(?:\[[a-z]+\])?>=)'
    r'(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?)'
    r'(?P<suffix>")'
)
PACKAGE_PATH = Path("integrations/agentbricks/pyproject.toml")
TEMPLATE_BASE_PATH = Path("integrations/agentbricks/src/databricks_agentbricks/templates")


def parse_version(value: str, *, allow_dev: bool = False) -> tuple[int, int, int, int, int]:
    match = VERSION_PATTERN.fullmatch(value)
    if not match or (match.group("dev") and not allow_dev):
        raise ValueError(f"unsupported version: {value!r}")
    stage = -1 if match.group("dev") else 0
    number = int(match.group("dev") or "0")
    return (int(match.group("major")), int(match.group("minor")), int(match.group("patch")), stage, number)


def parse_sha(value: str) -> str:
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"expected a 40-character commit SHA: {value!r}")
    return value.lower()


def plan_release(
    version: str, base_sha: str, base_version: str, release_sha: str | None = None
) -> dict[str, str | bool]:
    target = parse_version(version)
    current = parse_version(base_version, allow_dev=True)
    base = parse_sha(base_sha)
    branch_head = parse_sha(release_sha) if release_sha else None
    if target < current:
        raise ValueError(f"release {version} would regress from {base_version}")
    if target[2] > 0 and branch_head is None:
        raise ValueError("patch releases require an existing release branch SHA")
    if branch_head is not None and target[:2] != current[:2]:
        raise ValueError("existing release branch version must share the target major and minor")
    major, minor, patch = target[:3]
    # Patches reuse the same release branch for their major and minor series.
    return {
        "version": version,
        "branch": f"release/databricks-agentbricks/v{major}.{minor}",
        "tag": f"databricks-agentbricks-v{version}",
        "base_sha": branch_head or base,
        "branch_exists": branch_head is not None,
        "next_dev_version": f"{major}.{minor + 1}.0.dev0",
    }


def _replace_package_version(text: str, version: str) -> str:
    project_match = re.search(r"(?m)^\[project\]\s*$", text)
    if not project_match:
        raise ValueError("package metadata has no [project] section")
    next_section = re.search(r"(?m)^\[", text[project_match.end() :])
    section_end = project_match.end() + next_section.start() if next_section else len(text)
    project_text = text[project_match.end() : section_end]
    updated, count = PROJECT_VERSION_PATTERN.subn(
        lambda match: f'{match.group("prefix")}{version}{match.group("suffix")}', project_text
    )
    if count != 1:
        raise ValueError("package metadata must have exactly one [project].version")
    return text[: project_match.end()] + updated + text[section_end:]


def stamp_release(root: Path, version: str, *, templates: bool = False) -> None:
    parse_version(version, allow_dev=not templates)
    package_path = root / PACKAGE_PATH
    updates = {package_path: _replace_package_version(package_path.read_text(), version)}
    if templates:
        template_paths = sorted((root / TEMPLATE_BASE_PATH).glob("*/pyproject.toml"))
        if not template_paths:
            raise ValueError("expected Agent Bricks template pyproject.toml files")
        for path in template_paths:
            text = path.read_text()
            updated, count = TEMPLATE_DEPENDENCY_PATTERN.subn(
                lambda match: f'{match.group("prefix")}{version}{match.group("suffix")}', text
            )
            if count != 1:
                raise ValueError(f"expected exactly one Agent Bricks dependency in {path}")
            updates[path] = updated
    for path, updated in updates.items():
        if path.read_text() != updated:
            path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print a read-only plan for dry runs and actual cuts")
    plan.add_argument("--version", required=True)
    plan.add_argument("--base-sha", required=True, help="commit SHA of main for a new branch")
    plan.add_argument("--release-sha", help="head SHA of an existing release branch")
    plan.add_argument("--base-version", required=True, help="version at the selected base SHA")
    stamp = commands.add_parser("stamp", help="stamp the package and optionally its templates")
    stamp.add_argument("--target-version", required=True)
    stamp.add_argument("--root", type=Path, required=True)
    stamp.add_argument("--templates", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(plan_release(args.version, args.base_sha, args.base_version, args.release_sha)))
        else:
            stamp_release(args.root, args.target_version, templates=args.templates)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
