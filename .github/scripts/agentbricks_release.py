#!/usr/bin/env python3
"""Plan an Agent Bricks release, stamp its package metadata, and draft its release notes."""

import argparse
import json
import re
import subprocess
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
PACKAGE_ROOT = Path("integrations/agentbricks")
TAG_PREFIX = "databricks-agentbricks-v"
# Commits the release automation itself makes on release branches and main.
AUTOMATION_SUBJECT_PATTERN = re.compile(r"(Release Agent Bricks|Set Agent Bricks development version to) ")
# Run summaries do not autolink bare issue references the way release pages do.
PULL_REQUEST_REFERENCE_PATTERN = re.compile(r"(?<![\w/\[])#(?P<number>[1-9][0-9]*)\b")


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


def previous_release_tag(version: str, tags: list[str]) -> str | None:
    target = parse_version(version)
    released = []
    for tag in tags:
        if not tag.startswith(TAG_PREFIX):
            continue
        try:
            parsed = parse_version(tag[len(TAG_PREFIX) :])
        except ValueError:
            continue
        if parsed < target:
            released.append((parsed, tag))
    return max(released)[1] if released else None


def render_release_notes(
    version: str, previous_tag: str | None, commits: list[str], repository: str
) -> str:
    lines = [f"## databricks-agentbricks {version}", ""]
    changes = [subject for subject in commits if not AUTOMATION_SUBJECT_PATTERN.match(subject)]
    if changes:
        pull_url = f"https://github.com/{repository}/pull"
        linked = [
            PULL_REQUEST_REFERENCE_PATTERN.sub(
                lambda match: f"[#{match['number']}]({pull_url}/{match['number']})", subject
            )
            for subject in changes
        ]
        lines += ["### Changes", "", *(f"- {subject}" for subject in linked), ""]
    else:
        lines += ["No package changes since the previous release.", ""]
    if previous_tag:
        compare = f"https://github.com/{repository}/compare/{previous_tag}...{TAG_PREFIX}{version}"
        lines.append(f"**Full changelog:** {compare}")
    else:
        lines.append("First tagged release of databricks-agentbricks.")
    return "\n".join(lines) + "\n"


def release_notes(root: Path, version: str, release_sha: str, repository: str) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout

    head = parse_sha(release_sha)
    previous_tag = previous_release_tag(version, git("tag", "--list", f"{TAG_PREFIX}*").split())
    log_args = ["log", "--no-merges", "--format=%s"]
    if previous_tag:
        # Release branches carry cherry-picked fixes; skip ones the previous release already shipped.
        log_args += ["--cherry-pick", "--right-only", f"{previous_tag}...{head}"]
    else:
        log_args.append(head)
    commits = git(*log_args, "--", str(PACKAGE_ROOT)).splitlines()
    return render_release_notes(version, previous_tag, commits, repository)


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
    notes = commands.add_parser("notes", help="print Markdown release notes for a release commit")
    notes.add_argument("--version", required=True)
    notes.add_argument("--release-sha", required=True, help="commit the release tag points to")
    notes.add_argument("--root", type=Path, required=True, help="repository checkout with release tags")
    notes.add_argument("--repository", required=True, help="GitHub owner/name for the changelog link")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(plan_release(args.version, args.base_sha, args.base_version, args.release_sha)))
        elif args.command == "notes":
            print(release_notes(args.root, args.version, args.release_sha, args.repository), end="")
        else:
            stamp_release(args.root, args.target_version, templates=args.templates)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
