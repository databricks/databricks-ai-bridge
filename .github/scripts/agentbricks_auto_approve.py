"""Decide whether AgentBricksReviewer may auto-approve a PR.

The decision uses only facts the workflow can check itself: the posted review's
score and finding sections, and the files and lines the diff changes. It never
takes the model's word that a PR is safe to approve.

Usage: agentbricks_auto_approve.py --review REVIEW --diff DIFF [--min-score N] [--max-lines N]
Prints a JSON object {"approve": bool, "reasons": [...]}; reasons explain a refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

MERGE_SAFETY = re.compile(
    r"\*\*Merge safety:\*\*\s*(?P<score>\d+)/100\s*\((?P<risk>[A-Z ]+)\)"
    r"\s*·\s*reviewers:\s*(?P<delivered>\d+)/(?P<dispatched>\d+)"
)
EMPTY_SECTIONS = ("Blocking issues", "Security vulnerabilities", "Uncertain findings")
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)

# Only the agentbricks integration (its code, tests, and docs) and docs elsewhere can
# be auto-approved; the other packages ship on their own and need a human.
AUTO_APPROVE_PREFIXES = ("integrations/agentbricks/",)
DOC_SUFFIXES = frozenset({".md", ".rst"})

# Changes here need a human: CI and review rules, dependencies, and credential code.
PROTECTED_PREFIXES = (".github/", ".claude/")
PROTECTED_NAMES = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "CODEOWNERS",
    }
)
SENSITIVE_NAME = re.compile(r"credential|oauth|secret|token|(?:^|[_.-])auth(?:[_.-]|$)", re.IGNORECASE)


def changed_files_and_lines(diff: str) -> tuple[list[str], int]:
    """Paths touched by a unified diff (old and new names) and its changed line count."""
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", diff, re.MULTILINE):
        for path in match.groups():
            if path not in files:
                files.append(path)
    lines = sum(
        1
        for line in diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++ ", "--- "))
    )
    return files, lines


def protected_reason(path: str) -> str | None:
    pure = PurePosixPath(path)
    name = pure.name
    if path.startswith(PROTECTED_PREFIXES) or name == "REVIEW.md":
        return f"{path} is CI or review configuration"
    if name in PROTECTED_NAMES or name.endswith(".lock") or re.fullmatch(r"requirements.*\.(txt|in)", name):
        return f"{path} is a dependency manifest or lockfile"
    if SENSITIVE_NAME.search(name):
        return f"{path} looks like credential or auth code"
    if not path.startswith(AUTO_APPROVE_PREFIXES) and pure.suffix.lower() not in DOC_SUFFIXES:
        return f"{path} is outside integrations/agentbricks/ and isn't a doc"
    return None


def section_body(review: str, title: str) -> str | None:
    """Text under the heading named title, up to the next heading; None if missing."""
    headings = list(HEADING.finditer(review))
    for index, heading in enumerate(headings):
        if heading.group("title").strip().lower() == title.lower():
            end = headings[index + 1].start() if index + 1 < len(headings) else len(review)
            return review[heading.end() : end].strip()
    return None


def decide(review: str, diff: str, min_score: int = 95, max_lines: int = 200) -> dict:
    reasons: list[str] = []

    summary = MERGE_SAFETY.search(review)
    if not summary:
        reasons.append("the review has no merge safety line")
    else:
        score = int(summary.group("score"))
        delivered, dispatched = int(summary.group("delivered")), int(summary.group("dispatched"))
        if score < min_score:
            reasons.append(f"merge safety {score}/100 is below {min_score}")
        if delivered < 2 or delivered != dispatched:
            reasons.append(f"only {delivered} of {dispatched} reviewers delivered (need all, and at least 2)")

    for title in EMPTY_SECTIONS:
        body = section_body(review, title)
        if body is None:
            reasons.append(f'the review has no "{title}" section')
        elif body.rstrip(".").strip().lower() != "none":
            reasons.append(f'"{title}" is not empty')

    files, lines = changed_files_and_lines(diff)
    if not files:
        reasons.append("the diff changes no files")
    if lines > max_lines:
        reasons.append(f"{lines} changed lines is over the {max_lines}-line limit")
    reasons.extend(reason for path in files if (reason := protected_reason(path)))

    return {"approve": not reasons, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--min-score", type=int, default=95)
    parser.add_argument("--max-lines", type=int, default=200)
    args = parser.parse_args(argv)
    result = decide(
        args.review.read_text(encoding="utf-8"),
        args.diff.read_text(encoding="utf-8", errors="replace"),
        min_score=args.min_score,
        max_lines=args.max_lines,
    )
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
