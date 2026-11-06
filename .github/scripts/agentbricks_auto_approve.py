"""Classify a PR review and decide whether PRReviewer may auto-approve it.

The decision uses only facts the workflow can check itself: the posted review's
score and finding sections, and the files and lines the diff changes. The risk
label copies the model's score bands; approval adds deterministic safeguards.

Usage: agentbricks_auto_approve.py --review REVIEW --diff DIFF [--min-score N] [--max-lines N]
Prints JSON with the score, risk label, approval decision, and refusal reasons.
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
AGENTBRICKS_PREFIX = "integrations/agentbricks/"


def risk_label(score: int) -> str:
    """Map the model's merge-safety score to the requested three PR labels."""
    if score >= 90:
        return "review-risk: low"
    if score >= 70:
        return "review-risk: medium"
    return "review-risk: high"


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
    name = PurePosixPath(path).name
    if path.startswith(PROTECTED_PREFIXES):
        return f"{path} is CI or review configuration"
    if name in PROTECTED_NAMES or name.endswith(".lock") or re.fullmatch(r"requirements.*\.(txt|in)", name):
        return f"{path} is a dependency manifest or lockfile"
    if SENSITIVE_NAME.search(name):
        return f"{path} looks like credential or auth code"
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
    score: int | None = None
    label: str | None = None

    summary = MERGE_SAFETY.search(review)
    if not summary:
        reasons.append("the review has no merge safety line")
    else:
        score = int(summary.group("score"))
        label = risk_label(score)
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
    outside_scope = [path for path in files if not path.startswith(AGENTBRICKS_PREFIX)]
    if outside_scope:
        reasons.append(
            "the diff includes files outside integrations/agentbricks/**: "
            + ", ".join(outside_scope)
        )
    if lines > max_lines:
        reasons.append(f"{lines} changed lines is over the {max_lines}-line limit")
    reasons.extend(reason for path in files if (reason := protected_reason(path)))

    return {"score": score, "risk_label": label, "approve": not reasons, "reasons": reasons}


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
