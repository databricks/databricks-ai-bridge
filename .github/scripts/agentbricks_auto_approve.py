"""Decide whether AgentBricksReviewer may auto-approve a PR.

The decision combines the review's findings/coverage score and independent
change-safety assessment with deterministic path, component, and size checks.

Usage: agentbricks_auto_approve.py --review REVIEW --diff DIFF [--min-score N] [--max-lines N]
Prints a JSON object {"approve": bool, "reasons": [...]}; reasons explain a refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

REVIEW_SCORE = re.compile(
    r"\*\*Review score:\*\*\s*(?P<score>\d+)/100"
    r"\s*·\s*reviewers:\s*(?P<delivered>\d+)/(?P<dispatched>\d+)"
)
EMPTY_SECTIONS = ("Blocking issues", "Security vulnerabilities", "Uncertain findings")
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)
CHANGE_SAFETY = re.compile(
    r"^\*\*Change safety:\*\*\s*(?P<score>\d+)/100\s*\((?P<risk>LOW|MEDIUM|HIGH) RISK\)\s*$",
    re.MULTILINE,
)
RISK_LEVELS = ("low", "medium", "high")

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


def assess_change_risk(review: str, files: list[str], lines: int, diff: str = "") -> dict:
    """Keep contextual model assessment separate from findings, with risk floors."""
    matches = list(CHANGE_SAFETY.finditer(review))
    reasons: list[str] = []
    score = None
    level = "high"
    if len(matches) == 1:
        score = int(matches[0].group("score"))
        expected = "low" if score >= 90 else "medium" if score >= 60 else "high"
        if not 0 <= score <= 100 or matches[0].group("risk").lower() != expected:
            reasons.append("change-safety score or risk band is invalid")
            score = None
        else:
            level = expected
    else:
        reasons.append("the review must contain exactly one change-safety assessment")
    model_score = score

    def floor(minimum: str, reason: str):
        nonlocal level
        if RISK_LEVELS.index(minimum) > RISK_LEVELS.index(level):
            level = minimum
        reasons.append(reason)

    if not files:
        floor("high", "no changed paths are available")
    components = set()
    for path in files:
        pure = PurePosixPath(path)
        protected = protected_reason(path)
        if protected and "outside integrations/agentbricks/" not in protected:
            floor("high", protected)
        if pure.suffix.lower() in DOC_SUFFIXES or "tests" in pure.parts:
            continue
        parts = set(pure.parts)
        if parts & {"auth", "oauth", "credentials", "secrets", "migrations", "deployment", "deploy", "release", "releases"} or pure.stem.startswith("release_"):
            floor("high", f"{path} touches security, migration, deployment, or release behavior")
        elif pure.name == "__init__.py" or pure.stem.endswith(("_store", "_store_client")) or parts & {"cli", "api", "runtime", "databricks_agentkit", "checkpoint", "checkpoints", "storage", "memory", "sessions"}:
            floor("medium", f"{path} touches a public interface, runtime, or persisted state")
        if len(pure.parts) >= 2 and pure.parts[0] == "integrations":
            components.add(pure.parts[1])
        elif path.startswith("src/databricks_ai_bridge/"):
            components.add("core")
        elif path.startswith("databricks_mcp/"):
            components.add("mcp")
    if len(components) > 1:
        floor("medium", "production changes span multiple packages: " + ", ".join(sorted(components)))
    if lines > 1000 or len(files) > 30:
        floor("high", "the change exceeds 1000 lines or 30 paths")
    elif len(files) > 10:
        floor("medium", "the change exceeds 10 paths")
    if "GIT binary patch" in diff or re.search(r"^Binary files .+ differ$", diff, re.MULTILINE):
        floor("high", "binary changes need human review")
    if score is not None:
        score = min(score, {"low": 100, "medium": 89, "high": 59}[level])
    return {
        "level": level, "label": f"risk:{level}", "safety_score": score,
        "model_safety_score": model_score, "reasons": list(dict.fromkeys(reasons)),
    }


def decide(review: str, diff: str, min_score: int = 95, max_lines: int = 200,
           paths: list[str] | None = None) -> dict:
    reasons: list[str] = []

    summary = REVIEW_SCORE.search(review)
    if not summary:
        reasons.append("the review has no review score line")
    else:
        score = int(summary.group("score"))
        delivered, dispatched = int(summary.group("delivered")), int(summary.group("dispatched"))
        if score < min_score:
            reasons.append(f"review score {score}/100 is below {min_score}")
        if delivered < 2 or delivered != dispatched:
            reasons.append(f"only {delivered} of {dispatched} reviewers delivered (need all, and at least 2)")

    for title in EMPTY_SECTIONS:
        body = section_body(review, title)
        if body is None:
            reasons.append(f'the review has no "{title}" section')
        elif body.rstrip(".").strip().lower() != "none":
            reasons.append(f'"{title}" is not empty')

    files, lines = changed_files_and_lines(diff)
    if paths is not None:
        files = paths
    if not files:
        reasons.append("the diff changes no files")
    if lines > max_lines:
        reasons.append(f"{lines} changed lines is over the {max_lines}-line limit")
    reasons.extend(reason for path in files if (reason := protected_reason(path)))
    risk = assess_change_risk(review, files, lines, diff)
    if risk["level"] != "low":
        reasons.append(f"change risk is {risk['level']}; automatic approval requires low risk")

    return {"approve": not reasons, "reasons": reasons, "risk": risk}


def safety_section(review: str, decision: dict) -> str:
    risk = decision["risk"]
    summary = REVIEW_SCORE.search(review)
    review_score = f"{summary.group('score')}/100" if summary else "unavailable"
    safety = f"{risk['safety_score']}/100" if risk["safety_score"] is not None else "unrated"
    rationale = re.search(
        r"^\*\*Risk rationale:\*\*\s*(.*?)(?=\n\s*\n|\n## |\Z)", review,
        re.MULTILINE | re.DOTALL,
    )
    lines = [
        "### Safety assessment", "",
        "| Assessment | Result |", "|---|---|",
        f"| Review score (findings and coverage) | {review_score} |",
        f"| Change safety (impact of the change) | {safety} |",
        f"| Change risk | **{risk['level'].upper()}** · `{risk['label']}` |", "",
        "**Context:** " + (rationale[1].strip() if rationale else "The reviewer provided no change-risk rationale."),
    ]
    if risk["model_safety_score"] != risk["safety_score"]:
        lines += ["", f"Model change safety before CI adjustments: {risk['model_safety_score']}/100."]
    if risk["reasons"]:
        lines += ["", "**CI risk factors:**"] + ["- " + reason for reason in risk["reasons"]]
    lines += ["", "**Auto-approval content checks:** " + ("Passed." if decision["approve"] else "Not eligible.")]
    if decision["reasons"]:
        lines += ["- " + reason for reason in decision["reasons"]]
    lines += ["", "Final approval also requires automation to be enabled, an eligible author, a non-draft PR, and the reviewed head to remain current."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--min-score", type=int, default=95)
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--paths", type=Path, help="JSON changed paths from the frozen Git comparison")
    parser.add_argument("--safety-section", type=Path, help="Write the safety section for the PR comment")
    args = parser.parse_args(argv)
    review = args.review.read_text(encoding="utf-8")
    result = decide(
        review,
        args.diff.read_text(encoding="utf-8", errors="replace"),
        min_score=args.min_score,
        max_lines=args.max_lines,
        paths=json.loads(args.paths.read_text()) if args.paths else None,
    )
    if args.safety_section:
        args.safety_section.write_text(safety_section(review, result))
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
