"""Decide whether AgentBricksReviewer may auto-approve a PR.

Approval uses the review's findings/coverage score and deterministic path and
size checks. Change-risk classification is reported independently as advisory.

Usage: agentbricks_auto_approve.py --review REVIEW --diff DIFF [--min-score N]
       [--max-lines N]
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
# Reviewers write sections as Markdown headings or as a bold label on its own line.
HEADING = re.compile(r"^(?:#{1,6}\s+(?P<title>.+?)|\*\*(?P<label>[^*\n]+?):?\*\*:?)\s*$", re.MULTILINE)
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
CHANGE_RISK = re.compile(r"^\*\*Change risk:\*\*[ \t]*(?P<risk>[^\r\n]*)$", re.MULTILINE)
RISK_RATIONALE = re.compile(r"^\*\*Risk rationale:\*\*[ \t]*(?P<text>[^\r\n]*)$", re.MULTILINE)
RISK_LEVELS = ("low", "medium", "high")
RISK_NAMES = {"medium": "MODERATE"}
RISK_VALUES = {"LOW": "low", "MODERATE": "medium", "HIGH": "high", "UNRATED": "unrated"}

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
    """Text under the heading or bold label named title, up to the next one; None if missing."""
    headings = list(HEADING.finditer(review))
    for index, heading in enumerate(headings):
        if (heading.group("title") or heading.group("label")).strip().lower() == title.lower():
            end = headings[index + 1].start() if index + 1 < len(headings) else len(review)
            return review[heading.end() : end].strip()
    return None


def section_is_empty(body: str) -> bool:
    """A section starting with "None" and listing no findings; findings are always list items."""
    return body.lower().startswith("none") and not LIST_ITEM.search(body)


def assess_change_risk(review: str, files: list[str], lines: int, diff: str = "") -> dict:
    """Keep contextual model assessment separate from findings, with risk floors."""
    matches = list(CHANGE_RISK.finditer(review))
    rationales = list(RISK_RATIONALE.finditer(review))
    reasons: list[str] = []
    level = "unrated"
    if len(matches) != 1:
        reasons.append("the review must contain exactly one change-risk assessment")
    elif matches[0].group("risk").strip().upper() not in RISK_VALUES:
        reasons.append("change risk must be LOW, MODERATE, HIGH, or UNRATED")
    else:
        level = RISK_VALUES[matches[0].group("risk").strip().upper()]
        if level == "unrated":
            reasons.append("the reviewer could not assess change risk")
    model_level = level
    if len(rationales) != 1 or not rationales[0].group("text").strip():
        level = "unrated"
        reasons.append("the review must contain exactly one non-empty risk rationale")

    def floor(minimum: str, reason: str):
        nonlocal level
        if level != "unrated" and RISK_LEVELS.index(minimum) > RISK_LEVELS.index(level):
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
        floor("high", "binary changes have high risk")
    return {
        "level": level, "label": f"risk:{level}", "model_level": model_level,
        "reasons": list(dict.fromkeys(reasons)),
    }


def decide(review: str, diff: str, min_score: int = 95, max_lines: int = 500,
           paths: list[str] | None = None) -> dict:
    reasons: list[str] = []

    summary = REVIEW_SCORE.search(review)
    if not summary:
        reasons.append("the review has no review score line")
    else:
        score = int(summary.group("score"))
        delivered, dispatched = int(summary.group("delivered")), int(summary.group("dispatched"))
        if not 0 <= score <= 100:
            reasons.append("review score is outside 0–100")
        if score < min_score:
            reasons.append(f"review score {score}/100 is below {min_score}")
        if delivered < 2 or delivered != dispatched:
            reasons.append(f"only {delivered} of {dispatched} reviewers delivered (need all, and at least 2)")

    for title in EMPTY_SECTIONS:
        body = section_body(review, title)
        if body is None:
            reasons.append(f'the review has no "{title}" section')
        elif not section_is_empty(body):
            reasons.append(f'"{title}" is not empty')

    review_reasons = list(reasons)
    files, lines = changed_files_and_lines(diff)
    if paths is not None:
        files = paths
    if not files:
        reasons.append("the diff changes no files")
    if lines > max_lines:
        reasons.append(f"{lines} changed lines is over the {max_lines}-line limit")
    reasons.extend(reason for path in files if (reason := protected_reason(path)))
    risk = assess_change_risk(review, files, lines, diff)

    return {"review_passed": not review_reasons, "review_reasons": review_reasons,
            "approve": not reasons, "reasons": reasons, "risk": risk}


FINDING_SECTIONS = ("Blocking issues", "Security vulnerabilities", "Non-blocking notes", "Uncertain findings")
# Path reasons, grouped in the comment so each file list appears once.
PATH_REASONS = (
    (" is CI or review configuration", "CI or review configuration files"),
    (" is a dependency manifest or lockfile", "Dependency manifests or lockfiles"),
    (" looks like credential or auth code", "Credential or auth code files"),
    (" is outside integrations/agentbricks/ and isn't a doc", "Files outside `integrations/agentbricks/` that aren't docs"),
)
LIST_LIMIT = 5


def _code_list(paths: list[str]) -> str:
    shown = ", ".join(f"`{path.replace('`', '')}`" for path in paths[:LIST_LIMIT])
    return shown + (f", and {len(paths) - LIST_LIMIT} more" if len(paths) > LIST_LIMIT else "")


def _group_reasons(reasons: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split reasons into per-path groups (by kind) and the rest, keeping order."""
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    for reason in reasons:
        for suffix, kind in PATH_REASONS:
            if reason.endswith(suffix):
                groups.setdefault(kind, []).append(reason[: -len(suffix)].strip("`"))
                break
        else:
            other.append(reason)
    return groups, other


def unblock_steps(decision: dict) -> list[str]:
    """What the author can do about each reason auto-approval was refused."""
    groups, other = _group_reasons(decision["reasons"])
    in_risk, _ = _group_reasons(decision["risk"]["reasons"])
    steps: list[str] = []
    open_sections = [match[1].lower() for r in other if (match := re.match(r'"(.+)" is not empty', r))]
    low_score = next((match for r in other if (match := re.match(r"review score \d+/100 is below (\d+)", r))), None)
    if open_sections or low_score:
        what = " and ".join(open_sections) if open_sections else "findings"
        target = f" (the score must reach {low_score[1]})" if low_score else ""
        steps.append(f"Fix the {what} above, or reply explaining why they don't apply, then comment `/review`{target}.")
    if any(r.startswith("only ") and "reviewers delivered" in r for r in other):
        steps.append("A reviewer didn't finish: comment `/review force` to run the review again.")
    if any(r.startswith(("the review ", "review score is outside", "the diff changes no files")) for r in other):
        steps.append("The review output was incomplete: comment `/review force` to run it again.")
    if size := next((match for r in other if (match := re.match(r"(\d+) changed lines is over the (\d+)-line limit", r))), None):
        steps.append(f"Split the PR to at most {size[2]} changed lines (it has {size[1]}).")
    for kind, paths in groups.items():
        listed = "see **Risk**" if kind in in_risk else _code_list(paths)
        steps.append(f"{kind} can't be auto-approved; move them to a separate PR ({listed}).")
    return steps


def overview(review: str, decision: dict) -> str:
    """The top of the PR comment: score and risk, findings, risk items, and how to unblock approval."""
    risk = decision["risk"]
    risk_name = RISK_NAMES.get(risk["level"], risk["level"].upper())
    score = REVIEW_SCORE.search(review)
    head = (f"**Review score:** {score['score']}/100 · reviewers: {score['delivered']}/{score['dispatched']}"
            if score else "**Review score:** unavailable")
    lines = [f"{head} · **Change risk:** {risk_name} · `{risk['label']}`", "", "### Summary", ""]

    findings = [(title, body) for title in FINDING_SECTIONS
                if (body := section_body(review, title)) and body.rstrip(".").strip().lower() != "none"]
    if findings:
        for title, body in findings:
            lines += [f"**{title}**", body, ""]
    else:
        lines += ["No findings.", ""]
    approach = section_body(review, "Approach")
    if approach:
        lines += ["**Approach:** " + approach.split("\n\n", 1)[0].strip(), ""]

    lines += ["### Risk", ""]
    rationale = re.search(r"^\*\*Risk rationale:\*\*\s*(.*?)(?=\n\s*\n|\n## |\Z)", review, re.MULTILINE | re.DOTALL)
    lines.append(rationale[1].strip() if rationale else "The reviewer gave no risk rationale.")
    groups, other = _group_reasons(risk["reasons"])
    items = [f"{kind}: {_code_list(paths)}" for kind, paths in groups.items()] + other
    if items:
        if risk["model_level"] != risk["level"]:
            model_name = RISK_NAMES.get(risk["model_level"], risk["model_level"].upper())
            lines += ["", f"CI raised the reviewer's {model_name} to {risk_name}:"]
        lines += [""] + ["- " + item for item in items]

    lines += ["", "### AI approval", ""]
    label = "Review checks passed (`ai-approved`)" if decision["review_passed"] else "Review checks not passed"
    if decision["approve"]:
        lines.append(f"{label}, and the PR is eligible for auto-approval. github-actions[bot] approves it "
                     "if the author has write access, the PR isn't a draft, and the reviewed commit is still the head.")
    else:
        lines += [f"{label}; not eligible for auto-approval. To unblock:", ""] + ["- " + step for step in unblock_steps(decision)]
        lines += ["", "Or ask a maintainer to approve."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--min-score", type=int, default=95)
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument("--paths", type=Path, help="JSON changed paths from the frozen Git comparison")
    parser.add_argument("--overview", type=Path, help="Write the top of the PR comment: summary, risk, and approval")
    args = parser.parse_args(argv)
    review = args.review.read_text(encoding="utf-8")
    result = decide(
        review,
        args.diff.read_text(encoding="utf-8", errors="replace"),
        min_score=args.min_score,
        max_lines=args.max_lines,
        paths=json.loads(args.paths.read_text()) if args.paths else None,
    )
    if args.overview:
        args.overview.write_text(overview(review, result))
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
