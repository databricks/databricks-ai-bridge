"""Post AgentBricksReviewer findings as inline PR review comments.

Each finding whose `path:line` falls on a line GitHub can comment on (an added or
context line on the new side of the diff) becomes an inline comment in one
COMMENT review on the reviewed commit. In the summary comment, that finding
shrinks to one line linking to its inline comment. Findings that can't be
anchored, such as ones about files outside the diff, stay in the summary in full.
Inline comments from earlier runs are minimized as outdated so re-reviews don't
pile up.

Usage: agentbricks_inline_comments.py --repo OWNER/NAME --pr N --head SHA
           --review REVIEW --diff DIFF --out SUMMARY
Writes the summary text to SUMMARY. If posting fails, SUMMARY is the unchanged
review, so the summary comment still carries every finding.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FINDING_SECTIONS = ("Blocking issues", "Security vulnerabilities", "Non-blocking notes", "Uncertain findings")
HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
SECTION_LABEL = re.compile(r"^\*\*(?P<title>[^*]+?)\*\*:?\s*$")
BULLET = re.compile(r"^[-*]\s+(?P<text>.*)$")
TAGS = re.compile(r"^(?P<tags>(?:\[[^\]]+\]\s*)+)")
# The first `path:line` or `path:start-end` in a finding is its location.
LOCATION = re.compile(r"`?(?P<path>[\w.@+/-]+):(?P<start>\d+)(?:-(?P<end>\d+))?`?")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")
INLINE_MARKER = "<!-- agentbricks-reviewer-inline -->"
FINDING_MARKER = re.compile(r"<!-- agentbricks-reviewer-finding: (\d+) -->")
SUMMARY_LIMIT = 160


def commentable_lines(diff: str) -> dict[str, set[int]]:
    """New-side lines per file that GitHub accepts review comments on."""
    lines: dict[str, set[int]] = {}
    path = None
    in_header = False
    current = 0
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            path, in_header, current = None, True, 0
        elif in_header and raw.startswith("+++ "):
            # "+++ /dev/null" is a deleted file, which has no new-side lines.
            target = raw[4:].strip()
            path = target[2:] if target.startswith("b/") else None
        elif path is not None and (hunk := HUNK.match(raw)):
            in_header, current = False, int(hunk.group("start"))
        elif path is not None and not in_header and raw.startswith(("+", " ")):
            lines.setdefault(path, set()).add(current)
            current += 1
    return lines


def parse_findings(review: str) -> list[dict]:
    """Finding list items under the finding sections, with their line spans in the review."""
    findings: list[dict] = []
    review_lines = review.splitlines()
    section = None
    index = 0
    while index < len(review_lines):
        line = review_lines[index]
        heading = HEADING.match(line) or SECTION_LABEL.match(line.strip())
        if heading:
            title = heading.group("title").strip().rstrip(":").strip("*").strip()
            section = title if title in FINDING_SECTIONS else None
            index += 1
            continue
        bullet = BULLET.match(line)
        if section and bullet:
            end = index + 1
            # Continuation lines are indented; the next bullet, heading, or blank line ends it.
            while end < len(review_lines) and review_lines[end].startswith((" ", "\t")) and review_lines[end].strip():
                end += 1
            text = "\n".join([bullet.group("text")] + [part.strip() for part in review_lines[index + 1 : end]])
            findings.append({"section": section, "start": index, "end": end, "text": text})
            index = end
            continue
        index += 1
    return findings


def anchor(finding: dict, lines: dict[str, set[int]]) -> dict | None:
    """The review comment position for a finding, or None if it isn't on a commentable line."""
    location = LOCATION.search(finding["text"])
    if not location:
        return None
    path = location.group("path").removeprefix("./")
    available = lines.get(path)
    if not available:
        return None
    start = int(location.group("start"))
    end = int(location.group("end") or start)
    if start < end and all(n in available for n in range(start, end + 1)):
        return {"path": path, "start_line": start, "line": end, "start_side": "RIGHT", "side": "RIGHT"}
    for candidate in (start, end):
        if candidate in available:
            return {"path": path, "line": candidate, "side": "RIGHT"}
    return None


def short_form(text: str) -> str:
    """The finding's tags and trigger, short enough for one summary line."""
    first = text.split("\n", 1)[0]
    head = first.split("→", 1)[0].rstrip(" —-")
    if len(head) > SUMMARY_LIMIT:
        head = head[:SUMMARY_LIMIT].rsplit(" ", 1)[0] + "…"
    if head.count("`") % 2:
        head = head[: head.rfind("`")].rstrip() + "…"
    return head


def plan(review: str, diff: str, head_sha: str) -> tuple[list[dict], list[dict]]:
    """Inline comments to post, and the findings they came from."""
    lines = commentable_lines(diff)
    comments, anchored = [], []
    for number, finding in enumerate(parse_findings(review)):
        position = anchor(finding, lines)
        if position is None:
            continue
        tags = TAGS.match(finding["text"])
        title = f"**{tags.group('tags').strip()}** · {finding['section']}" if tags else f"**{finding['section']}**"
        body_text = finding["text"][tags.end() :] if tags else finding["text"]
        body = (
            f"{INLINE_MARKER}\n<!-- agentbricks-reviewer-finding: {number} -->\n"
            f"{title}\n\n{body_text.strip()}\n\n"
            f"<sub>AgentBricksReviewer finding for `{head_sha[:7]}`.</sub>"
        )
        comments.append({**position, "body": body})
        anchored.append({**finding, "number": number})
    return comments, anchored


def rewrite(review: str, anchored: list[dict], urls: dict[int, str]) -> str:
    """The review with each posted finding shrunk to a one-line link."""
    review_lines = review.splitlines()
    for finding in sorted(anchored, key=lambda item: item["start"], reverse=True):
        url = urls.get(finding["number"])
        if not url:
            continue
        review_lines[finding["start"] : finding["end"]] = [f"- {short_form(finding['text'])} → [inline comment]({url})"]
    return "\n".join(review_lines) + ("\n" if review.endswith("\n") else "")


def github(args: list[str], data: dict | None = None) -> object:
    result = subprocess.run(
        ["gh", "api", *args] + (["--input", "-"] if data is not None else []),
        input=json.dumps(data) if data is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout) if result.stdout.strip() else None


def minimize_earlier(repo: str, pr: int) -> None:
    """Collapse inline comments from earlier AgentBricksReviewer runs."""
    pages = github(["--paginate", "--slurp", f"repos/{repo}/pulls/{pr}/comments?per_page=100"])
    for page in pages or []:
        for comment in page:
            if comment["user"]["type"] == "Bot" and INLINE_MARKER in comment["body"]:
                github([
                    "graphql",
                    "-f",
                    "query=mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) "
                    "{ minimizedComment { isMinimized } } }",
                    "-f",
                    f"id={comment['node_id']}",
                ])


def post(repo: str, pr: int, head_sha: str, comments: list[dict]) -> dict[int, str]:
    """Post one COMMENT review and return each finding number's comment URL."""
    review = github(
        [f"repos/{repo}/pulls/{pr}/reviews"],
        {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": f"AgentBricksReviewer left {len(comments)} inline finding(s) on `{head_sha[:7]}`. "
            "The score and summary are in the review comment.",
            "comments": comments,
        },
    )
    pages = github(["--paginate", "--slurp", f"repos/{repo}/pulls/{pr}/reviews/{review['id']}/comments?per_page=100"])
    urls = {}
    for page in pages or []:
        for comment in page:
            if marker := FINDING_MARKER.search(comment["body"]):
                urls[int(marker.group(1))] = comment["html_url"]
    return urls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--head", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    review = args.review.read_text(encoding="utf-8")
    diff = args.diff.read_text(encoding="utf-8", errors="replace")
    summary = review
    try:
        minimize_earlier(args.repo, args.pr)
    except RuntimeError as error:
        print(f"::warning::Could not minimize earlier inline comments: {error}")
    comments, anchored = plan(review, diff, args.head)
    if comments:
        try:
            urls = post(args.repo, args.pr, args.head, comments)
            summary = rewrite(review, anchored, urls)
            print(f"Posted {len(urls)} inline finding(s).")
        except RuntimeError as error:
            print(f"::warning::Could not post inline findings; keeping them in the summary comment: {error}")
    else:
        print("No findings on commentable diff lines; keeping every finding in the summary comment.")
    args.out.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
