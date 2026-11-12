"""Post AgentBricksReviewer findings as inline PR review comments.

Each finding whose `path:line` falls on a line GitHub can comment on (an added or
context line on the new side of the diff) becomes an inline comment in one
COMMENT review on the reviewed commit. In the summary comment, each section's
posted findings are replaced by one line saying how many were left inline.
Findings that can't be anchored, such as ones about files outside the diff, stay
in the summary in full.
On a re-review, a finding that already has an open, unresolved inline comment
at the same file, line, and section isn't posted again: the earlier comment
keeps its thread and is updated to the current text. Earlier comments whose
findings are gone, or whose lines changed, are minimized as outdated; resolved
threads are left alone. The summary drops only findings that are visible inline.

Usage: agentbricks_inline_comments.py --repo OWNER/NAME --pr N --head SHA
           --review REVIEW --diff DIFF [--findings FINDINGS] [--out SUMMARY]
FINDINGS is JSON describing every finding (title, problem, fix, evidence) and,
for each one visible inline, its comment URL; the summary comment lists them
from it. SUMMARY is the review with inline findings replaced by counts. If
posting fails, those findings are not marked inline, so the summary keeps them.
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
FINDING_MARKER = re.compile(r"^<!-- agentbricks-reviewer-finding: (?P<section>.+?) -->$", re.MULTILINE)
# The bot's earlier inline comments, with where each sits on the current diff.
EXISTING_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isOutdated
          isResolved
          comments(first: 1) { nodes { id databaseId path line isMinimized body author { __typename } } }
        }
      }
    }
  }
}
"""
FOOTER = re.compile(r"\n*<sub>AgentBricksReviewer finding for `[0-9a-f]+`\.</sub>\s*$")
PAYLOAD_KEYS = ("path", "line", "side", "start_line", "start_side", "body")
# `path:line` (or a range) at the start of a finding, before its title.
LEADING_LOCATION = re.compile(r"^`?(?P<location>[\w.@+/-]+:\d+(?:-\d+)?)`?\s*(?:—|–|-|:)?\s*")
DETAIL_LABEL = re.compile(r"^\**(?P<label>Problem|Fix|Evidence)\**:\**\s*(?P<text>.*)$", re.IGNORECASE)
TITLE_LIMIT = 120


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


def split_finding(finding: dict) -> dict:
    """A finding's parts: tags, location, title, and the Problem/Fix/Evidence lines.

    Reviews in the older one-line format (trigger → consequence) have no labels;
    their title is the trigger and the rest of the text becomes the problem.
    """
    text = finding["text"]
    tags = TAGS.match(text)
    tag_names = re.findall(r"\[([^\]]+)\]", tags.group("tags")) if tags else []
    first, *rest = text[tags.end() if tags else 0 :].split("\n")
    location = LEADING_LOCATION.match(first)
    title = first[location.end() :] if location else first
    details = {"problem": [], "fix": [], "evidence": []}
    current = "problem"
    for line in rest:
        label = DETAIL_LABEL.match(line)
        if label:
            current = label.group("label").lower()
            line = label.group("text")
        elif re.match(r"^Verified-against:", line):
            current = "evidence"
        details[current].append(line)
    if not details["problem"] and not details["fix"] and (len(title) > TITLE_LIMIT or "→" in title):
        # Older one-line format: keep the whole sentence as the problem.
        details["problem"] = [title.strip()]
        title = title.split("→", 1)[0].strip()
    if len(title) > TITLE_LIMIT:
        title = title[:TITLE_LIMIT].rsplit(" ", 1)[0] + "…"
    if title.count("`") % 2:
        title = title[: title.rfind("`")].rstrip() + "…"
    return {
        "section": finding["section"],
        "severity": tag_names[0] if tag_names else "",
        "confidence": tag_names[1] if len(tag_names) > 1 else "",
        "location": location.group("location") if location else "",
        "title": title.strip(),
        **{key: " ".join(part.strip() for part in value if part.strip()) for key, value in details.items()},
    }


def render(parts: dict, head_sha: str) -> str:
    """An inline comment: title, severity line, problem, fix, and collapsed evidence."""
    meta = " · ".join(filter(None, (parts["severity"], parts["confidence"].lower(), parts["section"])))
    lines = [f"**{parts['title']}**", f"<sub>{meta}</sub>", ""]
    if parts["problem"]:
        lines += [parts["problem"], ""]
    if parts["fix"]:
        lines += [f"**Fix:** {parts['fix']}", ""]
    if parts["evidence"]:
        lines += ["<details><summary>Evidence</summary>", "", parts["evidence"], "", "</details>", ""]
    lines.append(f"<sub>AgentBricksReviewer finding for `{head_sha[:7]}`.</sub>")
    return "\n".join(lines)


def plan(review: str, diff: str, head_sha: str) -> tuple[list[dict], list[dict]]:
    """Inline comments to post, and the findings they came from."""
    lines = commentable_lines(diff)
    comments, anchored = [], []
    for finding in parse_findings(review):
        position = anchor(finding, lines)
        if position is None:
            continue
        body = (
            f"{INLINE_MARKER}\n<!-- agentbricks-reviewer-finding: {finding['section']} -->\n"
            + render(split_finding(finding), head_sha)
        )
        comments.append({**position, "body": body, "section": finding["section"], "finding": len(anchored)})
        anchored.append(finding)
    return comments, anchored


def rewrite(review: str, anchored: list[dict]) -> str:
    """The review with each section's posted findings replaced by a count."""
    review_lines = review.splitlines()
    by_section: dict[str, list[dict]] = {}
    for finding in anchored:
        by_section.setdefault(finding["section"], []).append(finding)
    replacements = []
    for findings in by_section.values():
        count = len(findings)
        note = f"- Left {count} inline comment{'s' if count != 1 else ''} on the changed lines."
        replacements += [(f["start"], f["end"], [note] if i == 0 else []) for i, f in enumerate(findings)]
    for start, end, lines in sorted(replacements, reverse=True):
        review_lines[start:end] = lines
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


def finding_key(path: str, line: int | None, section: str | None) -> tuple:
    return (path, line, section)


def reconcile(comments: list[dict], existing: list[dict]) -> tuple[list[dict], list[tuple[dict, dict]], list[dict]]:
    """Split planned comments into ones to post, earlier ones to reuse, and earlier ones to minimize.

    An earlier comment is reused, as an (earlier, planned) pair, when its thread is
    open, unresolved, and still on the current diff, and a planned finding has the
    same file, line, and section. Other earlier comments are minimized, except
    resolved threads, which stay as their reviewers left them.
    """
    planned = list(comments)
    reuse, minimize = [], []
    for comment in existing:
        key = finding_key(comment["path"], comment["line"], comment["section"])
        match = next((c for c in planned if finding_key(c["path"], c["line"], c["section"]) == key), None)
        reusable = not comment["outdated"] and not comment["resolved"] and comment["line"] is not None
        if reusable and match is not None:
            reuse.append((comment, match))
            planned.remove(match)
        elif not comment["minimized"] and not comment["resolved"]:
            minimize.append(comment)
    return planned, reuse, minimize


def existing_inline(repo: str, pr: int) -> list[dict]:
    """The bot's earlier AgentBricksReviewer inline comments on this PR, across every page of threads."""
    owner, name = repo.split("/", 1)
    found, cursor = [], None
    while True:
        args = ["graphql", "-f", f"query={EXISTING_QUERY}", "-f", f"owner={owner}", "-f", f"name={name}",
                "-F", f"number={pr}"] + (["-f", f"cursor={cursor}"] if cursor else [])
        threads = github(args)["data"]["repository"]["pullRequest"]["reviewThreads"]
        for thread in threads["nodes"]:
            for comment in thread["comments"]["nodes"]:
                if (comment.get("author") or {}).get("__typename") != "Bot" or INLINE_MARKER not in comment["body"]:
                    continue
                marker = FINDING_MARKER.search(comment["body"])
                found.append({
                    "id": comment["id"], "database_id": comment["databaseId"], "body": comment["body"],
                    "path": comment["path"], "line": comment["line"],
                    "section": marker["section"] if marker else None,
                    "outdated": thread["isOutdated"], "resolved": thread["isResolved"],
                    "minimized": comment["isMinimized"],
                })
        if not threads["pageInfo"]["hasNextPage"]:
            return found
        cursor = threads["pageInfo"]["endCursor"]


def set_minimized(comment: dict, minimized: bool) -> None:
    mutation = ("mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }"
                if minimized else "mutation($id: ID!) { unminimizeComment(input: {subjectId: $id}) { clientMutationId } }")
    github(["graphql", "-f", f"query={mutation}", "-f", f"id={comment['id']}"])


def post(repo: str, pr: int, head_sha: str, comments: list[dict]) -> dict[int, str]:
    """Post every inline comment in one COMMENT review; GitHub accepts all or none.

    Returns each posted comment's URL by its finding index, as far as GitHub
    reports them; a missing URL only drops a link from the summary.
    """
    review = github(
        [f"repos/{repo}/pulls/{pr}/reviews"],
        {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": f"AgentBricksReviewer left {len(comments)} inline comment(s) on `{head_sha[:7]}`. "
            "The score and summary are in the review comment.",
            "comments": [{key: c[key] for key in PAYLOAD_KEYS if key in c} for c in comments],
        },
    )
    urls: dict[int, str] = {}
    if not isinstance(review, dict) or "id" not in review:
        return urls
    try:
        pages = github(["--paginate", "--slurp", f"repos/{repo}/pulls/{pr}/reviews/{review['id']}/comments?per_page=100"])
    except RuntimeError as error:
        print(f"::warning::Posted inline findings but couldn't read their links: {error}")
        return urls
    pending = list(comments)
    for page in pages or []:
        for posted in page:
            match = next((c for c in pending if c["body"] == posted.get("body") and c["path"] == posted.get("path")), None)
            if match is not None:
                urls[match["finding"]] = posted["html_url"]
                pending.remove(match)
    return urls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--head", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--findings", type=Path, help="Write every finding, with inline comment URLs, as JSON")
    parser.add_argument("--out", type=Path, help="Write the review with inline findings replaced by counts")
    args = parser.parse_args(argv)

    review = args.review.read_text(encoding="utf-8")
    diff = args.diff.read_text(encoding="utf-8", errors="replace")
    comments, anchored = plan(review, diff, args.head)
    try:
        existing = existing_inline(args.repo, args.pr)
    except (RuntimeError, KeyError, TypeError) as error:
        print(f"::warning::Could not read earlier inline comments; posting every finding: {error}")
        existing = []
    to_post, reuse, stale = reconcile(comments, existing)
    for comment in stale:
        try:
            set_minimized(comment, True)
        except RuntimeError as error:
            print(f"::warning::Could not minimize an earlier inline comment: {error}")
    # Findings count as inline only once their comment is visible with current text;
    # anything else stays in the summary in full.
    visible: list[int] = []
    urls: dict[int, str] = {}
    for earlier, planned in reuse:
        try:
            if FOOTER.sub("", earlier["body"]) != FOOTER.sub("", planned["body"]):
                github(["--method", "PATCH", f"repos/{args.repo}/pulls/comments/{earlier['database_id']}"],
                       {"body": planned["body"]})
            if earlier["minimized"]:
                set_minimized(earlier, False)
            visible.append(planned["finding"])
            urls[planned["finding"]] = f"https://github.com/{args.repo}/pull/{args.pr}#discussion_r{earlier['database_id']}"
        except RuntimeError as error:
            print(f"::warning::Could not update an earlier inline comment; keeping its finding in the summary: {error}")
    if to_post:
        try:
            urls.update(post(args.repo, args.pr, args.head, to_post))
            visible += [c["finding"] for c in to_post]
        except RuntimeError as error:
            print(f"::warning::Could not post inline findings; keeping them in the summary comment: {error}")
    if not comments:
        print("No findings on commentable diff lines; keeping every finding in the summary comment.")
    print(f"Inline: {len(to_post)} new, {len(reuse)} reused, {len(stale)} minimized; {len(visible)} of {len(comments)} visible.")
    if args.out:
        args.out.write_text(rewrite(review, [anchored[i] for i in sorted(visible)]), encoding="utf-8")
    if args.findings:
        inline_ids = {id(anchored[i]): urls.get(i) for i in visible}
        records = []
        for finding in parse_findings(review):
            match = next((a for a in anchored if a["start"] == finding["start"]), None)
            shown = match is not None and id(match) in inline_ids
            records.append({**split_finding(finding), "inline": shown, "url": inline_ids.get(id(match)) if shown else None})
        args.findings.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
