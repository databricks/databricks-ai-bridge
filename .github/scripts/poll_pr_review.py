#!/usr/bin/env python3
"""Poll PRReviewer state for one exact pull-request head commit.

This helper is read-only. It never comments, edits labels, changes code, or pushes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

REVIEW_CONTEXT = "PRReviewer"
RISK_LABELS = {"review-risk: low", "review-risk: medium", "review-risk: high"}
FAILED_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "stale",
    "timed_out",
}


def gh_json(endpoint: str) -> Any:
    output = subprocess.check_output(["gh", "api", endpoint], text=True)
    return json.loads(output)


def gh_pages(endpoint: str) -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["gh", "api", "--paginate", "--slurp", endpoint], text=True
    )
    pages = json.loads(output)
    return [item for page in pages for item in page]


def workflow_state(statuses: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    states = [
        status.get("state")
        for status in statuses
        if status.get("context") == REVIEW_CONTEXT
    ]
    relevant_checks = [check for check in checks if check.get("name") == REVIEW_CONTEXT]
    if any(state in {"error", "failure"} for state in states) or any(
        check.get("conclusion") in FAILED_CONCLUSIONS for check in relevant_checks
    ):
        return "failed"
    if any(state == "pending" for state in states) or any(
        check.get("status") != "completed" for check in relevant_checks
    ):
        return "pending"
    if any(state == "success" for state in states) or any(
        check.get("status") == "completed" and check.get("conclusion") == "success"
        for check in relevant_checks
    ):
        return "success"
    return "unknown"


def classify(
    *,
    expected_sha: str,
    actual_sha: str,
    labels: list[str],
    comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": "pending",
        "expected_sha": expected_sha,
        "actual_sha": actual_sha,
        "risk_label": None,
        "review_body": None,
    }
    if actual_sha != expected_sha:
        result["state"] = "head_changed"
        return result

    risk_labels = sorted(RISK_LABELS.intersection(labels))
    if len(risk_labels) > 1:
        result["state"] = "inconsistent"
        result["risk_labels"] = risk_labels
        return result
    if risk_labels:
        result["risk_label"] = risk_labels[0]

    marker = f"<!-- pr-reviewer-sha: {expected_sha} -->"
    review_comments = [
        comment
        for comment in comments
        if comment.get("user", {}).get("type") == "Bot" and marker in (comment.get("body") or "")
    ]
    if review_comments:
        result["review_body"] = review_comments[-1].get("body")

    exact_reviews = [review for review in reviews if review.get("commit_id") == expected_sha]
    if any(review.get("state") == "CHANGES_REQUESTED" for review in exact_reviews):
        result["state"] = "changes_requested"
        return result
    approved = any(
        review.get("state") == "APPROVED"
        and review.get("user", {}).get("type") == "Bot"
        and "<!-- pr-reviewer-approval -->" in (review.get("body") or "")
        for review in exact_reviews
    )

    result["workflow_state"] = workflow_state(statuses, checks)
    if result["workflow_state"] == "failed":
        result["state"] = "failed"
    elif approved and result["review_body"]:
        result["state"] = "approved"
    elif (
        result["workflow_state"] == "success"
        and result["review_body"]
        and result["risk_label"]
    ):
        result["state"] = "reviewed_not_approved"
    return result


def snapshot(repo: str, pr_number: int, expected_sha: str) -> dict[str, Any]:
    pull = gh_json(f"repos/{repo}/pulls/{pr_number}")
    statuses = gh_json(f"repos/{repo}/commits/{expected_sha}/status").get("statuses", [])
    checks = gh_json(f"repos/{repo}/commits/{expected_sha}/check-runs").get("check_runs", [])
    comments = gh_pages(f"repos/{repo}/issues/{pr_number}/comments?per_page=100")
    reviews = gh_pages(f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100")
    result = classify(
        expected_sha=expected_sha,
        actual_sha=pull["head"]["sha"],
        labels=[label["name"] for label in pull.get("labels", [])],
        comments=comments,
        reviews=reviews,
        statuses=statuses,
        checks=checks,
    )
    result["url"] = pull.get("html_url")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="databricks/databricks-ai-bridge")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--sha", required=True, help="Exact PR head commit to wait for")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args(argv)
    if args.wait_seconds < 0 or args.poll_seconds < 1:
        parser.error("wait-seconds must be nonnegative and poll-seconds must be positive")

    deadline = time.monotonic() + args.wait_seconds
    while True:
        result = snapshot(args.repo, args.pr, args.sha)
        if result["state"] != "pending":
            print(json.dumps(result, indent=2, sort_keys=True))
            return {
                "approved": 0,
                "reviewed_not_approved": 0,
                "changes_requested": 0,
                "head_changed": 3,
                "inconsistent": 3,
                "failed": 4,
            }[result["state"]]
        if time.monotonic() >= deadline:
            result["state"] = "timeout" if args.wait_seconds else "pending"
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    sys.exit(main())
