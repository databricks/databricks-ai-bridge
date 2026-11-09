"""Reconcile risk and AI-review labels against the latest head and published bot assessment."""

import argparse
import json
import re
import subprocess

LABELS = {"risk:low": "0e8a16", "risk:medium": "fbca04", "risk:high": "d93f0b", "risk:unrated": "ededed"}
MANAGED_LABELS = {**LABELS, "ai-approved": "0e8a16"}
AI_APPROVED = re.compile(r"^<!-- agentbricks-reviewer-ai-approved: (true|false) -->$", re.MULTILINE)
RISK_LABEL = re.compile(r"^<!-- agentbricks-reviewer-risk-label: (risk:(?:low|medium|high|unrated)) -->$", re.MULTILINE)


def github(endpoint, data=None, method=None, paginate=False, allow_missing=False):
    command = ["gh", "api", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    if method:
        command += ["--method", method]
    if allow_missing:
        command += ["--include", "--silent"]
    if data is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command, input=json.dumps(data) if data is not None else None,
        capture_output=True, text=True,
    )
    if result.returncode:
        if allow_missing and re.search(r"^HTTP/[\d.]+ 404(?:\s|$)", result.stdout, re.MULTILINE):
            return None
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout) if result.stdout.strip() and not allow_missing else None


def assessment(api, repo, pr):
    head = api(f"repos/{repo}/pulls/{pr}")["head"]["sha"]
    pages = api(f"repos/{repo}/issues/{pr}/comments?per_page=100", paginate=True)
    candidates = []
    for page in pages:
        for comment in page:
            body = comment.get("body") or ""
            if (comment.get("user") or {}).get("login") != "github-actions[bot]":
                continue
            if f"<!-- agentbricks-reviewer-risk-sha: {head} -->" not in body.splitlines():
                continue
            matches = list(RISK_LABEL.finditer(body))
            if len(matches) == 1:
                verdicts = list(AI_APPROVED.finditer(body))
                passed = len(verdicts) == 1 and verdicts[0].group(1) == "true"
                candidates.append((comment["id"], matches[0].group(1), passed))
    identifier, label, passed = max(candidates, default=(None, None, False))
    return head, identifier, label, passed


def reconcile(api, repo, pr, ensure_label=lambda label: None, max_attempts=3):
    for _ in range(max_attempts):
        before = assessment(api, repo, pr)
        desired = before[2]
        wanted = {desired} if desired else set()
        if before[3]:
            wanted.add("ai-approved")
        for label in sorted(wanted):
            ensure_label(label)
        if wanted:
            api(f"repos/{repo}/issues/{pr}/labels", {"labels": sorted(wanted)}, method="POST")
        for label in MANAGED_LABELS:
            if label not in wanted:
                api(f"repos/{repo}/issues/{pr}/labels/{label}", method="DELETE", allow_missing=True)
        # A push or newly published review can race with any label mutation.
        if assessment(api, repo, pr) == before:
            return desired
    raise RuntimeError("PR head or assessment kept changing while reconciling risk labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    args = parser.parse_args()

    def ensure_label(label):
        subprocess.run(
            ["gh", "label", "create", label, "--repo", args.repo, "--color", MANAGED_LABELS[label],
             "--description", ("Passed AI findings and coverage checks; not a GitHub approval" if label == "ai-approved"
                               else f"Change risk assessed by AgentBricksReviewer: {label[5:]}"), "--force"],
            check=True,
        )

    label = reconcile(github, args.repo, args.pr, ensure_label)
    print(f"Risk labels reconciled: {label or 'no current-head assessment'}")


if __name__ == "__main__":
    main()
