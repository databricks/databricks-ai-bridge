"""Reserve a bounded automatic push-review attempt before starting the model."""

import argparse
import json
import re
import subprocess
import sys

MARKER = "<!-- agentbricks-reviewer-push-attempts:"
COUNT = re.compile(r"<!-- agentbricks-reviewer-push-attempts: (\d+) -->")


def plan(comments, limit, run_url):
    if limit < 0:
        raise ValueError("push-review limit must be nonnegative")
    records = [
        comment for comment in comments
        if comment["user"]["login"] == "github-actions[bot]"
        and comment["body"].startswith(MARKER)
    ]
    record = max(records, key=lambda comment: comment["id"], default=None)
    attempts = 0
    if record:
        match = COUNT.fullmatch(record["body"].splitlines()[0])
        if not match:
            raise ValueError("cannot read the recorded push-review count")
        attempts = int(match[1])
    if attempts >= limit:
        return {"skip": True, "attempts": attempts}
    attempts += 1
    body = (
        f"{MARKER} {attempts} -->\n"
        f"Automatic push review attempt **{attempts}/{limit}** reserved. "
        "Failed attempts count toward this limit. After the limit, comment "
        f"`/review` to request another review. [Latest attempt]({run_url})"
    )
    return {"skip": False, "attempts": attempts, "id": record["id"] if record else None, "body": body}


def request(endpoint, data=None, paginate=False):
    command = ["gh", "api", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    if data is not None:
        command += ["--method", "PATCH" if "/issues/comments/" in endpoint else "POST", "--input", "-"]
    result = subprocess.run(
        command, input=json.dumps(data) if data is not None else None,
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args(argv)
    pages = request(f"repos/{args.repo}/issues/{args.pr}/comments?per_page=100", paginate=True)
    decision = plan([comment for page in pages for comment in page], args.limit, args.run_url)
    if decision["skip"]:
        print(f"::notice::Automatic push-review limit reached ({decision['attempts']}/{args.limit}); comment /review to run again.", file=sys.stderr)
        print("skip=true")
        return
    endpoint = (
        f"repos/{args.repo}/issues/comments/{decision['id']}"
        if decision["id"] is not None
        else f"repos/{args.repo}/issues/{args.pr}/comments"
    )
    request(endpoint, {"body": decision["body"]})
    print("skip=false")


if __name__ == "__main__":
    main()
