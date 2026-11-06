---
name: pr-review-loop
description: Run a bounded authoring loop for AI Bridge PRReviewer feedback. Use when asked to iterate on a PR review.
---

# Bounded PR review loop

Use this skill only after the user has authorized work on a specific pull
request and branch. It does not grant permission to comment, edit, push, approve,
or merge. Preserve the user's existing authorization boundary each round.

The read-only poller is `.github/scripts/poll_pr_review.py`. It checks the exact
PR head SHA, PRReviewer status/check runs, the bot review marker, model-risk
label, review decisions, and the bot approval marker. It never mutates GitHub.

## Loop

1. Before each review round, capture the exact head SHA with `gh pr view` and
   confirm it is the commit just pushed by this authorized session. If another
   author moved the branch, stop for the user.
2. Let the automatic review run. Comment `/review` only when the user has
   authorized that external write and a manual trigger is needed.
3. Wait at most 20 minutes for that exact SHA:

   ```bash
   python3.12 .github/scripts/poll_pr_review.py --pr <number> --sha <40-char-sha> \
     --wait-seconds 1200 --poll-seconds 20
   ```

4. Handle the returned state:
   - `approved`: report success and stop.
   - `reviewed_not_approved`: read the returned exact-SHA review body. Verify
     each concrete finding against the branch. Fix clear, in-scope findings,
     test, and push only when already authorized. A high-risk label always
     requires human review before another change.
   - `changes_requested`: address only clear, in-scope requests. Ask a human
     before changing intended behavior or architecture.
   - `head_changed`, `inconsistent`, `failed`, or `timeout`: stop and report the
     state; do not guess or silently restart.
5. Capture the new SHA and repeat. Count the initial review as round one and
   stop after three total review rounds.

Stop for human input whenever the authoring agent disagrees with a finding,
believes it is a false positive, cannot identify a concrete fix, would need to
expand scope, or would change review/approval automation. Never dismiss,
override, or manufacture an approval. Never run an unbounded background loop.
