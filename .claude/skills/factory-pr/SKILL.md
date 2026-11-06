---
name: factory-pr
description: Produce a short intent-focused PR linked to reproducible evidence, respecting the repository template and existing Git history.
---

When the user or task owner authorizes PR publication, create or update one
ready-for-review PR. Reconcile the existing PR for the task and branch first.
Do not pass `--draft`; if the existing PR is a draft, mark that same PR ready
with the repository's GitHub/Gitspoke client instead of opening a duplicate.

Use the repository's configured GitHub account and inspect publication wrappers
before invoking them. Preserve branch history and use a fast-forward push.
Read the remote PR back and verify `isDraft=false`, the target repository, URL,
and published HEAD against the tested/reviewed commit. When the factory phase
request requires it, report the verified boolean as `pr_is_draft: false`.
Do not claim completion while the PR remains a draft. Publication does not
authorize merging, deployment, reviewer requests or announcements.

Lead with the concrete problem and resulting behavior. Keep the body short:
one or two sentences, the meaningful validation result, and links to the detailed
report, UI recording, and runtime evidence where applicable. Include size
(XS/S/M/L), material risks, and missing validation when useful or required by
the repository template. Do not bury limitations in an artifact.

The report should let a reviewer verify intent, acceptance results, exact tested
build, and operational impact. Include an interactive design diagram only when
it clarifies a decision; label illustrative content separately from measurements.

Preserve branch history: new commits for subsequent changes, fast-forward pushes,
and merges to incorporate upstream changes. Do not amend, rebase, squash existing
PR commits, or force-push. Recheck that the published head matches the evidence.
If new changes invalidate evidence, rerun the affected checks before handoff.

Return the PR URL, report and evidence references, current head, and remaining
review needs. A requested draft body or a failed create call is not a published PR.
