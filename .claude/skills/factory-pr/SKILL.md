---
name: factory-pr
description: Produce a short intent-focused PR linked to reproducible evidence, respecting the repository template and existing Git history.
---

Only publish when the user or task owner has authorized a PR. Check for an
existing PR for the task and branch before creating one. Repository guidance
and the explicit task scope determine reviewers, labels, and draft status.

Inspect Git wrappers before publication. In Universe, `git pp` creates changes
entirely under `experimental/` ready for review by default. When a draft is
required, use `git pp --no-pr` for the fast-forward push and then explicitly
create a draft with the repository's configured GitHub/Gitspoke client. Do not
assume a wrapper honors the task's draft requirement. Read the remote PR afterward
and verify draft status, head, target repository and requested reviewers.

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

Adapted from pstack's opening-a-pr playbook and Matt Pocock's concise PR approach.
See ../FACTORY-PROVENANCE.md for pinned sources and the history-preserving adaptation.
