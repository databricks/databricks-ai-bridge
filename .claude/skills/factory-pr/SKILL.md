---
name: factory-pr
description: Produce a short intent-focused PR linked to reproducible evidence, respecting the repository template and existing Git history.
---

Only publish when the user or task owner has authorized a PR. Check for an
existing PR for the task and branch before creating one. Repository guidance
and the explicit task scope determine reviewers, labels, and draft status.

Use the repository's configured GitHub account and reconcile any existing PR
for the branch before publishing. When a draft is requested, create it explicitly
as a draft and read it back to verify draft status, head, target repository and
requested reviewers. Inspect any publication wrapper before invoking it; do not
assume it preserves branch history or honors the requested draft status.

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
See [FACTORY-PROVENANCE.md](../FACTORY-PROVENANCE.md) for pinned sources and the
history-preserving adaptation.
