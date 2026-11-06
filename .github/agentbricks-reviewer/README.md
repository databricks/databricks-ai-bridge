# Team review criteria

The `agentbricks-review` skill defines review coordination and report
formatting. Root `REVIEW.md` holds repository-wide checks. A directory may add a
scoped `REVIEW.md`; reviews load the root file plus every file found while
walking from the repository root to each changed file's parent.

CI discovers and injects applicable files from its trusted checkout alongside
the skill contract. Changes to criteria in a reviewed PR take effect after they
reach the trusted branch. Each reviewer receives all applicable criteria; the
final report names their source paths, checked rule IDs, and coverage gaps. The
Codex skill path links to the existing Claude skill, so both runtimes use the
same review process.

The root file preserves the existing repository checks. The AgentBricks subtree
adds its own criteria and automation-eligibility note without changing workflow
triggers, approval behavior, or merge permissions.

Reviews now report two independent assessments: a review score
for findings and reviewer coverage, and change safety for the impact of the
change itself. Change safety is a 0–100 rubric, not a probability. Higher is
safer to automate: 90–100 is low, 60–89 medium, and 0–59 high risk. Missing or
invalid assessments withhold approval and receive high risk.

For example, `Review score: 100/100 · reviewers: 2/2` can appear alongside
`Change safety: 80/100 (MEDIUM RISK)` for a clean review of a consequential
change. Only change safety carries a risk band; the review score does not.

The reviewer considers actual behavior, affected users/packages, public API
compatibility, credentials/data writes, tests, and rollback. CI adds minimum
risk levels: protected CI/review/dependency/credential paths, migration/release
behavior, and binaries are high; public CLI/API/runtime/persistence paths and
multiple production packages are at least medium. More than 10 paths is at
least medium; more than 1000 lines or 30 paths is high. The configurable
`AI_REVIEW_AUTO_APPROVE_MAX_LINES` separately limits approval eligibility without
changing the risk classification. CI
caps the safety score at 89 or 59 when a minimum risk level applies and shows
the reasons in the review comment. Root and scoped REVIEW.md changes remain
protected; this does not add AGENTS.md or CLAUDE.md to the protected filenames.

The workflow labels completed current-head reviews with `risk:low`,
`risk:medium`, or `risk:high` regardless of the auto-approval setting. It replaces
the old risk label and clears stale labels after pushes, including when the
push-review budget is exhausted. Low risk is required for auto-approval but
does not override findings, coverage, author, scope, size, or commit checks.
