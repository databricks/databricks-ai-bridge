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
