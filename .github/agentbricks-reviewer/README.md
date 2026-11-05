# Team review criteria

The existing `agentbricks-review` skill defines review coordination and report
formatting. Root `REVIEW.md` holds the repository checks with stable `BRIDGE-*`
IDs. Both the local skill and GitHub reviewer use those checks.

CI injects `REVIEW.md` from its trusted checkout alongside the skill contract.
Changes to criteria in a reviewed PR take effect after they reach the trusted
branch. Each reviewer receives the criteria; the final report names the source,
checked rule IDs, and coverage gaps. The Codex skill path links to the existing
Claude skill, so both runtimes use the same review process.

This extracts the existing checks without changing their substance, workflow
triggers, approval behavior, or merge permissions.
