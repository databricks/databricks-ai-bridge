# Agent Bricks and AgentKit review checks

Scope: `integrations/agentbricks/**`. Apply these checks in addition to root
`REVIEW.md`, only when the changed behavior is relevant. Cite the rule ID and
verify the linked source, generated output, and exceptions before reporting.

## Automation policy

Automated review is enabled for this scope. A change whose entire diff is within
this directory may also be considered for automated approval by a separate,
deterministic CI gate after it is classified as low risk. This file supplies
review criteria; it does not classify risk, grant approval, or override the
workflow's human-review and protected-file safeguards.

## AGENTBRICKS-001: Keep CLI documentation and agent guidance in sync

When a change adds, removes, or modifies an `agentbricks` command, argument,
option, default, choice, requirement, or help string, update `cli.md`. For CLI
behavior changes, also inspect the bundled `agent-bricks-workflow` skill and
update affected workflow guidance, or explain in the PR description why no
skill change is needed. Purely internal changes with no CLI behavior or help
impact are exempt.

Check: `CONTRIBUTING.md` sections "Keeping docs in sync" and the changed Click
definitions under `src/databricks_agentbricks/cli/`.

## AGENTBRICKS-002: Keep templates compatible with the released SDK

Generated projects install the released `databricks-agentbricks` package, not
the source checkout that produced the template. Report a template that starts
using an unreleased AgentKit or runtime API without a compatible released
dependency plan; a newly generated project would otherwise fail to import or
run. A development-only scaffold with an explicit local or git source override
is valid and must not be shipped as a production template.

Check: `CONTRIBUTING.md` sections "Three kinds of change" and "Testing SDK /
runtime changes in a scaffold", the template dependency, and the referenced SDK
API's released version.

## AGENTBRICKS-003: Complete new framework onboarding

Adding an Agent Bricks framework requires the framework choice and template,
the corresponding public adapter symbols in `cli/doctor.py`, CLI and README
documentation, and doctor coverage. Report a missing piece only after verifying
the PR actually introduces a framework rather than extending an existing one.

Check: `CONTRIBUTING.md` section "Adding a new agent framework" and
`tests/unit_tests/doctor_test.py`.
