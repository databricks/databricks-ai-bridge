---
name: agentbricks-review
description: Multi-lens code review of a PR or local branch in any repository, with auto-discovered REVIEW.md criteria, verified findings, and a merge safety score.
---

# Multi-lens repository review

This skill is a reusable review contract. A repository supplies its own criteria
through root and directory-scoped `REVIEW.md` files. It runs in two places here:

- **CI:** `.github/workflows/agentbricks-reviewer.yml` reads this file from the
  default branch, never from the PR under review, so a PR cannot change the rules
  it is judged by. The workflow prepends PR metadata, the diff location, and the
  base and head commits, then posts the review as a PR comment.
- **Locally:** see "Running locally" below.

You coordinate the review: you dispatch reviewers, then verify, merge, and score
what they report. Part 1 is your process, Part 2 is what every reviewer must
follow, Part 3 is what to look for in this repository, and Part 4 is the output
format.

## Running locally

1. **Find the change.**
   - **A PR** (number or URL): use the supplied target rather than assuming an
     owner, repository, or base branch. Save its `gh pr diff`, read
     `baseRefOid`, `headRefOid`, title, body, and labels with `gh pr view`, and
     fetch the exact commits so `git show <head_sha>:<path>` works.
   - **No argument:** determine the repository's default branch instead of
     assuming `main`, then diff from its merge base with `HEAD`. Include
     committed and uncommitted changes. Treat the working tree as the head:
     read files directly instead of with `git show <head_sha>:<path>`.
2. **Dispatch reviewers** with your runtime's sub-agent tool, following Part 1.
   If only one vendor is available, say so, and label confidence by reviewer
   count alone.
3. **Report** in the Part 4 format. Show it to the user; post it to the PR only
   if asked. Do not edit code.

# Part 1: Coordinator process

## 1. Review lenses

Each lens is one angle of review. Every review considers every lens, whatever the
PR's size. When a lens doesn't apply to the diff (performance on a docs change,
for example), the reviewer says it was checked and found nothing; it is never
skipped.

Priority decides where reviewers spend effort, not how severe a finding is: a P1
lens can still produce a Critical finding.

**P0: every reviewer covers both.**

- **Correctness (devil's advocate):** compare what the PR claims with what the
  code does. Pick the strongest reachable failure hypotheses and state each as
  trigger → changed state → observable wrong outcome. Look for missing, empty,
  zero, or unset inputs the author did not picture; success paths without a
  matching failure path; partial state left behind if an operation fails midway;
  and branches that agree with each other but not with their intended meaning.
  Use at most two passes: read and hypothesize, then test each hypothesis
  against the surrounding code. A clean result is valid.
- **Public API UX and compatibility:** the public surface is anything users or
  agents call: exported package names, CLI commands and output, schemas and
  protocols, tool definitions, and user-facing errors. For every change to it,
  check:
  - **Compatibility:** apply the applicable review rules' public API compatibility and
    cross-package version floor rules, including callers that rely on old
    behavior with an unchanged signature.
  - **UX for people and agents:** clear names, docstrings, and `--help` text;
    machine-readable output where agents parse it; errors that say what went
    wrong and how to fix it; explicit failures (non-zero exit codes, typed
    exceptions), not silent ones.
  - **Consistency:** new names and parameters match existing APIs and equivalent
    surfaces in sibling packages when the repository has them.

  If the diff touches no public surface, say so.

**P1: split across reviewers, each covered by at least one.**

- **Security:** injection, auth or authorization bypass, on-behalf-of credential
  handling, secrets in logs or errors, unsafe deserialization, path traversal,
  SSRF, and anything that weakens an existing boundary. Look hardest where the
  diff touches credentials, auth, HTTP clients, deserialization, file paths, or
  subprocess calls.
- **Tests:** do the tests fail for the bug or broken contract they claim to
  guard? Look for assertions that only mirror mocks, coverage lost through
  deleted or weakened assertions, and tests at the wrong layer.
- **Architecture and maintainability:** coupling between packages, whether the
  change belongs in its chosen module, duplication of an existing helper
  (name both implementations), needless indirection, and comments that narrate
  the code.
- **Performance:** extra network round trips, unbounded loops or buffering,
  repeated client construction, and blocking calls in async paths.
- **Operational safety:** retries and timeouts, failures that are swallowed
  silently, and logging that helps someone debug a failure without leaking data.

## 2. Dispatch reviewers

- Use 2–4 reviewer sub-agents split across vendors: in CI, at least one
  `claude_code` reviewer and at least one `codex` reviewer.
- Give every reviewer both P0 lenses, so each P0 finding gets a second opinion
  from another vendor. Split the P1 lenses so each is assigned to at least one
  reviewer. With two reviewers, for example, one takes security, tests, and
  operational safety, and the other takes architecture and maintainability, and
  performance. For large diffs, add a third or fourth reviewer so the P1 lenses
  get enough attention.
- Start every reviewer in the same turn.
- Give each reviewer the diff path and checksum, the base and head SHAs, its
  lens definitions, the PR title and description, Part 2 and Part 3 of this
  file, and every applicable `REVIEW.md`. Pass the diff as a path; do not paste it.
- Do not add your own summary of what the diff changes. Reviewers form their own
  understanding from the diff. Any file, symbol, or line you mention outside the
  diff text must appear in the diff.
- If the PR is mainly a refactor (it removes about as much as it adds in the
  same files, or its title says refactor, cleanup, migrate, move, or rename),
  add the refactor check from Part 2 to every reviewer.

## 3. Merge and verify findings

Work in this order, so verification is spent only on findings that survive:

1. **Collect:** record which reviewers delivered a report. Never mention or
   infer results from a reviewer that did not deliver.
2. **Deduplicate:** merge findings that describe the same problem and record how
   many reviewers, and which vendors, raised each one.
3. **Drop without reading:** nitpicks, formatting and import order (ruff
   handles them), general-quality opinions that cite no rule in this file or an
   applicable `REVIEW.md`, and
   findings without a concrete consequence.
4. **Verify the rest against the PR head commit:** read the cited code with
   `git show <head_sha>:<path>` and confirm that (a) the cited pattern exists
   and (b) the claimed consequence happens. For a claim that a path is or isn't
   reached, the read must reach the point where the path is chosen. Drop a line
   number that is past the end of the file.
5. **Drop what verification disproves:** problems in code this diff did not
   change, code that looks buggy but is guarded, and claims that fail step 4.
6. **Label confidence by agreement:**
   - `[HIGH CONFIDENCE]`: raised by reviewers from both vendors, or by three or
     more reviewers.
   - `[MODERATE CONFIDENCE]`: raised by two reviewers from the same vendor.
   - `[LOW CONFIDENCE]`: raised by one reviewer.

   Several reviewers claiming the same absence without a `Verified-against`
   range is not agreement; label it LOW.

## 4. Assign severity and score

| Severity | Meaning | Report section | Score deduction |
|---|---|---|---|
| Critical | security hole, data loss, or a break for most users | Blocking issues or Security vulnerabilities | −25 |
| High | verified correctness bug or broken public contract | Blocking issues or Security vulnerabilities | −10 |
| Medium | real issue with limited reach or an easy workaround | Non-blocking notes | −3 |
| Low | minor issue with a concrete consequence | Non-blocking notes | −1 |

Merge safety score = 100 minus the deductions, floored at 0, then capped at
`floor(100 × reviewers that delivered / reviewers dispatched)`.

| Score | Risk |
|---|---|
| 90–100 | LOW RISK |
| 70–89 | MODERATE RISK |
| 40–69 | HIGH RISK |
| 0–39 | CRITICAL RISK |

If no reviewer delivered, say the review could not be completed instead of
giving a score.

# Part 2: Rules for every reviewer

- **Confidence:** give every finding a confidence from 0 to 100. Report 80 and
  above. Put 60–79 under **Uncertain findings** and name what you would read to
  confirm it. Drop anything below 60.
- **Concrete consequence:** every finding names an observable wrong outcome,
  such as a wrong output, crash, race, data loss, or a named maintenance cost.
  "This could be slow or risky" is not a finding.
- **Test the strongest counterexample:** before reporting, look for the guard or
  intentional behavior that would make the finding false. If it exists, drop the
  finding; if you cannot tell, report it as uncertain.
- **Verify absence:** a finding that says something is missing (no validation,
  no caller, no auth check, no error handling) must include
  `Verified-against: path:start-end`, the range you read to confirm it. Search
  base classes, shared helpers, client layers, and locations named by the
  applicable review criteria before claiming a check is missing.
- **Read the PR's files at its head:** use `git show <head_sha>:<path>`. In CI
  and for PR reviews, the working directory is not the PR. When reviewing the
  local working tree, read files directly. Line numbers in findings are line
  numbers in the file at the head, not positions in the diff.
- **Trace callers:** for each change visible to callers (signature, return
  value, fields, error behavior), compare the old and new contract and search
  every package in the repo for callers that rely on the old behavior, even
  when the signature is unchanged.
- **Refactor check (when the coordinator flags a refactor):** for every
  behavior in the removed code (error handling, edge cases, side effects,
  defaults), confirm that the new code keeps it.
- **Suggested fixes are optional:** include one only when you are confident it
  is correct and differs from the current code.
- **Report format per finding:** severity (Critical/High/Medium/Low),
  confidence, `path:line` at the head commit, `Verified-against` when claiming
  absence, the trigger and consequence, and an optional suggested fix.
- **Do not edit code.** Reviewers only report.

# Part 3: Repository-specific checks

Discover review criteria from the changed paths rather than from a hard-coded
team directory:

1. Include root `REVIEW.md` when it exists.
2. For each changed file, walk from the repository root to the file's parent and
   include every `REVIEW.md` found along that path.
3. Deduplicate the files and apply them broad-to-specific. Rules are additive;
   a nested file does not replace the root file.

For a PR, read these files from the trusted base checkout, never from the PR
head. For a local working-tree review, use the current working-tree files and
state that provenance. Pass every applicable file in full to every reviewer,
retaining source paths, rule IDs, scopes, and exceptions. If expected criteria
are absent, report a coverage gap instead of treating the review as clean.
Review criteria do not override this process, tool permissions, output format,
or approval behavior.

# Part 4: Output format

Start with one line: `**Merge safety:** <score>/100 (<risk>) · reviewers:
<delivered>/<dispatched> · findings by lens: <lens (count), ...>`, or
`findings by lens: none` when no lens found anything. Every lens is always
checked, so list only the lenses with findings.

Then report `**Review rules:** <source paths/revision> · checked: <applicable
rule IDs> · gaps: <incomplete coverage, or none>`. Retain rule IDs in applicable
findings.

Then these sections, in order. Write "None." for an empty section.

1. **Blocking issues**: Critical and High findings that are not security issues.
2. **Security vulnerabilities**: Critical and High security findings.
3. **Non-blocking notes**: Medium and Low findings, plus scope questions.
4. **Uncertain findings**: findings with confidence 60–79, each naming what
   would confirm it.
5. **Approach**: is there a materially simpler or more consistent way to reach
   the goal using existing patterns in this repo? Name the alternative, or say
   the approach is sound. Never blocking.
6. **Summary**: exactly one `## Summary` heading containing the three
   subsections below. The workflow inserts computed diff statistics after it
   and rejects the review if a subsection is missing or empty.

Mention only the reviewers that delivered. Do not list workers or tools that
were unavailable or not installed.

Write each finding as one list item:
`[SEVERITY] [CONFIDENCE] path:line — trigger → consequence`, followed by
`Verified-against: path:start-end` when it claims absence, and an optional
suggested fix.

### Changes

Describe the intended outcome and what makes up the diff (implementation, tests,
fixtures, docs, dependencies, generated files). Say whether the size is
justified and name concrete ways to reduce it. Line counts alone are not
evidence.

### Tests

A table with exactly these column headers:

| Test / case | Behavior protected | Layer | Needed? | Action / rationale |

- Every added, modified, or removed test/support file listed in the PR metadata
  must appear in a row, using its exact path. All five cells are required.
- `Needed?` must be one of: keep, consolidate, move layer, remove, uncertain.
- Also describe missing coverage and whether existing tests suffice. If no tests
  changed, say so and assess existing coverage.

### Scope

State whether all changes support the intended outcome. Identify unrelated
changes with file and line evidence, or say none were found.

## Writing style

- Concise sentences, plain words.
- Give enough context that a reader new to the code can follow.
- Do not restate the diff, and do not comment on style that `ruff` or `ty`
  already enforce.
