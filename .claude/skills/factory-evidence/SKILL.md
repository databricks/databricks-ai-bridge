---
name: factory-evidence
description: Gather reproducible runtime, UI and trace evidence for an engineering change, with exact build identity and explicit gaps.
---

Before implementation, map each acceptance criterion to an observable result.
After implementation, run the affected application or service and exercise the
real path. A mocked unit test alone does not prove an integration works.

Record the tested commit or source-bundle hash, environment, commands, expected
behavior, observed behavior, and result. Keep secrets, cookies, tokens, unrelated
customer data, and private conversations out of recordings and reports.

For UI behavior, capture a short continuous recording of the relevant journey
and an inspectable screenshot of its outcome. Check loading, empty, failure,
and narrow-screen states when affected. Recordings must show the tested build;
design diagrams and screenshots of mocks do not establish runtime behavior.

For service or agent behavior, inspect application traces or correlated request
logs from the actual run. Include trace/request IDs, relevant steps, timing,
errors, and the observed result. Inspect the trace contents: a trace ID by itself
is not proof of a successful path. Distinguish browser traces, application logs,
and distributed/MLflow traces. Never label one as another.

If UI or traces do not apply, provide a concrete reason. If they apply but are
unavailable, mark the evidence incomplete rather than substituting a diagram or
claiming success. Keep the detailed report separate from the concise PR body.

For a phased factory task, commit the reviewable report and runtime artifacts
during implementation, identifying their actual tested source hashes or prior
commit. Formal testing verifies the final implementation commit; save fresh
captures outside the checkout, compare them with committed artifacts, and report
the actual HEAD. Keep historical failures and build identities intact. PR links
must resolve to committed artifacts, not untracked worker files. Review notes
and draft PR text also stay outside the checkout to preserve the tested tree.

The checked-in factory.evidence module validates the evidence packet structure.
Run `python -m factory.evidence evidence.json` when this package is available,
or submit the packet to the authenticated session evidence-validation endpoint.
A structural pass does not verify links, replay the application, establish that
the artifacts are genuine, or approve the change. Human intent review remains.

When running a factory test phase, include the entire packet as the `evidence`
object inside the final `factory-result` JSON block. A separate JSON block or an
artifact link is not a substitute. Use the implemented full commit SHA for both
the result's `head` and the packet's `build`. Keep failed observations in their
original reports; a rerun must capture fresh evidence at the same commit.

Adapted from pstack's prove-it-works and show-me-your-work principles. See
../FACTORY-PROVENANCE.md for pinned sources.
