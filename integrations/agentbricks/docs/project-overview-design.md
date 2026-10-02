# Project overview: initial scope and follow-ups

The CLI now exposes a truthful resource inventory with `agentbricks status`. A UI overview should
reuse this read-only model rather than infer readiness from a saved binding. This document scopes
the requested resource, cost, evaluation and deployed-version panels. It does not imply that the
panels have shipped.

## First implementation slice: resources and links

Show the project framework, selected workspace, declared bindings and resolved resource identifiers
with links where the platform exposes a stable destination. Distinguish **configured**, **checking**,
**accessible**, **missing**, and **check failed**. An accessible resource does not prove the running
app's identity can invoke it; runtime validation belongs to a separate check. Show the last check
time, request identity, workspace and action for each unavailable resource. No automatic provisioning
or deletion occurs on page load. Refresh is a read operation.

The initial slice is a resources card and links to MLflow and Databricks Apps, with the CLI status
schema as its contract. The deployed UI needs a server route that exposes an allowlist of this
information to authorized project maintainers. It must not expose local filesystem paths,
credentials, arbitrary app environment variables, or administrative actions to chat end users.

| Panel | Truthful source | Missing/unavailable states | Scope and dependencies |
| --- | --- | --- | --- |
| Resources | `agent.toml` intent; workspace-specific provisioning receipts; read-only Apps, Session Store, Memory Store, and MLflow experiment APIs | Unverified binding; resource absent; API unavailable; insufficient permission; receipt unavailable for older projects | First slice. Resource IDs and supplied URLs; tool bindings remain unverified until invoked. Receipts are a cleanup aid, not an access-control authority. |
| Cost | Approved billing/usage system tables with explicit workspace/resource attribution and published price assumptions; trace token usage for diagnostic counts only | No billing permission; attribution unavailable; incomplete time window; delayed usage; unknown price | Follow-up owned with billing. Do not label token counts or partial traces as dollar cost. Show window, currency, source, coverage and refresh time before any total. |
| Evaluations | MLflow evaluation run ID, dataset identity, scorer names/versions, per-case outcomes and aggregate metrics from the starter or project-specific suite | No runs yet; run failed; different dataset/scorers; deleted or inaccessible experiment | First slice links to the existing MLflow result page. Inline comparisons require the same dataset/scorer version and an authorized read API; never imply the smoke dataset certifies production quality. |
| Version | Apps deployment metadata (deployment ID/state/source path), plus explicitly captured source revision and installed package/template version | No deployment; deployment failed; local changes; source revision not recorded; old receipt without version | First slice links to Apps deployment details. Follow-up captures commit SHA, dirty-tree marker and artifact identity at deploy; it must not substitute the local Git HEAD for the deployed revision. |

## Cleanup boundary

`agentbricks cleanup` previews resources in the explicitly selected workspace. `--apply` requires
confirmation (or `--yes` for automation). Only Apps with a local creation receipt are candidates;
the current remote service-principal identity must match before deletion. Managed Runtime Store
deletion also validates the API's app owner. A failed store delete retains the App for retry.
Results are saved after each resource, so partial failures can be retried independently.

Stores, experiments, tools, workspace source folders, legacy Lakebase projects and local files
are retained, including stores this project originally created. The current APIs cannot prove
these resources have no other consumers. The preview identifies those retained resources rather
than silently destroying shared data. Missing receipts, copied/adopted Apps and changed App identity
also fail closed. A future explicit store-deletion workflow requires service-supported ownership
and consumer checks; a display-name prefix is insufficient.

## Review checklist for the UI follow-up

- Validate fresh, partly configured, deployed and permission-denied states without provisioning.
- Establish maintainer authorization before exposing administrative project information.
- Verify each link and data field against the selected workspace and request identity.
- Show unavailable cost/version information explicitly; do not fill gaps with estimates.
- Keep chat execution and its errors usable if the overview API fails.

Source feedback: Chang Shi Lim, *Agentbricks_CLI_Feedback*, Wish List items 1–3 (UGW resources, Cost,
Evals, Version); Fabian Nobis, *Production Grade Document Chatbot*, paragraph beginning “One thing
I’m missing is scaffolding for an eval set.” Tracked in ML-70357.
