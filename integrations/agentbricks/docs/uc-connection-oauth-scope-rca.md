# RCA: Databricks Apps OAuth cannot call the UC Connection proxy

Status: confirmed platform authorization-contract mismatch
Affected flow: Agent Bricks user-principal UC Connections from Databricks Apps
Unaffected flow: existing bearer/PAT UC Connections invoked as the App principal
Implementation PR: [databricks/databricks-ai-bridge#656](https://github.com/databricks/databricks-ai-bridge/pull/656)

## Executive summary

Databricks Apps accepts `catalog.connections` as a user API scope and forwards a user token with
that scope. The UC Connection proxy does not accept it. The proxy rejects the request before calling
the third-party provider with:

```text
Invalid scope, required scopes: unity-catalog
```

Requesting `unity-catalog` from the App is not a workaround: Apps intentionally rejects that broad
scope. This is not a Linear authorization, DCR, Agent Bricks code-generation, or Connection-creation
failure. Both an existing OAuth Connection and a newly DCR-created OAuth Connection reach the same
runtime proxy boundary and fail there.

The short-term Agent Bricks contract is therefore bind-only bearer/PAT:

- remove inline DCR/Connection creation;
- bind existing UC HTTP Connections only;
- reject every `OAUTH_*` UC `credential_type` during bind with a platform-unsupported error;
- support the deployed App-principal path with existing `BEARER_TOKEN` Connections;
- retain MCP and HTTP transports, LangGraph and OpenAI Agents harnesses, and foreground and
  background execution coverage.

The preferred platform fix is for both Connection proxy routes to accept the granular
`catalog.connections` capability as an alternative to `unity-catalog`. Apps already issues the
granular scope, so this option requires no Apps allowlist change.

## Impact and reproduction

The OAuth matrix covered:

| Axis | Values |
| --- | --- |
| Connection setup | existing, inline DCR-created |
| Transport | MCP, HTTP |
| Agent harness | LangGraph, OpenAI Agents |
| Execution | foreground, background |

Provisioning succeeded for both setup modes. The existing Connection was bound, the inline
Connections were created and authorized, and all four Apps deployed. All 16 provider cells then
stopped at the same platform boundary before Linear received a request.

Boundary trace:

1. `ab auth connections bind` or the former inline create flow writes the same typed binding.
2. The deployed runtime selects request-user authentication.
3. Apps forwards a token with `catalog.connections`.
4. AgentKit calls `/api/2.0/unity-catalog/connections/{name}/proxy/...`.
5. The proxy rejects the token because it requires `unity-catalog` (or a route-specific legacy
   alternative), before provider credential exchange or invocation.

Changing existing versus inline setup cannot change steps 2-5. It only changes how the UC object was
provisioned.

## What is legacy and what is current

“Legacy/new” applies cleanly to routes, but not to the scope names themselves:

| Surface | Status | Authorization today |
| --- | --- | --- |
| `/api/2.0/unity-catalog/connections/{name}/proxy/...` | legacy UC alias; used by AgentKit in PR #656 | Edge mapping accepts `mcp.external` or broad `unity-catalog`; it does not accept `catalog.connections`. |
| `/api/2.0/ai-gateway/connections/...` | transitional route; removed | Superseded by the direct route below. |
| `/ai-gateway/connections/{name}/...` | current AI Gateway route | Edge requires `ai-gateway`; the handler then requires `ai-gateway` **and** `unity-catalog`. It does not accept `catalog.connections`. |

Scope history is counterintuitive:

- `catalog.connections` is the granular Connection scope that Apps onboarded in July 2025.
- `unity-catalog` is the later DeCo package/umbrella scope proposed as a replacement for multiple
  granular catalog scopes. It is broader, and Apps intentionally does not expose it for OBO.
- Both the legacy alias and current AI Gateway Connection handler still depend on the broad umbrella
  for this operation. In that compatibility sense, `unity-catalog` is the broad/legacy authorization
  expectation even though the package-scope mechanism itself was introduced later.

The legacy handler can rewrite an accepted request to the current AI Gateway route while preserving
the original token and headers. The downstream handler then enforces the direct route's stricter
`ai-gateway AND unity-catalog` contract. This silently expands the authorization requirement of the
caller-visible legacy endpoint. It conflicts with the internal API-scope principles that scopes
belong to caller-visible API surfaces, internal RPCs should not re-enforce caller scopes, and an
endpoint's required scopes must not expand during internal forwarding.

PAT, service-to-service, and internal tokens do not carry OAuth scopes and bypass the handler's OAuth
scope check. That explains why the App-principal bearer/PAT E2E succeeds while the user-principal
OAuth E2E fails.

## Root cause

The producer and consumer implement different least-privilege contracts:

- Apps producer contract: validate, consent, mint, and downscope `catalog.connections`; reject the
  broad `unity-catalog` scope.
- Connection proxy consumer contract: authorize the proxy operation with `unity-catalog` (plus
  `ai-gateway` on the current direct route); ignore `catalog.connections`.

The failure is deterministic and happens before provider OAuth/token exchange. Provider login and
the generated Agent Bricks tool code are not causal.

UC Connection CRUD is a separate boundary owned by Managed Catalog. Its APIs still declare broad
`unity-catalog` authorization even though Apps advertises `catalog.connections`; no backend mapping
for the granular scope was found. The observed E2E used the developer's workspace credential for
create/get/bind and failed only on App-runtime proxy invocation, so CRUD is not the live blocker in
this incident. It must nevertheless be reconciled before inline creation with an App OBO token can
be supported.

## Reconciliation options

### Option A — preferred: proxies accept the granular capability

Preserve existing broad-scope callers and add `catalog.connections` as a least-privilege alternative.

Required changes:

1. Legacy alias authorization:
   - accept `catalog.connections`, preserving `mcp.external`, `unity-catalog`, and `all-apis` as
     compatibility alternatives, in
     `common/authentication/apiscopes/AdditionalPathScopes.scala`;
   - update the generated Rust mirror/operation registration and authorization tests.
2. Current AI Gateway route:
   - retain the `ai-gateway` route scope;
   - change the direct handler check to
     `ai-gateway AND (catalog.connections OR unity-catalog OR all-apis)`;
   - carry trusted prevalidated/internal provenance when the legacy handler forwards the request,
     so an internal hop does not add the direct route's `ai-gateway` requirement; do not use a
     caller-spoofable header;
   - update scope/error tests and Connection proxy integration E2E.
3. Managed Catalog CRUD:
   - authorize create/update/delete with `catalog.connections`;
   - authorize get/list with `catalog.connections:read` while retaining broad-scope compatibility;
   - update endpoint declarations and authorization tests owned by `eng-data-security-team`.
4. Apps:
   - no issuer or allowlist change; `catalog.connections` and `ai-gateway` are already supported.
5. Agent Bricks after rollout:
   - re-enable request-user OAuth and decide separately whether inline DCR creation belongs in the
     CLI;
   - rerun the complete new/existing OAuth matrix.

For the current legacy MCP alias, requesting `mcp.external` may be a narrow tactical workaround, but
it does not cover generic HTTP Connection use and keeps Agent Bricks coupled to a legacy route.

### Option B — not recommended: Apps issues `unity-catalog`

This is not a one-line allowlist update. Apps would need to allow and issue `unity-catalog`, add its
security/consent metadata and scope-modifier restrictions, update workspace allowed-scope policy
handling and frontend types, and run consent/downscope E2E. Direct AI Gateway callers would request
`ai-gateway + unity-catalog`; with today's internal forwarding, the legacy alias effectively needs
both too. Proxy and CRUD code could keep their current broad checks, but the internal endpoint-
contract expansion would remain. Apps owners have an active least-privilege objection to exposing
this broad scope, so prefer Option A while a resource-specific capability can authorize the
operation.

## Owners and escalation path

### Databricks Apps OBO scope producer

Declared owner: `eng-lakehouse-apps-team`

1. Theo Fernandez (`theo-fernandez_data`) — current allowlist, validation, and downscoping owner;
   primary contact for the `unity-catalog` security decision.
2. Aakrati Talati (`aakrati-talati_data`) — owner of the canonical Apps OBO scope-onboarding SOP and
   approver of the original `catalog.connections` rollout.
3. Jerry Liang (`jerry-liang_data`) — Apps auth/control-plane and effective-scope propagation.

### UC Connection proxy consumer

Declared subdirectory owner: `eng-ml-rag-platform`; current AI Gateway ownership also involves
`eng-ai-governance-and-observability-team`.

1. Sunish Sheth (`sunish-sheth_data`) — strongest cross-surface DRI; authored the legacy alias,
   current route, scope enforcement, and caller migration.
2. Nisha Balaji (`nisha-balaji_data`) — current AI Gateway implementation/reliability DRI and owner
   of recent Connection-proxy egress work.
3. Bryan McQuade (`bryan-mcquade_data`) — Auth Serving scope-semantics DRI and owner of the internal
   API-scope principles.

Managed Catalog Connection CRUD is separately owned by `eng-data-security-team` and is not the
primary escalation target for this proxy-only failure.

## Escalation request

Ask the Apps and proxy owners to agree on Option A and answer:

1. Can the legacy alias accept `catalog.connections` as an alternative without changing existing
   `unity-catalog`/`mcp.external` callers?
2. Can the current route require `ai-gateway AND (catalog.connections OR unity-catalog)`?
3. Can legacy-to-current internal forwarding carry trusted prevalidated provenance instead of
   re-enforcing the direct route's scopes?
4. Will Managed Catalog map CRUD to `catalog.connections` and read methods to its `:read` modifier?
5. Which team owns the shared auth mapping and generated-operation updates?
6. What workspace and token evidence should gate rollout before Agent Bricks re-enables OAuth?

## Short-term verification

The retained bearer/PAT path uses an existing GitHub UC HTTP Connection and direct hierarchy grants
to the App service principal (`USE_CATALOG`, `USE_SCHEMA`, `USE_CONNECTION`). The earlier smoke run
passed LangGraph and OpenAI Agents in foreground and background execution. PR #656 narrows the live
suite to the eight supported cells: existing × MCP/HTTP × LangGraph/OpenAI × foreground/background.

## Sources

- AgentKit legacy proxy call:
  `integrations/agentbricks/src/databricks_agentkit/auth/connections.py`
- Apps allowlist and validation:
  `apps/commons/src/conf/AppsCommonConf.scala`,
  `apps/src/utils/AppsValidationUtils.scala`
- Apps OBO creation/downscoping:
  `apps/commons/src/serviceprincipal/AppAuthManager.scala`,
  `apps/src/workflow/tasks/CreateOAuth.scala`,
  `apps/src/workflow/tasks/DownscopeOAuthScopes.scala`
- Legacy alias scope mapping:
  `common/authentication/apiscopes/AdditionalPathScopes.scala`,
  `deco/api/scopes/scopes/scopes.go`
- Legacy handler:
  `langchain/langchain-core/src/handlers/ExternalMcpHandler.scala`
- Current route and handler:
  `ai-gateway/route-conf.jsonnet`,
  `ai-gateway/src/ucconnections/UcConnectionsProxyHandler.scala`
- Managed Catalog CRUD declarations:
  `managed-catalog/api/endpoints/service.proto`
- Apps `catalog.connections` rollout:
  [universe PR #1201371](https://github.com/databricks-eng/universe/pull/1201371)
- AI Gateway scope decision:
  [go/ai-gateway-scope-decision](https://docs.google.com/document/d/15qbMpvb8DhdtEarkNTWfeb5rPwcYObAwYVvHNQTrlh8/edit)
- Databricks API-scope principles:
  [What are Databricks API scopes?](https://docs.google.com/document/d/1gIBHxA_QF8BuwQNoNAk0R4LZCqzQUBRsh8joEQ1UIu4/edit)
- OBO MCP scope model:
  [On-Behalf-Of MCP Flow](https://docs.google.com/document/d/1yDoHEmdVyyQEAX91eEEG_8C0Nr9yVr44YfTYdj1cAnQ/edit)
- Public reproduction of Apps rejecting `unity-catalog`:
  [databricks-sdk-go#1528](https://github.com/databricks/databricks-sdk-go/issues/1528)
