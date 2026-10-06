# PR #686 regression report

Recorded 2026-10-06 with live Sonnet 4.5 models and real managed session/memory stores on e2-dogfood-apps. No model responses or cloud stores were mocked.

Before: `ed3ed6b742ac1633cd05afa335870abaf98f6233`. After: `c71b6b2451b4e3216e6bc1f1e1f83e577678bd54`.

Both videos show the same generated project locally, its real deployment, then that project locally again. Deployment and CLI-hint chapters display captured command output; browser chapters are continuous screen recordings.

## Before

[686-before-local-deploy-local.mp4](686-before-local-deploy-local.mp4)

- 0:00 — local-initial
- 0:38 — cli-deploy
- 0:46 — deployed
- 1:45 — local-again
- 2:24 — cli-hints

## After

[686-after-local-deploy-local.mp4](686-after-local-deploy-local.mp4)

- 0:00 — local-initial
- 0:38 — cli-deploy
- 0:46 — deployed
- 1:51 — local-again
- 2:32 — cli-hints

## Results

- 169 automated tests passed: 53 CLI, 28 store tests, 38 generated LangGraph tests, and 50 LangGraph/OpenAI UI/runtime tests on current main plus the PR.
- 69 browser checks: 67 pass; two expected failures reproduce the old deployed behavior. All 35 fixed-version checks pass.
- Before deployment: local history and sample approval work on both revisions.
- Deployed before: API succeeds but history returns zero items; pending approval is missing after refresh.
- Deployed after: API history returns both messages; pending approval survives refresh and can be approved/resumed.
- All three invocation modes retain conversation context. Three turns yield three assistant replies after history reload.
- Deployed memory write/read works before and after. Returning to local development keeps memory off, resets local history after restart, and leaves saved cloud data intact.
- Served deployed JavaScript hashes match the generated source exactly.
- Applying the fixed history test to the baseline fails as expected.

The recordings test the exact PR head. Its merge with main (`209dec3ea126fb769a9c5ee468aa745658f95c8e`) was checked separately in a temporary checkout without modifying the PR branch.

Scope: the PR changes CLI setup wording; browser capability wording is unchanged. Local memory is still disabled and local history is still lost on restart. Deployed process restart/crash recovery and cross-user access were not part of these tests.

## Focused PR demonstrations

The PR description links only the two changed behaviors. These clips are trimmed from the deployed browser chapters above, with BEFORE / AFTER labels:

- [Saved messages: empty chat before, restored messages after (15 seconds)](686-history-before-after.mp4)
- [Pending approval: missing before, restored and approved after (19 seconds)](686-approval-before-after.mp4)

The recordings are at original speed, with brief end-frame holds. The event-log pane and old captions are cropped out; new labels sit outside the application UI. No UI contents were simulated or replaced. [Source cut ranges and hashes](short-video-manifest.json).
