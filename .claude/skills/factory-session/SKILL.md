---
name: factory-session
description: Create or locate an assignee-owned Omnigent session through an already configured Software Factory app, preserving request deduplication and team visibility.
---

Use only when asked to launch or locate a coding session. A worker already in a
factory session should use factory-author and continue that session instead.

Use the operator-provided FACTORY_ORIGIN. Authenticate through that Databricks
App; never invent proxy identity headers or put bearer tokens in a prompt, URL,
log, shell history or issue. Read GET /api/factory/sessions/capabilities and
confirm the signed-in assignee, native runtimes, and OAuth connection. Each
engineer must connect their own account; the app SP owns storage, not the human
session. If disconnected, direct the owner to Settings → Connect Omnigent.

Route universe to native Arclet; route databricks-ai-bridge to native Databricks
Sandbox (databricks-lakebox). AI Bridge needs the owner's linked GitHub account
in the execution workspace. An explicit git_credential_id selects a linked
credential there; never export, enumerate secrets or try another person's
credentials. Do not attach an Arclet bootstrap to Lakebox.

POST /api/factory/sessions with request_id, repository, harness (codex/claude),
assignee, title, prompt, workflow_mode (manual/phased), and share_with_team.
Keep one stable request_id per task. For a Jira handoff, use
POST /api/factory/sessions/from-jira with dispatch_id, issue_key, summary,
description, repository, harness, workflow_mode, and the actual assignee.
The caller must be that assignee; a coordinator may use only a previously
consented grant for the same owner. Do not substitute the app operator.

Explain sharing before launch: the configured named team can view/comment on
the app card and view the Omnigent session; only the owner steers or approves.
Use share_with_team=false for a private manual launch. Inspect native_sharing
on GET /api/factory/board/{receipt_id}; pending is not confirmed access.

Poll GET /api/factory/sessions/{receipt_id} until session_url appears, then
return both the card and native conversation links. Accepted/queued input is
not task completion. Reuse the same receipt after a lost response; never retry
with a fresh ID to bypass uncertainty. For phased work, one conversation carries
intake → design → implement → test → review → draft PR with owner gates.
Read the checkout's factory-author, factory-evidence and factory-pr skills;
report their source commit and paths. Keep the session available for steering.
