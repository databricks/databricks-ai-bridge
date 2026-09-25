# Agent Bricks Sandbox Token Environment Spec

## Goal

Let code running in `system.ai.sandbox` call Databricks workspace APIs by opting the sandbox into
the already-minted Databricks credential. New `ab tools add sandbox` bindings opt in by default.

## Manifest contract

New bindings write the protected policy next to their fixed downscope:

```toml
policy = { downscope = [{ resource = "workspace:/Workspace/Shared", permission = "read_only" }], databricks_access_token_included = true }
```

- `--no-databricks-access-token-included` creates a binding with the policy set to `false`.
- Existing manifests that omit the field continue to parse as `false`.
- The field is valid only for `source.kind = "sandbox"` and must be a TOML boolean.

## Runtime contract

Both the LangGraph and OpenAI adapters add protected MCP metadata on every sandbox call:

```json
{
  "downscope": {"workspace_paths": [{"path": "/Workspace/Shared", "permission": "read_only"}]},
  "databricks_access_token_included": true
}
```

Caller-provided metadata cannot override either protected policy field.

Identity selection remains independent of token exposure:

- `auth = "user"` resolves the request-user/OBO workspace client.
- `auth = "app"` resolves the Databricks App service-principal workspace client.

The sandbox service mints the token represented by those credentials and exposes it as
`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_AUTH_TYPE` inside sandbox execution.

## Validation

Unit tests cover manifest compatibility, CLI default/opt-out behavior, and protected metadata in
both adapters. Live E2E calls `WorkspaceClient().current_user.me()` from the sandbox with a
workspace-path downscope across these axes:

| Mode | App/SP | User/OBO |
| --- | --- | --- |
| `ab dev` | required | required, with local-profile caveat recorded |
| `ab deploy` | required | required; authoritative OBO proof |

The E2E report records pass/fail evidence without storing tokens or credential-bearing headers.
