# Custom Microsoft Agent Framework server

A minimal foreground-only FastAPI server. `POST /invocations` accepts `{"input":[{"role":"user","content":"Hello"}]}`.
Run `agentbricks dev` or `agentbricks deploy <app-name> --source .`.

This custom HTTP server does not use the managed invocation runtime, conversation history, memory,
or browser UI. Use `agentbricks init --framework maf` for the managed integrations.
