# AgentKit runtime distribution

This project builds `databricks-agentkit-runtime`, which owns the
`databricks_agentkit` Python namespace. It has no CLI entry point or templates.
The parent `databricks-agentbricks` project owns the CLI and depends on this SDK.
Both wheels share the same release version and must be published together.

Install `[openai]` or `[langgraph]` for a framework adapter, `[sandbox]` for the
orchestrator transports, and `[genie]` for Genie. The normal CLI installation
retains tracking, provisioning, and existing template dependencies.
