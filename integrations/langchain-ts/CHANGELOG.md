# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-26

### Added

- Initial release of `@databricks/langchain-ts`
- `ChatDatabricks` class implementing LangChain `BaseChatModel` interface
- Support for three endpoint APIs:
  - `chat-completions`: OpenAI-compatible chat completions for Foundation Models
  - `chat-agent`: Databricks agent chat completion for deployed agents
  - `responses`: Rich output with reasoning, citations, and function calls
- Streaming and non-streaming response support
- Tool/function calling support with `bindTools()`
- Automatic authentication via Databricks SDK (environment variables, CLI config, OAuth, etc.)
- Explicit authentication option via `auth` parameter
- Model parameters: `temperature`, `maxTokens`, `stop`, `extraParams`
- Call-time parameter overrides
