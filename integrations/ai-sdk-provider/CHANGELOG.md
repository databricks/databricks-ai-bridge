# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-28

### Changed

- **BREAKING**: Upgraded to AI SDK v6 (`@ai-sdk/provider@3.0.5`, `@ai-sdk/provider-utils@4.0.10`)


## [0.2.3] - 2026-01-26

### Added

- Added `callOptionsToResponsesArgs` for converting AI SDK call options to Responses API parameters (temperature, topP, maxOutputTokens, responseFormat, provider-specific options)
- Added `callOptionsToFmapiArgs` for converting AI SDK call options to FMAPI (Chat Completions) parameters (temperature, topP, topK, maxOutputTokens, stopSequences, responseFormat, provider-specific options)
- Both language models now return warnings for unsupported options (presencePenalty, frequencyPenalty, seed)

## [0.2.2] - 2026-01-26

### Changed

- Eased schema for finish_reason in chat completion and responses models

## [0.2.1] - 2026-01-21

### Changed

- Updated README.md

## [0.2.0] - 2026-01-20

### Added

- Added `chatCompletions` provider method as the new name for accessing FM API endpoints
- Added `responses` provider method as the new name for accessing Responses endpoints
- Added tool calling support for Responses endpoints

### Changed

- FMAPI: Removed XML tag parsing for tool calls since this is not part of the public API
- FMAPI: System messages now preserve `role: 'system'` instead of being converted to `role: 'user'`
- FMAPI: Tool messages now use OpenAI `tool_call_id` format instead of XML serialization

### Removed

- `fmapi()` provider method - use `chatCompletions()` instead
- `responsesAgent()` provider method - use `responses()` instead

## [0.1.1] - 2026-01-15

## [0.1.0] - 2026-01-15

### Added

- Initial release of @databricks/ai-sdk-provider
- Support for Chat Agent endpoint (`agent/v2/chat`)
- Support for Responses Agent endpoint (`agent/v1/responses`)
- Support for FM API endpoint (`llm/v1/chat`)
- Stream and non-stream (generate) support for all three endpoint types
- Custom tool calling mechanism for Databricks agents
- Stream processing and transformation utilities
- MCP (Model Context Protocol) approval utilities
- Three export paths: main, tools, and mcp
- Full TypeScript support with dual ESM/CJS formats
- Comprehensive documentation and examples

### Changed

### Fixed

### Deprecated

### Removed

### Security
