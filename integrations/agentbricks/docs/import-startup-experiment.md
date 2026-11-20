# Optional import and dependency experiment

This draft reduces eager package imports across databricks-openai,
databricks-langchain and databricks-mcp. Public exports resolve lazily and are
cached; vector-only utilities load numerical dependencies on demand. MCP resource
inspection loads MLflow model resources when called, and AgentKit loads Genie
only when configured. Existing public utility import paths remain available.

The dependency experiment replaces direct full-MLflow requirements with
`mlflow-skinny` and removes redundant full-MLflow requirements from AgentKit's
framework extras. SQLAlchemy/Alembic remain test dependencies for local MLflow
trace readback. Transitive packages can still install full MLflow.

This is separate from the sandbox runtime and is not required to run it. It does
not establish an end-to-end startup improvement or remove arbitrary framework
initialization. Before release, verify a clean Skinny-only installation against
all tracing/model logging APIs and supported minimum dependency versions.

Validation: eight fresh-process/public-export import checks and 77 vector utility
and vector-store tests passed; Ruff checks and formatting passed. A broader
OpenAI tool suite passed 55 tests before being interrupted in an SDK retry wait;
the MCP behavior suite was also interrupted in an SDK retry wait. Those suites
are incomplete, not passing evidence. Live workspace tests were not rerun.

## Follow-up sandbox measurements (November 13)

The recorded LangGraph demo still installed the released integration wheels, so
these import changes were not present in its worker. Applying equivalent import
changes to those same installed versions produced 13.95 s cold initialization
versus 13.30 s baseline: no demonstrated standalone improvement. Full MLflow and
the installed Spark/Arrow paths still dominated much of initialization.

A separate feasibility experiment using tracing-only MLflow, lazy imports, and
omitting unused Connect/Arrow packages reached 9.19 s and 204 MiB peak RSS versus
353 MiB baseline. Real streamed tool calls, managed conversation history, and
persisted model/tool traces passed. These were single direct-worker cold samples,
not end-to-end latency percentiles. That experiment bypassed some declared package
requirements and is not the dependency contract in this PR.

The next prototype must express the reduced worker dependencies in package
metadata and validate a clean locked installation. Detailed timings, caveats,
source identity and inspected traces are in the
[startup investigation](https://github.com/databricks/databricks-ai-bridge/blob/codex/agentbricks-sandbox-warm-pool/integrations/agentbricks/docs/sandbox-worker-startup.md).
