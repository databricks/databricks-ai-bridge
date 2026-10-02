# Evaluate the project agent

Start this project in another terminal with `agentbricks --profile <profile> dev`.
Then run from the project directory:

```bash
uv run python evals/run.py
```

This sends each case to **your running project agent** at `/api/invocations`: its instructions,
model and tools run normally. The dataset is offline (checked into this project); model/tool
calls still use the configured workspace and can incur normal charges. Each case gets a fresh
session so cases cannot inherit conversation history. The script never overrides the agent model.

The starter supports the OpenAI Agents SDK and LangGraph **Agent Bricks server** templates.
Custom HTTP server templates need an evaluator matching their own input/output contract.

Results, per-case input/output/errors, scorer feedback and aggregate pass rates are saved in a
local MLflow experiment. The script prints the run ID and UI command:

```bash
uv run mlflow ui --backend-store-uri sqlite:///.agentbricks/evaluations.db
```

Open http://127.0.0.1:5000, select `agentbricks-starter-evaluation`, then the printed run.
A failed invocation, empty answer, approval interruption or unmet expectation causes a nonzero exit
status. Authentication/HTTP failures remain visible as failed cases; they are not counted as passes.
These simple deterministic smoke checks are a starting point, not a measure of domain quality.

## Extend the dataset

Add one JSON object per line to `evals/cases.jsonl`. `inputs.query` is sent to the agent.
`expectations.contains` is a list of case-insensitive substrings required in the final assistant
answer. For example:

```json
{"inputs":{"query":"What is 3 + 4? Reply with the number only."},"expectations":{"contains":["7"]}}
```

Change the prompts and expectations for your agent's real use cases. Custom scorers, tool-quality
checks and LLM judges can be added to `run_evaluation` or a project-specific copy; see
[MLflow GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/).

## Evaluate a deployment or share results

```bash
uv run python evals/run.py --app <deployment-name> --profile <oauth-profile>
uv run python evals/run.py --url http://127.0.0.1:8000 --data evals/cases.jsonl --timeout 180
```

`--app` uses the selected OAuth profile for Databricks Apps; `--url` is unauthenticated and intended
for local testing. To log evaluations in your chosen workspace rather than locally:

```bash
uv run python evals/run.py --app <deployment-name> --profile <oauth-profile> \
  --tracking-uri databricks://<oauth-profile> --experiment /Users/<you>/agent-evaluations
```

Keep the local `.agentbricks` directory to retain your evaluation history. Project cleanup retains
local results, workspace experiments, session/memory stores and tools because other projects may
use them. The evaluator invokes the actual agent, so use isolated test resources for tool cases
that can write data or perform external actions.
