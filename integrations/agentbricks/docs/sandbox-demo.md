# Run the sandbox prototype demo

Use the **sandbox runtime PR alone**; the import/MLflow experiment is optional.
Prerequisites: Git, `uv`, the Databricks CLI, and a workspace with Apps, sandbox
preview and managed Runtime Store APIs enabled. Model/tool access runs as the
App service principal. Authentication and provisioning use your selected profile.

## 1. Install the local CLI

Use a fresh directory so this does not change an existing checkout:

```sh
git clone --branch codex/agentbricks-sandbox-runtime-prototype \
  https://github.com/databricks/databricks-ai-bridge.git agentbricks-sandbox-demo
cd agentbricks-sandbox-demo
agentbricks_repo="$PWD"
uv venv --python 3.12 .venv-sandbox
uv pip install --python .venv-sandbox/bin/python -e 'integrations/agentbricks[sandbox]'
source .venv-sandbox/bin/activate
agentbricks --version
```

Log in once if needed (replace the host with your enabled workspace):

```sh
databricks auth login --profile DEFAULT --host https://YOUR-WORKSPACE
```

## 2. Create an agent and package this SDK into its source

```sh
agentbricks -p DEFAULT init .agentbricks-playground/demo-agent \
  --framework langgraph --name sb-demo
uv build integrations/agentbricks --wheel \
  --out-dir .agentbricks-playground/demo-agent/vendor
cd .agentbricks-playground/demo-agent
uv add --python 3.12 --no-sync --extra langgraph ./vendor/databricks_agentbricks-*.whl
```

The wheel in `vendor/` and the updated `uv.lock` travel with the agent. This is
necessary: installing the CLI from this branch does **not** automatically replace
the released SDK in an agent's dependency lock. The sandbox installs this wheel
from the copied source; no local absolute-path dependency needs to be reachable.
For OpenAI, use `--framework openai` at init and `--extra openai` in `uv add`.
Start with a fresh `vendor/` so the glob selects only one SDK wheel.

Use sandbox-local chat history for this minimal demo and avoid provisioning
additional Memory/Session Stores:

```sh
agentbricks -p DEFAULT memory unbind
agentbricks -p DEFAULT sessions unbind
# Optional tool, if your workspace supports it:
agentbricks -p DEFAULT tools add mcp system.ai.web_search --auth app
```

The two unbind commands only edit this new project's metadata. Tracing uses the
template's default experiment. Lakebase Runtime Store is still provisioned for
the orchestrator. Sandbox-local history survives worker restart but not loss of
the sandbox disk. Existing accessible Session Store bindings can be used instead.

## 3. Deploy and invoke

Choose an unused deployment name if `sb-demo` already belongs to someone else:

```sh
agentbricks -p DEFAULT deploy sb-demo --sandbox --source . --sandbox-idle-timeout 1800
agentbricks -p DEFAULT invoke sb-demo --session-id demo-1 \
  --json '{"message":"Hello! What can you help with?"}' --stream
```

Open the URL in the deploy success panel. New Chat prepares a dedicated sandbox:
create/resume, source copy, dependency install/build, and worker initialization.
The spinner stays visible until ready. Send another message to reuse the worker;
create another chat to see a separate sandbox. The Sandbox panel shows phase
timings, source link, sandbox ID and Stop/Resume controls. The model selector can
switch to another available workspace model if the default is rate-limited.

For local source development before deploying, `uv sync --locked --python 3.12`
installs the agent locally and `agentbricks -p DEFAULT dev` starts its ordinary
local template server. That command does **not** exercise sandbox orchestration;
use the deployed App for the sandbox demo.

## Iterate and debug

After editing agent source, repeat `deploy --sandbox`. After changing AgentKit
or its adapters, build a new wheel into `vendor/`, update the dependency/lock with
`uv add`, and redeploy. The project wheel is a development artifact, not an image.
Use a new chat for the new revision; older sessions remain pinned to their source.
Use the actual App/environment/session identifiers returned by the UI/deploy for:

```sh
agentbricks -p DEFAULT environment ssh ENVIRONMENT \
  --app agent-bricks-sb-demo --session-id SESSION
```

The developer needs `CAN_MANAGE` on the App. This is a terminal bridge to an
already allocated session sandbox.

This demo supports the updated OpenAI/LangGraph callable adapters. Previously
generated projects need migration; arbitrary HTTP servers are not automatically
launched. See [compatibility notes](sandbox-prototype-changes.md).
