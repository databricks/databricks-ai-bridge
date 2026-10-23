# Model and prompt upgrades

`agentbricks experimental models upgrade` searches candidate models and prompt rewrites for each
LLM call against the user's own eval set; `models apply` then switches the agent's model services
and prompt aliases to the winner. `experimental prompts` only declares which prompts the agent
loads. The commands are experimental: confirm options with
`agentbricks experimental models upgrade --help` before relying on them.

## Prerequisites in the project

Each LLM call site must call a Unity Gateway model service by role, through
`resolve_model_service("<role>")` from `databricks_agentkit.runtime.tool_manifest`, and the role must
be bound with `agentbricks experimental models bind <catalog.schema.name> --role <role>`. A call that
names a model directly is invisible to the search. A prompt the search rewrites must be loaded
from the Prompt Registry by alias (`@production`). Prompts bound with
`agentbricks experimental prompts bind` are the default set;
`upgrade --prompt prompts:/<name>@<alias>` selects prompts for one run without changing
`agent.toml`, so don't add bindings just to run a search. Deployment creates the declared model services and grants the app
access.

Tracing must be bound (`agentbricks tracing bind`); the search logs next to the trace experiment.

## The eval module

`models upgrade` imports four `module:attr` references from the project, usually `agent/eval.py`:

- `--predict`: a synchronous callable taking one record's `inputs` dict and returning the answer.
  Wrap an async agent in `asyncio.run`. Enable the framework's MLflow autolog so traces carry token
  usage; without it, cost is estimated from model choice alone.
- `--train-data` / `--val-data`: lists of `{"inputs": {...}, "expectations": {...}}`. Use records
  that resemble real traffic. The recommendation and the reported medians come from the validation
  records.
- `--scorer`: MLflow scorers or `(inputs, expectations, answer) -> float` callables.

Do not invent eval records or expectations the user hasn't provided or approved; ask for them.

## Running, applying, and undoing

A search runs as a serverless job by default and can take hours; `--run-on local` runs it in the
CLI's process. It evaluates temporary `<service>_exp_<id>` copies, so production traffic is
untouched until the recommendation is applied. Check progress with `models status`.

`models apply` repoints production model services and moves prompt aliases; `models rollback` undoes
the last apply. Both change what the deployed agent serves, so run them, or `upgrade --apply ask` /
`--apply auto`, only when the user has asked for the switch. Report the summary's recommendation,
score change, and median quality, latency, and cost changes, and say which roles and prompts would
change.
