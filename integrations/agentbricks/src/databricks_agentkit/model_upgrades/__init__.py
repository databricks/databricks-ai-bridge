"""Joint model + prompt optimization for agents that call AI Gateway model services.

Bring your own agent: a ``predict(inputs: dict)`` callable that loads its prompts from the MLflow
Prompt Registry and calls one or more Unity Catalog AI Gateway model services
(``catalog.schema.name``) through the OpenAI-compatible gateway. Point the optimizer at the prompts
and services to tune; it evaluates candidates against temporary ``<service>_exp`` clones, so the
agent's production services are untouched until ``promote_to_prod``.

    from databricks_agentkit.model_upgrades import optimize_prompts_and_models, promote_to_prod
    from mlflow.genai.scorers import Correctness

    result = optimize_prompts_and_models(
        predict,
        train_data,  # [{"inputs": {...}, "expectations": {"expected_response": ...}}, ...]
        val_data,
        prompt_uris=["prompts:/cat.schema.supervisor@production"],
        gateway_endpoints={"cat.schema.supervisor_llm": ["claude-sonnet-4-5", "claude-haiku-4-5"]},
        scorers=[Correctness(model="databricks:/databricks-gpt-5-4-nano")],
        max_metric_calls=200,
    )
    promote_to_prod(result)

Needs the ``upgrade`` extra (``gepa`` and full ``mlflow``): ``pip install
'databricks-agentbricks[upgrade]'``. `agentbricks experimental models upgrade` drives this for Agent Bricks
projects.
"""

from databricks_agentkit.model_upgrades.optimization import (
    Result,
    optimize_prompts_and_models,
    promote_to_prod,
    score,
    setup_endpoints,
)

__all__ = [
    "Result",
    "optimize_prompts_and_models",
    "promote_to_prod",
    "score",
    "setup_endpoints",
]
