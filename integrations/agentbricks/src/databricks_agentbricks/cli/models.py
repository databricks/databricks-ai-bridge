"""`agentbricks experimental models` — bind the agent's LLM calls to swappable model services; upgrade them.

Each LLM call site in the agent (a *role*, e.g. ``router`` and ``writer`` in a compound agent, or
the single ``agent`` role in a one-model agent) calls a user-owned Unity Catalog AI Gateway *model
service* (``catalog.schema.name``) declared in agent.toml's ``[model_services.<role>]`` table,
instead of a hardcoded ``system.ai.*`` model. `agentbricks deploy` creates each service (routed to
its binding's default model) and grants the app access; from then on the models behind them are
repointed here, with no code change or redeploy:

- ``models upgrade`` uploads the project to the workspace and submits a serverless Databricks job
  that searches models for every role, and rewritten prompts, jointly on the user's eval set. The
  search never runs on this machine (it can take hours);
- ``models status`` shows the latest run and, once it finishes, its recommendation per role;
- ``models apply`` promotes that recommendation (quality / latency / cost weighted) with the
  optimizer's ``promote_to_prod``; ``models set`` switches one role to a named model;
  ``models rollback`` undoes the last apply or set: every model it switched and every prompt alias
  it moved (or, with ``--role``, just that role's last model switch).

Runs and switches are recorded in ``.agentbricks/model_upgrades.json``.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.runtime.model_services import destination_model

_BIND_COMMAND = "agentbricks experimental models bind <catalog>.<schema>.<name> [--role <role>] --default system.ai.<model>"


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Agent Bricks project containing agent.toml.",
    )(function)


def _yes_option(function):
    return click.option(
        "--yes", "-y", is_flag=True, help="Switch without asking for confirmation."
    )(function)


def _role_option(function):
    return click.option(
        "--role",
        default=None,
        help="Which bound LLM call site (e.g. router). Optional when only one role is bound.",
    )(function)


def _bound_project(source: pathlib.Path):
    """The project at ``source``; errors when it has no model-service binding."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    if not project.model_services:
        raise AgentCliError(
            "This agent has no model service bound.",
            hint=f"Run `{_BIND_COMMAND}`, then `agentbricks deploy`.",
        )
    return project


def _pick_role(project, role: Optional[str]) -> str:
    """``role`` if it's bound; the only bound role when ``role`` is omitted; otherwise an error."""
    roles = list(project.model_services)
    if role is not None:
        if role not in project.model_services:
            raise AgentCliError(
                f"No model service is bound to role '{role}'.",
                hint=f"Bound roles: {', '.join(roles)}.",
            )
        return role
    if len(roles) == 1:
        return roles[0]
    raise AgentCliError(
        "This agent binds several model services; say which one.",
        hint=f"Pass --role with one of: {', '.join(roles)}.",
    )


def _current_model(client, service: str) -> str:
    try:
        resolved = client.get_model_service(service)
    except AgentCliError as exc:
        if exc.error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
            raise AgentCliError(
                f"Model service '{service}' doesn't exist yet.",
                hint="Run `agentbricks deploy` to create it from agent.toml.",
            ) from exc
        raise
    model = destination_model(resolved)
    if model is None:
        raise AgentCliError(f"Model service '{service}' has no foundation-model destination.")
    return model


def _switch(obj, project, service: str, previous: str, model: str, reason: str, report=None):
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    with render.status(f"Switching '{service}' to {model}…"):
        obj.client().set_model_service_model(service, model)
    model_upgrade.record_change(
        project.root,
        service=service,
        previous=previous,
        model=model,
        reason=reason,
        report=report,
    )


def _confirm(obj, yes: bool, question: str) -> bool:
    """True to go ahead. JSON output never prompts: it goes ahead only with --yes."""
    if yes:
        return True
    if obj.output == "json":
        return False
    return click.confirm(question, default=False)


def _registry_mlflow(obj):
    """MLflow pointed at the workspace and its Unity Catalog prompt registry, honoring --profile."""
    import os  # noqa: PLC0415

    from databricks_agentbricks.cli import tracing  # noqa: PLC0415

    if obj.profile:
        # promote_to_prod's workspace client and MLflow both resolve auth from the environment.
        os.environ["DATABRICKS_CONFIG_PROFILE"] = obj.profile
    mlflow = tracing._mlflow()
    tracing._set_tracking_uri(mlflow, obj.profile)
    mlflow.set_registry_uri("databricks-uc")
    return mlflow


def _promote(obj, report) -> list[dict]:
    """Apply an upgrade report with the optimizer's own ``promote_to_prod``.

    It repoints every model service whose model changed, registers each rewritten prompt as a new
    version and moves its alias there (the prior version keeps ``@production_previous``), and rolls
    all of it back if any step fails. Returns the prompt-alias moves it made, as
    {name, alias, prior_version}, so `models rollback` can move them back.
    """
    import contextlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    try:
        from databricks_agentkit.model_upgrades import promote_to_prod  # noqa: PLC0415
    except ImportError as exc:
        raise AgentCliError(
            "Applying an upgrade requires the 'upgrade' extra.",
            hint=model_upgrade.UPGRADE_EXTRA_HINT,
        ) from exc
    _registry_mlflow(obj)
    result = model_upgrade.load_promotion(str(report.mlflow_run_id))
    # promote_to_prod prints its plan; keep stdout clean for -o json.
    with render.status("Applying the recommendation…"):
        with contextlib.redirect_stdout(sys.stderr):
            promote_to_prod(result)
    return [
        {"name": target.name, "alias": target.alias, "prior_version": target.prior_version}
        for target in result.prompt_targets
        if target.alias
        and result.best_candidate.get(f"prompt:{target.short_name}") not in (None, target.template)
    ]


def _restore_prompts(obj, prompts: list[dict]) -> None:
    """Move each prompt's alias back to the version it pointed at before an apply."""
    mlflow = _registry_mlflow(obj)
    with render.status("Moving prompt aliases back…"):
        for prompt in prompts:
            mlflow.genai.set_prompt_alias(
                name=prompt["name"], alias=prompt["alias"], version=prompt["prior_version"]
            )


@click.group()
def models() -> None:
    """Choose, evaluate, and upgrade the models behind your agent.

    Bind each of the agent's LLM calls to a model service it calls through the AI Gateway, then
    switch the models behind those services at any time — by name, or by letting `models upgrade`
    search models and prompts for every call together on your eval set. Switches take effect
    without a code change or redeploy.
    """


@models.command("bind")
@click.argument("service")
@click.option(
    "--role",
    default=None,
    help="Name for the LLM call site this service backs, e.g. router or writer (default: agent). "
    "Bind one service per call site in a compound agent.",
)
@click.option(
    "--default",
    "default_model",
    default=None,
    help="system.ai.* model `agentbricks deploy` routes the service to when it creates it.",
)
@_source_option
@click.pass_obj
def models_bind(
    obj, service: str, role: Optional[str], default_model: Optional[str], source: pathlib.Path
) -> None:
    """Bind model service SERVICE (catalog.schema.name) to one of the agent's LLM calls.

    This only edits agent.toml — it does not create the service. `agentbricks deploy` creates it if it
    doesn't exist (routed to --default), grants the app's service principal EXECUTE on it, and points
    the agent at it via AGENT_MODEL_SERVICE_<ROLE>.
    """
    from databricks_agentbricks.agent_project import (  # noqa: PLC0415
        DEFAULT_MODEL_ROLE,
        AgentProject,
    )
    from databricks_agentbricks.model_upgrade import system_ai_name  # noqa: PLC0415
    from databricks_agentkit.runtime.tool_manifest import model_service_env  # noqa: PLC0415

    role = role or DEFAULT_MODEL_ROLE
    project = AgentProject.load(source)
    project.bind_model_service(
        service, system_ai_name(default_model) if default_model else None, role=role
    )
    project.write()
    binding = project.model_services[role]
    if obj.output == "json":
        render.emit_json(
            {
                "role": role,
                "model_service": service,
                "default": binding.default,
                "manifest": str(project.path),
            }
        )
        return
    fields = {"agent.toml": str(project.path), "Env var": model_service_env(role)}
    if binding.default:
        fields["Default model"] = binding.default
    render.success(
        f"Bound model service '{service}' to role '{role}'",
        fields=fields,
        next_steps=[
            ("agentbricks deploy <name>", "Create it if missing and grant the app access"),
            (
                "agentbricks experimental models upgrade --candidates <model>",
                "Then evaluate other models",
            ),
        ],
    )


@models.command("unbind")
@_role_option
@_source_option
@click.pass_obj
def models_unbind(obj, role: Optional[str], source: pathlib.Path) -> None:
    """Remove a model-service binding from agent.toml.

    The service itself is left in place. After the next deploy that LLM call uses its own default
    model directly again.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    removed = False
    if project.model_services:
        role = _pick_role(project, role)
        removed = project.unbind_model_service(role)
        project.write()
    if obj.output == "json":
        render.emit_json({"role": role, "removed": removed})
        return
    render.success(f"Unbound role '{role}'" if removed else "No model service was bound")


@models.command("bind-prompt")
@click.argument("prompt")
@click.option(
    "--key",
    default=None,
    help="Name for this prompt in agent.toml (default: the prompt's own name, e.g. writer).",
)
@_source_option
@click.pass_obj
def models_bind_prompt(obj, prompt: str, key: Optional[str], source: pathlib.Path) -> None:
    """Declare MLflow Prompt Registry prompt PROMPT (catalog.schema.name) that the agent loads.

    This only edits agent.toml. `agentbricks deploy` grants the app's service principal what the
    Prompt Registry requires to load prompts (USE SCHEMA, EXECUTE, CREATE FUNCTION, and MANAGE on
    the prompt's schema), and `agentbricks experimental models upgrade` optimizes bound prompts.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    key = project.bind_prompt(prompt, key)
    project.write()
    if obj.output == "json":
        render.emit_json({"key": key, "prompt": prompt, "manifest": str(project.path)})
        return
    render.success(
        f"Bound prompt '{prompt}' as '{key}'",
        fields={"agent.toml": str(project.path)},
        next_steps=[("agentbricks deploy <name>", "Grant the app access to it")],
    )


@models.command("unbind-prompt")
@click.argument("key")
@_source_option
@click.pass_obj
def models_unbind_prompt(obj, key: str, source: pathlib.Path) -> None:
    """Remove prompt binding KEY from agent.toml (the prompt itself is left in place)."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    removed = project.unbind_prompt(key)
    project.write()
    if obj.output == "json":
        render.emit_json({"key": key, "removed": removed})
        return
    render.success(f"Unbound prompt '{key}'" if removed else f"No prompt was bound as '{key}'")


@models.command("list")
@click.pass_obj
def models_list(obj) -> None:
    """List the system.ai.* chat models you can route the agent to."""
    names = obj.client().list_chat_model_services()
    if obj.output == "json":
        render.emit_json(names)
        return
    render.resource_table(
        "AI Gateway Models · system.ai", [("Model", "left")], [[n] for n in names]
    )


@models.command("status")
@_source_option
@click.pass_obj
def models_status(obj, source: pathlib.Path) -> None:
    """Show each bound model service, the model behind it now, and the latest upgrade run."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    client = obj.client()
    bound = {
        role: {"model_service": b.name, "model": _current_model(client, b.name)}
        for role, b in project.model_services.items()
    }
    history = model_upgrade.read_history(project.root)
    last_change = history["changes"][-1] if history["changes"] else None
    run = _refresh_latest_run(obj, project) if history["runs"] else None
    if obj.output == "json":
        render.emit_json(
            {
                "model_services": bound,
                "prompts": project.prompts,
                "last_change": last_change,
                "latest_run": run,
            }
        )
        return
    render.resource_table(
        "Model services",
        [("Role", "left"), ("Model service", "left"), ("Current model", "left")],
        [[role, b["model_service"], b["model"]] for role, b in bound.items()],
    )
    if project.prompts:
        render.resource_table(
            "Prompts",
            [("Key", "left"), ("Prompt", "left")],
            [[key, name] for key, name in project.prompts.items()],
        )
    fields = {}
    if last_change:
        fields["Last switch"] = (
            f"{last_change.get('model_service')}: {last_change.get('previous_model')} → "
            f"{last_change.get('model')} ({last_change.get('reason')})"
        )
    if run:
        fields["Latest upgrade run"] = (
            f"{run.get('state')} · {run.get('run_page_url') or run.get('run_id')}"
        )
    render.success("Model status", fields=fields)
    report = run.get("report") if run else None
    if report:
        _render_report(model_upgrade.UpgradeReport.from_json(report))


@models.command("set")
@click.argument("model")
@_role_option
@_yes_option
@_source_option
@click.pass_obj
def models_set(obj, model: str, role: Optional[str], yes: bool, source: pathlib.Path) -> None:
    """Switch one bound model service to MODEL (a system.ai.* name)."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415
    from databricks_agentbricks.model_upgrade import system_ai_name  # noqa: PLC0415

    project = _bound_project(source)
    service = project.model_services[_pick_role(project, role)].name
    model = system_ai_name(model)
    previous = _current_model(obj.client(), service)
    if previous == model:
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": model, "changed": False})
            return
        render.success(f"'{service}' already routes to {model}")
        return
    if not _confirm(obj, yes, f"Switch '{service}' from {previous} to {model}?"):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": previous, "changed": False})
            return
        raise click.Abort()
    _switch(obj, project, service, previous, model, reason="set")
    model_upgrade.record_action(
        project.root,
        "set",
        models=[{"model_service": service, "previous_model": previous, "model": model}],
    )
    if obj.output == "json":
        render.emit_json(
            {"model_service": service, "previous_model": previous, "model": model, "changed": True}
        )
        return
    render.success(
        f"Switched '{service}' to {model}",
        fields={"Previous model": previous},
        next_steps=[("agentbricks experimental models rollback", "Switch back")],
    )


@models.command("rollback")
@click.option(
    "--role",
    default=None,
    help="Roll back only this role's last model switch. Without it, undo the whole last apply or "
    "set: every model it switched and every prompt alias it moved.",
)
@_yes_option
@_source_option
@click.pass_obj
def models_rollback(obj, role: Optional[str], yes: bool, source: pathlib.Path) -> None:
    """Undo the last `models apply` or `models set`.

    Switches every model service the action changed back to its previous model, and moves each
    prompt alias it moved (e.g. @production) back to the version it pointed at before, the one
    @production_previous marks. Run it again to undo the action before that. With --role, only
    that role's last model switch is undone and prompts are left alone.
    """
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    if role is not None:
        _rollback_role(obj, project, project.model_services[_pick_role(project, role)].name, yes)
        return
    found = model_upgrade.last_undoable_action(project.root)
    if found is None:
        raise AgentCliError("No recorded switch or upgrade to roll back.")
    index, action = found
    client = obj.client()
    moved = [
        m["model_service"]
        for m in action["models"]
        if _current_model(client, m["model_service"]) != m["model"]
    ]
    if moved:
        raise AgentCliError(
            f"{', '.join(moved)} switched models after the last {action['kind']}.",
            hint="Roll those back one at a time with `agentbricks experimental models rollback --role <role>`, "
            "or switch them with `agentbricks experimental models set`.",
        )
    undo = [
        f"{m['model_service']}: {m['model']} → {m['previous_model']}" for m in action["models"]
    ] + [f"{p['name']}@{p['alias']} → version {p['prior_version']}" for p in action["prompts"]]
    if not _confirm(obj, yes, f"Undo the last {action['kind']}: " + "; ".join(undo) + "?"):
        if obj.output == "json":
            render.emit_json({"changed": False})
            return
        raise click.Abort()
    for m in action["models"]:
        _switch(
            obj, project, m["model_service"], m["model"], m["previous_model"], reason="rollback"
        )
    if action["prompts"]:
        _restore_prompts(obj, action["prompts"])
    model_upgrade.mark_rolled_back(project.root, index)
    if obj.output == "json":
        render.emit_json(
            {
                "changed": True,
                "undid": action["kind"],
                "models": action["models"],
                "prompts": action["prompts"],
            }
        )
        return
    render.success(f"Rolled back the last {action['kind']}", fields={"Undid": "; ".join(undo)})


def _rollback_role(obj, project, service: str, yes: bool) -> None:
    """Switch one model service back to the model it used before its last switch."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    history = [
        h
        for h in model_upgrade.read_history(project.root)["changes"]
        if h.get("model_service") == service
    ]
    previous = history[-1].get("previous_model") if history else None
    if not previous:
        raise AgentCliError(f"No recorded switch of '{service}' to roll back.")
    current = _current_model(obj.client(), service)
    if not _confirm(obj, yes, f"Switch '{service}' from {current} to {previous}?"):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": current, "changed": False})
            return
        raise click.Abort()
    _switch(obj, project, service, current, previous, reason="rollback")
    if obj.output == "json":
        render.emit_json(
            {
                "model_service": service,
                "previous_model": current,
                "model": previous,
                "changed": True,
            }
        )
        return
    render.success(f"Rolled '{service}' back to {previous}", fields={"Previous model": current})


def _parse_weights(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 3 or any(p < 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise click.BadParameter("use three non-negative numbers summing to 1, e.g. 0.7,0.2,0.1")
    return parts  # ty: ignore[invalid-return-type]


def _parse_candidates(values: tuple[str, ...], roles: list[str]) -> dict[str, list[str]]:
    """``-c router=m1,m2 -c writer=m3`` → {role: [system.ai names]}.

    The ``role=`` prefix is optional when only one role is bound.
    """
    from databricks_agentbricks.model_upgrade import system_ai_name  # noqa: PLC0415

    parsed: dict[str, list[str]] = {}
    for value in values:
        role, sep, models_part = value.partition("=")
        if not sep:
            if len(roles) != 1:
                raise click.BadParameter(
                    f"'{value}': this agent binds several model services, so prefix the candidates "
                    f"with their role, e.g. {roles[0]}={value}",
                    param_hint="--candidates",
                )
            role, models_part = roles[0], value
        role = role.strip()
        if role not in roles:
            raise click.BadParameter(
                f"'{role}' isn't a bound role (bound: {', '.join(roles)}).",
                param_hint="--candidates",
            )
        role_models = parsed.setdefault(role, [])
        for name in models_part.split(","):
            if name.strip() and system_ai_name(name.strip()) not in role_models:
                role_models.append(system_ai_name(name.strip()))
    return parsed


_TERMINAL_STATES = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


def _state_name(value) -> Optional[str]:
    return getattr(value, "value", None) or (str(value) if value is not None else None)


def _run_state(run) -> tuple[Optional[str], Optional[str]]:
    """``(life_cycle_state, result_state)`` of a job run, as plain strings."""
    state = getattr(run, "state", None)
    return (
        _state_name(getattr(state, "life_cycle_state", None)),
        _state_name(getattr(state, "result_state", None)),
    )


def _refresh_latest_run(obj, project) -> Optional[dict]:
    """The latest recorded upgrade run, with its job state refreshed and its report fetched once done."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    runs = model_upgrade.read_history(project.root)["runs"]
    if not runs:
        return None
    entry = dict(runs[-1])
    if entry.get("report") or entry.get("state") in {"FAILED", "CANCELED", "TIMEDOUT"}:
        return entry
    life_cycle, result = _run_state(obj.client().get_run(int(entry["run_id"])))
    entry["state"] = result or life_cycle
    if life_cycle in _TERMINAL_STATES and result == "SUCCESS":
        report = model_upgrade.fetch_report(
            obj.profile, str(project.trace_experiment_name), entry["upgrade_id"]
        )
        if report is not None:
            entry["report"] = report.to_json()
    model_upgrade.update_run(
        project.root, entry["upgrade_id"], state=entry["state"], report=entry.get("report")
    )
    return entry


def _render_report(report) -> None:
    """The upgrade's result: what would change per role, the score change, and what to do next."""
    # Every candidate is a system.ai model, so the tables drop the shared prefix to stay readable.
    short = lambda model: model.removeprefix("system.ai.")  # noqa: E731
    render.resource_table(
        f"Recommendation · {report.train_records} train / {report.val_records} eval records",
        [
            ("Role", "left"),
            ("Model service", "left"),
            ("Current", "left"),
            ("Recommended", "left"),
            ("", "left"),
        ],
        [
            [
                role,
                rec.service,
                short(rec.current_model),
                short(rec.recommended_model),
                "switch" if rec.changed else "keep",
            ]
            for role, rec in report.recommendations.items()
        ],
    )
    for role, rec in report.recommendations.items():
        if len(rec.model_scores) > 1:
            render.resource_table(
                f"Candidates · {role}",
                [("Model", "left"), ("Best score", "right"), ("", "left")],
                [
                    [short(model), f"{score:.3f}", "current" if model == rec.current_model else ""]
                    for model, score in sorted(rec.model_scores.items(), key=lambda kv: -kv[1])
                ],
            )
    delta = report.best_score - report.baseline_score
    fields = {
        "Score (quality, latency, cost)": (
            f"{report.baseline_score:.3f} → {report.best_score:.3f} ({delta:+.3f})"
        ),
        "Models switched": str(sum(rec.changed for rec in report.recommendations.values())),
        "Prompts rewritten": ", ".join(report.prompt_changes) or "none",
    }
    if report.mlflow_run_id:
        fields["MLflow run"] = report.mlflow_run_id
    render.success(
        "Upgrade recommendation" if report.changed else "Already on the best models and prompts",
        fields=fields,
        next_steps=[("agentbricks experimental models apply", "Apply it")]
        if report.changed
        else None,
    )


def _apply_report(obj, project, report, yes: bool) -> dict:
    """Promote ``report`` (asking first unless ``yes``) and record it for `models rollback`.

    Returns ``{"changed": bool, ...}``; ``declined`` is set when the user said no.
    """
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    client = obj.client()
    # promote_to_prod diffs against the models each service had when the search started; if one
    # was switched since, applying would silently stomp that switch.
    moved = [
        rec.service
        for rec in report.recommendations.values()
        if _current_model(client, rec.service) != rec.current_model
    ]
    if moved:
        raise AgentCliError(
            f"{', '.join(moved)} switched models after this upgrade run started.",
            hint="Run `agentbricks experimental models upgrade` again so the recommendation reflects them.",
        )
    changes = [
        f"{rec.service}: {rec.current_model.removeprefix('system.ai.')} → "
        f"{rec.recommended_model.removeprefix('system.ai.')}"
        for rec in report.recommendations.values()
        if rec.changed
    ] + [f"new version of prompt {name}" for name in report.prompt_changes]
    if not changes:
        return {"changed": False}
    if not _confirm(obj, yes, "Apply " + "; ".join(changes) + "?"):
        return {"changed": False, "declined": True}
    prompt_moves = _promote(obj, report)
    model_upgrade.record_action(
        project.root,
        "apply",
        models=[
            {
                "model_service": rec.service,
                "previous_model": rec.current_model,
                "model": rec.recommended_model,
            }
            for rec in report.recommendations.values()
            if rec.changed
        ],
        prompts=prompt_moves,
    )
    for rec in report.recommendations.values():
        if rec.changed:
            model_upgrade.record_change(
                project.root,
                service=rec.service,
                previous=rec.current_model,
                model=rec.recommended_model,
                reason="upgrade",
                report=report,
            )
    return {
        "changed": True,
        "applied": changes,
        "models": {
            role: {
                "model_service": rec.service,
                "previous_model": rec.current_model,
                "model": rec.recommended_model,
            }
            for role, rec in report.recommendations.items()
            if rec.changed
        },
        "prompts": report.prompt_changes,
    }


def _render_applied(result: dict) -> None:
    if not result["changed"]:
        render.success(
            "Not applied; nothing changed"
            if result.get("declined")
            else "The agent is already on the recommended models and prompts",
            next_steps=[("agentbricks experimental models apply", "Apply it later")]
            if result.get("declined")
            else None,
        )
        return
    fields = {"Applied": "; ".join(result["applied"])}
    if result["prompts"]:
        fields["Prior prompt versions"] = "aliased @production_previous"
    render.success(
        "Upgraded the agent",
        fields=fields,
        next_steps=[("agentbricks experimental models rollback", "Undo all of it")],
    )


@models.command("upgrade")
@click.option(
    "--candidates",
    "-c",
    "candidates",
    multiple=True,
    required=True,
    help="system.ai.* models to evaluate for one role, as ROLE=MODEL[,MODEL...] (e.g. "
    "router=claude-haiku-4-5,gpt-5-4-nano). Repeat per role; the ROLE= prefix is optional when "
    "only one role is bound. Each role's current model is always included as its baseline. See "
    "`agentbricks experimental models list`.",
)
@click.option(
    "--prompt",
    "prompt_uris",
    multiple=True,
    help="MLflow Prompt Registry prompt to optimize alongside the model, as a prompts:/ URI (e.g. "
    "prompts:/main.my_agent.system@production). Repeat for more. Defaults to the prompts bound in "
    "agent.toml (`models bind-prompt`) at @production. The agent must load it with load_prompt and "
    "call .format() on each call.",
)
@click.option(
    "--models-only",
    is_flag=True,
    help="Search models only; leave the agent's prompts as they are.",
)
@click.option(
    "--predict",
    "predict_fn",
    required=True,
    help="Your agent's predict_fn, as module:attr in the project (e.g. agent.eval:predict). A sync "
    "callable taking one eval record's inputs dict.",
)
@click.option(
    "--train-data",
    required=True,
    help="Training records, as module:attr: a list of {'inputs': ..., 'expectations': ...}.",
)
@click.option(
    "--val-data",
    required=True,
    help="Validation records, as module:attr, in the same shape as --train-data.",
)
@click.option(
    "--scorer",
    "scorers",
    multiple=True,
    required=True,
    help="Scorer, or list of scorers, as module:attr (repeat for more). MLflow scorers or "
    "(inputs, expectations, answer) -> float callables.",
)
@click.option(
    "--budget",
    type=click.IntRange(min=1),
    default=None,
    help="Evaluation budget (agent runs). Default: 10 x the number of eval records.",
)
@click.option(
    "--weights",
    default="0.7,0.2,0.1",
    show_default=True,
    help="Quality, latency, cost weights for picking the winner.",
)
@click.option(
    "--timeout-hours",
    type=click.FloatRange(min=0.1),
    default=6.0,
    show_default=True,
    help="Cancel the job if it runs longer than this.",
)
@click.option(
    "--run-on",
    type=click.Choice(["job", "local"]),
    default="job",
    show_default=True,
    help="Where the search runs: a serverless Databricks job, or this process (needs the 'upgrade' "
    "extra and the project's dependencies installed here).",
)
@click.option(
    "--apply",
    "apply_mode",
    type=click.Choice(["never", "ask", "auto"]),
    default="never",
    show_default=True,
    help="What to do with the recommendation: never (review it, then `models apply`), ask (wait "
    "for it and ask before changing anything), or auto (wait for it and apply without asking).",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for the job to finish and show its recommendation (implied by --apply ask/auto).",
)
@_source_option
@click.pass_obj
def models_upgrade(
    obj,
    candidates: tuple[str, ...],
    prompt_uris: tuple[str, ...],
    models_only: bool,
    predict_fn: str,
    train_data: str,
    val_data: str,
    scorers: tuple[str, ...],
    budget: Optional[int],
    weights: str,
    timeout_hours: float,
    run_on: str,
    apply_mode: str,
    wait: bool,
    source: pathlib.Path,
) -> None:
    """Search models for each of the agent's LLM calls, and with --prompt its prompts, jointly.

    Uploads the project to your workspace and submits a serverless one-time run. The job imports
    your predict_fn, eval data, and scorers from the project and runs the whole agent on each eval
    record per candidate combination, against temporary copies of the model services (production
    traffic is untouched). The winner balances quality, latency, and cost (--weights); at equal
    quality the cheaper models win.

    By default the search runs as a Databricks job, since it can take hours, and this command
    returns once the job is submitted: check on it with `agentbricks experimental models status`, then switch to
    its recommendation with `agentbricks experimental models apply`. `--run-on local` runs it here instead.
    Nothing changes until the recommendation is applied, by `models apply` or `--apply ask/auto`.
    """
    import uuid  # noqa: PLC0415

    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    parsed_weights = _parse_weights(weights)
    bad_uris = [uri for uri in prompt_uris if not uri.startswith("prompts:/")]
    if bad_uris:
        raise click.BadParameter(
            f"{', '.join(bad_uris)}: use a prompts:/ URI, e.g. prompts:/cat.schema.name@production",
            param_hint="--prompt",
        )
    project = _bound_project(source)
    if not project.trace_experiment_name:
        raise AgentCliError(
            "models upgrade logs next to the agent's trace experiment, but tracing isn't bound.",
            hint="Run `agentbricks tracing bind --experiment-name <path>` first.",
        )
    if models_only and prompt_uris:
        raise click.UsageError("--models-only can't be combined with --prompt.")
    if not prompt_uris and not models_only:
        # The prompts the agent declares it loads: optimize them alongside the models by default.
        prompt_uris = tuple(f"prompts:/{name}@production" for name in project.prompts.values())
    services = {role: b.name for role, b in project.model_services.items()}
    by_role = _parse_candidates(candidates, list(services))
    client = obj.client()
    # Fails fast if a service isn't deployed yet.
    current = {role: _current_model(client, service) for role, service in services.items()}
    # A model service can only route to models its owner can execute; catch that now rather than
    # an hour into the job.
    names = sorted({name for models_ in by_role.values() for name in models_})
    denied = [name for name in names if client.can_execute_model(name) is False]
    if denied:
        raise AgentCliError(
            f"You can't route a model service to: {', '.join(denied)}.",
            hint="A model service can only route to models its owner holds EXECUTE on. Ask a "
            "workspace admin to grant EXECUTE on those system.ai models, or pick others "
            "(`agentbricks experimental models list`).",
        )
    config = model_upgrade.JobConfig(
        upgrade_id=uuid.uuid4().hex[:12],
        services=services,
        candidates=by_role,
        prompt_uris=list(prompt_uris),
        predict_fn=predict_fn,
        train_data=train_data,
        val_data=val_data,
        scorers=list(scorers),
        trace_experiment=str(project.trace_experiment_name),
        budget=budget,
        weights=parsed_weights,
    )
    if run_on == "local":
        run = _run_locally(obj, project, config, services, current, by_role)
    else:
        run = _run_as_job(
            obj, client, project, config, services, current, by_role, timeout_hours,
            wait=wait or apply_mode != "never",
        )  # fmt: skip
        if run is None:  # submitted, not waited on
            return

    report = model_upgrade.UpgradeReport.from_json(run["report"]) if run.get("report") else None
    if report is None:
        raise AgentCliError(
            f"The upgrade run finished as {run.get('state')} without a recommendation.",
            hint=f"See the run: {run.get('run_page_url') or run.get('run_id')}",
        )
    applied = None
    if apply_mode != "never" and report.changed:
        if obj.output != "json":
            _render_report(report)
        applied = _apply_report(obj, project, report, yes=apply_mode == "auto")
    if obj.output == "json":
        render.emit_json({"upgrade_id": config.upgrade_id, "run": run, "applied": applied})
        return
    if applied is None:
        _render_report(report)
    else:
        _render_applied(applied)


def _run_locally(obj, project, config, services, current, by_role) -> dict:
    """Run the search in this process and record it like a finished job run."""
    import os  # noqa: PLC0415

    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    if obj.profile:
        # The search's workspace client and MLflow both resolve auth from the environment.
        os.environ["DATABRICKS_CONFIG_PROFILE"] = obj.profile
    with render.progress("Running the search locally (this can take hours)…"):
        report = model_upgrade.run_and_record(project.root, config)
    entry = {
        "upgrade_id": config.upgrade_id,
        "run_on": "local",
        "run_id": None,
        "model_services": services,
        "current_models": current,
        "candidates": by_role,
        "prompt_uris": config.prompt_uris,
        "state": "SUCCESS",
        "report": report.to_json(),
    }
    model_upgrade.record_run(project.root, entry)
    return entry


def _run_as_job(
    obj, client, project, config, services, current, by_role, timeout_hours, *, wait: bool
) -> Optional[dict]:
    """Upload the project and submit the search as a serverless job; with ``wait``, return the
    finished run, else report the submission and return None."""
    import time  # noqa: PLC0415

    from databricks_agentbricks import model_upgrade  # noqa: PLC0415
    from databricks_agentbricks.databricks_cli import _databricks  # noqa: PLC0415

    ws_path = model_upgrade.workspace_project_path(
        client.current_user, str(project.deployment_name or project.root.name)
    )
    with render.status(f"Uploading the project to {ws_path}…"):
        _databricks(
            ["sync", str(project.root), ws_path, "--exclude", "uv.lock", "--exclude", ".venv"],
            obj.profile,
            capture=True,
            action="Could not upload the project for the upgrade job.",
        )
    with render.status("Submitting the upgrade job…"):
        run_id, run_url = model_upgrade.submit_upgrade_run(
            client, workspace_path=ws_path, config=config, timeout_hours=timeout_hours
        )
    model_upgrade.record_run(
        project.root,
        {
            "upgrade_id": config.upgrade_id,
            "run_on": "job",
            "run_id": run_id,
            "run_page_url": run_url,
            "model_services": services,
            "current_models": current,
            "candidates": by_role,
            "prompt_uris": config.prompt_uris,
            "state": "PENDING",
        },
    )
    if not wait:
        if obj.output == "json":
            render.emit_json(
                {
                    "upgrade_id": config.upgrade_id,
                    "run_id": run_id,
                    "run_page_url": run_url,
                    "run": None,
                }
            )
            return None
        render.success(
            "Submitted the model upgrade job",
            fields={
                "Run": run_url or str(run_id),
                **{f"Candidates ({role})": ", ".join(m) for role, m in by_role.items()},
            },
            next_steps=[
                (
                    "agentbricks experimental models status",
                    "Check on it and see its recommendation",
                ),
                (
                    "agentbricks experimental models apply",
                    "Apply the recommendation once it's done",
                ),
            ],
        )
        return None
    with render.progress("Waiting for the upgrade job (this can take hours; Ctrl-C detaches)…"):
        while True:
            life_cycle, _ = _run_state(client.get_run(run_id))
            if life_cycle in _TERMINAL_STATES:
                break
            time.sleep(30)
    return _refresh_latest_run(obj, project) or {}


@models.command("apply")
@_yes_option
@_source_option
@click.pass_obj
def models_apply(obj, yes: bool, source: pathlib.Path) -> None:
    """Apply the latest finished upgrade run's recommendation: every model and prompt it changed."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    run = _refresh_latest_run(obj, project)
    if run is None:
        raise AgentCliError(
            "No upgrade run to apply.",
            hint="Run `agentbricks experimental models upgrade -c <model>` first.",
        )
    if not run.get("report"):
        raise AgentCliError(
            f"The latest upgrade run is {run.get('state')}; there's no recommendation to apply yet.",
            hint="Check on it with `agentbricks experimental models status`.",
        )
    report = model_upgrade.UpgradeReport.from_json(run["report"])
    result = _apply_report(obj, project, report, yes)
    if result.get("declined") and obj.output != "json":
        raise click.Abort()
    if obj.output == "json":
        render.emit_json(result)
        return
    _render_applied(result)
