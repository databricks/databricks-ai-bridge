"""Resolve published session/tracing bindings and grant only the target App access."""

from databricks.sdk.errors import NotFound, ResourceAlreadyExists
from databricks.sdk.service.ml import ExperimentAccessControlRequest, ExperimentPermissionLevel

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.conversation_store_client import SessionStoreClient
from databricks_agentbricks.errors import AgentCliError


def provision_features(workspace, features, principal, *, profile=None):
    store_name = features["sessions"].get("store")
    if store_name:
        stores = SessionStoreClient(ApiClientProvider(profile))
        resolved = stores.reconcile(store_name)
        if error := stores.grant(resolved.store_name, principal):
            raise AgentCliError(f"Could not grant App access to session store: {error}")
    name = features["tracing"].get("experiment_name")
    if name:
        workspace.workspace.mkdirs(name.rsplit("/", 1)[0])
        try:
            experiment = workspace.experiments.get_by_name(name).experiment
        except NotFound:
            experiment = None
        if experiment is None:
            try:
                experiment_id = workspace.experiments.create_experiment(name).experiment_id
            except ResourceAlreadyExists:
                experiment_id = workspace.experiments.get_by_name(name).experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
        workspace.experiments.update_permissions(
            experiment_id,
            access_control_list=[
                ExperimentAccessControlRequest(
                    service_principal_name=principal,
                    permission_level=ExperimentPermissionLevel.CAN_EDIT,
                )
            ],
        )
        features["tracing"].update(
            experiment_id=experiment_id,
            url=f"{workspace.config.host.rstrip('/')}/#mlflow/experiments/{experiment_id}/traces",
        )
    return features
