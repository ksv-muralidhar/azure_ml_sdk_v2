import os
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Workspace


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
azure_identity_logger = logging.getLogger('azure.identity')
azure_identity_logger.setLevel(logging.ERROR)

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_location = os.getenv("WORKSPACE_LOCATION")
workspace_name = os.getenv("WORKSPACE_NAME")

ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), 
                     subscription_id, resource_group)

def list_workspaces(ml_client):
    logging.info("Entering list_workspaces()")
    try:
        ws_list = []
        for ws in ml_client.workspaces.list():
            ws_list.append(ws.name)
    except Exception as e:
        logging.critical(f"Error in list_workspaces(): {e}")
        raise
    logging.info("Exiting list_workspaces()")
    return ws_list

def get_or_create_workspace(ml_client, workspace_name, workspace_location,
                            subscription_id, resource_group):
    logging.info("Entering get_or_create_workspace()")
    try:
        ws_list = list_workspaces(ml_client)
        if workspace_name not in ws_list:

            ws = Workspace(
                name=workspace_name,
                location=workspace_location,
                display_name=workspace_name,
                description=workspace_name,
                hbi_workspace=False
            )

            ws = ml_client.workspaces.begin_create(ws).result()
            logging.info("Successfully created workspace")
        else:
            ws = MLClient(DefaultAzureCredential(),
                          subscription_id="subscription_id",
                          resource_group_name=resource_group,
                          workspace_name=workspace_name,
                         )
            logging.info("Workspace already exists")
    except Exception  as e:
        logging.critical(f"Error in get_or_create_workspaces(): {e}")
        raise
    logging.info("Exiting get_or_create_workspace()")

if __name__ == "__main__":
    get_or_create_workspace(ml_client, workspace_name, workspace_location,
                            subscription_id, resource_group)
