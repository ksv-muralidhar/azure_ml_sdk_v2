import os
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml.entities import AccountKeyConfiguration


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
azure_identity_logger = logging.getLogger('azure.identity')
azure_identity_logger.setLevel(logging.ERROR)

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_location = os.getenv("WORKSPACE_LOCATION")
training_datastore_name = os.getenv("TRAINING_DATASTORE_NAME")
training_blob_storage_account_name = os.getenv("TRAINING_BLOB_STORAGE_ACCOUNT_NAME")
training_blob_container_name = os.getenv("TRAINING_BLOB_CONTAINER_NAME")
training_storage_account_key = os.getenv("TRAINING_STORAGE_ACCOUNT_ACCOUNT_KEY")

ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), subscription_id, resource_group, 
                     workspace_name=workspace_name, workspace_location=workspace_location)

def get_or_create_datastore(ml_client, training_datastore_name, training_blob_storage_account_name, 
                            training_blob_container_name, training_storage_account_key):
    logging.info("Entering get_or_create_datastore()")
    try:
        store = AzureBlobDatastore(name=training_datastore_name,
                                   description="",
                                   account_name=training_blob_storage_account_name,
                                   container_name=training_blob_container_name,
                                   protocol="https",
                                   credentials=AccountKeyConfiguration(account_key=training_storage_account_key)
                                  )

        ml_client.create_or_update(store)
        logging.info("Successfully created datastore")
    except Exception as e:
        logging.critical(f"Error in get_or_create_datastore(): {e}")
        raise
    logging.info("Exiting get_or_create_datastore()")

if __name__ == "__main__":
    get_or_create_datastore(ml_client, training_datastore_name, training_blob_storage_account_name, 
                            training_blob_container_name, training_storage_account_key)
   