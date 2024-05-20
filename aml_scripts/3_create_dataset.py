import os
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
azure_identity_logger = logging.getLogger('azure.identity')
azure_identity_logger.setLevel(logging.ERROR)

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_location = os.getenv("WORKSPACE_LOCATION")
training_datastore_name = os.getenv("TRAINING_DATASTORE_NAME")
training_dataset_filename = os.getenv("TRAINING_DATASET_FILENAME")
training_dataset_name = os.getenv("TRAINING_DATASET_NAME")

ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), 
                     subscription_id, resource_group, 
                     workspace_name=workspace_name, workspace_location=workspace_location)

def get_or_create_dataset(ml_client, training_datastore_name, training_dataset_filename, 
                          training_dataset_name):
    logging.info("Entering get_or_create_dataset()")
    try:
        data_path = f'azureml://datastores/{training_datastore_name}/paths/{training_dataset_filename}'
        
        data_set = Data(path=data_path,
                        type=AssetTypes.URI_FILE,
                        description="",
                        name=training_dataset_name
                        )

        ml_client.data.create_or_update(data_set)
        logging.info("Successfully created/retrieved dataset")
    except Exception as e:
        logging.critical(f"Error in  get_or_create_dataset(): {e}")
    logging.info("Exiting get_or_create_dataset()")

if __name__ == "__main__":
    get_or_create_dataset(ml_client, training_datastore_name, 
                          training_dataset_filename, 
                          training_dataset_name)
 