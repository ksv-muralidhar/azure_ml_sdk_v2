from dotenv import load_dotenv
import os
import logging
from training_scripts.utils.logger import Logger
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from datetime import datetime



logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
azure_identity_logger = logging.getLogger('azure.identity')
azure_identity_logger.setLevel(logging.ERROR)
os.environ['AZURE_LOG_LEVEL'] = 'ERROR'

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_location = os.getenv("WORKSPACE_LOCATION")
environment_name = os.getenv('ENVIRONMENT_NAME')
training_dataset_name = os.getenv("TRAINING_DATASET_NAME")
model_performance_threshold = float(os.getenv("MODEL_PERFORMANCE_THRESHOLD"))
registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
new_model_name_to_save = os.getenv("CLOUDPICKLE_MODEL_NAME_TO_SAVE")

sp_azure_client_id = os.getenv('AZURE_CLIENT_ID')
sp_azure_tenant_id = os.getenv('AZURE_TENANT_ID')
sp_azure_client_secret = os.getenv('AZURE_CLIENT_SECRET')

cpu_compute_target = "train-singlenode-compute"

timestamp = f'{datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "")}'
parent_mlflow_exp_name = f'training_{timestamp}'

env_vars_for_amlcompute = {'TIMESTAMP': timestamp, 
                           'TRAINING_DATASET_NAME': training_dataset_name,
                           'MODEL_PERFORMANCE_THRESHOLD': model_performance_threshold,
                           'SUBSCRIPTION_ID': subscription_id,
                           'RESOURCE_GROUP': resource_group,
                           'WORKSPACE_NAME': workspace_name, 
                           'AZURE_CLIENT_ID': sp_azure_client_id,
                           'AZURE_TENANT_ID': sp_azure_tenant_id,
                           'AZURE_CLIENT_SECRET': sp_azure_client_secret,
                           'REGISTERED_MODEL_NAME': registered_model_name,
                           'CLOUDPICKLE_MODEL_NAME_TO_SAVE': new_model_name_to_save}


ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), subscription_id, resource_group, 
                     workspace_name=workspace_name, workspace_location=workspace_location)

def get_latest_environment(ml_client, env_name):
    logger = None
    try:
        logger = Logger()
        logger.log_message("Entering get_latest_environment()")
        environments = ml_client.environments.list(name=env_name)
        latest_environment = max(environments, key=lambda env: env.version)
        logger.log_message(f"Retrieved Environment with Version: {latest_environment.version}")
        logger.log_message("Exiting get_latest_environment()")
        return latest_environment
    except Exception as e:
        logger.log_message(f"Error in get_latest_environment(): {e}", 'CRITICAL')
        raise

def run_training_job(ml_client, cpu_compute_target: str, min_instances: int, 
                     max_instances: int, timestamp: str, parent_mlflow_exp_name: str,
                     environment_name: str):
    logger = None
    try:
        logger = Logger()
        logger.log_message("Entering run_training_job()")
        try:
            ml_client.compute.get(cpu_compute_target)
        except:
            logger.log_message("Creating a new cpu compute target.")
            compute = AmlCompute(name=cpu_compute_target, size="Standard_A2m_v2", 
                         min_instances=min_instances, max_instances=max_instances)

            ml_client.compute.begin_create_or_update(compute).result()
            logger.log_message("Successfully Created a new cpu compute target.")
            
        command_job = command(code=".",
                            command="python initiate_model_training_from_local.py",
                            environment=get_latest_environment(ml_client, environment_name),
                            compute=cpu_compute_target,
                            name=f"hyperparameter_tuning_{timestamp}",
                            experiment_name=parent_mlflow_exp_name,
                            environment_variables=env_vars_for_amlcompute
                            )

        job = ml_client.jobs.create_or_update(command_job)
        job = ml_client.jobs.get(job.name)
        ml_client.jobs.stream(job.name)

        logger.log_message("Exiting run_training_job()")
    except Exception as e:
        logger.log_message("Error in run_training_job(): {e}", 'CRITICAL')
        raise

if __name__== '__main__':
    run_training_job(ml_client=ml_client, cpu_compute_target=cpu_compute_target,
                        min_instances=0, max_instances=1, 
                        timestamp=timestamp, parent_mlflow_exp_name=parent_mlflow_exp_name,
                        environment_name=environment_name)
    