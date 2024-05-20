from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os
import datetime
from training_scripts.utils.logger import Logger
import tempfile
from azure.ai.ml.entities import (ManagedOnlineEndpoint,
                                  ManagedOnlineDeployment,
                                  CodeConfiguration)


subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_location = os.getenv("WORKSPACE_LOCATION")
registered_env_name = os.getenv('ENVIRONMENT_NAME')
registered_model_name = os.getenv('REGISTERED_MODEL_NAME')
cloudpickle_model_name = os.getenv("CLOUDPICKLE_MODEL_NAME_TO_SAVE")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_name = os.getenv("ENDPOINT_NAME")

ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), subscription_id, resource_group, 
                            workspace_name=workspace_name, workspace_location=workspace_location)


def get_current_deployed_resource_name(ml_client, resource_type: str, registered_resource_name: str):
    logger = None
    try:
        logger = Logger()
        logger.log_message(f'Entering get_current_deployed_resource_name(): {resource_type}')
        latest_resource_name = None
        try:
            if resource_type not in ['model', 'environment']:
                resource_type = 'environment'
            
            if resource_type == 'model':
                resources = ml_client.models.list(name=registered_resource_name)
                next(resources)
                resources = ml_client.models.list(name=registered_resource_name)
            else:
                resources = ml_client.environments.list(name=registered_resource_name)
                next(resources)
                resources = ml_client.environments.list(name=registered_resource_name)
            
            latest_version = max(resources, key=lambda d: d.version)
            latest_version = str(latest_version.version)
        except:
            latest_version = None

        if latest_version is None:
            logger.log_message(f'No current deployed {resource_type} was found')
        else:
            latest_resource_name = f'azureml:{registered_resource_name}:{latest_version}'
            logger.log_message(f'Current deployed {resource_type} name: {latest_resource_name}')
        logger.log_message(f'Exiting get_current_deployed_resource_name(): {resource_type}')
        return latest_resource_name
    except Exception as e:
        logger.log_message(f'Entering get_current_deployed_resource_name(): {resource_type}: {e}')
        raise


def deploy_endpoint(ml_client, registered_model_name: str, 
                    registered_env_name: str, 
                    endpoint_name: str, deployment_name: str):
    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering deploy_endpoint()')
        latest_model = get_current_deployed_resource_name(ml_client=ml_client, 
                                                        resource_type='model', 
                                                        registered_resource_name=registered_model_name)

        latest_env = get_current_deployed_resource_name(ml_client=ml_client, 
                                                        resource_type='environment', 
                                                        registered_resource_name=registered_env_name)

        deployment_desc = f'Model = {latest_model.replace("azureml:", "").replace(":", "-")}'
        endpoint_desc = f'Deployment = {deployment_name}, Model = {latest_model.replace("azureml:", "").replace(":", "-")}'

        endpoint = ManagedOnlineEndpoint(name = endpoint_name, auth_mode="key", 
                                         description=endpoint_desc)

        ml_client.online_endpoints.begin_create_or_update(endpoint=endpoint).result()
        deployed_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        if deployed_endpoint.provisioning_state.lower() != 'succeeded':
            raise Exception("Endpoint Creation/Updation Failed")
        else:
            logger.log_message("Endpoint Creation/Updation Successful")
        
        
        deployment = ManagedOnlineDeployment(name=deployment_name,
                                            endpoint_name=endpoint_name,
                                            description=deployment_desc,
                                            model=latest_model,
                                            environment=latest_env,
                                            code_configuration=CodeConfiguration(code=".", scoring_script="scoring_script.py"),
                                            instance_type="Standard_DS2_v2",
                                            instance_count=1,
                                            environment_variables={'CLOUDPICKLE_MODEL_NAME_TO_SAVE': cloudpickle_model_name}
                                            )
        
        
        ml_client.online_deployments.begin_create_or_update(deployment=deployment).result()
        created_deployment = ml_client.online_deployments.get(name=deployment_name)
        if created_deployment.provisioning_state.lower() != 'succeeded':
            raise Exception("Deployment Creation/Updation Failed")
        else:
            logger.log_message("Deployment Creation/Updation Successful")

        
        logger.log_message('Exiting deploy_endpoint()')
    except Exception as e:
        logger.log_message('Exiting deploy_endpoint()', 'CRITICAL')
        raise

if __name__== '__main__':
    deploy_endpoint(ml_client=ml_client, 
                    registered_model_name=registered_model_name, 
                    registered_env_name=registered_env_name, 
                    endpoint_name=endpoint_name, 
                    deployment_name=deployment_name) 
 