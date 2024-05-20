from training_scripts.utils.logger import Logger
import tempfile
import cloudpickle
import os
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


def get_current_deployed_model(ml_client, registered_model_name: str, abs_mlflow_model_artifact_path: str):
    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering get_current_deployed_model()')
        current_model = None
        try:
            model_versions = ml_client.models.list(name=registered_model_name)
            next(model_versions)
            model_versions = ml_client.models.list(name=registered_model_name)
            latest_version = max(model_versions, key=lambda d: d.version)
            latest_version = str(latest_version.version)
            with tempfile.TemporaryDirectory() as temp_dir:
                ml_client.models.download(name=registered_model_name, 
                                          version=latest_version, 
                                          download_path=temp_dir)

                model_path = os.path.join(temp_dir, abs_mlflow_model_artifact_path)
                with open(model_path, 'rb') as f:
                    current_model = cloudpickle.load(f)
            logger.log_message(f'Found current deployed model version {latest_version}')
        except:
            current_model = None
        if current_model is None:
            logger.log_message('No current deployed model was found')
        else:
            logger.log_message('Successfully loaded the current deployed model')
        logger.log_message('Exiting get_current_deployed_model()')
        return current_model
    except Exception as e:
        logger.log_message(f'Error in get_current_deployed_model(): {e}')
        raise


def register_new_model(ml_client, new_model_mlflow_run_id: str, 
                       abs_mlflow_model_artifact_path: str, 
                       registered_model_name: str):

    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering register_new_model()')
    
        model_uri = f"runs:/{new_model_mlflow_run_id}/{abs_mlflow_model_artifact_path}"

        file_model = Model(
            path=model_uri,
            type=AssetTypes.CUSTOM_MODEL,
            name=registered_model_name
            )
        ml_client.models.create_or_update(file_model)
        logger.log_message('Successfully registered new model')
        logger.log_message('Exiting register_new_model()')
    except Exception as e:
        logger.log_message(f'Error in register_new_model(): {e}', 'CRITICAL')
        raise
    