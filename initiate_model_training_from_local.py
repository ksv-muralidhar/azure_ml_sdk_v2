from training_scripts.utils.logger import Logger
import logging
import warnings
import os
from datetime import datetime
from training_scripts.data_preprocessing import DataPreprocessor
from training_scripts.data_ingestion import get_data, get_train_test_data
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import tempfile
from training_scripts.model_training import model_training
from training_scripts.model_evaluate_save import evaluate_save_log_model
from training_scripts.register_model import get_current_deployed_model, register_new_model


def disable_logging():
    try:
        mlflow_logger = logging.getLogger("mlflow")
        mlflow_logger.setLevel(logging.ERROR)
        azure_identity_logger = logging.getLogger('azure.identity')
        azure_identity_logger.setLevel(logging.ERROR)
        sklearn_logger = logging.getLogger("sklearn")
        sklearn_logger.setLevel(logging.ERROR)
        xgb_logger = logging.getLogger("xgboost")
        xgb_logger.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
        warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
        warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
        os.environ['AZURE_LOG_LEVEL'] = 'ERROR'
    except:
        pass

def main():
    logger = None
    try:
        logger = Logger()
        
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")
        workspace_location = os.getenv("WORKSPACE_LOCATION")
        training_dataset_name = os.getenv("TRAINING_DATASET_NAME")
        registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
        new_model_name_to_save = os.getenv("CLOUDPICKLE_MODEL_NAME_TO_SAVE")
        model_performance_threshold = float(os.getenv("MODEL_PERFORMANCE_THRESHOLD"))
        CLASS_NAMES = {0: 'Weak', 1: 'Strong', 2: 'Very strong'}

        training_dataset_version = "latest"
        target_column='strength'
        test_data_frac=0.2
        hyperopt_n_trials = 5
        hyperopt_cv_splits = 3

        new_model_evaluation_mlflow_run_name = "new_model"
        new_model_n_evaluation_splits = 10
        new_model_evaluation_split_sample_frac=0.5
        mlflow_new_model_artifact_path = "model"
        current_model_evaluation_mlflow_run_name = "current_deployed_model"
        current_model_n_evaluation_splits = 10
        current_model_evaluation_split_sample_frac = 0.5
        absolute_mlflow_model_artifact_path = f"{mlflow_new_model_artifact_path}/{new_model_name_to_save}"
        '''
        PARENT_MLFLOW_EXP_NAME env var is not present in secrets.env. It is passed from the Command
        executing this script.
        '''
        # need not set new experiment with name parent_mlflow_exp_name
        # as the AmlCompute Command creates an experiment with given experiment_name
        timestamp = os.getenv('TIMESTAMP')
        if timestamp is None:
           timestamp = f'{datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "")}'
           os.environ['TIMESTAMP'] = timestamp
        parent_mlflow_exp_name = f'training_{timestamp}'




        disable_logging()

        ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), subscription_id, resource_group, 
                            workspace_name=workspace_name, workspace_location=workspace_location)


        data = get_data(ml_client, training_dataset_name, version=training_dataset_version)

        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.fit_transform(data)

        x_train, x_test, y_train, y_test = get_train_test_data(data = data, target=target_column, 
                                                                test_size=test_data_frac)

        new_model = model_training(x_train=x_train, y_train=y_train, x_test=x_test, 
                                                y_test=y_test, n_trials=hyperopt_n_trials, 
                                                cv_splits=hyperopt_cv_splits, 
                                                parent_mlflow_exp_name=parent_mlflow_exp_name)

        new_model_run_id, new_model_test_score = evaluate_save_log_model(model=new_model, x_train=x_train, y_train=y_train, 
                                                    x_test=x_test, y_test=y_test, parent_mlflow_exp_name=parent_mlflow_exp_name, 
                                                    model_performance_threshold=model_performance_threshold,
                                                    n_eval_splits=new_model_n_evaluation_splits,
                                                    eval_split_sample_frac=new_model_evaluation_split_sample_frac,
                                                    run_name=new_model_evaluation_mlflow_run_name,
                                                    log_model=True, save_model=True, evaluate_train=True, 
                                                    model_name_to_save=new_model_name_to_save,
                                                    mlflow_model_artifact_path=mlflow_new_model_artifact_path)
        
        current_deployed_model = get_current_deployed_model(ml_client=ml_client, 
                                                            registered_model_name=registered_model_name, 
                                                            abs_mlflow_model_artifact_path=absolute_mlflow_model_artifact_path)

        if current_deployed_model is not None:
            _, current_model_test_score = evaluate_save_log_model(model=current_deployed_model, x_train=None, y_train=None, 
                                                        x_test=x_test, y_test=y_test, parent_mlflow_exp_name=parent_mlflow_exp_name, 
                                                        model_performance_threshold=model_performance_threshold,
                                                        n_eval_splits=current_model_n_evaluation_splits,
                                                        eval_split_sample_frac=current_model_evaluation_split_sample_frac,
                                                        run_name=current_model_evaluation_mlflow_run_name,
                                                        log_model=False, save_model=False, evaluate_train=False, 
                                                        model_name_to_save=None,
                                                        mlflow_model_artifact_path=None)

            if (new_model_test_score > current_model_test_score) and (new_model_test_score >= model_performance_threshold):
                logger.log_message(f'''New Model Score ({new_model_test_score}) > Current Model Score ({current_model_test_score}) 
                and > Performance Threshold ({model_performance_threshold}). So REGISTERING The New Model''')

                register_new_model(ml_client=ml_client, new_model_mlflow_run_id=new_model_run_id,
                                abs_mlflow_model_artifact_path=absolute_mlflow_model_artifact_path,
                                registered_model_name=registered_model_name)
            else:
                logger.log_message(f'New Model Score ({new_model_test_score}) < Current Model Score ({current_model_test_score}). So NOT REGISTERING The New Model')
        else:
            if new_model_test_score >= model_performance_threshold:
                logger.log_message(f'New Model Score ({new_model_test_score}) >= Performance Threshold ({model_performance_threshold}). So Registering The New Model')
                register_new_model(ml_client=ml_client, new_model_mlflow_run_id=new_model_run_id,
                                    abs_mlflow_model_artifact_path=absolute_mlflow_model_artifact_path,
                                    registered_model_name=registered_model_name)
    except Exception as e:
        logger.log_message(f"Error in application: {e}", 'CRITICAL')
        raise

if __name__ == "__main__":
    main()
    