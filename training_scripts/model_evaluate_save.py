import pandas as pd
from training_scripts.utils.logger import Logger
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import cloudpickle
import numpy as np


def evaluate_save_log_model(model, x_train: pd.DataFrame, y_train: pd.Series, 
                   x_test: pd.DataFrame, y_test: pd.Series, parent_mlflow_exp_name: str,
                   model_performance_threshold: float, n_eval_splits: int, 
                   eval_split_sample_frac: float, run_name: str, 
                   log_model: bool, save_model: bool, evaluate_train: bool,
                   model_name_to_save: str, mlflow_model_artifact_path: str):
    '''
    Function to evaluate the model on training and test sets.
    '''
    logger = None
    try:
        logger = Logger()
        logger.log_message(f'Entering evaluate_save_log_model(): {run_name}')
        _, classifier = model
        timestamp = os.getenv('TIMESTAMP')

        # need not set new experiment when using AmlCompute Command as it creates an experiment with given experiment_name
        # only set exp when running locally
        try:
            mlflow.set_experiment(experiment_name=parent_mlflow_exp_name)
        except:
            pass


        train_scores = []
        test_scores = []
        training_sample_size = 0
        for n_split in range(n_eval_splits):
            x_test_split, _, y_test_split, _ = train_test_split(x_test, y_test, stratify=y_test, train_size=eval_split_sample_frac)
            test_split_pred_prob = classifier.predict_proba(x_test_split)

            if evaluate_train:
                training_sample_size = len(x_train)
                x_train_split, _, y_train_split, _ = train_test_split(x_train, y_train, stratify=y_train, train_size=eval_split_sample_frac)
                train_split_pred_prob = classifier.predict_proba(x_train_split)
                train_scores.append(roc_auc_score(y_train_split, train_split_pred_prob, multi_class="ovr"))

            test_scores.append(roc_auc_score(y_test_split, test_split_pred_prob, multi_class="ovr"))

        if save_model:
            with open(model_name_to_save, "wb") as f:
                cloudpickle.dump(model, f)
        
        
        with mlflow.start_run(run_name=f'{run_name}_{timestamp}') as run:
            if log_model:
                mlflow.log_artifact(model_name_to_save, artifact_path=mlflow_model_artifact_path)
            avg_test_score = np.mean(test_scores)
            std_test_score = np.std(test_scores)
            mlflow.log_param("training_sample_size", training_sample_size)
            mlflow.log_metric("Test ROC AUC", float(avg_test_score))
            mlflow.log_metric("Performance Threshold", model_performance_threshold)
            mlflow.log_metric("Test ROC AUC STDEV", float(std_test_score))
            if evaluate_train:
                avg_train_score = np.mean(train_scores)
                std_train_score = np.std(train_scores)
                adj_test_score = avg_test_score + (0.5 * (avg_test_score / avg_train_score))
                mlflow.log_metric("Adusted Test ROC AUC", float(adj_test_score))
                mlflow.log_metric("Train ROC AUC", float(avg_train_score))
                mlflow.log_metric("Train ROC AUC STDEV", float(std_train_score))

        best_model_run_id = run.info.run_id

        
        logger.log_message(f'Exiting evaluate_save_log_model(): {run_name}')
        return best_model_run_id, avg_test_score
    except Exception as e:
        logger.log_message(f'Encountered an unexpected error in evaluate_save_log_model(): {run_name}: {e}', 'CRITICAL')
        raise
    