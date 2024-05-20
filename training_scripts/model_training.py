import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, 
                                   OrdinalEncoder, LabelEncoder)
from sklearn.utils import class_weight
import mlflow
import os
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from hyperopt import tpe, fmin, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from training_scripts.utils.find_class_weights import find_class_weights
from training_scripts.utils.logger import Logger
from training_scripts.utils import hyperopt_search_space
import cloudpickle
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from sklearn.calibration import CalibratedClassifierCV


def find_best_model_params(x: pd.DataFrame, y: pd.Series, cv_splits: int, n_trials: int, 
                           parent_mlflow_exp_name: str):
    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering find_best_model_params()')
        x_train = x.copy()
        y_train = y.copy()
        
        x_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        timestamp = os.getenv('TIMESTAMP')
        
        
        # need not set new experiment when using AmlCompute Command as it creates an experiment with given experiment_name
        # only set exp when running locally
        try:
            mlflow.set_experiment(experiment_name=parent_mlflow_exp_name)
        except:
            pass

        
        def objective(params):
            train_scores = []
            val_scores = []

            mlflow_model_name = "-"
            mlflow_c = "-"
            mlflow_solver = "-"
            mlflow_penalty = "-"
            mlflow_max_depth = "-"
            mlflow_criterion = "-"
            mlflow_n_estimators = "-"
            mlflow_eta = "-"

            skfold = StratifiedKFold(n_splits=cv_splits)

            for train_idx, test_idx in skfold.split(x_train, y_train):

                x_train_hopt = x_train.iloc[train_idx].copy()
                y_train_hopt = y_train[train_idx].copy()
                training_sample_size = len(x_train_hopt)

                x_val_hopt = x_train.iloc[test_idx].copy()
                y_val_hopt = y_train[test_idx].copy()
                
                class_weight_param = find_class_weights(y=y_train_hopt)
                sample_weight_param = class_weight.compute_sample_weight(class_weight="balanced", y=y_train_hopt)

                mlflow_model_name = params['classifier']['type']
                # CLASSIFIER
                classifier = None
                if params['classifier']['type'] == hyperopt_search_space.classifier_params[1]:  # logistic
                    C = params['classifier']['C']
                    solver = params['classifier']['solver_chooser']['solver']
                    penalty = params['classifier']['solver_chooser']['penalty']
                    classifier = LogisticRegression(C=C,
                                                    penalty=penalty,
                                                    solver=solver,
                                                    random_state=42,
                                                    max_iter=10000,
                                                    class_weight=class_weight_param)

                    mlflow_c = C
                    mlflow_solver = solver
                    mlflow_penalty = penalty

                if params['classifier']['type'] == hyperopt_search_space.classifier_params[0]:  # dtree
                    max_depth = params['classifier']['max_depth']
                    criterion = params['classifier']['criterion']
                    classifier = DecisionTreeClassifier(max_depth=max_depth,
                                                        criterion=criterion,
                                                        random_state=42,
                                                        class_weight=class_weight_param)

                    mlflow_max_depth = max_depth
                    mlflow_criterion = criterion

                if params['classifier']['type'] == hyperopt_search_space.classifier_params[2]:  # adaboost
                    n_estimators = params['classifier']['n_estimators']
                    ada_learning_rate = params['classifier']['learning_rate']
                    classifier = AdaBoostClassifier(n_estimators=n_estimators,
                                                    learning_rate=ada_learning_rate,
                                                    random_state=42)

                    mlflow_n_estimators = n_estimators
                    mlflow_eta = ada_learning_rate

                if params['classifier']['type'] == hyperopt_search_space.classifier_params[3]:  # rf
                    n_estimators = params['classifier']['n_estimators']
                    max_depth = params['classifier']['max_depth']
                    criterion = params['classifier']['criterion']
                    classifier = RandomForestClassifier(max_depth=max_depth,
                                                        n_estimators=n_estimators,
                                                        criterion=criterion,
                                                        random_state=42,
                                                        class_weight=class_weight_param)

                    mlflow_max_depth = max_depth
                    mlflow_criterion = criterion
                    mlflow_n_estimators = n_estimators

                if params['classifier']['type'] == hyperopt_search_space.classifier_params[4]:  # xgb
                    n_estimators = params['classifier']['n_estimators']
                    max_depth = params['classifier']['max_depth']
                    xgb_learning_rate = params['classifier']['learning_rate']
                    classifier = XGBClassifier(max_depth=max_depth,
                                            n_estimators=n_estimators,
                                            random_state=42,
                                            learning_rate=xgb_learning_rate, verbosity=0)

                    mlflow_max_depth = max_depth
                    mlflow_n_estimators = n_estimators
                    mlflow_eta = xgb_learning_rate

                if isinstance(classifier, XGBClassifier):
                    classifier.fit(x_train_hopt, y_train_hopt, sample_weight=sample_weight_param,
                                eval_set=[[x_val_hopt, y_val_hopt]], early_stopping_rounds=5, eval_metric='mlogloss', verbose=False)

                elif isinstance(classifier, AdaBoostClassifier):
                    classifier.fit(x_train_hopt, y_train_hopt, sample_weight=sample_weight_param)

                else:
                    classifier.fit(x_train_hopt, y_train_hopt)

                train_pred = classifier.predict_proba(x_train_hopt)
                train_scores.append(roc_auc_score(y_train_hopt, train_pred, multi_class="ovr"))

                val_pred = classifier.predict_proba(x_val_hopt)
                val_scores.append(roc_auc_score(y_val_hopt, val_pred, multi_class="ovr"))


            avg_train_score = np.mean(train_scores)
            std_train_score = np.std(train_scores)
            avg_val_score = np.mean(val_scores)
            std_val_score = np.std(val_scores)
            adj_val_score = avg_val_score + (0.5 * (avg_val_score / avg_train_score))

            with mlflow.start_run(run_name=mlflow_model_name, nested=True):
                mlflow.log_param("model_name", mlflow_model_name)
                mlflow.log_param("training_sample_size", training_sample_size)
                mlflow.log_param("C", float(mlflow_c) if type(mlflow_c)!=str else mlflow_c)
                mlflow.log_param("solver", mlflow_solver)
                mlflow.log_param("penalty", mlflow_penalty)
                mlflow.log_param("max_depth", 
                                            int(mlflow_max_depth) if type(mlflow_max_depth)!=str 
                                            else mlflow_max_depth)
                mlflow.log_param("criterion", mlflow_criterion)
                mlflow.log_param("n_estimators", 
                                               int(mlflow_n_estimators) if type(mlflow_n_estimators)!=str
                                                else mlflow_n_estimators)
                mlflow.log_param("eta", float(mlflow_eta) if type(mlflow_eta)!=str else mlflow_eta)

                mlflow.log_metric("Adusted Test ROC AUC", float(adj_val_score))
                mlflow.log_metric("Train ROC AUC", float(avg_train_score))
                mlflow.log_metric("Train ROC AUC STDEV", float(std_train_score))
                mlflow.log_metric("Test ROC AUC", float(avg_val_score))
                mlflow.log_metric("Test ROC AUC STDEV", float(std_val_score))

            return {'loss': -adj_val_score, 'train_score': avg_train_score, 
                    'val_score': avg_val_score, 'status': STATUS_OK}

        
        with mlflow.start_run(run_name=f"hyperparameter_tuning_{timestamp}"):
            model_trials = Trials()
            best_trial = fmin(
                fn=objective,
                space=hyperopt_search_space.hyperopt_search_space,
                algo=tpe.suggest,
                max_evals=n_trials,
                trials=model_trials,
                catch_eval_exceptions=False
                )
            

        logger.log_message(f"Best Trial: {best_trial}")
        logger.log_message('Exiting find_best_model_params()')
        return best_trial
    except Exception as e:
        logger.log_message(f'Error in find_best_model_params(): {e}', 'CRITICAL')
        raise




def model_training(x_train: pd.DataFrame, y_train: pd.Series, 
                   x_test: pd.DataFrame, y_test: pd.Series, 
                   cv_splits: int, n_trials: int, parent_mlflow_exp_name: str):
    """
    Trains a model from scratch using hyperopt hyperparameter tuning.
    """
    logger = None
    try:
        logger = Logger()
        x_train = x_train.copy()
        y_train = y_train.copy()
        x_test = x_test.copy()
        y_test = y_test.copy()

        logger.log_message('Entering model_training()')

        classifier = None

        # LABEL ENCODER
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train))

        best_trial = find_best_model_params(x=x_train, y=y_train, cv_splits=cv_splits, n_trials=n_trials, 
                                                    parent_mlflow_exp_name=parent_mlflow_exp_name)
        
        class_weight_param = find_class_weights(y=y_train)
        sample_weight_param = class_weight.compute_sample_weight(class_weight="balanced", y=y_train)
        
        # CLASSIFIER
        if hyperopt_search_space.classifier_params[best_trial['classifier']] == 'dtree':
            max_depth = hyperopt_search_space.max_depth_params[best_trial['dtree_max_depth']]
            criterion = hyperopt_search_space.criterion_params[best_trial['dtree_criterion']]
            classifier = DecisionTreeClassifier(max_depth=max_depth,
                                                criterion=criterion,
                                                random_state=42,
                                                class_weight=class_weight_param)

        if hyperopt_search_space.classifier_params[best_trial['classifier']] == 'adaboost':
            n_estimators = hyperopt_search_space.n_estimator_params[best_trial['ada_n_estimators']]
            ada_learning_rate = hyperopt_search_space.learning_rate_params[best_trial['ada_learning_rate']]
            classifier = AdaBoostClassifier(n_estimators=n_estimators,
                                            learning_rate=ada_learning_rate,
                                            random_state=42)

        if hyperopt_search_space.classifier_params[best_trial['classifier']] == 'rf':
            n_estimators = hyperopt_search_space.n_estimator_params[best_trial['rf_n_estimators']]
            max_depth = hyperopt_search_space.max_depth_params[best_trial['rf_max_depth']]
            criterion = hyperopt_search_space.criterion_params[best_trial['rf_criterion']]
            classifier = RandomForestClassifier(max_depth=max_depth,
                                                n_estimators=n_estimators,
                                                criterion=criterion,
                                                random_state=42,
                                                class_weight=class_weight_param)

        if hyperopt_search_space.classifier_params[best_trial['classifier']] == 'xgb':
            n_estimators = hyperopt_search_space.n_estimator_params[best_trial['xgb_n_estimators']]
            max_depth = hyperopt_search_space.max_depth_params[best_trial['xgb_max_depth']]
            xgb_learning_rate = hyperopt_search_space.learning_rate_params[best_trial['xgb_learning_rate']]
            classifier = XGBClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    random_state=42,
                                    learning_rate=xgb_learning_rate, verbosity=0)

        if hyperopt_search_space.classifier_params[best_trial['classifier']] == 'logistic':  # logistic
            C = hyperopt_search_space.C_params[best_trial['C']]
            solver = None
            penalty = None
            
            if best_trial['solver_chooser'] == 0:
                solver = hyperopt_search_space.solver0_params[best_trial['solver0']]
                penalty = hyperopt_search_space.penalty0_params[best_trial['penalty0']]
            if best_trial['solver_chooser'] == 1:
                solver = hyperopt_search_space.solver1_params[best_trial['solver1']]
                penalty = hyperopt_search_space.penalty1_params[best_trial['penalty1']]
            classifier = LogisticRegression(C=C, solver=solver, penalty=penalty,
                                            random_state=42, max_iter=10000,
                                            class_weight=class_weight_param)

        logger.log_message(f'Best Model: {classifier}')
        

        if isinstance(classifier, XGBClassifier):
            classifier.fit(x_train, y_train, sample_weight=sample_weight_param,
                        eval_set=[[x_test, y_test]], early_stopping_rounds=5, eval_metric='mlogloss', verbose=False)

        elif isinstance(classifier, AdaBoostClassifier):
            classifier.fit(x_train, y_train, sample_weight=sample_weight_param)

        else:
            classifier.fit(x_train, y_train)

        logger.log_message('Calibrating Classifier')
        calibrated_classifier = CalibratedClassifierCV(estimator=classifier, method='isotonic', cv='prefit')
        calibrated_classifier.fit(x_test, y_test)
        logger.log_message('Successfully Calibrated Classifier')
        best_model = (label_encoder, calibrated_classifier)

        logger.log_message('Exiting model_training()')

        return best_model
    except Exception as e:
        logger.log_message(f"Error in model_training(): {e}")
        raise
    