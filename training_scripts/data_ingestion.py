from training_scripts.utils.logger import Logger
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(ml_client, training_dataset_name: str, version: str):
    '''
    Read the input data from CSV as a data frame
    '''
    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering get_data()')
        if version == "latest":
            dataset_versions = ml_client.data.list(name=training_dataset_name)
            latest_version = max(dataset_versions, key=lambda d: d.version)
            data_set = ml_client.data.get(training_dataset_name, version=latest_version.version)
            logger.log_message(f'Retrieved Latest Dataset With Version: {latest_version.version}', 'DEBUG')
        else:
            data_set = ml_client.data.get(training_dataset_name, version=version)
            logger.log_message(f'Retrieved Dataset With Version: {version}', 'DEBUG')
        
        data = pd.read_csv(data_set.path)
        logger.log_message('Exiting get_data()')
        return data
    except Exception as e:
        logger.log_message(f'Encountered an unexpected error in get_data()\n{e}', 'CRITICAL')
        raise # log the error and raise it to abort the execution


def get_train_test_data(data: pd.DataFrame, target: str,
                        test_size:float, random_state:int=42):
    logger = None
    try:
        logger = Logger()
        logger.log_message('Entering get_train_test_data()')
        y = data[target].copy() # strength is the target column
        x = data.drop(columns=[target, 'password']) # drop cols 'strength' and 'password'
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        logger.log_message(f'\nx_train_shape: {x_train.shape}\nx_test_shape: {x_test.shape}\n' +
        f'y_train_shape: {y_train.shape}\ny_test_shape: {y_test.shape}\n', 'DEBUG')
        logger.log_message('Exiting get_train_test_data()')
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.log_message(f'Encountered an unexpected error in get_train_test_data()\n{e}', 'CRITICAL')
        raise # log the error and raise it to abort the execution
