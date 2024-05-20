from training_scripts.utils.logger import Logger
import pandas as pd
import re


class DataPreprocessor:
    def __init__(self):
        pass
    
    def fit(self, x: pd.DataFrame):
        return self
    
    @staticmethod
    def transform(x: pd.DataFrame):
        logger = None
        try:
            logger = Logger()
            logger.log_message('Entering DataPreprocessor transform()')
            x = x.copy()
            x['len'] = x['password'].map(len) # Add password length
            # add count of numbers in password
            x['num_count'] = x['password'].map(lambda x: len(re.findall(r'\d', x))) 
            # add count of uppercase chars in password
            x['uppercase_count'] = x['password'].map(lambda x: len(re.findall(r'[A-Z]', x)))
            # add count of special chars in password
            x['special_char_count'] = x['password'].map(lambda x: len(re.findall(r'[^\w]|_', x)))
            logger.log_message('Exiting DataPreprocessor transform()')
            return x
        except Exception as e:
            logger.log_message(f'Encountered an unexpected error in DataPreprocessor transform()\n{e}', 'CRITICAL')
            raise # log the error and raise it to abort the execution
    
    def fit_transform(self, x: pd.DataFrame):
        x = self.transform(x)
        return x
    
    @staticmethod
    def __repr__():
        return f"DataPreprocessor()"
