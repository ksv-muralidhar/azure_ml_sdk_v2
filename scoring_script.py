import os
import logging
import json
import cloudpickle
import pandas as pd
import numpy as np
from training_scripts.data_preprocessing import DataPreprocessor
from training_scripts.utils.logger import Logger


def init():
    logger = None
    try:
        logger = Logger()
        logger.log_message("Entering init()")
        global label_encoder_model
        global data_preprocessor
        global cloudpickle_model_name
        global class_names
        global init_error
        
        class_names = {0: 'Weak', 1: 'Strong', 2: 'Very strong'}
        cloudpickle_model_name = os.getenv("CLOUDPICKLE_MODEL_NAME_TO_SAVE")
        data_preprocessor = DataPreprocessor()
        init_error = 0
        
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), cloudpickle_model_name)
        
        with open(model_path, 'rb') as f:
            label_encoder_model = cloudpickle.load(f)
        logger.log_message("init() Successful")
        logger.log_message("Exiting init()")
    except Exception as e:
        init_error = str(e)


def run(raw_data):
    logger = None
    status = 200
    error_msg = ""
    pred_confidence = []
    input_pwd = []
    result = []
    try:        
        data = json.loads(raw_data)["data"]  
        if type(data) == str:
            input_pwd = [data]
            data = pd.DataFrame(data, index=[0], columns=['password'])
        if type(data) == list:
            input_pwd = data
            data = pd.DataFrame(data, columns=['password'])
        
        logger = Logger()
        logger.log_message("Succesfully Parsed Input Request")

        
        if init_error != 0:
            raise Exception(f"init error(): {init_error}")

        label_encoder, model = label_encoder_model
        data = data_preprocessor.transform(data)
        data.drop(columns=['password'], inplace=True)
        pred_prob = model.predict_proba(data)
        pred_confidence = [*np.max(pred_prob, axis=1)]
        pred_label = [*np.argmax(pred_prob, axis=1)]
        
        result = [class_names[l] for l in pred_label]
        logger.log_message("Request Processed Successfully")
        logger.log_message("Exiting run()")
    except Exception as e:
        status = 500
        error_msg = str(e)
    return {"input": input_pwd, "strength": result, "confidence": pred_confidence, "status": status, "message": error_msg}
