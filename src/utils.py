import os
import sys
import numpy as np
import pandas as pd
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.info("Error is raised: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(self, X_train, y_train, X_test, y_test, models):
    try:
        evaluation_report = {}

        for i, key in enumerate(list(models.keys())):
            model = models[key]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_train_score = r2_score(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            y_test_score = r2_score(y_test, y_test_pred)

            evaluation_report[key] = y_test_score
        
        return evaluation_report
    
    except Exception as e:
        logging.info("Error is raised: {0}".format(e))
        raise CustomException(e, sys)
    


