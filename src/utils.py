import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    '''
    Saves any object to a file using dill so it can be loaded and reused later
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        y_train_log = np.log(y_train)

        for i in range (len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train_log)
            y_train_pred = np.exp(model.predict(X_train))
            y_test_pred = np.exp(model.predict(X_test))

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    #for commit