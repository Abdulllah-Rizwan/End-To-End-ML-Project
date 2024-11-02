from src.exception import CustomException
from src.logger import logging
from sklearn import metrics
import pandas as pd
import numpy as np
import dill
import sys
import os

def save_file(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, 'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_model(X_train,X_test,y_train,y_test,models):
    report = {}
    
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test)
            
            r2_score = metrics.r2_score(y_test, y_pred)
            
            report[list(models.keys())[i]] = r2_score
            
            return report
        
    except Exception as e:
        raise CustomException(e,sys)