from src.exception import CustomException
from src.logger import logging
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