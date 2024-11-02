from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_file, evaluate_model
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from catboost import CatBoostRegressor
from dataclasses import dataclass
from xgboost import XGBRegressor
from src.logger import logging
from sklearn import metrics
import sys
import os

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor):
        
        try:
            logging.info('Splitting Training and Testing Data..')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'KNN': KNeighborsRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }
            
            model_report: dict = evaluate_model(X_train = X_train, X_test=X_test,y_train = y_train,y_test=y_test,models=models)
            
            best_scores = max(sorted(list(model_report.values())))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_scores)
            ]
            
            best_model = models[best_model_name]
            
            if best_scores < 0.6: raise CustomException('No Best Model found')
            
            logging.info('Best Model  on testing data')
            
            save_file(
                file_path=self.model_trainer_config.model_trainer_path,
                obj=best_model
                )
            
            predicted = best_model.predict(X_test)
            
            r2_score = metrics.r2_score(y_test,predicted)
            
            return r2_score
            
        except Exception as e:
            raise CustomException(e,sys)    
    