from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_file
from src.logger import logging
import pandas as pd
import numpy as np
import sys
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features = [
                'writing_score', 
                'reading_score', 
                'math_score',
                'average_scores'
                ]
            categorical_features = [
                'gender',
                'race_ethnicity',
                'lunch',
                'test_preparation_course',
                'parental_level_of_education'
            ]
            
            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Standard Scaling is done for numerical features')
            
            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )
            logging.info('One Hot Encoding is done for categorical features')
            
            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_features),
                ('categorical_pipeline',categorical_pipeline,categorical_features)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)

    def instantiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read test and training data successfully')
            
            logging.info('Obtaining Preprocessor Object')
            preprocesing_obj = self.get_data_transformer_obj()
            
            target_column = 'total_scores'
            numerical_column = [               
                'writing_score', 
                'reading_score', 
                'math_score',
                'average_scores']
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info('Applying preprocessor object on train and test dataframe')
            input_feature_train_arr = preprocesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info('Saved Preprocessor file')
            
            save_file(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)