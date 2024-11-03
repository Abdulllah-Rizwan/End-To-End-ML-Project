from src.exception import CustomException
from src.utils import load_object
import pandas as pd
import sys
import os
import os

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:    
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class Customdata:
    def __init__(
        self,
        gender,
        lunch,
        race_ethnicity,
        parental_level_of_education,
        test_preparation_course,
        writing_score,
        reading_score,

    ):
        self.gender = gender
        
        self.lunch = lunch
        
        self.race_ethnicity = race_ethnicity
        
        self.parental_level_of_education = parental_level_of_education
        
        self.reading_score = reading_score
        
        self.writing_score = writing_score
                
        self.test_preparation_course = test_preparation_course
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)