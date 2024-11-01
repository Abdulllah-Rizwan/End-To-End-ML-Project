from sklearn.model_selection import train_test_split
from src.exception import CustomException
from dataclasses import dataclass
from src.logger import logging
import pandas as pd
import sys
import os

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion')
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Dataset Has been read')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split initialized')
            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion is completed')
            
            return ( 
                    self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path 
                    )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()