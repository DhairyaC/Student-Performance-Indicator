import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        '''
        This function initiates the data ingestion.

        returns: Tuple of three paths (train, test, and raw data)
        '''
        logging.info("Initiating the data ingestion process")
        try:
            # Read data from the source (database, local files, etc)
            df = pd.read_csv("data/StudentsPerformance.csv")
            logging.info("Data is imported and read as a dataframe")

            # Creating directories
            logging.info("Creating artifacts directory and save the raw data")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into train and test
            logging.info("Initiating train/test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(f"Exception is raised: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    di_obj = DataIngestion()
    train_data, test_data = di_obj.initiate_data_ingestion()

    dt_obj = DataTransformation()
    dt_obj.initiate_data_transformation(train_data, test_data)







