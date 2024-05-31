import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTranformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_tranformer_config = DataTranformationConfig()

    def get_preprocessor(self):
        '''
        This function defines: 
        1. Numerical and Categorical features
        2. Preprocessing pipelines for numerical and categorical features

        returns: Preprocessor of type ColumnTransformer that contains the two preprocessing pipelines for each feature type 
        '''
        try:
            logging.info("Defining numerical and categorical features")
            num_features = ["writing score", "reading_score"]
            cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            logging.info("Creating data transformation pipeline for each feature")
            num_pipeline = Pipeline(
                steps = [
                    "Imputer", SimpleImputer(strategy='median'),
                    "StandardScaler", StandardScaler()
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    "Imputer", SimpleImputer(strategy='most_frequent'),
                    "OneHotEncoder", OneHotEncoder(),
                    "StandardScaler", StandardScaler()
                ]
            )
            logging.info("Data transformation pipeline created")

            preprocessor = ColumnTransformer(
                ("Numerical Feature Transformation", num_pipeline, num_features),
                ("Categorical Feature Transformation", cat_pipeline, cat_features)
            )
            logging.info("Preprocessor created")

            return preprocessor
        except Exception as e:
            logging.info(f"Error is raised: {e}")
            raise CustomException(e, sys)