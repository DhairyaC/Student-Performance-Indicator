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
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    def get_preprocessor(self):
        '''
        This function defines: 
        1. Numerical and Categorical features
        2. Preprocessing pipelines for numerical and categorical features

        returns: Preprocessor of type ColumnTransformer that contains the two preprocessing pipelines for each feature type 
        '''
        try:
            logging.info("Defining numerical and categorical features")
            num_features = ["writing score", "reading score"]
            cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            logging.info("Creating data transformation pipeline for each feature")
            num_pipeline = Pipeline(
                steps = [
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("StandardScaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("StandardScaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Data transformation pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical Feature Transformation", num_pipeline, num_features),
                    ("Categorical Feature Transformation", cat_pipeline, cat_features)
                ]
            )

            logging.info("Preprocessor created")

            return preprocessor
        
        except Exception as e:
            logging.info(f"Error is raised: {e}")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        '''
        This function performs data transformation on the train and test data.

        returns: 
            train_arr: transformed array of train set
            test_arr: transformed array of test set
            preprocessor_obj: preprocessor object used to transform the data
        '''
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            preprocessing_obj = self.get_preprocessor()
            logging.info("Preprocessor is loaded")

            target_feature = "math score"

            input_train_df = train_df.drop(columns=[target_feature], axis = 1)
            input_test_df = test_df.drop(columns=[target_feature], axis = 1)

            output_train_df = train_df[target_feature]
            output_test_df = test_df[target_feature]

            logging.info("Applying preprocessing object on train and test df")
            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(output_train_df)]
            test_arr = np.c_[input_test_arr, np.array(output_test_df)]

            logging.info("Data has been transformed")
            logging.info("Saving preprocessor object")

            save_object(
                obj = preprocessing_obj,
                file_path = self.data_transformer_config.preprocessor_path
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_path
            )
        
        except Exception as e:
            logging.info("Error is raised: {e}")
            raise CustomException(e, sys)