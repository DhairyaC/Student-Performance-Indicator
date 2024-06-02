import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor 
)

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into X and Y")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor()
            }
            logging.info("All the models initialized")

            logging.info("Evaluating the models and generating model evaluation report")
            evaluation_report: dict = evaluate_models(self, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            logging.info("Fetching the model with best r2_score")
            best_model_score = max(sorted(evaluation_report.values()))
            best_model_name = list(evaluation_report.keys())[list(evaluation_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                logging.info("No best model found")

            logging.info("Best model found and saving the model")
            model = models[best_model_name]

            save_object(
                obj = model, 
                file_path = self.model_trainer_config.trained_model_path
            )
            logging.info("Model saved")

            return evaluation_report, model
        
        except Exception as e:
            logging.info("Error is raised: {e}")
            raise CustomException(e, sys)