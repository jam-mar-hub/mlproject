import os
import sys
from dataclasses import dataclass

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # --- 1. DATA PREPARATION ---
            # X = All columns except the last two
            X_train = train_array[:, :-2]
            X_test = test_array[:, :-2]

            # y = The last two columns (Status and Time)
            
            y_train_status = train_array[:, -2].astype(bool) 
            y_train_time = train_array[:, -1]               
            
            y_test_status = test_array[:, -2].astype(bool)
            y_test_time = test_array[:, -1]

            # Create the structured array expected by the model
            y_train = Surv.from_arrays(event=y_train_status, time=y_train_time)
            y_test = Surv.from_arrays(event=y_test_status, time=y_test_time)

            logging.info("Input data formatted for Survival Analysis")

            # --- 2. SINGLE MODEL DEFINITION ---
            models = {
                "Random Survival Forest": RandomSurvivalForest(
                    random_state=42, 
                    n_jobs=-1  
                )
            }

            # --- 3. HYPERPARAMETER GRID (GRID SEARCH) ---
    
            params = {
                "Random Survival Forest": {
                    'n_estimators': [50, 100, 150],      
                    'min_samples_leaf': [5, 10, 20],     
                    'max_features': ['sqrt'],            
                    'max_depth': [None, 10]            
                }
            }

            # --- 4. EVALUATION VIA UTILS (With GridCV) ---
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Retrieve the score of the (single) model
            best_model_score = max(sorted(model_report.values()))

            # Retrieve the model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            logging.info(f"Random Forest trained. Best C-Index found: {best_model_score}")

            if best_model_score < 0.5:
                raise CustomException("Model is not better than random guessing")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # --- 5. FINAL SCORE ---
            # Return the C-Index on the test set
            final_score = best_model.score(X_test, y_test)
            return final_score
            
        except Exception as e:
            raise CustomException(e, sys)