import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation configuration
        '''
        try:
            # 1. Column Definition
            numerical_columns = [
                "BM_BLAST", 
                "WBC", 
                "ANC", 
                "MONOCYTES", 
                "HB", 
                "PLT", 
                "Nmut"  
            ]
            
            categorical_columns = [
                "CENTER"
            ]

            # 2. Numerical Pipeline
            # Median Imputation + Standardization
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()) 
                ]
            )

            # 3. Categorical Pipeline
            # Mode Imputation + OneHotEncoding

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # handle_unknown='ignore' allows the model to handle new categories in test data
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # 4. Creating the Global Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Reading train and test files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # --- PRELIMINARY CLEANING ---
            # Cleaning targets before transformation
            logging.info("Cleaning targets before transformation")
            
            for df in [train_df, test_df]:
                # Drop rows where survival info is missing
                df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
                # Ensure boolean is readable (0/1) for models
                df['OS_STATUS'] = df['OS_STATUS'].astype(int) 
                df['OS_YEARS'] = pd.to_numeric(df['OS_YEARS'], errors='coerce')

            # --- DEFINING TARGETS AND FEATURES ---
            target_column_names = ["OS_STATUS", "OS_YEARS"]
            
            # Dropping ID and CYTOGENETICS from training dataset
            drop_columns = target_column_names + ["ID", "CYTOGENETICS"]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_names]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_names]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Feature Transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # --- SPARSE MATRIX BUG FIX ---
            # If OneHotEncoder returns a sparse matrix, convert to dense
            # otherwise np.c_ will fail with a dimension error.
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # --- FINAL CONCATENATION ---
            # Concatenating features and targets into a large numpy array
            # The last 2 columns will be OS_STATUS and OS_YEARS
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)