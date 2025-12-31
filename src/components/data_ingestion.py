import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # 1. READING RAW DATA
            logging.info("Reading the 3 raw datasets")

            df_clinical = pd.read_csv('notebook/data/X_train/clinical_train.csv')
            df_molecular = pd.read_csv('notebook/data/X_train/molecular_train.csv')
            df_target = pd.read_csv('notebook/data/target_train.csv')

            logging.info("Calculating Nmut (mutations per patient)")
            molecular_counts = df_molecular.groupby('ID').size().reset_index(name='Nmut')

            logging.info("Merging datasets into a single dataframe")
            
            df = pd.merge(df_clinical, df_target, on='ID')
            df = pd.merge(df, molecular_counts, on='ID', how='left')
      
            df['Nmut'] = df['Nmut'].fillna(0)

            logging.info(f"Merged Dataframe shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

