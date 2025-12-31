import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        '''
        Cette fonction lance tout le processus d'entraînement de A à Z.
        '''
        try:
            print(">> 1. Starting Data Ingestion")
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()

            print(">> 2. Starting Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            print(">> 3. Starting Model Training")
            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            print("\n" + "="*50)
            print(f" TRAINING PIPELINE COMPLETED SUCCESSFULL")
            print(f" Final C-Index Score: {score:.4f}")
            print("="*50 + "\n")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()