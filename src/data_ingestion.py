import sys
import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, TRAIN_PATH, TEST_PATH
import kaggle

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            kaggle.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"Error authenticating Kaggle API: {e}")
            raise CustomException(str(e), sys)

    def download_kaggle_dataset(self):
        try:
            dataset = "gauravmalik26/food-delivery-dataset"
            # Download the dataset files to the output directory
            kaggle.api.dataset_download_files(
                dataset,
                path=self.output_dir,
                unzip=True
            )
            logger.info(f"Dataset {dataset} downloaded successfully")

            # Look for train.csv and test.csv specifically
            train_file = None
            test_file = None
            
            for file in os.listdir(self.output_dir):
                if file == 'train.csv':
                    train_file = os.path.join(self.output_dir, file)
                elif file == 'test.csv':
                    test_file = os.path.join(self.output_dir, file)
                elif file not in ['train.csv', 'test.csv']:
                    # Remove unwanted files like Sample_Submission.csv
                    os.remove(os.path.join(self.output_dir, file))
                    logger.info(f"Removed unwanted file: {file}")

            if not train_file or not test_file:
                raise Exception("train.csv or test.csv not found in downloaded dataset")

            # Load the train and test CSV files
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded train.csv with {len(train_df)} rows and test.csv with {len(test_df)} rows")

            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error downloading dataset from Kaggle: {e}")
            raise CustomException(str(e), sys)

    def save_data(self, train_df, test_df):
        try:
            # Save the train and test data to the specified paths
            train_df.to_csv(TRAIN_PATH, index=False)
            test_df.to_csv(TEST_PATH, index=False)
            logger.info(f"Data saved successfully. Train rows: {len(train_df)}, Test rows: {len(test_df)}")
        except Exception as e:
            logger.error(f"Error while saving data: {e}")
            raise CustomException(str(e), sys)

    def run(self):
        try:
            logger.info("Data Ingestion Pipeline Started...")
            train_df, test_df = self.download_kaggle_dataset()
            self.save_data(train_df, test_df)
            logger.info("End of Data Ingestion Pipeline")
        except Exception as e:
            logger.error(f"Error in Data Ingestion Pipeline: {e}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion(RAW_DIR)
    data_ingestion.run()