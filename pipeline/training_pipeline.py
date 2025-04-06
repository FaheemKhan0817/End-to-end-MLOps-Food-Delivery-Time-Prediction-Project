from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.feature_store import RedisFeatureStore
from config.paths_config import RAW_DIR, TRAIN_PATH, TEST_PATH
from src.logger import get_logger
from comet_ml import Experiment  
import os

logger = get_logger(__name__)

if __name__ == "__main__":
    # Initialize a single Comet ML Experiment for the entire pipeline
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="food-delivery-time-prediction",
        workspace="faheemkhan0817"  # Replace with your workspace
    )
    experiment.set_name("Full_Training_Pipeline")  # Optional: custom name

    try:
        # Data Ingestion
        with experiment.context_manager("data_ingestion"):
            data_ingestion = DataIngestion(RAW_DIR)
            data_ingestion.experiment = experiment  # Pass the experiment object
            data_ingestion.run()
        
        # Data Processing
        feature_store = RedisFeatureStore()
        with experiment.context_manager("data_processing"):
            data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
            data_processor.experiment = experiment  # Pass the experiment object
            data_processor.run()

        # Model Training
        with experiment.context_manager("model_training"):
            model_trainer = ModelTraining(feature_store)
            model_trainer.experiment = experiment  # Pass the experiment object
            model_trainer.run()

        logger.info("Entire Training Pipeline Completed Successfully")
        experiment.end()

    except Exception as e:
        logger.error(f"Error in Training Pipeline: {e}")
        experiment.log_other("pipeline_error", str(e))
        experiment.end()
        raise