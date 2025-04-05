import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_PATH, SCALER_PATH, MODEL_DIR
import os
import joblib
import sys
from comet_ml import Experiment  

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path=MODEL_DIR):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None
        self.scaler = StandardScaler()
        self.top_12_features = [
            'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
            'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
            'Delivery_person_Age', 'prep_traffic', 'City'
        ]
        os.makedirs(self.model_save_path, exist_ok=True)
        self.experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="food-delivery-time-prediction",
            workspace="faheem-khan0817"  
        )
        logger.info("Model Training initialized with Comet ML...")

    def load_data_from_redis(self, entity_ids):
        try:
            logger.info("Extracting data from Redis")
            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"Features not found for entity_id: {entity_id}")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis: {e}")
            raise CustomException(str(e), sys.exc_info())

    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            if not entity_ids:
                raise ValueError("No entity IDs found in Redis")

            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            if not train_data or not test_data:
                raise ValueError("No data retrieved from Redis")

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df[self.top_12_features]
            X_test = test_df[self.top_12_features]
            y_train = train_df["Time_taken(min)"]
            y_test = test_df["Time_taken(min)"]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.experiment.log_parameter("X_train_shape", X_train_scaled.shape)
            self.experiment.log_parameter("X_test_shape", X_test_scaled.shape)

            logger.info(f"Prepared data - X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def hyperparameter_tuning_xgb(self, X_train, y_train):
        try:
            param_grid_xgb = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

            xgb_reg = xgb.XGBRegressor(random_state=42)
            random_search = RandomizedSearchCV(
                xgb_reg, param_grid_xgb, n_iter=20, cv=5, scoring='r2', 
                n_jobs=-1, random_state=42
            )
            random_search.fit(X_train, y_train)

            self.experiment.log_parameters(param_grid_xgb)
            self.experiment.log_parameters(random_search.best_params_)
            self.experiment.log_metric("best_cv_r2", random_search.best_score_)

            logger.info(f"Best XGBoost parameters: {random_search.best_params_}")
            logger.info(f"Best CV R2 Score: {random_search.best_score_:.4f}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error while hyperparameter tuning XGBoost: {e}")
            raise CustomException(str(e), sys.exc_info())

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            best_xgb = self.hyperparameter_tuning_xgb(X_train, y_train)
            best_xgb.fit(X_train, y_train)
            y_pred = best_xgb.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.experiment.log_metric("test_rmse", rmse)
            self.experiment.log_metric("test_mae", mae)
            self.experiment.log_metric("test_r2", r2)

            logger.info(f"XGBoost Model Performance - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

            self.save_model(best_xgb)
            self.save_scaler()

            return best_xgb, rmse, mae, r2
        except Exception as e:
            logger.error(f"Error while training and evaluating model: {e}")
            raise CustomException(str(e), sys.exc_info())

    def save_model(self, model):
        try:
            joblib.dump(model, MODEL_PATH, compress=3)
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            logger.info(f"Model saved at {MODEL_PATH}, Size: {file_size:.2f} MB")
            self.experiment.log_model("XGBoost_Model", MODEL_PATH)
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException(str(e), sys.exc_info())

    def save_scaler(self):
        try:
            joblib.dump(self.scaler, SCALER_PATH, compress=3)
            file_size = os.path.getsize(SCALER_PATH) / (1024 * 1024)
            logger.info(f"Scaler saved at {SCALER_PATH}, Size: {file_size:.2f} MB")
            self.experiment.log_model("Scaler", SCALER_PATH)
        except Exception as e:
            logger.error(f"Error while saving scaler: {e}")
            raise CustomException(str(e), sys.exc_info())

    def run(self):
        try:
            logger.info("Starting Model Training Pipeline...")
            with self.experiment.context_manager("model_training"):
                X_train, X_test, y_train, y_test = self.prepare_data()
                best_model, rmse, mae, r2 = self.train_and_evaluate(X_train, y_train, X_test, y_test)
            logger.info("End of Model Training Pipeline...")
            self.experiment.end()
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline: {e}")
            self.experiment.log_other("error", str(e))
            raise CustomException(str(e), sys.exc_info())

if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()