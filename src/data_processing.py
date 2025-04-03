import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys
import joblib

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None  # Train data
        self.test_data = None  # Test data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_store = feature_store
        self.top_12_features = [
            'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
            'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
            'Delivery_person_Age', 'prep_traffic', 'City'
        ]
        logger.info("Your Data Processing is initialized...")

    def load_data(self):
        """Load train and test data from CSV files."""
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Read the train and test data successfully")
        except Exception as e:
            logger.error(f"Error while reading data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def preprocess_data(self, df):
        """Preprocess the input DataFrame (train or test)."""
        try:
            # Rename Weatherconditions column
            df.rename(columns={'Weatherconditions': 'Weather_conditions'}, inplace=True)

            # Extract Time_taken(min) for train data only
            if 'Time_taken(min)' in df.columns:
                df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))

            # Extract Weather conditions
            df['Weather_conditions'] = df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip() if pd.notnull(x) else x)

            # Extract city code from Delivery person ID
            df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]

            # Drop unnecessary columns
            df.drop(['ID', 'Delivery_person_ID'], axis=1, inplace=True)

            # Change data types
            df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce').astype('float64')
            df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce').astype('float64')
            df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce').astype('float64')
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y")

            # Replace string 'NaN' with np.nan
            df.replace('NaN', float(np.nan), regex=True, inplace=True)

            # Handle null values
            df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age'].dropna()))
            df['Weather_conditions'] = df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions'].dropna()))
            df['City'] = df['City'].fillna(df['City'].mode()[0])
            df['Festival'] = df['Festival'].fillna(df['Festival'].mode()[0])
            df['multiple_deliveries'] = df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0])
            df['Road_traffic_density'] = df['Road_traffic_density'].fillna(df['Road_traffic_density'].mode()[0])
            df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())

            logger.info("Data preprocessing completed for DataFrame")
            return df

        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def label_encoding(self, df):
        """Apply label encoding to categorical columns."""
        try:
            categorical_columns = df.select_dtypes(include='object').columns
            label_encoder = LabelEncoder()
            df[categorical_columns] = df[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))
            logger.info("Label encoding completed for DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error while label encoding: {e}")
            raise CustomException(str(e), sys.exc_info())

    def feature_engineering(self, df):
        """Perform feature engineering on the input DataFrame."""
        try:
            # Date features
            df["day"] = df.Order_Date.dt.day
            df["month"] = df.Order_Date.dt.month
            df["quarter"] = df.Order_Date.dt.quarter
            df["year"] = df.Order_Date.dt.year
            df['day_of_week'] = df.Order_Date.dt.day_of_week.astype(int)
            df["is_month_start"] = df.Order_Date.dt.is_month_start.astype(int)
            df["is_month_end"] = df.Order_Date.dt.is_month_end.astype(int)
            df["is_quarter_start"] = df.Order_Date.dt.is_quarter_start.astype(int)
            df["is_quarter_end"] = df.Order_Date.dt.is_quarter_end.astype(int)
            df["is_year_start"] = df.Order_Date.dt.is_year_start.astype(int)
            df["is_year_end"] = df.Order_Date.dt.is_year_end.astype(int)
            df['is_weekend'] = np.where(df['day_of_week'].isin([5, 6]), 1, 0)

            # Calculate order_prepare_time
            df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'].fillna('00:00:00'))
            df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'].fillna('00:00:00'))
            df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']
            df['Time_Order_picked_base'] = df['Order_Date'] + df['Time_Order_picked']
            mask = df['Time_Order_picked'] < df['Time_Orderd']
            df['Time_Order_picked_formatted'] = df['Time_Order_picked_base'].copy()
            df.loc[mask, 'Time_Order_picked_formatted'] += pd.Timedelta(days=1)
            df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
            df['order_prepare_time'] = df['order_prepare_time'].fillna(df['order_prepare_time'].median())
            df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 
                     'Time_Order_picked_base', 'Time_Order_picked_formatted', 'Order_Date'], 
                    axis=1, inplace=True)

            # Calculate distance
            restaurant_coordinates = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
            delivery_location_coordinates = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
            df['distance'] = np.array([geodesic(restaurant, delivery).kilometers for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])

            # Label encode categorical columns before engineered features
            df = self.label_encoding(df)

            # Additional engineered features (now safe with numeric Road_traffic_density)
            df['distance_traffic'] = df['distance'] * df['Road_traffic_density']
            df['distance_deliveries'] = df['distance'] * df['multiple_deliveries']
            df['prep_traffic'] = df['order_prepare_time'] * df['Road_traffic_density']
            df['age_ratings'] = df['Delivery_person_Age'] * df['Delivery_person_Ratings']
            df['prep_distance'] = df['order_prepare_time'] * df['distance']

            # Handle outliers
            for col in ['distance', 'order_prepare_time', 'Delivery_person_Age', 'multiple_deliveries']:
                upper_limit = df[col].quantile(0.99)
                df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

            logger.info("Feature engineering completed for DataFrame")
            return df

        except Exception as e:
            logger.error(f"Error while feature engineering: {e}")
            raise CustomException(str(e), sys.exc_info())

    def split_and_scale(self):
        """Split train data and scale features."""
        try:
            # Split features and target
            X = self.data.drop('Time_taken(min)', axis=1)
            y = self.data['Time_taken(min)']

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Select top 12 features
            self.X_train = self.X_train[self.top_12_features]
            self.X_test = self.X_test[self.top_12_features]
            self.test_data = self.test_data[self.top_12_features]

            # Scale features
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.test_data = scaler.transform(self.test_data)

            # Save scaler for later use
            joblib.dump(scaler, SCALER_PATH)
            logger.info("Data split and scaled successfully")
        except Exception as e:
            logger.error(f"Error while splitting and scaling data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def store_features_in_redis(self):
        """Store processed train features in Redis."""
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = f"train_{idx}"
                features = {col: row[col] for col in self.top_12_features}
                features['Time_taken(min)'] = row['Time_taken(min)']
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Train data features stored in Redis")
        except Exception as e:
            logger.error(f"Error while storing features in Redis: {e}")
            raise CustomException(str(e), sys.exc_info())

    def retrieve_features_from_redis(self, entity_id):
        """Retrieve features from Redis by entity ID."""
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        logger.warning(f"No features found for entity_id: {entity_id}")
        return None

    def run(self):
        """Execute the full preprocessing pipeline."""
        try:
            logger.info("Starting Data Processing Pipeline...")
            self.load_data()

            # Process train data
            self.data = self.preprocess_data(self.data)
            self.data = self.feature_engineering(self.data)

            # Process test data
            self.test_data = self.preprocess_data(self.test_data)
            self.test_data = self.feature_engineering(self.test_data)

            # Split and scale
            self.split_and_scale()

            # Store features in Redis
            self.store_features_in_redis()

            logger.info("End of Data Processing Pipeline...")
        except Exception as e:
            logger.error(f"Error in Data Processing Pipeline: {e}")
            raise CustomException(str(e), sys.exc_info())

if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()
    print(data_processor.retrieve_features_from_redis("train_0"))