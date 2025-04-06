# from flask import Flask, request, render_template, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# from config.paths_config import MODEL_PATH, SCALER_PATH
# from src.logger import get_logger
# from src.feature_store import RedisFeatureStore
# from alibi_detect.cd import KSDrift
# import os

# app = Flask(__name__, template_folder='templates', static_folder='static')
# logger = get_logger(__name__)

# # Redis configuration
# REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
# REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
# REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# # Define the top 12 features
# TOP_12_FEATURES = [
#     'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
#     'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
#     'Delivery_person_Age', 'prep_traffic', 'City'
# ]

# # Load model and scaler
# try:
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     logger.info("Model and scaler loaded successfully from %s and %s", MODEL_PATH, SCALER_PATH)
# except Exception as e:
#     logger.error(f"Error loading model or scaler: {e}")
#     raise

# # Initialize RedisFeatureStore and fetch reference data
# try:
#     feature_store = RedisFeatureStore(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
#     entity_ids = feature_store.get_all_entity_ids()
#     if not entity_ids:
#         raise ValueError("No reference data found in Redis")
#     batch_features = feature_store.get_batch_features(entity_ids[:1000])
#     reference_df = pd.DataFrame([features for features in batch_features.values()], columns=TOP_12_FEATURES)
#     reference_data = scaler.transform(reference_df)
#     logger.info(f"Fetched and transformed {len(reference_data)} reference rows from RedisFeatureStore")
# except Exception as e:
#     logger.error(f"Error fetching reference data from Redis: {e}")
#     raise

# # Mapping dictionaries for dropdowns
# WEATHER_CONDITIONS = {'Sunny': 0, 'Cloudy': 1, 'Fog': 2, 'Sandstorms': 3, 'Stormy': 4, 'Windy': 5}
# TRAFFIC_DENSITY = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
# VEHICLE_CONDITION = {'Poor': 0, 'Good': 1, 'Excellent': 2}
# FESTIVAL = {'No': 0, 'Yes': 1}
# CITY = {'Urban': 0, 'Semi-Urban': 1, 'Metropolitan': 2}

# # Initialize drift detector
# ksd = KSDrift(x_ref=reference_data, p_val=0.05)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form.to_dict()
#         logger.info(f"Received form data: {data}")

#         def validate_dropdown(value, mapping, field_name):
#             if value not in mapping:
#                 raise ValueError(f"Invalid value '{value}' for {field_name}. Valid options: {list(mapping.keys())}")
#             return mapping[value]

#         input_data = {
#             'multiple_deliveries': float(data['multiple_deliveries']),
#             'Road_traffic_density': validate_dropdown(data['Road_traffic_density'], TRAFFIC_DENSITY, 'Road_traffic_density'),
#             'Vehicle_condition': validate_dropdown(data['Vehicle_condition'], VEHICLE_CONDITION, 'Vehicle_condition'),
#             'Delivery_person_Ratings': float(data['Delivery_person_Ratings']),
#             'distance_deliveries': float(data['distance_deliveries']),
#             'Weather_conditions': validate_dropdown(data['Weather_conditions'], WEATHER_CONDITIONS, 'Weather_conditions'),
#             'Festival': validate_dropdown(data['Festival'], FESTIVAL, 'Festival'),
#             'distance_traffic': float(data['distance_traffic']),
#             'distance': float(data['distance']),
#             'Delivery_person_Age': float(data['Delivery_person_Age']),
#             'prep_traffic': float(data['prep_traffic']),
#             'City': validate_dropdown(data['City'], CITY, 'City')
#         }
#         df = pd.DataFrame([input_data], columns=TOP_12_FEATURES)

#         # Scale incoming data
#         features_scaled = scaler.transform(df)

#         # Data drift detection
#         drift_result = ksd.predict(features_scaled)
#         drift_detected = drift_result['data']['is_drift']
#         if drift_detected:
#             logger.warning(f"Data drift detected! p-values: {drift_result['data']['p_val']}")
#         else:
#             logger.info("No data drift detected.")

#         # Predict
#         prediction = float(model.predict(features_scaled)[0])
#         logger.info(f"Prediction: {prediction}")

#         # Return prediction and drift status
#         return jsonify({
#             'prediction': round(prediction, 2),
#             'drift_detected': bool(drift_detected),
#             'drift_p_values': drift_result['data']['p_val'].tolist()  # Convert to list for JSON
#         })
#     except KeyError as e:
#         logger.error(f"Missing field: {e}")
#         return jsonify({'error': f"Missing required field: {e}"}), 400
#     except ValueError as e:
#         logger.error(f"Invalid input: {e}")
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(debug=True, host='0.0.0.0', port=port) 


############################################################################################################################################

# for deployment purpose i am out of aws credits so i am removing data drift detection and redis feature store 

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from config.paths_config import MODEL_PATH, SCALER_PATH
from src.logger import get_logger
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
logger = get_logger(__name__)

# Define the top 12 features
TOP_12_FEATURES = [
    'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
    'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
    'Delivery_person_Age', 'prep_traffic', 'City'
]

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully from %s and %s", MODEL_PATH, SCALER_PATH)
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Mapping dictionaries for dropdowns
WEATHER_CONDITIONS = {'Sunny': 0, 'Cloudy': 1, 'Fog': 2, 'Sandstorms': 3, 'Stormy': 4, 'Windy': 5}
TRAFFIC_DENSITY = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
VEHICLE_CONDITION = {'Poor': 0, 'Good': 1, 'Excellent': 2}
FESTIVAL = {'No': 0, 'Yes': 1}
CITY = {'Urban': 0, 'Semi-Urban': 1, 'Metropolitan': 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        logger.info(f"Received form data: {data}")

        def validate_dropdown(value, mapping, field_name):
            if value not in mapping:
                raise ValueError(f"Invalid value '{value}' for {field_name}. Valid options: {list(mapping.keys())}")
            return mapping[value]

        input_data = {
            'multiple_deliveries': float(data['multiple_deliveries']),
            'Road_traffic_density': validate_dropdown(data['Road_traffic_density'], TRAFFIC_DENSITY, 'Road_traffic_density'),
            'Vehicle_condition': validate_dropdown(data['Vehicle_condition'], VEHICLE_CONDITION, 'Vehicle_condition'),
            'Delivery_person_Ratings': float(data['Delivery_person_Ratings']),
            'distance_deliveries': float(data['distance_deliveries']),
            'Weather_conditions': validate_dropdown(data['Weather_conditions'], WEATHER_CONDITIONS, 'Weather_conditions'),
            'Festival': validate_dropdown(data['Festival'], FESTIVAL, 'Festival'),
            'distance_traffic': float(data['distance_traffic']),
            'distance': float(data['distance']),
            'Delivery_person_Age': float(data['Delivery_person_Age']),
            'prep_traffic': float(data['prep_traffic']),
            'City': validate_dropdown(data['City'], CITY, 'City')
        }
        df = pd.DataFrame([input_data], columns=TOP_12_FEATURES)

        # Scale incoming data
        features_scaled = scaler.transform(df)

        # Predict
        prediction = float(model.predict(features_scaled)[0])
        logger.info(f"Prediction: {prediction}")

        # Return prediction
        return jsonify({
            'prediction': round(prediction, 2)
        })
    except KeyError as e:
        logger.error(f"Missing field: {e}")
        return jsonify({'error': f"Missing required field: {e}"}), 400
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)