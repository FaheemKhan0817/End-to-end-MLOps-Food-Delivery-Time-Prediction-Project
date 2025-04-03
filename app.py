from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from config.paths_config import MODEL_PATH, SCALER_PATH
from src.logger import get_logger

app = Flask(__name__, template_folder='templates', static_folder='static')
logger = get_logger(__name__)

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Define the top 12 features
TOP_12_FEATURES = [
    'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
    'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
    'Delivery_person_Age', 'prep_traffic', 'City'
]

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
        # Get form data
        data = request.form.to_dict()
        logger.info(f"Received form data: {data}")

        # Validate and map dropdown values
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

        # Scale the input data
        scaled_data = scaler.transform(df)
        logger.info("Input data scaled successfully")

        # Predict and convert float32 to Python float
        prediction = float(model.predict(scaled_data)[0])
        logger.info(f"Prediction: {prediction}")

        return jsonify({'prediction': round(prediction, 2)})
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
    app.run(debug=True, host='0.0.0.0', port=5000)