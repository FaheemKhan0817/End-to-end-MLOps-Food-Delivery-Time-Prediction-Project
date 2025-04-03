import unittest
from app import app  # Import directly from app.py
import json

class TestRoutes(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True
        # Sample valid input data
        self.valid_data = {
            'multiple_deliveries': '1',
            'Road_traffic_density': 'Low',
            'Vehicle_condition': 'Good',
            'Delivery_person_Ratings': '4.5',
            'distance_deliveries': '10.5',
            'Weather_conditions': 'Sunny',
            'Festival': 'No',
            'distance_traffic': '15.2',
            'distance': '5.0',
            'Delivery_person_Age': '30',
            'prep_traffic': '12.0',
            'City': 'Urban'
        }
        # Sample invalid data (missing field)
        self.invalid_data = {
            'multiple_deliveries': '1',
            'Road_traffic_density': 'Low',
            # Missing Vehicle_condition
            'Delivery_person_Ratings': '4.5',
            'distance_deliveries': '10.5',
            'Weather_conditions': 'Sunny',
            'Festival': 'No',
            'distance_traffic': '15.2',
            'distance': '5.0',
            'Delivery_person_Age': '30',
            'prep_traffic': '12.0',
            'City': 'Urban'
        }

    def test_index_route(self):
        """Test the index route returns a 200 status and HTML content."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
        self.assertIn(b'Food Delivery Time Prediction', response.data)

    def test_predict_success(self):
        """Test a successful prediction with valid data."""
        response = self.app.post('/predict', data=self.valid_data)
        self.assertEqual(response.status_code, 200)
        
        json_data = json.loads(response.data)
        self.assertIn('prediction', json_data)
        self.assertIsInstance(json_data['prediction'], (int, float))
        self.assertGreaterEqual(json_data['prediction'], 0)

    def test_predict_missing_field(self):
        """Test prediction with a missing field returns an error."""
        response = self.app.post('/predict', data=self.invalid_data)
        self.assertEqual(response.status_code, 400)
        
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('Vehicle_condition', json_data['error'])

    def test_predict_invalid_numeric(self):
        """Test prediction with invalid numeric input returns an error."""
        invalid_data = self.valid_data.copy()
        invalid_data['Delivery_person_Ratings'] = 'invalid'
        response = self.app.post('/predict', data=invalid_data)
        self.assertEqual(response.status_code, 400)
        
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('invalid', json_data['error'].lower())

    def test_predict_invalid_dropdown(self):
        """Test prediction with an invalid dropdown value returns an error."""
        invalid_data = self.valid_data.copy()
        invalid_data['Weather_conditions'] = 'Rainy'
        response = self.app.post('/predict', data=invalid_data)
        self.assertEqual(response.status_code, 400)
        
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('Weather_conditions', json_data['error'])
        self.assertIn('Rainy', json_data['error'])

if __name__ == '__main__':
    unittest.main()