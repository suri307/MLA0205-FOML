"""
Automated Test Suite for Smart Home Energy Prediction Application
=================================================================
Validates Bayesian model, inference correctness, database integrity,
Flask route status codes, HTML template rendering, and chart generation.
"""

import os
import unittest
import json
import io

import bayesian_model
import database
import recommendations
import charts
from app import app


class TestSmartHomeEnergyApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()
        # Initialize database
        database.init_db()

    def test_01_bayesian_network_structure_and_validity(self):
        """Verifies that the Bayesian Network is valid according to pgmpy."""
        self.assertTrue(bayesian_model.BAYESIAN_MODEL.check_model(), "Bayesian Network CPDs or DAG failed check_model()")
        nodes = list(bayesian_model.BAYESIAN_MODEL.nodes())
        required_nodes = ['Temperature', 'Day_Type', 'Time_Of_Day', 'Occupancy', 'AC_Usage', 'Appliance_Usage', 'Energy_Consumption']
        for n in required_nodes:
            self.assertIn(n, nodes, f"Missing node {n} in Bayesian Network")

    def test_02_preprocessing_and_inference_high_consumption(self):
        """Verifies preprocessing and inference for high energy demand scenario."""
        clean, evid = bayesian_model.preprocess_household_data(
            temperature_raw=38.0,
            occupants_raw=6,
            ac_usage_raw='Heavy',
            appliance_usage_raw='High',
            time_of_day_raw='Evening',
            day_type_raw='Weekend',
            appliances_list=['Geyser / Water Heater', 'Washing Machine']
        )
        self.assertEqual(clean['temperature_cat'], 'Hot')
        self.assertEqual(clean['occupants_cat'], 'High')
        self.assertEqual(evid['AC_Usage'], 'Heavy')

        pred = bayesian_model.predict_energy_consumption(evid)
        self.assertEqual(pred['predicted_category'], 'High')
        total_p = sum(pred['probabilities'].values())
        self.assertAlmostEqual(total_p, 100.0, places=1)
        self.assertGreater(pred['probabilities']['High'], pred['probabilities']['Low'])

    def test_03_preprocessing_and_inference_low_consumption(self):
        """Verifies preprocessing and inference for low energy demand scenario."""
        clean, evid = bayesian_model.preprocess_household_data(
            temperature_raw=17.5,
            occupants_raw=1,
            ac_usage_raw='Off',
            appliance_usage_raw='Low',
            time_of_day_raw='Night',
            day_type_raw='Weekday',
            appliances_list=['Refrigerator']
        )
        self.assertEqual(clean['temperature_cat'], 'Cool')
        self.assertEqual(clean['occupants_cat'], 'Low')
        self.assertEqual(evid['AC_Usage'], 'Off')

        pred = bayesian_model.predict_energy_consumption(evid)
        self.assertEqual(pred['predicted_category'], 'Low')
        total_p = sum(pred['probabilities'].values())
        self.assertAlmostEqual(total_p, 100.0, places=1)
        self.assertGreater(pred['probabilities']['Low'], pred['probabilities']['High'])

    def test_04_recommendations_generation(self):
        """Verifies decision support recommendations engine."""
        clean, evid = bayesian_model.preprocess_household_data(35.0, 4, 'Heavy', 'High', 'Evening', 'Weekend', ['Geyser / Water Heater'])
        pred = bayesian_model.predict_energy_consumption(evid)
        recs, disclaimer = recommendations.generate_recommendations(clean, pred)

        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertTrue(any('Thermostat' in r['title'] for r in recs))
        self.assertIn("Disclaimer", disclaimer)

    def test_05_database_operations(self):
        """Verifies database insertion, query, stats, and delete."""
        clean, evid = bayesian_model.preprocess_household_data(28.0, 3, 'Moderate', 'Medium', 'Afternoon', 'Weekday', ['Refrigerator', 'Television / Entertainment'])
        pred = bayesian_model.predict_energy_consumption(evid)

        initial_count = len(database.get_all_predictions())
        new_id = database.insert_prediction(clean, pred)
        self.assertIsNotNone(new_id)

        updated_preds = database.get_all_predictions()
        self.assertEqual(len(updated_preds), initial_count + 1)

        single = database.get_prediction_by_id(new_id)
        self.assertIsNotNone(single)
        self.assertEqual(single['temperature_cat'], 'Moderate')

        stats = database.get_summary_stats()
        self.assertGreater(stats['total'], 0)

    def test_06_flask_routes(self):
        """Verifies all Flask web pages and status codes."""
        # 1. GET /
        resp_index = self.client.get('/')
        self.assertEqual(resp_index.status_code, 200)
        self.assertIn(b"Smart Home Energy Consumption Prediction", resp_index.data)

        # 2. POST /predict
        resp_predict = self.client.post('/predict', data={
            'temperature': '32.0',
            'occupants': '4',
            'ac_usage': 'Heavy',
            'appliance_usage': 'High',
            'time_of_day': 'Evening',
            'day_type': 'Weekend',
            'appliances': ['Geyser / Water Heater', 'Washing Machine']
        }, follow_redirects=True)
        self.assertEqual(resp_predict.status_code, 200)
        self.assertIn(b"Module 2: Bayesian Network Prediction", resp_predict.data)
        self.assertIn(b"Posterior Marginal Probabilities", resp_predict.data)

        # 3. POST /api/predict (JSON)
        resp_api = self.client.post('/api/predict', json={
            'temperature': 22.0,
            'occupants': 2,
            'ac_usage': 'Moderate',
            'appliance_usage': 'Medium',
            'time_of_day': 'Morning',
            'day_type': 'Weekday'
        })
        self.assertEqual(resp_api.status_code, 200)
        data = json.loads(resp_api.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('probabilities', data['prediction_result'])

        # 4. GET /history
        resp_history = self.client.get('/history')
        self.assertEqual(resp_history.status_code, 200)
        self.assertIn(b"Prediction History Logs", resp_history.data)

        # 5. GET /analytics
        resp_analytics = self.client.get('/analytics')
        self.assertEqual(resp_analytics.status_code, 200)
        self.assertIn(b"Matplotlib Graphical Visualizations", resp_analytics.data)

        # 6. GET /export/csv
        resp_csv = self.client.get('/export/csv')
        self.assertEqual(resp_csv.status_code, 200)
        self.assertEqual(resp_csv.content_type, 'text/csv; charset=utf-8')
        self.assertIn(b"Prediction_ID,Date_Time", resp_csv.data)

        # 7. GET /model-info
        resp_info = self.client.get('/model-info')
        self.assertEqual(resp_info.status_code, 200)
        self.assertIn(b"Bayesian Network Architecture", resp_info.data)

    def test_07_chart_generation(self):
        """Verifies Matplotlib chart generation."""
        df = database.export_dataframe()
        paths = charts.generate_all_charts(df)
        self.assertIsNotNone(paths['distribution_chart'])
        self.assertIsNotNone(paths['daily_trend_chart'])
        self.assertIsNotNone(paths['tod_breakdown_chart'])


if __name__ == '__main__':
    unittest.main()
