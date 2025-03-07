import unittest
import numpy as np
import pandas as pd
from src.data.preprocess import load_and_merge_data, prepare_lstm_data
from src.data.augmentation import augment_time_series_data

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic data for testing
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='H')
        self.data = pd.DataFrame({
            'time': self.dates,
            'xray_flux_short': np.random.exponential(1e-6, 100),
            'xray_flux_long': np.random.exponential(1e-7, 100),
            'speed': 400 + np.random.normal(0, 50, 100),
            'density': 5 + np.random.normal(0, 2, 100),
            'temperature': 1e5 + np.random.normal(0, 2e4, 100),
            'bz': np.random.normal(0, 5, 100),
            'bt': 5 + np.random.normal(0, 2, 100)
        })
    
    def test_prepare_lstm_data(self):
        # Test LSTM data preparation
        X_train, y_train, X_test, y_test, scaler, log_transform = prepare_lstm_data(
            self.data, 
            sequence_length=12, 
            forecast_horizon=6,
            test_split=0.2
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[1], 12)  # sequence_length
        self.assertEqual(y_train.shape[1], 6)   # forecast_horizon
        
        # Check train/test split
        total_samples = len(X_train) + len(X_test)
        expected_test_samples = int(total_samples * 0.2)
        self.assertAlmostEqual(len(X_test), expected_test_samples, delta=1)
    
    def test_augmentation(self):
        # Create sample data
        X = np.random.random((10, 12, 5))  # 10 samples, 12 timesteps, 5 features
        y = np.random.random((10, 6))      # 10 samples, 6 horizon steps
        
        # Test augmentation
        X_aug, y_aug = augment_time_series_data(X, y, augmentation_factor=3)
        
        # Check augmented data size
        self.assertGreater(len(X_aug), len(X))
        self.assertEqual(len(X_aug), len(y_aug))
        
        # Check shapes are preserved
        self.assertEqual(X_aug.shape[1:], X.shape[1:])
        self.assertEqual(y_aug.shape[1:], y.shape[1:])

if __name__ == '__main__':
    unittest.main() 