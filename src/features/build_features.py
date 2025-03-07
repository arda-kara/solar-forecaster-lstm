"""
Script to extract features from processed satellite data for space weather forecasting.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats

def create_time_windows(data, window_size=24, forecast_horizon=6):
    """
    Create time windows for sequence prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed and merged satellite data
    window_size : int
        Size of the input window in hours
    forecast_horizon : int
        How many hours ahead to predict
    
    Returns:
    --------
    tuple
        (X, y) where X is the input sequences and y is the target values
    """
    print(f"Creating time windows with window_size={window_size} and forecast_horizon={forecast_horizon}...")
    
    # Ensure data is sorted by time if it has a time column
    if 'time' in data.columns:
        data = data.sort_values('time')
        time_values = data['time'].values
        # Remove time column for feature creation
        feature_data = data.drop('time', axis=1)
    else:
        feature_data = data.copy()
        time_values = None
    
    # Get the target column (assuming it's the first column for now)
    # In a real implementation, this should be a parameter
    target_column = feature_data.columns[0]
    
    # Create empty lists to store sequences and targets
    X, y = [], []
    
    # Create sliding windows
    for i in range(len(feature_data) - window_size - forecast_horizon + 1):
        # Input sequence
        X.append(feature_data.iloc[i:i+window_size].values)
        
        # Target sequence (can be single value or sequence depending on the task)
        if forecast_horizon == 1:
            # Single step prediction
            y.append(feature_data[target_column].iloc[i+window_size])
        else:
            # Multi-step prediction
            y.append(feature_data[target_column].iloc[i+window_size:i+window_size+forecast_horizon].values)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} samples with shape X: {X.shape}, y: {y.shape}")
    
    return X, y

def extract_flare_features(data):
    """
    Extract features specifically related to solar flare prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed satellite data
    
    Returns:
    --------
    pd.DataFrame
        Data with additional flare-related features
    """
    print("Extracting flare-specific features...")
    
    # TODO: Implement actual feature extraction
    # This would include:
    # - Calculating rate of change in X-ray flux
    # - Extracting features from solar wind parameters
    # - Creating lag features
    # - Calculating rolling statistics
    
    # Placeholder for demonstration
    print("This is a placeholder. Actual implementation will extract meaningful flare features.")
    
    # Add some dummy features to the data
    data_copy = data.copy()
    data_copy['xray_flux_change'] = np.random.random(len(data))
    data_copy['xray_flux_rolling_max'] = np.random.random(len(data))
    data_copy['bz_rolling_min'] = np.random.random(len(data))
    data_copy['speed_acceleration'] = np.random.random(len(data))
    
    print(f"Added 4 new features to the data")
    
    return data_copy

def save_features(X, y, output_dir='data/processed'):
    """
    Save the feature matrices for model training.
    
    Parameters:
    -----------
    X : np.ndarray
        Input sequences
    y : np.ndarray
        Target values
    output_dir : str
        Directory to save the features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{output_dir}/X_features.npy", X)
    np.save(f"{output_dir}/y_targets.npy", y)
    
    print(f"Saved features to {output_dir}/X_features.npy and {output_dir}/y_targets.npy")

def extract_solar_flare_features(data):
    """
    Extract enhanced features for solar flare prediction with better engineering.
    """
    print("Extracting solar flare features...")
    
    # Make a copy of the data
    df = data.copy()
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Add time-based features
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_year'] = df['time'].dt.dayofyear
    
    # Add solar cycle position approximation (simplified)
    # Assuming a ~11 year cycle with the last solar minimum around Dec 2019
    days_since_min = (df['time'] - pd.Timestamp('2019-12-01')).dt.days
    df['solar_cycle_pos'] = np.sin(2 * np.pi * days_since_min / (11 * 365.25))
    
    # Calculate rolling statistics for X-ray flux
    if 'xray_flux_short' in df.columns:
        # Log transform the flux values (common in solar physics)
        df['log_xray_flux'] = np.log10(df['xray_flux_short'].clip(1e-9, 1e-3))
        
        # Rolling means at different windows
        for window in [3, 6, 12, 24]:
            df[f'xray_flux_mean_{window}h'] = df['xray_flux_short'].rolling(window=window, min_periods=1).mean()
            df[f'log_flux_mean_{window}h'] = df['log_xray_flux'].rolling(window=window, min_periods=1).mean()
        
        # Rolling max (important for flare detection)
        for window in [6, 12, 24]:
            df[f'xray_flux_max_{window}h'] = df['xray_flux_short'].rolling(window=window, min_periods=1).max()
        
        # Rolling standard deviation (volatility)
        for window in [6, 12, 24]:
            df[f'xray_flux_std_{window}h'] = df['xray_flux_short'].rolling(window=window, min_periods=1).std()
        
        # Rate of change features
        df['flux_diff_1h'] = df['xray_flux_short'].diff(periods=1)
        df['flux_diff_3h'] = df['xray_flux_short'].diff(periods=3)
        df['flux_diff_6h'] = df['xray_flux_short'].diff(periods=6)
        
        # Acceleration (second derivative)
        df['flux_accel'] = df['flux_diff_1h'].diff(periods=1)
        
        # Replace infinity with NaN and then fill NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().bfill()  # Using non-deprecated methods
        
        # Clip extreme values
        for col in df.columns:
            if col != 'time' and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                q_low = df[col].quantile(0.001)
                q_high = df[col].quantile(0.999)
                df[col] = df[col].clip(q_low, q_high)
    
    # Calculate solar wind features if available
    if all(col in df.columns for col in ['speed', 'bz', 'bt', 'density']):
        # Solar wind pressure (proxy)
        df['solar_wind_pressure'] = df['speed'] * df['density'] / 1000
        
        # IMF features
        df['bt_bz_ratio'] = df['bt'] / (df['bz'].abs() + 1e-5)  # Avoid division by zero
        
        # Southward IMF (important for geomagnetic activity)
        df['bz_south'] = df['bz'].clip(upper=0).abs()
        
        # Interaction terms
        df['speed_bz_product'] = df['speed'] * df['bz']
        df['speed_density_product'] = df['speed'] * df['density']
        
        # Rolling statistics for solar wind
        for window in [6, 12, 24]:
            df[f'speed_mean_{window}h'] = df['speed'].rolling(window=window, min_periods=1).mean()
            df[f'bz_min_{window}h'] = df['bz'].rolling(window=window, min_periods=1).min()
            df[f'bt_mean_{window}h'] = df['bt'].rolling(window=window, min_periods=1).mean()
    
    # Save the features
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/solar_flare_features.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Features extracted and saved to {output_file}")
    return df

def create_flare_labels(data, threshold=1e-5, prediction_window=24, output_dir='data/processed'):
    """
    Create binary labels for solar flare prediction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with time series data
    threshold : float
        X-ray flux threshold for defining a flare event
    prediction_window : int
        Time window (in hours) for prediction
    output_dir : str
        Directory to save the processed data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with flare labels
    """
    print(f"Creating flare labels with threshold {threshold}...")
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    # Create a future max column
    df['future_max_flux'] = df['xray_flux_short'].rolling(window=f'{prediction_window}h', min_periods=1).max().shift(-prediction_window)
    
    # Create binary label
    df['flare_label'] = (df['future_max_flux'] > threshold).astype(int)
    
    # Reset index
    df.reset_index(inplace=True)
    
    # Save labeled data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'solar_flare_labels.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Flare labels created and saved to {output_file}")
    return df

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/merged_satellite_data.csv"
    
    # Create the file if it doesn't exist (for demonstration)
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        # Create a placeholder DataFrame
        dates = pd.date_range(start='2020-01-01', periods=100, freq='H')
        data = {
            'time': dates,
            'xray_flux_short': np.random.random(100) * 1e-6,
            'xray_flux_long': np.random.random(100) * 1e-7,
            'bz': np.random.normal(0, 5, 100),
            'bt': np.random.normal(5, 2, 100),
            'speed': np.random.normal(400, 50, 100),
            'density': np.random.normal(5, 2, 100),
            'temperature': np.random.normal(1e5, 3e4, 100)
        }
        df = pd.DataFrame(data)
        df.to_csv(input_file, index=False)
    
    # Load the merged data
    data = pd.read_csv(input_file)
    
    # Extract additional features
    data_with_features = extract_flare_features(data)
    
    # Create time windows for sequence prediction
    X, y = create_time_windows(data_with_features, window_size=24, forecast_horizon=6)
    
    # Save the features
    save_features(X, y)
    
    # Extract features
    feature_data = extract_solar_flare_features(data)
    
    # Create labels
    labeled_data = create_flare_labels(feature_data) 