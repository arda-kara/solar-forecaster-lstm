"""
Script to preprocess the raw NASA satellite data for space weather forecasting.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import json
import datetime
import joblib

def preprocess_goes_data(input_file, output_file=None):
    """
    Preprocess GOES X-ray flux data.
    
    Parameters:
    -----------
    input_file : str
        Path to the raw GOES data file
    output_file : str, optional
        Path to save the processed data. If None, a default path is used.
    
    Returns:
    --------
    pd.DataFrame
        Processed GOES data
    """
    print(f"Preprocessing GOES X-ray flux data from {input_file}...")
    
    # Read the raw data
    df = pd.read_csv(input_file)
    
    # Convert time column to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Handle missing values
    for col in df.columns:
        if col != 'time':
            # Fill missing values with forward fill, then backward fill
            df[col] = df[col].ffill().bfill()
            
            # If there are still NaNs, fill with column mean
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    
    # Resample to a consistent time interval (hourly)
    if 'time' in df.columns:
        df = df.set_index('time')
        df = df.resample('H').mean()
        df = df.reset_index()
    
    # Add derived features
    if 'xray_flux_short' in df.columns and 'xray_flux_long' in df.columns:
        # Add ratio of short to long wavelength flux (indicator of flare hardness)
        df['xray_ratio'] = df['xray_flux_short'] / df['xray_flux_long']
        
        # Add log-transformed flux (common in solar physics)
        df['log_xray_flux_short'] = np.log10(df['xray_flux_short'].clip(1e-9, None))
        df['log_xray_flux_long'] = np.log10(df['xray_flux_long'].clip(1e-10, None))
        
        # Add rate of change
        df['xray_short_delta'] = df['xray_flux_short'].diff().fillna(0)
        df['xray_long_delta'] = df['xray_flux_long'].diff().fillna(0)
        
        # Add exponential moving averages (important for trend detection)
        df['xray_short_ema12'] = df['xray_flux_short'].ewm(span=12).mean()
        df['xray_short_ema24'] = df['xray_flux_short'].ewm(span=24).mean()
        
        # Add volatility measures
        df['xray_short_volatility'] = df['xray_flux_short'].rolling(window=12).std().fillna(0)
    
    # Save processed data if output_file is provided
    if output_file is None:
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/goes_xray_processed.csv'
    
    df.to_csv(output_file, index=False)
    print(f"Processed GOES data saved to {output_file}")
    
    return df

def preprocess_ace_data(input_file, output_file=None):
    """
    Preprocess ACE solar wind data.
    
    Parameters:
    -----------
    input_file : str
        Path to the raw ACE data file
    output_file : str, optional
        Path to save the processed data. If None, a default path is used.
    
    Returns:
    --------
    pd.DataFrame
        Processed ACE data
    """
    print(f"Preprocessing ACE solar wind data from {input_file}...")
    
    # Read the raw data
    df = pd.read_csv(input_file)
    
    # Convert time column to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Handle missing values
    for col in df.columns:
        if col != 'time':
            # Fill missing values with forward fill, then backward fill
            df[col] = df[col].ffill().bfill()
            
            # If there are still NaNs, fill with column mean
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    
    # Resample to a consistent time interval (hourly)
    if 'time' in df.columns:
        df = df.set_index('time')
        df = df.resample('H').mean()
        df = df.reset_index()
    
    # Add derived features
    if 'speed' in df.columns and 'density' in df.columns:
        # Add dynamic pressure (important for magnetospheric compression)
        # P = n * m_p * v^2, where m_p is proton mass
        proton_mass = 1.6726219e-27  # kg
        df['dynamic_pressure'] = df['density'] * proton_mass * (df['speed'] * 1000)**2 * 1e9  # nPa
        
        # Add Alfvén Mach number (important for solar wind-magnetosphere coupling)
        if 'bt' in df.columns:
            mu0 = 4 * np.pi * 1e-7  # H/m
            df['alfven_speed'] = df['bt'] * 1e-9 / np.sqrt(mu0 * df['density'] * 1.6726219e-27) * 1e-3  # km/s
            df['alfven_mach'] = df['speed'] / df['alfven_speed']
        
        # Add solar wind electric field (important for geomagnetic activity)
        if 'bz' in df.columns:
            df['e_field'] = -df['speed'] * df['bz'] * 1e-3  # mV/m
    
    # Save processed data if output_file is provided
    if output_file is None:
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/ace_solarwind_processed.csv'
    
    df.to_csv(output_file, index=False)
    print(f"Processed ACE data saved to {output_file}")
    
    return df

def load_and_merge_data(goes_file, ace_file, output_file=None):
    """
    Load and merge GOES and ACE data.
    
    Parameters:
    -----------
    goes_file : str
        Path to the GOES data file
    ace_file : str
        Path to the ACE data file
    output_file : str, optional
        Path to save the merged data. If None, a default path is used.
    
    Returns:
    --------
    pd.DataFrame
        Merged data
    """
    print(f"Loading and merging data...")
    
    # Preprocess individual datasets
    goes_data = preprocess_goes_data(goes_file)
    ace_data = preprocess_ace_data(ace_file)
    
    # Merge on time column
    if 'time' in goes_data.columns and 'time' in ace_data.columns:
        # Convert to datetime if not already
        goes_data['time'] = pd.to_datetime(goes_data['time'])
        ace_data['time'] = pd.to_datetime(ace_data['time'])
        
        # Merge with outer join to keep all timestamps
        merged_data = pd.merge(goes_data, ace_data, on='time', how='outer', suffixes=('_goes', '_ace'))
        
        # Sort by time
        merged_data = merged_data.sort_values('time')
        
        # Fill missing values
        merged_data = merged_data.ffill().bfill()
    else:
        print("Warning: 'time' column not found in one or both datasets. Using simple concatenation.")
        merged_data = pd.concat([goes_data, ace_data], axis=1)
    
    # Add time-based features
    if 'time' in merged_data.columns:
        # Extract hour of day (captures diurnal patterns)
        merged_data['hour_of_day'] = merged_data['time'].dt.hour
        merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data['hour_of_day'] / 24)
        merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data['hour_of_day'] / 24)
        
        # Extract day of year (captures seasonal patterns)
        merged_data['day_of_year'] = merged_data['time'].dt.dayofyear
        merged_data['day_sin'] = np.sin(2 * np.pi * merged_data['day_of_year'] / 365.25)
        merged_data['day_cos'] = np.cos(2 * np.pi * merged_data['day_of_year'] / 365.25)
        
        # Solar cycle position (approximate)
        # Assuming solar cycle 24 minimum in Dec 2019 and cycle length of 11 years
        cycle_start = pd.Timestamp('2019-12-01')
        cycle_length_days = 11 * 365.25
        merged_data['days_since_cycle_start'] = (merged_data['time'] - cycle_start).dt.total_seconds() / (24 * 3600)
        merged_data['solar_cycle_phase'] = (merged_data['days_since_cycle_start'] % cycle_length_days) / cycle_length_days
        merged_data['solar_cycle_sin'] = np.sin(2 * np.pi * merged_data['solar_cycle_phase'])
        merged_data['solar_cycle_cos'] = np.cos(2 * np.pi * merged_data['solar_cycle_phase'])
    
    # Save merged data if output_file is provided
    if output_file is None:
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/merged_space_weather_data.csv'
    
    merged_data.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    return merged_data

def prepare_lstm_data(data, sequence_length=24, forecast_horizon=12, target_column='xray_flux_short', 
                     test_split=0.2, output_dir='data/processed', stride=1, augment_data=True):
    """
    Prepare data for LSTM model with improved preprocessing.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    sequence_length : int
        Length of input sequences
    forecast_horizon : int
        Number of time steps to forecast
    target_column : str
        Name of the target column
    test_split : float
        Fraction of data to use for testing
    output_dir : str
        Directory to save processed data
    stride : int
        Stride for creating sequences
    augment_data : bool
        Whether to augment data if training set is small
        
    Returns:
    --------
    dict
        Dictionary containing processed data and metadata
    """
    print(f"Preparing data for LSTM model with sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the data to avoid modifying the original
    data = data.copy()
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values before handling:\n{missing_values}")
    
    # Handle missing values
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            # For time series, forward fill then backward fill is often better than mean
            data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
    
    # Check if any missing values remain
    remaining_missing = data.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after filling")
        # Fill any remaining missing values with column means
        data = data.fillna(data.mean())
    
    # Apply log transformation to X-ray flux if it's the target (common for solar data)
    log_transform_applied = False
    if target_column in ['xray_flux_short', 'xray_flux_long'] and 'xray' in target_column.lower():
        print(f"Applying log transformation to {target_column}")
        # Add small constant to avoid log(0)
        epsilon = 1e-10
        data[target_column] = np.log10(data[target_column] + epsilon)
        log_transform_applied = True
    
    # Print data statistics before scaling
    print(f"Data statistics before scaling:")
    print(data.describe())
    
    # Identify feature columns (all except target)
    feature_columns = [col for col in data.columns if col != target_column and col != 'time_tag']
    
    # Save column names for later reference
    if 'time_tag' in data.columns:
        time_column = 'time_tag'
        time_values = data[time_column].values
    else:
        time_column = None
        time_values = None
    
    # Use RobustScaler for better handling of outliers
    from sklearn.preprocessing import RobustScaler
    
    # Scale features
    feature_scaler = RobustScaler()
    if feature_columns:
        data[feature_columns] = feature_scaler.fit_transform(data[feature_columns])
    
    # Scale target separately
    target_scaler = RobustScaler()
    data[[target_column]] = target_scaler.fit_transform(data[[target_column]])
    
    # Print data statistics after scaling
    print(f"Data statistics after scaling:")
    print(data.describe())
    
    # Create sequences for multi-horizon prediction
    X, y = [], []
    
    # Determine the maximum index to avoid going out of bounds
    max_idx = len(data) - forecast_horizon
    
    # Create sequences with the specified stride
    for i in range(0, max_idx - sequence_length + 1, stride):
        # Input sequence
        X.append(data.iloc[i:i+sequence_length][feature_columns + [target_column]].values)
        
        # Target sequence (multi-horizon)
        y.append(data.iloc[i+sequence_length:i+sequence_length+forecast_horizon][target_column].values)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Check if we have enough sequences
    if len(X) < 100:
        print(f"Warning: Only {len(X)} sequences created. Consider using a smaller stride or sequence length.")
        
        # Automatically adjust stride if needed
        if stride > 1 and len(data) > sequence_length + forecast_horizon:
            new_stride = max(1, stride // 2)
            print(f"Adjusting stride from {stride} to {new_stride} to create more sequences")
            
            # Recreate sequences with smaller stride
            X, y = [], []
            for i in range(0, max_idx - sequence_length + 1, new_stride):
                X.append(data.iloc[i:i+sequence_length][feature_columns + [target_column]].values)
                y.append(data.iloc[i+sequence_length:i+sequence_length+forecast_horizon][target_column].values)
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Created {len(X)} sequences with adjusted stride")
    
    # Print sequence shapes
    print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
    
    # Split into training and testing sets using time-based split
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Print split information
    print(f"Training set: {len(X_train)} sequences")
    print(f"Testing set: {len(X_test)} sequences")
    
    # Apply data augmentation if training set is small and augmentation is enabled
    if augment_data and len(X_train) < 500:
        try:
            from src.data.augmentation import augment_time_series_data
            print("Applying data augmentation to increase training set size")
            X_train_aug, y_train_aug = augment_time_series_data(X_train, y_train, augmentation_factor=3)
            
            # Check if augmentation was successful
            if len(X_train_aug) > len(X_train):
                X_train, y_train = X_train_aug, y_train_aug
                print(f"After augmentation: {len(X_train)} training sequences")
            else:
                print("Augmentation did not increase dataset size, using original data")
        except Exception as e:
            print(f"Error during data augmentation: {e}")
            print("Using original data without augmentation")
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save time values if available
    if time_values is not None:
        test_time_values = time_values[split_idx + sequence_length:split_idx + sequence_length + len(y_test)]
        if len(test_time_values) == len(y_test):
            np.save(os.path.join(output_dir, 'test_time_values.npy'), test_time_values)
    
    # Save metadata
    metadata = {
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'log_transform_applied': log_transform_applied,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'stride': stride,
        'test_split': test_split,
        'augmentation_applied': augment_data and len(X_train) > len(X) * (1 - test_split),
        'feature_scaler_type': 'RobustScaler',
        'target_scaler_type': 'RobustScaler',
        'feature_scaler_params': {
            'center': True,
            'scale': True
        },
        'target_scaler_params': {
            'center': True,
            'scale': True
        },
        'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save scalers for later use
    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
    
    # Save metadata as JSON
    import json
    with open(os.path.join(output_dir, 'preprocessing_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Processed data and metadata saved to {output_dir}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metadata': metadata,
        'time_values': time_values
    }

if __name__ == "__main__":
    # Example usage
    goes_file = "data/raw/goes_xray_2020-01-01_to_2020-01-07.csv"
    ace_file = "data/raw/ace_solarwind_2020-01-01_to_2020-01-07.csv"
    
    merged_data = load_and_merge_data(goes_file, ace_file)
    X_train, y_train, X_test, y_test, scaler, log_transform_applied = prepare_lstm_data(merged_data) 