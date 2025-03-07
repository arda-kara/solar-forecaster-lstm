import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

def augment_time_series_data(X, y, augmentation_factor=3):
    """
    Apply various time series-specific augmentations to increase dataset size.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input sequences of shape (n_samples, sequence_length, n_features)
    y : numpy.ndarray
        Target values of shape (n_samples, forecast_horizon)
    augmentation_factor : int
        Factor by which to increase the dataset size
        
    Returns:
    --------
    tuple
        (X_augmented, y_augmented) - Augmented datasets
    """
    print(f"Augmenting time series data with factor {augmentation_factor}")
    print(f"Original shapes - X: {X.shape}, y: {y.shape}")
    
    n_samples, seq_length, n_features = X.shape
    forecast_horizon = y.shape[1] if len(y.shape) > 1 else 1
    
    # Initialize augmented datasets
    X_aug = [X]
    y_aug = [y]
    
    # Calculate standard deviation for each feature for jittering
    feature_stds = np.std(X.reshape(-1, n_features), axis=0)
    
    # Ensure no zero standard deviations
    feature_stds = np.maximum(feature_stds, 0.001)
    
    # Calculate target standard deviation for scaling
    if len(y.shape) > 1:
        target_stds = np.std(y, axis=0)
        target_stds = np.maximum(target_stds, 0.001)
    else:
        target_stds = np.maximum(np.std(y), 0.001)
    
    # 1. Jittering (add noise)
    for i in range(augmentation_factor // 3 + 1):
        # Vary noise level for different augmentations
        noise_level = 0.01 + (i * 0.01)
        
        # Add noise to input sequences
        noise = np.random.normal(0, noise_level, X.shape)
        # Scale noise by feature standard deviation
        for j in range(n_features):
            noise[:, :, j] *= feature_stds[j]
        
        X_jittered = X + noise
        
        # Add smaller noise to targets (since they're predictions)
        if len(y.shape) > 1:
            target_noise = np.random.normal(0, noise_level/2, y.shape)
            # Scale noise by target standard deviation
            for j in range(forecast_horizon):
                target_noise[:, j] *= target_stds[j] * 0.5
            y_jittered = y + target_noise
        else:
            target_noise = np.random.normal(0, noise_level/2, y.shape) * target_stds * 0.5
            y_jittered = y + target_noise
        
        X_aug.append(X_jittered)
        y_aug.append(y_jittered)
    
    # 2. Scaling (multiply by random factor)
    for i in range(augmentation_factor // 3 + 1):
        # Generate random scaling factors
        scale_factors = np.random.uniform(0.8, 1.2, (n_samples, 1, n_features))
        
        # Apply scaling to input sequences
        X_scaled = X * scale_factors
        
        # Apply proportional scaling to targets for regression tasks
        if len(y.shape) > 1:
            # Calculate average scaling factor across features
            avg_scale = np.mean(scale_factors, axis=2)
            y_scaled = y * avg_scale[:, 0, np.newaxis]
        else:
            avg_scale = np.mean(scale_factors, axis=2)
            y_scaled = y * avg_scale[:, 0]
        
        X_aug.append(X_scaled)
        y_aug.append(y_scaled)
    
    # 3. Time Warping (using cubic interpolation for smoothness)
    for i in range(augmentation_factor // 3 + 1):
        X_warped = np.zeros_like(X)
        
        for j in range(n_samples):
            # Create time points
            orig_time = np.arange(seq_length)
            
            # Generate random warping
            # More sophisticated warping with smooth transitions
            num_knots = 4  # Number of control points for warping
            knot_positions = np.linspace(0, seq_length-1, num_knots)
            knot_warping = np.random.uniform(0.8, 1.2, num_knots)
            
            # Create cubic spline for smooth warping
            warping_function = CubicSpline(knot_positions, knot_warping * knot_positions)
            
            # Apply warping function
            warped_time = warping_function(orig_time)
            
            # Ensure warped time stays within bounds
            warped_time = np.clip(warped_time, 0, seq_length-1)
            
            # Apply warping to each feature
            for k in range(n_features):
                # Create cubic spline of original data
                cs = CubicSpline(orig_time, X[j, :, k])
                # Sample at warped time points
                X_warped[j, :, k] = cs(warped_time)
        
        # For targets, we don't warp but adjust based on the end of the warped input
        # This simulates how the warping affects the forecast
        warp_factor = np.mean(warped_time[-5:] / orig_time[-5:])
        
        if len(y.shape) > 1:
            # Apply a diminishing warp effect on forecast horizons
            # (effect is stronger on near-term forecasts)
            warp_factors = np.ones(forecast_horizon)
            for h in range(forecast_horizon):
                # Diminishing effect as we go further in the forecast
                warp_factors[h] = 1.0 + (warp_factor - 1.0) * (1.0 - h/forecast_horizon)
            
            y_warped = y * warp_factors.reshape(1, -1)
        else:
            y_warped = y * warp_factor
        
        X_aug.append(X_warped)
        y_aug.append(y_warped)
    
    # 4. Magnitude Warping (using Savitzky-Golay filter for smoothness)
    for i in range(augmentation_factor // 3 + 1):
        X_mag_warped = np.zeros_like(X)
        
        for j in range(n_samples):
            # Create smooth magnitude warping factors
            for k in range(n_features):
                # Generate random warping curve
                raw_warping = np.random.uniform(0.9, 1.1, seq_length // 4)
                
                # Interpolate to full length
                full_idx = np.linspace(0, len(raw_warping)-1, seq_length)
                warping_curve = np.interp(full_idx, np.arange(len(raw_warping)), raw_warping)
                
                # Smooth the warping curve
                if seq_length > 5:  # Need at least 5 points for savgol_filter
                    window_length = min(seq_length // 2 * 2 + 1, 11)  # Must be odd
                    warping_curve = savgol_filter(warping_curve, window_length, 3)
                
                # Apply warping
                X_mag_warped[j, :, k] = X[j, :, k] * warping_curve
        
        # For targets, apply a similar but different warping
        if len(y.shape) > 1:
            # Generate smooth warping factors for targets
            y_mag_warped = np.zeros_like(y)
            
            for j in range(n_samples):
                # Generate random warping curve for targets
                raw_warping = np.random.uniform(0.9, 1.1, forecast_horizon // 2 + 1)
                
                # Interpolate to full length
                full_idx = np.linspace(0, len(raw_warping)-1, forecast_horizon)
                warping_curve = np.interp(full_idx, np.arange(len(raw_warping)), raw_warping)
                
                # Smooth if possible
                if forecast_horizon > 5:
                    window_length = min(forecast_horizon // 2 * 2 + 1, 5)  # Must be odd
                    warping_curve = savgol_filter(warping_curve, window_length, 1)
                
                # Apply warping
                y_mag_warped[j, :] = y[j, :] * warping_curve
        else:
            # For single-value targets, just apply a random factor
            warp_factors = np.random.uniform(0.9, 1.1, n_samples)
            y_mag_warped = y * warp_factors
        
        X_aug.append(X_mag_warped)
        y_aug.append(y_mag_warped)
    
    # 5. Add synthetic seasonal patterns
    for i in range(augmentation_factor // 3 + 1):
        X_seasonal = X.copy()
        
        for j in range(n_samples):
            # Add sine wave with random phase and amplitude
            for k in range(n_features):
                # Random amplitude (scaled by feature std)
                amplitude = np.random.uniform(0.05, 0.15) * feature_stds[k]
                
                # Random frequency
                frequency = np.random.uniform(1, 3)
                
                # Random phase
                phase = np.random.uniform(0, 2*np.pi)
                
                # Generate seasonal pattern
                seasonal_pattern = amplitude * np.sin(frequency * np.linspace(0, 2*np.pi, seq_length) + phase)
                
                # Add to data
                X_seasonal[j, :, k] += seasonal_pattern
        
        # For targets, extend the seasonal pattern
        if len(y.shape) > 1:
            y_seasonal = y.copy()
            
            for j in range(n_samples):
                # Use the same frequency but continue the pattern
                for h in range(forecast_horizon):
                    # Random amplitude (scaled by target std)
                    amplitude = np.random.uniform(0.05, 0.15) * target_stds[h if len(target_stds) > 1 else 0]
                    
                    # Random frequency (same as used for input)
                    frequency = np.random.uniform(1, 3)
                    
                    # Random phase (same as used for input)
                    phase = np.random.uniform(0, 2*np.pi)
                    
                    # Continue the pattern for this horizon
                    t = seq_length + h
                    seasonal_value = amplitude * np.sin(frequency * (2*np.pi * t/seq_length) + phase)
                    
                    # Add to target
                    y_seasonal[j, h] += seasonal_value
        else:
            # For single-value targets
            y_seasonal = y.copy()
            
            for j in range(n_samples):
                # Random amplitude
                amplitude = np.random.uniform(0.05, 0.15) * target_stds
                
                # Random frequency
                frequency = np.random.uniform(1, 3)
                
                # Random phase
                phase = np.random.uniform(0, 2*np.pi)
                
                # Continue the pattern for the target
                t = seq_length
                seasonal_value = amplitude * np.sin(frequency * (2*np.pi * t/seq_length) + phase)
                
                # Add to target
                y_seasonal[j] += seasonal_value
        
        X_aug.append(X_seasonal)
        y_aug.append(y_seasonal)
    
    # Combine all augmentations
    X_combined = np.vstack(X_aug)
    y_combined = np.vstack(y_aug) if len(y.shape) > 1 else np.hstack(y_aug)
    
    print(f"Augmented shapes - X: {X_combined.shape}, y: {y_combined.shape}")
    print(f"Created {len(X_combined)} sequences from original {len(X)} sequences")
    
    return X_combined, y_combined 