import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, max_error
import datetime

def custom_huber_loss(delta=1.0):
    """
    Create a custom Huber loss function with the given delta.
    
    Parameters:
    -----------
    delta : float
        Threshold at which the loss changes from quadratic to linear
        
    Returns:
    --------
    function
        Huber loss function
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        small_error_loss = 0.5 * tf.square(error)
        big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small_error, small_error_loss, big_error_loss)
    
    return loss

def create_lstm_model(input_shape, output_size=1, model_type='regression'):
    """
    Create an improved LSTM model with better architecture for time series forecasting.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    output_size : int
        Number of output units (forecast horizon)
    model_type : str
        Type of model ('regression' or 'classification')
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM model
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add batch normalization to standardize inputs
    normalized = tf.keras.layers.BatchNormalization()(inputs)
    
    # First LSTM layer with dropout and recurrent dropout
    lstm1 = tf.keras.layers.LSTM(
        128, 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(normalized)
    
    # Add batch normalization between LSTM layers
    bn1 = tf.keras.layers.BatchNormalization()(lstm1)
    
    # Second LSTM layer
    lstm2 = tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(bn1)
    
    # Add batch normalization after LSTM
    bn2 = tf.keras.layers.BatchNormalization()(lstm2)
    
    # Dense hidden layer with dropout
    dense1 = tf.keras.layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(bn2)
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
    
    # Second dense hidden layer with reduced size
    dense2 = tf.keras.layers.Dense(
        32, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(dropout1)
    
    # Output layer
    if model_type == 'regression':
        # For regression, use linear activation
        outputs = tf.keras.layers.Dense(output_size)(dense2)
    else:
        # For classification, use softmax activation
        outputs = tf.keras.layers.Dense(output_size, activation='softmax')(dense2)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with appropriate loss function and optimizer
    if model_type == 'regression':
        # Try different loss function names for compatibility with different TensorFlow versions
        try:
            # First try 'huber' (newer versions)
            model.compile(
                loss='huber',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['mae', 'mse']
            )
            print("Using 'huber' loss function")
        except ValueError:
            try:
                # Then try 'huber_loss' (older versions)
                model.compile(
                    loss='huber_loss',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['mae', 'mse']
                )
                print("Using 'huber_loss' loss function")
            except ValueError:
                # Fall back to custom implementation
                print("Built-in Huber loss not available, using custom implementation")
                model.compile(
                    loss=custom_huber_loss(delta=1.0),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['mae', 'mse']
                )
    else:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    
    return model

def train_model(X_train, y_train, X_val=None, y_val=None, model=None, 
               epochs=200, batch_size=32, model_path='models/lstm_space_weather_model.keras',
               early_stopping_patience=30):
    """
    Train the LSTM model with improved training process.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data
    y_train : numpy.ndarray
        Training target data
    X_val : numpy.ndarray, optional
        Validation input data
    y_val : numpy.ndarray, optional
        Validation target data
    model : tensorflow.keras.Model, optional
        Model to train. If None, a new model is created.
    epochs : int
        Maximum number of epochs to train
    batch_size : int
        Batch size for training
    model_path : str
        Path to save the best model
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped
        
    Returns:
    --------
    tuple
        (model, history)
    """
    # Create validation data if not provided
    if X_val is None or y_val is None:
        # Use 20% of training data for validation
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
    
    # Print data shapes and statistics
    print(f"Training data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_train min: {X_train.min()}, max: {X_train.max()}, mean: {X_train.mean()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}, mean: {y_train.mean()}")
    
    # Create model if not provided
    if model is None:
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        model = create_lstm_model(input_shape, output_size)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001  # Smaller improvements will still count
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,  # More gradual reduction
        patience=early_stopping_patience // 4,  # More frequent LR adjustments
        min_lr=1e-6,
        verbose=1,
        min_delta=0.0001  # Smaller improvements will still count
    )
    
    # Ensure directory exists for model checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Add learning rate warmup and cycling
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr * (1.0 + 0.05 * epoch)  # Gradual warmup
        elif epoch % 30 == 0 and epoch > 0:
            return lr * 1.5  # Larger boost every 30 epochs to escape local minima
        else:
            return lr
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    # Add TensorBoard logging if available
    callbacks = [early_stopping, reduce_lr, model_checkpoint, lr_schedule]
    try:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard_callback)
        print(f"TensorBoard logs will be saved to {log_dir}")
    except:
        print("TensorBoard logging not available, continuing without it")
    
    # Print training information
    print(f"Training model with {len(X_train)} samples, validating with {len(X_val)} samples")
    print(f"Maximum epochs: {epochs}, Early stopping patience: {early_stopping_patience}")
    print(f"Batch size: {batch_size}")
    print(f"Model will be saved to: {model_path}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True  # Ensure data is shuffled for each epoch
    )
    
    # Load the best model
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded best model from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path} due to error: {e}")
            print("Using last epoch model instead")
    
    print(f"Model trained for {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler=None, log_transform=False, 
                  save_dir='reports/figures', save_metrics=True, time_values=None):
    """
    Evaluate model performance with comprehensive metrics and visualizations.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X_test : numpy.ndarray
        Test input data
    y_test : numpy.ndarray
        Test target data
    scaler : sklearn.preprocessing.Scaler, optional
        Scaler used to transform the data
    log_transform : bool
        Whether log transformation was applied to the target
    save_dir : str
        Directory to save visualizations and metrics
    save_metrics : bool
        Whether to save metrics to a file
    time_values : numpy.ndarray, optional
        Time values corresponding to test data for time-based plots
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and predictions
    """
    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Print test data statistics
    print(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"X_test min: {X_test.min()}, max: {X_test.max()}, mean: {X_test.mean()}")
    print(f"y_test min: {y_test.min()}, max: {y_test.max()}, mean: {y_test.mean()}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check and fix dimensions
    print(f"Original shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
    
    # Ensure y_test and y_pred have compatible shapes
    # Case 1: Multi-horizon prediction (both are 2D arrays)
    if len(y_test.shape) == 2 and len(y_pred.shape) == 2:
        if y_test.shape[1] != y_pred.shape[1]:
            # If horizons don't match, use the minimum horizon
            min_horizon = min(y_test.shape[1], y_pred.shape[1])
            y_test = y_test[:, :min_horizon]
            y_pred = y_pred[:, :min_horizon]
            print(f"Adjusted shapes to match horizons - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
    # Case 2: y_test is 2D but y_pred is 1D
    elif len(y_test.shape) == 2 and len(y_pred.shape) == 1:
        # Reshape y_pred to match y_test's first dimension
        if y_test.shape[0] == y_pred.shape[0]:
            # Single horizon prediction
            y_pred = y_pred.reshape(-1, 1)
            y_test = y_test[:, 0].reshape(-1, 1)  # Take only first horizon from y_test
            print(f"Reshaped y_pred to match y_test - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
        else:
            # Handle case where dimensions completely mismatch
            print(f"Warning: Incompatible shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
            # Reshape both to 1D for basic metrics calculation
            y_test = y_test.flatten()
            y_pred = y_pred.flatten()
            # Truncate to the minimum length
            min_len = min(len(y_test), len(y_pred))
            y_test = y_test[:min_len]
            y_pred = y_pred[:min_len]
            print(f"Flattened and truncated to compatible shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
    # Case 3: y_pred is 2D but y_test is 1D
    elif len(y_test.shape) == 1 and len(y_pred.shape) == 2:
        # Reshape y_test to match y_pred's first dimension
        if y_test.shape[0] == y_pred.shape[0]:
            # Take only first horizon from y_pred
            y_pred = y_pred[:, 0]
            print(f"Taking first horizon from y_pred - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
        else:
            # Handle case where dimensions completely mismatch
            print(f"Warning: Incompatible shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
            # Reshape both to 1D for basic metrics calculation
            y_test = y_test.flatten()
            y_pred = y_pred.flatten()
            # Truncate to the minimum length
            min_len = min(len(y_test), len(y_pred))
            y_test = y_test[:min_len]
            y_pred = y_pred[:min_len]
            print(f"Flattened and truncated to compatible shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
    
    # Print prediction statistics before inverse transform
    print(f"y_pred min: {y_pred.min()}, max: {y_pred.max()}, mean: {y_pred.mean()}")
    
    # Inverse transform if needed
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        try:
            print("Applying inverse transform with scaler...")
            # For multi-horizon predictions, reshape is needed
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                # Create dummy arrays for inverse transform
                y_test_dummy = np.zeros((y_test.shape[0], scaler.n_features_in_))
                y_pred_dummy = np.zeros((y_pred.shape[0], scaler.n_features_in_))
                
                # Assume the target is the first column for simplicity
                # In a real implementation, you'd use the actual target column index
                y_test_dummy[:, 0] = y_test[:, 0]  # Just use first horizon for metrics
                y_pred_dummy[:, 0] = y_pred[:, 0]
                
                # Inverse transform
                y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 0]
                y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, 0]
            else:
                # Single target case - reshape to 2D if needed
                y_test_2d = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
                y_pred_2d = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
                
                # Create dummy arrays if needed
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 1:
                    y_test_dummy = np.zeros((y_test_2d.shape[0], scaler.n_features_in_))
                    y_pred_dummy = np.zeros((y_pred_2d.shape[0], scaler.n_features_in_))
                    
                    y_test_dummy[:, 0] = y_test_2d.flatten()
                    y_pred_dummy[:, 0] = y_pred_2d.flatten()
                    
                    y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 0]
                    y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, 0]
                else:
                    # Direct inverse transform if scaler expects single feature
                    y_test_inv = scaler.inverse_transform(y_test_2d).flatten()
                    y_pred_inv = scaler.inverse_transform(y_pred_2d).flatten()
            
            print(f"After inverse transform - y_test_inv min: {y_test_inv.min()}, max: {y_test_inv.max()}, mean: {y_test_inv.mean()}")
            print(f"After inverse transform - y_pred_inv min: {y_pred_inv.min()}, max: {y_pred_inv.max()}, mean: {y_pred_inv.mean()}")
        except Exception as e:
            print(f"Error during inverse transform: {e}")
            print("Using scaled values for evaluation")
            y_test_inv = y_test
            y_pred_inv = y_pred
    else:
        print("No scaler provided, using original values")
        y_test_inv = y_test
        y_pred_inv = y_pred
    
    # If log transform was applied, reverse it
    if log_transform:
        try:
            print("Reversing log transform...")
            y_test_inv = 10 ** y_test_inv
            y_pred_inv = 10 ** y_pred_inv
            print(f"After log reverse - y_test_inv min: {y_test_inv.min()}, max: {y_test_inv.max()}, mean: {y_test_inv.mean()}")
            print(f"After log reverse - y_pred_inv min: {y_pred_inv.min()}, max: {y_pred_inv.max()}, mean: {y_pred_inv.mean()}")
        except Exception as e:
            print(f"Error reversing log transform: {e}")
    
    # Ensure y_test_inv and y_pred_inv have compatible shapes for metrics calculation
    if len(y_test_inv.shape) != len(y_pred_inv.shape):
        if len(y_test_inv.shape) > len(y_pred_inv.shape):
            y_test_inv = y_test_inv.flatten()
        else:
            y_pred_inv = y_pred_inv.flatten()
    
    # Ensure same length for metrics calculation
    min_len = min(len(y_test_inv.flatten()), len(y_pred_inv.flatten()))
    y_test_flat = y_test_inv.flatten()[:min_len]
    y_pred_flat = y_pred_inv.flatten()[:min_len]
    
    print(f"Final shapes for metrics calculation - y_test: {y_test_flat.shape}, y_pred: {y_pred_flat.shape}")
    
    # Check for NaN or Inf values
    if np.isnan(y_test_flat).any() or np.isnan(y_pred_flat).any():
        print("Warning: NaN values detected in test or prediction data")
        # Replace NaN with zeros for metrics calculation
        y_test_flat = np.nan_to_num(y_test_flat)
        y_pred_flat = np.nan_to_num(y_pred_flat)
    
    if np.isinf(y_test_flat).any() or np.isinf(y_pred_flat).any():
        print("Warning: Inf values detected in test or prediction data")
        # Replace Inf with large values for metrics calculation
        y_test_flat = np.nan_to_num(y_test_flat)
        y_pred_flat = np.nan_to_num(y_pred_flat)
    
    # Calculate basic metrics
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    
    # R² score
    r2 = r2_score(y_test_flat, y_pred_flat)
    
    # Correlation coefficient
    correlation = np.corrcoef(y_test_flat, y_pred_flat)[0, 1]
    
    # Calculate additional metrics
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test_flat - y_pred_flat) / (np.abs(y_test_flat) + epsilon))) * 100
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = 100 * np.mean(2 * np.abs(y_pred_flat - y_test_flat) / (np.abs(y_test_flat) + np.abs(y_pred_flat) + epsilon))
    
    # Median Absolute Error
    median_ae = median_absolute_error(y_test_flat, y_pred_flat)
    
    # Max Error
    max_error_val = max_error(y_test_flat, y_pred_flat)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")
    print(f"Median Absolute Error: {median_ae:.6f}")
    print(f"Max Error: {max_error_val:.6f}")
    
    # Calculate errors
    errors = y_test_flat - y_pred_flat
    
    # Create a dictionary to store metrics
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'correlation': float(correlation),
        'mape': float(mape),
        'smape': float(smape),
        'median_ae': float(median_ae),
        'max_error': float(max_error_val)
    }
    
    # Save metrics to file if requested
    if save_metrics and save_dir:
        import json
        metrics_file = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
    
    # Create visualizations
    if save_dir:
        # 1. Scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_flat, y_pred_flat, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test_flat.min(), y_pred_flat.min())
        max_val = max(y_test_flat.max(), y_pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        plt.text(
            0.05, 0.95, 
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nCorr: {correlation:.4f}", 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Time series plot if time values are provided
        if time_values is not None and len(time_values) == len(y_test_flat):
            plt.figure(figsize=(12, 6))
            plt.plot(time_values, y_test_flat, label='Actual', marker='o', markersize=4, alpha=0.7)
            plt.plot(time_values, y_pred_flat, label='Predicted', marker='x', markersize=4, alpha=0.7)
            plt.title('Time Series of Actual and Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'time_series.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Error distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Error Distribution')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_flat, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Error by forecast horizon (for multi-horizon predictions)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1 and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Calculate MAE for each horizon
            horizon_mae = []
            for h in range(min(y_test.shape[1], y_pred.shape[1])):
                horizon_mae.append(mean_absolute_error(y_test[:, h], y_pred[:, h]))
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(horizon_mae)+1), horizon_mae, 'o-')
            plt.title('Error by Forecast Horizon')
            plt.xlabel('Forecast Horizon')
            plt.ylabel('Mean Absolute Error')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, len(horizon_mae)+1))
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'error_by_horizon.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Return evaluation results
    return {
        'y_test': y_test_inv,
        'y_pred': y_pred_inv,
        'metrics': metrics,
        'errors': errors
    }

def create_attention_seq2seq_model(input_shape, output_size=12, model_type='regression'):
    """
    Create a Seq2Seq model with attention for multi-horizon solar flare prediction.
    """
    # Encoder
    encoder_inputs = tf.keras.Input(shape=input_shape, name='encoder_inputs')
    
    # Bidirectional LSTM encoder - removed recurrent_dropout for better compatibility
    encoder = tf.keras.layers.Bidirectional(
        LSTM(64, return_sequences=True, return_state=True,
             kernel_regularizer=tf.keras.regularizers.L2(1e-6)),
        name='encoder_lstm'
    )(encoder_inputs)
    
    # We get 2 states for each direction (forward and backward)
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder
    
    # Concatenate the states from both directions
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # Decoder setup
    decoder_lstm = LSTM(128, return_sequences=True, return_state=True,
                       kernel_regularizer=tf.keras.regularizers.L2(1e-6))
    
    # Initialize decoder with encoder states
    decoder_inputs = tf.keras.layers.RepeatVector(output_size)(state_h)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    # Add dropout after LSTM
    decoder_outputs = Dropout(0.3)(decoder_outputs)
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()([decoder_outputs, encoder_outputs])
    
    # Combine attention with decoder output
    decoder_combined = tf.keras.layers.Concatenate()([decoder_outputs, attention])
    decoder_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation='relu')
    )(decoder_combined)
    decoder_dense = tf.keras.layers.BatchNormalization()(decoder_dense)
    decoder_dense = tf.keras.layers.Dropout(0.2)(decoder_dense)
    
    # Output layer
    if model_type == 'classification':
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='sigmoid')
        )(decoder_dense)
        model = tf.keras.Model(inputs=encoder_inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='linear')
        )(decoder_dense)
        # Reshape to match expected output shape
        outputs = tf.keras.layers.Reshape((output_size,))(outputs)
        model = tf.keras.Model(inputs=encoder_inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
    
    print(model.summary())
    return model

def train_with_curriculum_learning(X_train, y_train, X_val, y_val, model=None, 
                                  forecast_horizons=[1, 3, 6, 12], 
                                  epochs_per_horizon=25,
                                  batch_size=32, 
                                  model_path='models/lstm_space_weather_model.keras',
                                  early_stopping_patience=10):
    """
    Train with curriculum learning - start with easier short-term predictions 
    and gradually increase difficulty.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data
    y_train : numpy.ndarray
        Training target data (multi-horizon)
    X_val : numpy.ndarray
        Validation input data
    y_val : numpy.ndarray
        Validation target data (multi-horizon)
    model : tensorflow.keras.Model, optional
        Model to train. If None, a new model is created.
    forecast_horizons : list
        List of forecast horizons to use in curriculum
    epochs_per_horizon : int
        Number of epochs to train for each horizon
    batch_size : int
        Batch size for training
    model_path : str
        Path to save the final model
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped
        
    Returns:
    --------
    tuple
        (model, history)
    """
    print("Training with curriculum learning strategy...")
    
    # Validate input shapes
    if len(X_train.shape) != 3:
        raise ValueError(f"Expected X_train to have 3 dimensions (samples, time_steps, features), got shape {X_train.shape}")
    
    if len(y_train.shape) != 2:
        raise ValueError(f"Expected y_train to have 2 dimensions (samples, forecast_horizon), got shape {y_train.shape}")
    
    # Validate that the maximum forecast horizon doesn't exceed the target shape
    max_horizon = max(forecast_horizons)
    if max_horizon > y_train.shape[1]:
        raise ValueError(f"Maximum forecast horizon ({max_horizon}) exceeds target shape ({y_train.shape[1]})")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create model if not provided
    if model is None:
        model = create_attention_seq2seq_model(input_shape, output_size=y_train.shape[1])
    
    # Validate model output shape
    if model.output_shape[1] != y_train.shape[1]:
        raise ValueError(f"Model output shape ({model.output_shape[1]}) doesn't match target shape ({y_train.shape[1]})")
    
    # Initialize combined history
    combined_history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    # Train with progressively longer forecast horizons
    for i, horizon in enumerate(forecast_horizons):
        print(f"\nStage {i+1}/{len(forecast_horizons)}: Training with {horizon}-step horizon")
        
        # For the first few horizons, we'll use only part of the target sequence
        if horizon < y_train.shape[1]:
            y_train_curr = y_train[:, :horizon]
            y_val_curr = y_val[:, :horizon]
            
            # We need to adjust the model output shape for this stage
            if i > 0:  # Skip for the first iteration as we just created the model
                # Get the current model's weights
                weights = model.get_weights()
                
                # Create a new model with the current horizon
                temp_model = create_attention_seq2seq_model(input_shape, output_size=horizon)
                
                # Try to set weights from the previous model where shapes match
                try:
                    temp_model.set_weights(weights)
                    model = temp_model
                    print(f"Adjusted model output shape to {horizon}")
                except ValueError as e:
                    print(f"Could not transfer weights: {e}")
                    print("Creating new model for this stage")
                    model = create_attention_seq2seq_model(input_shape, output_size=horizon)
        else:
            y_train_curr = y_train
            y_val_curr = y_val
        
        # Validate shapes before training
        print(f"Training shapes - X_train: {X_train.shape}, y_train_curr: {y_train_curr.shape}")
        print(f"Model output shape: {model.output_shape}")
        
        if model.output_shape[1] != y_train_curr.shape[1]:
            raise ValueError(f"Model output shape ({model.output_shape[1]}) doesn't match current target shape ({y_train_curr.shape[1]})")
        
        # Train for this stage
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=early_stopping_patience // 2,
                min_lr=0.0001
            )
        ]
        
        history = model.fit(
            X_train, y_train_curr,
            validation_data=(X_val, y_val_curr),
            epochs=epochs_per_horizon,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Add to combined history
        for key in combined_history:
            if key in history.history:
                combined_history[key].extend(history.history[key])
        
        # Optionally adjust learning rate between stages
        K = tf.keras.backend
        lr = K.get_value(model.optimizer.learning_rate)
        K.set_value(model.optimizer.learning_rate, lr * 0.8)  # Reduce learning rate between stages
    
    # For the final stage, ensure we have a model that can predict the full horizon
    if model.output_shape[1] != y_train.shape[1]:
        print("Creating final model for full forecast horizon")
        final_model = create_attention_seq2seq_model(input_shape, output_size=y_train.shape[1])
        
        # Train the final model with the knowledge gained from curriculum learning
        final_history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_horizon // 2,  # Fewer epochs for final training
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path, 
                    monitor='val_loss',
                    save_best_only=True
                )
            ],
            verbose=1
        )
        
        # Add to combined history
        for key in combined_history:
            if key in final_history.history:
                combined_history[key].extend(final_history.history[key])
        
        model = final_model
    
    # Save final model
    model.save(model_path)
    print(f"Curriculum learning completed, final model saved to {model_path}")
    return model, combined_history

def physics_guided_loss(y_true, y_pred, continuity_threshold=0.5, decay_threshold=0.3, 
                       continuity_weight=0.1, decay_weight=0.1):
    """
    Custom loss function with physics-based constraints for solar flare prediction.
    
    Parameters:
    -----------
    y_true : tf.Tensor
        True target values
    y_pred : tf.Tensor
        Predicted target values
    continuity_threshold : float
        Threshold for penalizing physically impossible rapid changes
    decay_threshold : float
        Threshold for penalizing rapid decay
    continuity_weight : float
        Weight for the continuity penalty term
    decay_weight : float
        Weight for the decay rate penalty term
    
    Returns:
    --------
    tf.Tensor
        Total loss value
    """
    # Standard MSE loss
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Temporal continuity constraint - penalize physically impossible rapid changes
    # For multi-horizon predictions, check consecutive timesteps
    if len(tf.shape(y_pred)) > 1 and tf.shape(y_pred)[1] > 1:
        # Calculate differences between consecutive predictions
        pred_diffs = tf.abs(y_pred[:, 1:] - y_pred[:, :-1])
        
        # Penalize large jumps (more than continuity_threshold increase in one timestep)
        # This is based on typical solar flare evolution physics
        continuity_penalty = tf.reduce_mean(
            tf.maximum(0.0, pred_diffs - continuity_threshold)
        ) * continuity_weight
    else:
        continuity_penalty = 0.0
    
    # Solar flare decay rate constraint
    # Flares typically decay more slowly than they rise
    if len(tf.shape(y_pred)) > 1 and tf.shape(y_pred)[1] > 2:
        # Find where predictions are decreasing
        decreasing_mask = tf.cast(y_pred[:, 1:] < y_pred[:, :-1], tf.float32)
        
        # Calculate decay rates (negative differences)
        decay_rates = (y_pred[:, 1:] - y_pred[:, :-1]) * decreasing_mask
        
        # Penalize rapid decay (more than decay_threshold drop in one timestep)
        rapid_decay_penalty = tf.reduce_mean(
            tf.maximum(0.0, tf.abs(decay_rates) - decay_threshold)
        ) * decay_weight
    else:
        rapid_decay_penalty = 0.0
    
    total_loss = mse_loss + rapid_decay_penalty + continuity_penalty
    
    return total_loss

def create_pretrained_model(input_shape, output_size=12):
    """
    Create a model based on a pre-trained architecture and fine-tune it for solar data.
    Uses a 1D CNN architecture which is more appropriate for time series data.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # 1D CNN layers for time series feature extraction
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add LSTM layer to capture temporal dependencies
    x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Add solar-specific layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer for multi-horizon prediction
    outputs = tf.keras.layers.Dense(output_size)(x)
    
    # Create model
    model = tf.keras.Model(inputs, outputs)
    
    # Compile with our physics-guided loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=lambda y_true, y_pred: physics_guided_loss(
            y_true, y_pred, 
            continuity_threshold=0.5, 
            decay_threshold=0.3
        ),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def train_with_transfer_learning(X_train, y_train, X_val, y_val, model_path='models/transfer_model.keras',
                                early_stopping_patience=10):
    """
    Train using transfer learning approach.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data
    y_train : numpy.ndarray
        Training target data
    X_val : numpy.ndarray
        Validation input data
    y_val : numpy.ndarray
        Validation target data
    model_path : str
        Path to save the best model
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped
        
    Returns:
    --------
    tuple
        (model, history)
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = y_train.shape[1]
    
    # Create model with frozen base
    model = create_pretrained_model(input_shape, output_size)
    
    # First stage: train only the top layers
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path.replace('.keras', '_stage1.keras'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
    )
    
    # Second stage: fine-tune upper layers of the base model
    for layer in model.layers[-20:]:
        layer.trainable = True
        
    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=lambda y_true, y_pred: physics_guided_loss(
            y_true, y_pred, 
            continuity_threshold=0.5, 
            decay_threshold=0.3
        ),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,  # Smaller batch size for fine-tuning
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return model, combined_history

if __name__ == "__main__":
    # Example usage with proper error handling
    try:
        # Import visualization module
        import sys
        sys.path.append('src')
        from visualization.visualize import visualize_model_performance
        import datetime
        
        # Define file paths
        data_dir = 'data/processed'
        X_train_path = os.path.join(data_dir, 'X_train.npy')
        y_train_path = os.path.join(data_dir, 'y_train.npy')
        X_test_path = os.path.join(data_dir, 'X_test.npy')
        y_test_path = os.path.join(data_dir, 'y_test.npy')
        model_path = os.path.join('models', 'lstm_solar_forecaster.keras')
        results_dir = os.path.join('reports', 'figures')
        
        # Create directories if they don't exist
        for directory in [data_dir, 'models', results_dir, 'logs', 'logs/fit']:
            os.makedirs(directory, exist_ok=True)
        
        # Check if data files exist
        if all(os.path.exists(f) for f in [X_train_path, y_train_path, X_test_path, y_test_path]):
            # Load data
            print("Loading existing data files...")
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
            X_test = np.load(X_test_path)
            y_test = np.load(y_test_path)
            
            # Try to load metadata if available
            metadata_path = os.path.join(data_dir, 'preprocessing_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: {metadata}")
                log_transform_applied = metadata.get('log_transform_applied', False)
            else:
                log_transform_applied = False
                print("No metadata file found, assuming no log transform was applied.")
            
            # Print data shapes
            print(f"Data loaded successfully with shapes:")
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            # Check if we have enough training data
            if X_train.shape[0] < 500:
                print(f"Warning: Training set is small ({X_train.shape[0]} samples). Augmenting data...")
                try:
                    from src.data.augmentation import augment_time_series_data
                    X_train, y_train = augment_time_series_data(X_train, y_train, augmentation_factor=4)
                    print(f"Data augmented. New shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
                except ImportError:
                    print("Could not import augmentation module. Proceeding with original data.")
            
            # Train model with increased patience and epochs
            print("Training model with increased patience...")
            model, history = train_model(
                X_train, y_train, X_test, y_test, 
                model_path=model_path,
                epochs=200,  # Increased from 100
                early_stopping_patience=30  # Increased from 15
            )
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Evaluate model with enhanced metrics and visualizations
            print("Evaluating model with enhanced metrics...")
            evaluation = evaluate_model(
                model, 
                X_test, 
                y_test, 
                log_transform=log_transform_applied,
                save_dir=results_dir,
                save_metrics=True
            )
            
            # Create advanced visualizations
            print("Creating advanced visualizations...")
            visualize_model_performance(
                evaluation['y_test'],
                evaluation['y_pred'],
                metrics=evaluation['metrics'],
                save_dir=os.path.join(results_dir, 'advanced_visualizations'),
                model_name='standard_lstm'
            )
            
            # Save evaluation results
            np.save(os.path.join(results_dir, 'y_test.npy'), evaluation['y_test'])
            np.save(os.path.join(results_dir, 'y_pred.npy'), evaluation['y_pred'])
            np.save(os.path.join(results_dir, 'errors.npy'), evaluation['errors'])
            
            # Try different model architectures and compare
            print("\nTraining and evaluating alternative model architectures for comparison...")
            
            # 1. Train with curriculum learning
            print("\n1. Training with curriculum learning...")
            curriculum_model_path = os.path.join('models', 'curriculum_solar_forecaster.keras')
            curriculum_model, curriculum_history = train_with_curriculum_learning(
                X_train, y_train, X_test, y_test, 
                model_path=curriculum_model_path,
                early_stopping_patience=20  # Increased patience
            )
            
            # Evaluate curriculum model
            print("Evaluating curriculum learning model...")
            curriculum_results_dir = os.path.join(results_dir, 'curriculum_model')
            os.makedirs(curriculum_results_dir, exist_ok=True)
            
            curriculum_evaluation = evaluate_model(
                curriculum_model, 
                X_test, 
                y_test, 
                log_transform=log_transform_applied,
                save_dir=curriculum_results_dir,
                save_metrics=True
            )
            
            # Create advanced visualizations for curriculum model
            visualize_model_performance(
                curriculum_evaluation['y_test'],
                curriculum_evaluation['y_pred'],
                metrics=curriculum_evaluation['metrics'],
                save_dir=os.path.join(curriculum_results_dir, 'advanced_visualizations'),
                model_name='curriculum_lstm'
            )
            
            # 2. Train with transfer learning approach
            print("\n2. Training with transfer learning approach...")
            transfer_model_path = os.path.join('models', 'transfer_solar_forecaster.keras')
            transfer_model, transfer_history = train_with_transfer_learning(
                X_train, y_train, X_test, y_test, 
                model_path=transfer_model_path,
                early_stopping_patience=20  # Increased patience
            )
            
            # Evaluate transfer learning model
            print("Evaluating transfer learning model...")
            transfer_results_dir = os.path.join(results_dir, 'transfer_model')
            os.makedirs(transfer_results_dir, exist_ok=True)
            
            transfer_evaluation = evaluate_model(
                transfer_model, 
                X_test, 
                y_test, 
                log_transform=log_transform_applied,
                save_dir=transfer_results_dir,
                save_metrics=True
            )
            
            # Create advanced visualizations for transfer learning model
            visualize_model_performance(
                transfer_evaluation['y_test'],
                transfer_evaluation['y_pred'],
                metrics=transfer_evaluation['metrics'],
                save_dir=os.path.join(transfer_results_dir, 'advanced_visualizations'),
                model_name='transfer_learning'
            )
            
            # Compare model performances
            print("\nModel Performance Comparison:")
            print("-" * 50)
            print(f"{'Metric':<15} {'Standard LSTM':<15} {'Curriculum':<15} {'Transfer':<15}")
            print("-" * 50)
            
            for metric in ['rmse', 'mae', 'r2', 'correlation']:
                print(f"{metric:<15} {evaluation['metrics'][metric]:<15.4f} "
                      f"{curriculum_evaluation['metrics'][metric]:<15.4f} "
                      f"{transfer_evaluation['metrics'][metric]:<15.4f}")
            
            print("-" * 50)
            
            # Save comparison to file
            comparison = {
                'standard_lstm': evaluation['metrics'],
                'curriculum_learning': curriculum_evaluation['metrics'],
                'transfer_learning': transfer_evaluation['metrics']
            }
            
            with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
                json.dump(comparison, f, indent=4)
            
            print(f"Model comparison saved to {os.path.join(results_dir, 'model_comparison.json')}")
            
            # Create a combined visualization comparing all models
            print("\nCreating combined model comparison visualizations...")
            
            # Prepare data for combined visualization
            all_models_data = {
                'Standard LSTM': {
                    'y_test': evaluation['y_test'],
                    'y_pred': evaluation['y_pred'],
                    'metrics': evaluation['metrics']
                },
                'Curriculum Learning': {
                    'y_test': curriculum_evaluation['y_test'],
                    'y_pred': curriculum_evaluation['y_pred'],
                    'metrics': curriculum_evaluation['metrics']
                },
                'Transfer Learning': {
                    'y_test': transfer_evaluation['y_test'],
                    'y_pred': transfer_evaluation['y_pred'],
                    'metrics': transfer_evaluation['metrics']
                }
            }
            
            # Create combined error by horizon plot if we have multi-horizon predictions
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                plt.figure(figsize=(14, 8))
                
                # Calculate MAE for each horizon for each model
                for model_name, model_data in all_models_data.items():
                    # Ensure we have compatible shapes
                    if len(model_data['y_test'].shape) > 1 and len(model_data['y_pred'].shape) > 1:
                        min_horizon = min(model_data['y_test'].shape[1] if len(model_data['y_test'].shape) > 1 else 1,
                                        model_data['y_pred'].shape[1] if len(model_data['y_pred'].shape) > 1 else 1)
                        
                        # Reshape if needed
                        if len(model_data['y_test'].shape) == 1:
                            y_test_model = model_data['y_test'].reshape(-1, 1)
                        else:
                            y_test_model = model_data['y_test'][:, :min_horizon]
                            
                        if len(model_data['y_pred'].shape) == 1:
                            y_pred_model = model_data['y_pred'].reshape(-1, 1)
                        else:
                            y_pred_model = model_data['y_pred'][:, :min_horizon]
                        
                        # Calculate MAE for each horizon
                        horizon_mae = []
                        for h in range(min_horizon):
                            horizon_mae.append(mean_absolute_error(
                                y_test_model[:, h].flatten(), 
                                y_pred_model[:, h].flatten()
                            ))
                        
                        plt.plot(range(1, len(horizon_mae)+1), horizon_mae, 'o-', label=f'{model_name} MAE')
                
                plt.xlabel('Forecast Horizon')
                plt.ylabel('Mean Absolute Error')
                plt.title('Error by Forecast Horizon - Model Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(range(1, min_horizon+1))
                
                combined_dir = os.path.join(results_dir, 'combined_comparison')
                os.makedirs(combined_dir, exist_ok=True)
                plt.savefig(os.path.join(combined_dir, 'horizon_error_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create combined average prediction plot
                plt.figure(figsize=(14, 8))
                
                for model_name, model_data in all_models_data.items():
                    # Ensure we have compatible shapes
                    if len(model_data['y_test'].shape) > 1 and len(model_data['y_pred'].shape) > 1:
                        min_horizon = min(model_data['y_test'].shape[1] if len(model_data['y_test'].shape) > 1 else 1,
                                        model_data['y_pred'].shape[1] if len(model_data['y_pred'].shape) > 1 else 1)
                        
                        # Reshape if needed
                        if len(model_data['y_test'].shape) == 1:
                            y_test_model = model_data['y_test'].reshape(-1, 1)
                        else:
                            y_test_model = model_data['y_test'][:, :min_horizon]
                            
                        if len(model_data['y_pred'].shape) == 1:
                            y_pred_model = model_data['y_pred'].reshape(-1, 1)
                        else:
                            y_pred_model = model_data['y_pred'][:, :min_horizon]
                        
                        # Calculate means
                        mean_actual = np.mean(y_test_model, axis=0)
                        mean_pred = np.mean(y_pred_model, axis=0)
                        
                        # Plot only predicted means (actual should be the same for all models)
                        if model_name == 'Standard LSTM':
                            plt.plot(range(1, len(mean_actual)+1), mean_actual, 'k-', linewidth=2, label='Actual (Mean)')
                        
                        plt.plot(range(1, len(mean_pred)+1), mean_pred, '--', linewidth=2, label=f'{model_name} (Mean)')
                
                plt.xlabel('Forecast Horizon')
                plt.ylabel('Value')
                plt.title('Average Prediction by Model')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(combined_dir, 'average_prediction_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print("Model training, evaluation, and comparison completed successfully.")
            print(f"All results saved to {results_dir}")
            
        else:
            print("Data files not found. Please run data preprocessing first.")
            print("Missing files:")
            for f in [X_train_path, y_train_path, X_test_path, y_test_path]:
                if not os.path.exists(f):
                    print(f"  - {f}")
            
            # Create dummy data for demonstration purposes
            print("\nCreating dummy data for demonstration...")
            sequence_length = 24
            n_features = 10
            forecast_horizon = 12
            n_samples = 2000  # Increased from 1000
            
            X_train = np.random.random((n_samples, sequence_length, n_features))
            y_train = np.random.random((n_samples, forecast_horizon))
            X_test = np.random.random((int(n_samples * 0.2), sequence_length, n_features))
            y_test = np.random.random((int(n_samples * 0.2), forecast_horizon))
            
            # Add some patterns to make the dummy data more realistic
            for i in range(n_samples):
                # Add trend
                trend = np.linspace(0, 0.5, sequence_length).reshape(-1, 1)
                X_train[i, :, 0] += trend.flatten()  # Add trend to first feature
                
                # Add seasonality
                season = 0.2 * np.sin(np.linspace(0, 4*np.pi, sequence_length))
                X_train[i, :, 1] += season  # Add seasonality to second feature
                
                # Make target related to input
                if i < n_samples:
                    base_level = np.mean(X_train[i, -5:, 0])  # Use last 5 timesteps of first feature
                    for h in range(forecast_horizon):
                        # Target follows input with some noise and trend
                        y_train[i, h] = base_level + 0.1 * h + np.random.normal(0, 0.05)
            
            # Do the same for test data
            for i in range(int(n_samples * 0.2)):
                trend = np.linspace(0, 0.5, sequence_length).reshape(-1, 1)
                X_test[i, :, 0] += trend.flatten()
                
                season = 0.2 * np.sin(np.linspace(0, 4*np.pi, sequence_length))
                X_test[i, :, 1] += season
                
                base_level = np.mean(X_test[i, -5:, 0])
                for h in range(forecast_horizon):
                    y_test[i, h] = base_level + 0.1 * h + np.random.normal(0, 0.05)
            
            # Save dummy data
            np.save(X_train_path, X_train)
            np.save(y_train_path, y_train)
            np.save(X_test_path, X_test)
            np.save(y_test_path, y_test)
            
            # Save metadata
            with open(os.path.join(data_dir, 'preprocessing_metadata.json'), 'w') as f:
                json.dump({
                    'log_transform_applied': False,
                    'target_column': 'synthetic_target',
                    'sequence_length': sequence_length,
                    'forecast_horizon': forecast_horizon,
                    'feature_names': [f'feature_{i}' for i in range(n_features)],
                    'target_column_idx': 0,
                    'scaler_type': 'None',
                    'train_size': n_samples,
                    'test_size': int(n_samples * 0.2),
                    'stride': 1,
                    'augmentation_applied': False,
                    'is_synthetic': True
                }, f)
            
            print(f"Dummy data created and saved with shapes:")
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            # Train model with dummy data
            print("Training model with dummy data...")
            model, history = train_model(X_train, y_train, X_test, y_test, model_path=model_path, epochs=50)
            
            # Evaluate model with enhanced metrics
            print("Evaluating model with enhanced metrics...")
            evaluation = evaluate_model(
                model, 
                X_test, 
                y_test, 
                save_dir=results_dir,
                save_metrics=True
            )
            
            # Create advanced visualizations
            print("Creating advanced visualizations...")
            visualize_model_performance(
                evaluation['y_test'],
                evaluation['y_pred'],
                metrics=evaluation['metrics'],
                save_dir=os.path.join(results_dir, 'advanced_visualizations'),
                model_name='standard_lstm'
            )
            
            print("Model training and evaluation with dummy data completed successfully.")
            print(f"Results saved to {results_dir}")
            print("Note: This is using synthetic data with patterns for demonstration purposes.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 