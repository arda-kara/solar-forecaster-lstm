import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
from scipy import stats

def plot_xray_flux_time_series(data, save_dir=None):
    """
    Plot GOES X-ray flux time series.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing GOES X-ray flux data
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    if 'time' not in data.columns or ('xray_flux_short' not in data.columns and 'xray_flux_long' not in data.columns):
        print("Required columns not found in data.")
        return
    
    plt.figure(figsize=(12, 6))
    
    if 'xray_flux_short' in data.columns:
        plt.semilogy(data['time'], data['xray_flux_short'], label='Short Wavelength (0.5-4.0 Å)')
    
    if 'xray_flux_long' in data.columns:
        plt.semilogy(data['time'], data['xray_flux_long'], label='Long Wavelength (1.0-8.0 Å)')
    
    # Add GOES flare classification lines
    plt.axhline(y=1e-4, color='r', linestyle='--', alpha=0.7, label='X-class threshold')
    plt.axhline(y=1e-5, color='orange', linestyle='--', alpha=0.7, label='M-class threshold')
    plt.axhline(y=1e-6, color='g', linestyle='--', alpha=0.7, label='C-class threshold')
    plt.axhline(y=1e-7, color='b', linestyle='--', alpha=0.7, label='B-class threshold')
    
    plt.xlabel('Time')
    plt.ylabel('X-ray Flux (W/m²)')
    plt.title('GOES X-ray Flux')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'xray_flux_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_solar_wind_parameters(data, save_dir=None):
    """
    Plot ACE solar wind parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing ACE solar wind data
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    if 'time' not in data.columns:
        print("Time column not found in data.")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Plot speed
    if 'speed' in data.columns:
        axes[0].plot(data['time'], data['speed'], 'b-')
        axes[0].set_ylabel('Speed (km/s)')
        axes[0].set_title('Solar Wind Speed')
        axes[0].grid(True, alpha=0.3)
    
    # Plot density
    if 'density' in data.columns:
        axes[1].plot(data['time'], data['density'], 'g-')
        axes[1].set_ylabel('Density (n/cm³)')
        axes[1].set_title('Solar Wind Density')
        axes[1].grid(True, alpha=0.3)
    
    # Plot temperature
    if 'temperature' in data.columns:
        axes[2].plot(data['time'], data['temperature'], 'r-')
        axes[2].set_ylabel('Temperature (K)')
        axes[2].set_title('Solar Wind Temperature')
        axes[2].grid(True, alpha=0.3)
    
    # Plot IMF components
    if 'bz' in data.columns and 'bt' in data.columns:
        axes[3].plot(data['time'], data['bz'], 'r-', label='Bz')
        axes[3].plot(data['time'], data['bt'], 'b-', label='Bt')
        axes[3].set_ylabel('IMF (nT)')
        axes[3].set_title('Interplanetary Magnetic Field')
        axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[3].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'solar_wind_parameters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_model_performance(history, save_dir=None):
    """
    Plot model training history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    if 'mae' in history:
        plt.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_results(y_true, y_pred, dates=None, forecast_horizon=12, save_dir=None):
    """
    Plot multi-horizon prediction results.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    dates : numpy.ndarray, optional
        Dates corresponding to the predictions
    forecast_horizon : int
        Number of steps in the forecast horizon
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    # For multi-horizon predictions
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        plt.figure(figsize=(14, 7))
        
        # Plot actual vs predicted for the first few test samples
        num_samples = min(5, len(y_true))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            plt.plot(range(forecast_horizon), y_true[i], 'b-', label='Actual')
            plt.plot(range(forecast_horizon), y_pred[i], 'r--', label='Predicted')
            plt.title(f'Sample {i+1}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'multi_horizon_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Plot time series if dates are provided
    if dates is not None:
        plt.figure(figsize=(14, 7))
        
        # For single-step prediction or using just the first step of multi-horizon
        y_true_plot = y_true[:, 0] if len(y_true.shape) > 1 else y_true
        y_pred_plot = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        
        plt.plot(dates, y_true_plot, 'b-', label='Actual')
        plt.plot(dates, y_pred_plot, 'r--', label='Predicted')
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'time_series_prediction.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Plot overall comparison
    plt.figure(figsize=(10, 10))
    
    # Flatten arrays for scatter plot
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Create scatter plot
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient and R² to plot
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    r2 = r2_score(y_true_flat, y_pred_flat)
    plt.annotate(f'Correlation: {correlation:.4f}\nR²: {r2:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_dir=None):
    """
    Plot feature importance for interpretable models.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_attention_weights(attention_weights, input_sequence, output_sequence, save_dir=None):
    """
    Plot attention weights for sequence-to-sequence models.
    
    Parameters:
    -----------
    attention_weights : numpy.ndarray
        Attention weights with shape (output_len, input_len)
    input_sequence : list
        Input sequence
    output_sequence : list
        Output sequence
    save_dir : str, optional
        Directory to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', annot=False)
    
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.title('Attention Weights')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_flare_prediction_metrics(y_true, y_pred_prob, output_dir='reports/figures'):
    """
    Plot metrics for binary flare prediction.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True binary labels
    y_pred_prob : numpy.ndarray
        Predicted probabilities
    output_dir : str
        Directory to save the plots
    """
    # Create ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save ROC curve
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save precision-recall curve
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix at threshold 0.5
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_performance(y_test, y_pred, metrics=None, time_values=None, 
                               save_dir='reports/figures/model_performance', 
                               model_name='lstm_model'):
    """
    Create advanced visualizations for model evaluation results.
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        Actual target values
    y_pred : numpy.ndarray
        Predicted target values
    metrics : dict, optional
        Dictionary of evaluation metrics
    time_values : numpy.ndarray, optional
        Time values for time series plots
    save_dir : str
        Directory to save visualizations
    model_name : str
        Name of the model for labeling
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure arrays are flattened for single-target predictions
    if len(y_test.shape) == 1 or (len(y_test.shape) == 2 and y_test.shape[1] == 1):
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        # For multi-horizon, we'll use the first horizon for some plots
        y_test_flat = y_test[:, 0]
        y_pred_flat = y_pred[:, 0]
    
    # Calculate metrics if not provided
    if metrics is None:
        metrics = {
            'mse': mean_squared_error(y_test_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)),
            'mae': mean_absolute_error(y_test_flat, y_pred_flat),
            'r2': r2_score(y_test_flat, y_pred_flat),
            'correlation': np.corrcoef(y_test_flat, y_pred_flat)[0, 1]
        }
    
    # 1. Enhanced scatter plot with hexbin for density
    plt.figure(figsize=(12, 10))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, label='Predictions')
    
    # Add perfect prediction line
    min_val = min(y_test_flat.min(), y_pred_flat.min())
    max_val = max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_flat, y_pred_flat)
    plt.plot(np.array([min_val, max_val]), 
             slope * np.array([min_val, max_val]) + intercept, 
             'g-', label=f'Regression Line (slope={slope:.2f})')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics annotation
    metrics_text = "\n".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', fontsize=9)
    
    # Hexbin plot for density
    plt.subplot(2, 2, 2)
    hb = plt.hexbin(y_test_flat, y_pred_flat, gridsize=30, cmap='Blues')
    plt.colorbar(hb, label='Count')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Density of Predictions')
    plt.grid(True, alpha=0.3)
    
    # Error histogram with KDE
    plt.subplot(2, 2, 3)
    errors = y_pred_flat - y_test_flat
    sns.histplot(errors, kde=True, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # Add error statistics
    error_stats = f"Mean Error: {np.mean(errors):.4f}\nStd Dev: {np.std(errors):.4f}"
    plt.annotate(error_stats, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', fontsize=9)
    
    # Q-Q plot for error normality
    plt.subplot(2, 2, 4)
    stats.probplot(errors, plot=plt)
    plt.title('Q-Q Plot of Errors')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time series visualization if we have multi-horizon predictions
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        # Plot for selected samples
        num_samples = min(5, y_test.shape[0])
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            
            # Plot actual values
            plt.plot(range(y_test.shape[1]), y_test[i], 'b-', label='Actual', linewidth=2)
            
            # Plot predicted values
            plt.plot(range(y_test.shape[1]), y_pred[i], 'r--', label='Predicted', linewidth=2)
            
            # Calculate error metrics for this sample
            sample_mae = mean_absolute_error(y_test[i], y_pred[i])
            sample_rmse = np.sqrt(mean_squared_error(y_test[i], y_pred[i]))
            
            plt.title(f'Sample {i+1} - MAE: {sample_mae:.4f}, RMSE: {sample_rmse:.4f}')
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_sample_forecasts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot average prediction with confidence intervals
        plt.figure(figsize=(12, 6))
        
        # Calculate mean and standard deviation across samples
        mean_actual = np.mean(y_test, axis=0)
        std_actual = np.std(y_test, axis=0)
        mean_pred = np.mean(y_pred, axis=0)
        std_pred = np.std(y_pred, axis=0)
        
        # Plot mean values
        plt.plot(range(y_test.shape[1]), mean_actual, 'b-', label='Actual (Mean)', linewidth=2)
        plt.plot(range(y_test.shape[1]), mean_pred, 'r--', label='Predicted (Mean)', linewidth=2)
        
        # Add confidence intervals (±1 std)
        plt.fill_between(range(y_test.shape[1]), 
                        mean_actual - std_actual,
                        mean_actual + std_actual,
                        alpha=0.2, color='blue', label='Actual ±1σ')
        
        plt.fill_between(range(y_test.shape[1]), 
                        mean_pred - std_pred,
                        mean_pred + std_pred,
                        alpha=0.2, color='red', label='Predicted ±1σ')
        
        plt.xlabel('Forecast Horizon')
        plt.ylabel('Value')
        plt.title('Average Forecast with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{model_name}_average_forecast.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot error by horizon
        plt.figure(figsize=(12, 6))
        
        # Calculate MAE and RMSE for each horizon
        horizon_mae = [mean_absolute_error(y_test[:, h], y_pred[:, h]) for h in range(y_test.shape[1])]
        horizon_rmse = [np.sqrt(mean_squared_error(y_test[:, h], y_pred[:, h])) for h in range(y_test.shape[1])]
        
        plt.plot(range(1, y_test.shape[1]+1), horizon_mae, 'bo-', label='MAE')
        plt.plot(range(1, y_test.shape[1]+1), horizon_rmse, 'ro-', label='RMSE')
        
        plt.xlabel('Forecast Horizon')
        plt.ylabel('Error')
        plt.title('Forecast Error by Horizon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, y_test.shape[1]+1))
        plt.savefig(os.path.join(save_dir, f'{model_name}_error_by_horizon.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Time series visualization if time values are provided
    if time_values is not None:
        plt.figure(figsize=(15, 6))
        
        # If we have multi-horizon predictions, use the first horizon
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_plot = y_test[:, 0]
            y_pred_plot = y_pred[:, 0]
        else:
            y_test_plot = y_test_flat
            y_pred_plot = y_pred_flat
        
        # Ensure time_values matches the length of the data
        if len(time_values) >= len(y_test_plot):
            time_values = time_values[:len(y_test_plot)]
            
            plt.plot(time_values, y_test_plot, 'b-', label='Actual')
            plt.plot(time_values, y_pred_plot, 'r--', label='Predicted')
            
            # Calculate rolling error
            window = min(24, len(y_test_plot) // 10)  # Use about 10% of data points for window
            if window > 0:
                rolling_mae = [mean_absolute_error(y_test_plot[max(0, i-window):i+1], 
                                                y_pred_plot[max(0, i-window):i+1]) 
                            for i in range(len(y_test_plot))]
                
                # Plot rolling MAE on secondary y-axis
                ax2 = plt.gca().twinx()
                ax2.plot(time_values, rolling_mae, 'g-', alpha=0.5, label='Rolling MAE')
                ax2.set_ylabel('Rolling MAE', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
            
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Time Series of Actual vs Predicted Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_time_series.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Create a heatmap of prediction accuracy by value range
    plt.figure(figsize=(12, 10))
    
    # Define bins for actual values
    n_bins = 10
    actual_bins = np.linspace(y_test_flat.min(), y_test_flat.max(), n_bins+1)
    
    # Create a matrix to store average errors by bin
    error_matrix = np.zeros((n_bins, n_bins))
    count_matrix = np.zeros((n_bins, n_bins))
    
    # Calculate absolute errors
    abs_errors = np.abs(y_pred_flat - y_test_flat)
    
    # Bin the data and calculate average error in each bin
    for i in range(len(y_test_flat)):
        actual_bin = np.digitize(y_test_flat[i], actual_bins) - 1
        pred_bin = np.digitize(y_pred_flat[i], actual_bins) - 1
        
        # Ensure bin indices are within bounds
        actual_bin = min(actual_bin, n_bins-1)
        pred_bin = min(pred_bin, n_bins-1)
        
        error_matrix[actual_bin, pred_bin] += abs_errors[i]
        count_matrix[actual_bin, pred_bin] += 1
    
    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    avg_error_matrix = error_matrix / count_matrix
    
    # Create heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(count_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Value Bin')
    plt.ylabel('Actual Value Bin')
    plt.title('Count of Predictions by Value Range')
    
    plt.subplot(2, 1, 2)
    sns.heatmap(avg_error_matrix, annot=True, fmt='.3f', cmap='Reds')
    plt.xlabel('Predicted Value Bin')
    plt.ylabel('Actual Value Bin')
    plt.title('Average Absolute Error by Value Range')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced visualizations saved to {save_dir}") 