import yaml
import os

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration parameters
    """
    if not os.path.exists(config_path):
        # Create default configuration
        config = {
            'data': {
                'sequence_length': 24,
                'forecast_horizon': 12,
                'test_split': 0.2,
                'log_transform': True,
                'target_column': 'xray_flux_long'
            },
            'model': {
                'type': 'ensemble',  # 'lstm', 'attention', 'gru', 'ensemble'
                'use_curriculum': True,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 15
            },
            'paths': {
                'data_dir': 'data',
                'model_dir': 'models',
                'reports_dir': 'reports/figures'
            }
        }
        
        # Save default configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 