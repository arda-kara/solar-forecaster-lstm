import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from src.models.lstm_model import physics_guided_loss

def create_gru_model(input_shape, output_size=12, model_type='regression'):
    """
    Create a GRU-based model for space weather forecasting.
    This provides a different architecture for the ensemble.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    output_size : int
        Number of output units (forecast horizon)
    model_type : str
        Type of model: 'regression' or 'classification'
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled GRU model
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # First GRU layer
    x = GRU(64, activation='tanh', return_sequences=True, 
           recurrent_dropout=0.1,
           kernel_regularizer=tf.keras.regularizers.L2(1e-6))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second GRU layer
    x = GRU(32, activation='tanh', return_sequences=False,
           recurrent_dropout=0.1,
           kernel_regularizer=tf.keras.regularizers.L2(1e-6))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-6))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    if model_type == 'classification':
        outputs = Dense(output_size, activation='sigmoid')(x)
    else:
        outputs = Dense(output_size, activation='linear')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if model_type == 'classification':
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=physics_guided_loss,
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
    
    return model 