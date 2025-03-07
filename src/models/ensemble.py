import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.lstm_model import create_lstm_model, create_attention_seq2seq_model
from src.models.gru_model import create_gru_model

def create_model_ensemble(input_shape, forecast_horizon=12):
    """
    Create an ensemble of different model architectures.
    """
    # Create diverse base models
    model1 = create_attention_seq2seq_model(input_shape, output_size=forecast_horizon)
    model2 = create_lstm_model(input_shape, output_size=forecast_horizon)
    model3 = create_gru_model(input_shape, output_size=forecast_horizon)  # A GRU-based model
    
    # Create a meta-learner that combines predictions
    inputs = tf.keras.Input(shape=input_shape)
    
    # Get predictions from each model
    pred1 = model1(inputs)
    pred2 = model2(inputs)
    pred3 = model3(inputs)
    
    # Combine predictions using a learnable weighting
    combined_preds = tf.keras.layers.Concatenate()([
        tf.keras.layers.Reshape((forecast_horizon, 1))(pred1),
        tf.keras.layers.Reshape((forecast_horizon, 1))(pred2),
        tf.keras.layers.Reshape((forecast_horizon, 1))(pred3)
    ])
    
    # Meta-learner weights the models differently at each time step
    weighted_preds = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='linear')
    )(combined_preds)
    
    # Final output
    outputs = tf.keras.layers.Reshape((forecast_horizon,))(weighted_preds)
    
    # Create and compile ensemble model
    ensemble = tf.keras.Model(inputs=inputs, outputs=outputs)
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return ensemble, [model1, model2, model3]

def train_ensemble(X_train, y_train, X_val, y_val, ensemble_path='models/ensemble_model.keras'):
    """
    Train the ensemble model using a two-stage approach.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    forecast_horizon = y_train.shape[1]
    
    # Create ensemble and base models
    ensemble, base_models = create_model_ensemble(input_shape, forecast_horizon)
    
    # First stage: Train base models independently
    for i, model in enumerate(base_models):
        print(f"Training base model {i+1}...")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(f'models/base_model_{i+1}.keras', save_best_only=True)
            ]
        )
    
    # Freeze base models
    for model in base_models:
        for layer in model.layers:
            layer.trainable = False
    
    # Second stage: Train ensemble (meta-learner) with frozen base models
    ensemble.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            ModelCheckpoint(ensemble_path, save_best_only=True)
        ]
    )
    
    return ensemble 