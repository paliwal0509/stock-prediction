"""Training module"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import create_sequences, scale_features, split_data


def train_model(model, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray = None, y_test: np.ndarray = None,
                epochs: int = 50, batch_size: int = 32):
    """
    Train a model on stock data
    
    Args:
        model: Model instance (LSTM, RandomForest, or XGBoost)
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        epochs: Number of epochs (for LSTM)
        batch_size: Batch size (for LSTM)
    
    Returns:
        Training history
    """
    # Determine model type
    model_type = model.__class__.__name__
    
    if model_type == "LSTMModel":
        # Scale features for LSTM using ALL data (to handle sequence boundary)
        scaler = MinMaxScaler()
        X_all_scaled = scaler.fit_transform(np.vstack([X_train, X_test]))
        
        # Split back after scaling
        split_idx = len(X_train)
        X_train_scaled = X_all_scaled[:split_idx]
        X_test_scaled = X_all_scaled[split_idx:]
        
        # Create sequences - need enough data for sequences
        seq_length = model.seq_length
        
        # For training: create sequences from training data
        X_seq, y_seq = create_sequences(X_train_scaled, y_train, seq_length)
        
        # For test: we need to include some training data as context
        # Combine last seq_length-1 training points with test data
        if X_test is not None and len(X_test_scaled) > 0:
            # Use last (seq_length-1) points from training as context
            context_len = min(seq_length - 1, len(X_train_scaled))
            context = X_train_scaled[-context_len:] if context_len > 0 else np.array([])
            X_test_with_context = np.vstack([context, X_test_scaled]) if context_len > 0 else X_test_scaled
            
            # Create corresponding y values with context (use last training y values)
            y_context = y_train[-context_len:] if context_len > 0 else np.array([])
            y_test_with_context = np.concatenate([y_context, y_test]) if context_len > 0 else y_test
            
            if len(X_test_with_context) >= seq_length:
                X_test_seq, y_test_seq = create_sequences(X_test_with_context, y_test_with_context, seq_length)
            else:
                X_test_seq, y_test_seq = None, None
        else:
            X_test_seq, y_test_seq = None, None
        
        # Rebuild model with correct input shape based on actual data
        n_features = X_seq.shape[2]
        model.n_features = n_features
        model._build_model()
        
        # Train
        history = model.train(X_seq, y_seq, X_test_seq, y_test_seq, epochs, batch_size)
        
    else:
        # For tree-based models (RandomForest, XGBoost)
        # Scale features
        if X_test is not None:
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
            history = model.train(X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            history = model.train(X_train_scaled, y_train)
        
        # Attach scaler to model for saving
        model.scaler = scaler
    
    return history


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }