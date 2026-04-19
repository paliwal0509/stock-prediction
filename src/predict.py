"""Prediction module"""
import numpy as np
import pandas as pd
from pathlib import Path


def make_prediction(model, data, scaler=None):
    """
    Make predictions on stock data
    
    Args:
        model: Trained model
        data: Input data (DataFrame or array)
        scaler: Optional scaler for preprocessing
    
    Returns:
        Predictions array
    """
    # Determine model type
    model_type = model.__class__.__name__
    
    if isinstance(data, pd.DataFrame):
        # Extract features from DataFrame
        X = data.values
    else:
        X = data
    
    # Scale if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions


def predict_future(model, last_sequence: np.ndarray, n_steps: int = 1,
                   scaler=None) -> np.ndarray:
    """
    Predict future stock prices
    
    Args:
        model: Trained model
        last_sequence: Last known sequence
        n_steps: Number of future steps to predict
        scaler: Optional scaler
    
    Returns:
        Array of future predictions
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_steps):
        # Reshape for prediction
        if len(current_seq.shape) == 2:
            current_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        
        # Predict next value
        pred = model.predict(current_seq)[0, 0]
        predictions.append(pred)
        
        # Update sequence for next prediction
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred
    
    return np.array(predictions)


def save_predictions(predictions: np.ndarray, actual: np.ndarray = None,
                     output_path: str = "predictions.csv"):
    """
    Save predictions to CSV
    
    Args:
        predictions: Predicted values
        actual: Actual values (optional)
        output_path: Output file path
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {'predicted': predictions}
    if actual is not None:
        data['actual'] = actual
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    
    print(f"Predictions saved to {path}")


def load_predictions(file_path: str) -> pd.DataFrame:
    """
    Load predictions from CSV
    
    Args:
        file_path: Path to predictions file
    
    Returns:
        DataFrame with predictions
    """
    return pd.read_csv(file_path)