"""LSTM Model for stock prediction"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib


class LSTMModel:
    """LSTM neural network for time series prediction"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 11):
        """
        Initialize LSTM model
        
        Args:
            seq_length: Number of time steps in each sequence
            n_features: Number of input features
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler = None
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture"""
        self.model = Sequential([
            Input(shape=(self.seq_length, self.n_features)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, verbose: int = 1):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str):
        """Save model to disk"""
        self.model.save(f"{path}.h5")
        print(f"Model saved to {path}.h5")
    
    def load(self, path: str):
        """Load model from disk"""
        # Handle path - remove .h5 suffix if already present to avoid double extension
        if path.endswith('.h5'):
            base_path = path[:-3]  # Remove '.h5'
        else:
            base_path = path
        
        # Load model with compile=False to avoid deserialization issues
        self.model = tf.keras.models.load_model(f"{base_path}.h5", compile=False)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        print(f"Model loaded from {base_path}.h5")
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


def create_lstm_model(seq_length: int = 60, n_features: int = 11) -> Sequential:
    """
    Factory function to create LSTM model
    
    Args:
        seq_length: Sequence length
        n_features: Number of features
    
    Returns:
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model