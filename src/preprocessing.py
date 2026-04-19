"""Data preprocessing utilities"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess stock data
    
    Args:
        df: Raw stock data DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Ensure we have required columns
    required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
    
    return df


def create_features(df: pd.DataFrame, target_col: str = 'Close', 
                   lookback: int = 60) -> tuple:
    """
    Create features for model training
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column to predict
        lookback: Number of time steps to look back
    
    Returns:
        Tuple of (X, y) arrays
    """
    # Technical indicators
    df = add_technical_indicators(df)
    
    # Features and target
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA_5', 'MA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
    
    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Handle missing values from indicators
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to DataFrame"""
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Price change
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
    """
    Split data into train and test sets
    
    Args:
        X: Features array
        y: Target array
        test_size: Proportion of data for testing
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray, 
                  scaler_type: str = 'minmax') -> tuple:
    """
    Scale features using specified scaler
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: 'minmax' or 'standard'
    
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Fit on training data only
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 60) -> tuple:
    """
    Create sequences for LSTM model
    
    Args:
        X: Features array
        y: Target array
        seq_length: Length of sequences
    
    Returns:
        Tuple of (X_seq, y_seq)
    """
    X_seq, y_seq = [], []
    
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)