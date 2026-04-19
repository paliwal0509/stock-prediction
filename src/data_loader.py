"""Data loading utilities"""
import yfinance as yf
import pandas as pd
from pathlib import Path


def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading {symbol} data from {start} to {end}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    print(f"Downloaded {len(df)} records")
    return df


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load stock data from CSV file
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with stock data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(path, parse_dates=True)
    
    # Standardize column names
    column_map = {
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume',
        'Adj Close': 'Adj_Close'
    }
    
    # Try to find and rename columns
    for old_col, new_col in column_map.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    return df


def save_data(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")