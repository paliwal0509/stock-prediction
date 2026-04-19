# Stock Prediction System

AI-powered stock price prediction using machine learning models (LSTM, Random Forest).

## Features

- Data loading from CSV files or Yahoo Finance
- Data preprocessing and feature engineering
- Multiple ML models: LSTM, Random Forest, XGBoost
- Visualization of predictions vs actual prices
- CLI interface for easy usage
- Model evaluation metrics (MAE, RMSE, R²)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Download Stock Data

```bash
python main.py download --symbol AAPL --start 2020-01-01 --end 2024-01-01
```

### Train Model

```bash
python main.py train --data data/AAPL.csv --model lstm
```

### Make Predictions

```bash
python main.py predict --data data/AAPL.csv --model models/lstm_model.h5
```

### Visualize Results

```bash
python main.py visualize --data data/AAPL.csv --predictions predictions.csv
```

## Project Structure

```
stock_prediction/
├── data/                  # Data storage
├── models/                # Trained models
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py   # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   ├── train.py           # Training logic
│   ├── predict.py         # Prediction logic
│   └── visualize.py       # Visualization
├── main.py                # CLI entry point
├── requirements.txt
└── README.md
```

## Models

- **LSTM**: Long Short-Term Memory neural network for time series
- **Random Forest**: Ensemble learning method
- **XGBoost**: Gradient boosting algorithm

## License

MIT