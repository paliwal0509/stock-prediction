# Stock Prediction System - Project Instructions

## Project Overview
AI-powered stock price prediction using machine learning models (LSTM, Random Forest, XGBoost).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Stock Data
```bash
python main.py download --symbol AAPL --start 2020-01-01 --end 2024-01-01
```

### 3. Train a Model
```bash
python main.py train --data data/AAPL.csv --model lstm
```

### 4. Make Predictions
```bash
python main.py predict --data data/AAPL.csv --model models/lstm_model.h5
```

## Project Structure
- `main.py` - CLI entry point
- `src/data_loader.py` - Data loading utilities
- `src/preprocessing.py` - Data preprocessing
- `src/models/` - ML models (LSTM, Random Forest, XGBoost)
- `src/train.py` - Training logic
- `src/predict.py` - Prediction logic
- `src/visualize.py` - Visualization

## Available Models
- `lstm` - LSTM neural network
- `rf` - Random Forest
- `xgboost` - XGBoost

## Notes
- Python 3.8+ required
- TensorFlow required for LSTM model
- Data saved to `data/` folder
- Models saved to `models/` folder