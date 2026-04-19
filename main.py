"""Stock Prediction CLI - Main entry point"""
import argparse
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import download_stock_data, load_csv_data
from src.preprocessing import preprocess_data, create_features, split_data
from src.models.lstm_model import LSTMModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.train import train_model
from src.predict import make_prediction
from src.visualize import plot_predictions, plot_training_history


def download_command(args):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading {args.symbol} from {args.start} to {args.end}...")
    df = download_stock_data(args.symbol, args.start, args.end)
    
    output_path = Path(args.output) if args.output else Path("data") / f"{args.symbol}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df


def train_command(args):
    """Train a model on stock data"""
    print(f"Loading data from {args.data}...")
    df = load_csv_data(args.data)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Select model
    model_map = {
        "lstm": LSTMModel,
        "rf": RandomForestModel,
        "xgboost": XGBoostModel
    }
    
    if args.model not in model_map:
        print(f"Error: Unknown model '{args.model}'. Available: {list(model_map.keys())}")
        return
    
    model_class = model_map[args.model]
    n_features = X.shape[1]
    model = model_class() if args.model != "lstm" else model_class(seq_length=60, n_features=n_features)
    
    print(f"Training {args.model} model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    save_path = Path(args.output) if args.output else Path("models") / f"{args.model}_model"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")
    
    if history and args.model == "lstm":
        plot_training_history(history)


def predict_command(args):
    """Make predictions using a trained model"""
    print(f"Loading data from {args.data}...")
    df = load_csv_data(args.data)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    X, y = create_features(df)
    
    # Load model
    model_path = Path(args.model)
    model_type = model_path.stem.split("_")[0]
    
    model_map = {
        "lstm": LSTMModel,
        "rf": RandomForestModel,
        "xgboost": XGBoostModel
    }
    
    model_class = model_map.get(model_type, LSTMModel)
    model = model_class()
    model.load(str(model_path))
    
    print("Making predictions...")
    
    # Handle LSTM differently - needs sequences
    if model_type == "lstm":
        from sklearn.preprocessing import MinMaxScaler
        from src.preprocessing import create_sequences
        
        # Scale all data together
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences using the model's seq_length
        seq_length = model.seq_length
        if len(X_scaled) > seq_length:
            X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
            predictions = model.predict(X_seq)
            # Align predictions with original y (skip first seq_length points)
            y_aligned = y[seq_length:]
        else:
            print("Not enough data for sequences")
            predictions = np.array([])
            y_aligned = y
    else:
        # For tree-based models
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use model's scaler if available, otherwise use new scaler
        if hasattr(model, 'scaler') and model.scaler is not None:
            X_scaled = model.scaler.transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled)
        y_aligned = y
    
    # Save predictions
    output_path = Path(args.output) if args.output else Path("predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    pred_df = pd.DataFrame({"actual": y_aligned, "predicted": predictions})
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Visualize
    plot_predictions(y_aligned, predictions)


def visualize_command(args):
    """Visualize predictions vs actual"""
    import pandas as pd
    
    if args.predictions:
        df = pd.read_csv(args.predictions)
        plot_predictions(df["actual"].values, df["predicted"].values)
    elif args.data:
        df = load_csv_data(args.data)
        df = preprocess_data(df)
        X, y = create_features(df)
        # Show last portion of actual data
        plot_predictions(y[-100:], y[-100:], title="Actual Stock Prices")


def main():
    parser = argparse.ArgumentParser(description="Stock Prediction System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download stock data")
    download_parser.add_argument("--symbol", required=True, help="Stock symbol (e.g., AAPL)")
    download_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    download_parser.add_argument("--output", help="Output file path")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to CSV data file")
    train_parser.add_argument("--model", required=True, choices=["lstm", "rf", "xgboost"], 
                             help="Model type")
    train_parser.add_argument("--output", help="Model output path")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--data", required=True, help="Path to CSV data file")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--output", help="Predictions output path")
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize results")
    vis_parser.add_argument("--data", help="Path to data file")
    vis_parser.add_argument("--predictions", help="Path to predictions CSV")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()