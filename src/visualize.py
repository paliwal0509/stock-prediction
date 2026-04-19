"""Visualization module"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                     title: str = "Stock Price Prediction", save_path: str = None):
    """
    Plot actual vs predicted stock prices
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Limit to last 200 points for readability
    n = min(200, len(actual))
    idx = range(len(actual) - n, len(actual))
    
    plt.plot(idx, actual[-n:], label='Actual', color='blue', linewidth=2)
    plt.plot(idx, predicted[-n:], label='Predicted', color='red', 
             linestyle='--', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}'
    plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path: str = None):
    """
    Plot training history for LSTM model
    
    Args:
        history: Training history object
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss', 
                     linewidth=2, linestyle='--')
    axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    if 'val_mae' in history.history:
        axes[1].plot(history.history['val_mae'], label='Validation MAE', 
                     linewidth=2, linestyle='--')
    axes[1].set_title('Model MAE', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: list, importance: np.ndarray,
                            title: str = "Feature Importance", save_path: str = None):
    """
    Plot feature importance for tree-based models
    
    Args:
        feature_names: List of feature names
        importance: Feature importance values
        title: Plot title
        save_path: Optional path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_stock_data(df, column: str = 'Close', title: str = "Stock Prices",
                    save_path: str = None):
    """
    Plot stock price data
    
    Args:
        df: DataFrame with stock data
        column: Column to plot
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(df.index, df[column], linewidth=1.5, color='blue')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_error_distribution(actual: np.ndarray, predicted: np.ndarray,
                            save_path: str = None):
    """
    Plot prediction error distribution
    
    Args:
        actual: Actual values
        predicted: Predicted values
        save_path: Optional path to save figure
    """
    errors = actual - predicted
    
    plt.figure(figsize=(10, 6))
    
    sns.histplot(errors, kde=True, bins=30, color='steelblue')
    
    plt.title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    stats_text = f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}'
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 verticalalignment='top', horizontalalignment='right',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")