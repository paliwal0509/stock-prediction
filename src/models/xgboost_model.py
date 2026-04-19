"""XGBoost Model for stock prediction"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


class XGBoostModel:
    """XGBoost regressor for stock price prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        self.scaler = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: int = 1):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Verbosity level
        
        Returns:
            Training metrics
        """
        if verbose:
            print(f"Training XGBoost with {self.n_estimators} rounds...")
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        if verbose:
            print(f"Training MAE: {train_mae:.4f}")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
        
        # Evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            if verbose:
                print(f"Validation MAE: {val_mae:.4f}")
                print(f"Validation RMSE: {val_rmse:.4f}")
                print(f"Validation R²: {val_r2:.4f}")
            
            return {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            }
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.feature_importances_
    
    def save(self, path: str):
        """Save model to disk"""
        self.model.save_model(f"{path}_model.json")
        
        if self.scaler is not None:
            with open(f"{path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        
        print(f"Model saved to {path}_model.json")
    
    def load(self, path: str):
        """Load model from disk"""
        # Handle path - remove _model.json suffix if already present to avoid double extension
        if path.endswith('_model.json'):
            base_path = path[:-11]  # Remove '_model.json'
        else:
            base_path = path
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{base_path}_model.json")
        
        scaler_path = f"{base_path}_scaler.pkl"
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            pass
        
        print(f"Model loaded from {base_path}_model.json")
    
    def summary(self):
        """Print model info"""
        print(f"XGBoost Regressor")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  learning_rate: {self.learning_rate}")
        print(f"  subsample: {self.subsample}")
        print(f"  colsample_bytree: {self.colsample_bytree}")


def create_xgboost(n_estimators: int = 100, max_depth: int = 6,
                   learning_rate: float = 0.1) -> xgb.XGBRegressor:
    """
    Factory function to create XGBoost model
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
    
    Returns:
        XGBRegressor instance
    """
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )