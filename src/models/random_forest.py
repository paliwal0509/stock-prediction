"""Random Forest Model for stock prediction"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle


class RandomForestModel:
    """Random Forest regressor for stock price prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, random_state: int = 42):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = None
        self.feature_names = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: int = 1):
        """
        Train the Random Forest model
        
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
            print(f"Training Random Forest with {self.n_estimators} trees...")
        
        self.model.fit(X_train, y_train)
        
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
        # Save model
        with open(f"{path}_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler if exists
        if self.scaler is not None:
            with open(f"{path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        
        print(f"Model saved to {path}_model.pkl")
    
    def load(self, path: str):
        """Load model from disk"""
        import os
        
        # If the exact file exists, use it directly
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to load scaler with same base name
            base_path = path[:-4] if path.endswith('.pkl') else path
            scaler_path = f"{base_path}_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                pass
            
            print(f"Model loaded from {path}")
        else:
            # Try adding _model.pkl suffix
            model_file = f"{path}_model.pkl"
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            scaler_path = f"{path}_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                pass
            
            print(f"Model loaded from {model_file}")
    
    def summary(self):
        """Print model info"""
        print(f"Random Forest Regressor")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  min_samples_split: {self.min_samples_split}")
        print(f"  n_features: {self.model.n_features_in_}")


def create_random_forest(n_estimators: int = 100, max_depth: int = 10) -> RandomForestRegressor:
    """
    Factory function to create Random Forest model
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth
    
    Returns:
        RandomForestRegressor instance
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )