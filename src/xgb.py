import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from typing import Dict, List, Optional, Union, Tuple
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomXGBoost:
    """
    A customizable XGBoost implementation for regression tasks with various features
    including hyperparameter optimization, cross-validation, and visualization tools.
    """
    
    def __init__(
        self,
        target_column: str,
        scale_features: bool = True,
        scale_target: bool = True,
        xgb_params: Optional[Dict] = None,
        validation_size: float = 0.2,
        random_state: int = 42,
        early_stopping_rounds: int = 50
    ):
        """
        Initialize the CustomXGBoost model.
        
        Args:
            target_column: Name of the target variable
            scale_features: Whether to scale features using MinMaxScaler
            scale_target: Whether to scale target variable
            xgb_params: Custom XGBoost parameters
            validation_size: Size of validation set
            random_state: Random seed for reproducibility
            early_stopping_rounds: Number of rounds for early stopping
        """
        self.target_column = target_column
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.validation_size = validation_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler() if scale_features else None
        self.target_scaler = MinMaxScaler() if scale_target else None
        
        # Set default XGBoost parameters if none provided
        self.xgb_params = xgb_params if xgb_params is not None else {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        
        self.model = None
        self.feature_names = None
        self.metrics_history = None
        
    def preprocess_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data by scaling features and target.
        
        Args:
            data: Input DataFrame containing features and target
            
        Returns:
            Tuple of processed features and target arrays
        """
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column].values.reshape(-1, 1)
        
        self.feature_names = X.columns.tolist()
        
        if self.scale_features:
            X = self.feature_scaler.fit_transform(X)
        else:
            X = X.values
            
        if self.scale_target:
            y = self.target_scaler.fit_transform(y)
        
        return X, y.ravel()
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary of optimized parameters
        """
        def objective(trial):
            param = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
            }
            
            model = xgb.XGBRegressor(**param, random_state=self.random_state)
            score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
            return -score
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        self.xgb_params.update(study.best_params)
        return study.best_params
    
    def fit(
        self,
        data: pd.DataFrame,
        optimize: bool = False,
        n_trials: int = 100,
        plot_training: bool = True
    ) -> None:
        """
        Fit the XGBoost model to the data.
        
        Args:
            data: Input DataFrame
            optimize: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials
            plot_training: Whether to plot training progress
        """
        X, y = self.preprocess_data(data)
        
        if optimize:
            self.optimize_hyperparameters(X, y, n_trials)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, random_state=self.random_state
        )
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model = xgb.XGBRegressor(
            **self.xgb_params,
            random_state=self.random_state
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True,
        )
        
        self.metrics_history = {
            'train': self.model.evals_result()['validation_0']['rmse'],
            'val': self.model.evals_result()['validation_1']['rmse']
        }
        
        if plot_training:
            self.plot_training_progress()
    
    def predict(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of predictions
        """
        X = data.drop(columns=[self.target_column])
        
        if self.scale_features:
            X = self.feature_scaler.transform(X)
        
        predictions = self.model.predict(X)
        
        if self.scale_target:
            predictions = predictions.reshape(-1, 1)
            predictions = self.target_scaler.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            data: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        X, y_true = self.preprocess_data(data)
        y_pred = self.model.predict(X)
        
        if self.scale_target:
            y_true = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def plot_training_progress(self) -> None:
        """Plot training and validation RMSE over training iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history['train'], label='Training RMSE')
        plt.plot(self.metrics_history['val'], label='Validation RMSE')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_feature_importance(
        self,
        importance_type: str = 'weight',
        max_features: int = 20
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_type: Type of feature importance ('weight', 'gain', or 'cover')
            max_features: Maximum number of features to plot
        """
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        importance = pd.DataFrame(
            {'feature': importance.keys(), 'importance': importance.values()}
        ).sort_values('importance', ascending=False).head(max_features)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title(f'Feature Importance ({importance_type})')
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the model and scalers to disk."""
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'params': self.xgb_params
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load the model and scalers from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_names = model_data['feature_names']
        self.xgb_params = model_data['params']

    def get_fitness_score(self, data):
        metrics = self.model.evals_result()
        print(1/metrics['validation_1']['rmse'][-1])
        return 1/metrics['validation_1']['rmse'][-1]

# Example usage:
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("../data/concrete/Concrete_Data_Yeh.csv")
    target = 'csMPa'
    
    # Initialize model with custom parameters
    xgb_params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    model = CustomXGBoost(
        target_column=target,
        scale_features=True,
        scale_target=True,
        xgb_params=xgb_params,
        validation_size=0.2,
        early_stopping_rounds=30
    )
    
    # Train model with hyperparameter optimization
    model.fit(data, optimize=True, n_trials=50, plot_training=True)
    
    # Evaluate model
    metrics = model.evaluate(data)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Plot feature importance
    model.plot_feature_importance(importance_type='gain')
    
    # Save model
    model.save_model('xgboost_model.joblib')