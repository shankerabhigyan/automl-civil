import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Union, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

class AutoPreprocessor:
    def __init__(self, 
                 target_column: str,
                 mode: str = 'auto',
                 max_features: int = 15,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.target_column = target_column
        self.mode = mode
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.summary_stats = None
        self.feature_importances = None
        self.selected_features = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _detect_data_type(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect the type of each column"""
        data_types = {}
        for column in df.columns:
            if column == self.target_column:
                continue
                
            unique_ratio = df[column].nunique() / len(df)
            if unique_ratio < 0.05 or df[column].dtype == 'object':
                data_types[column] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                data_types[column] = 'datetime'
            else:
                data_types[column] = 'numerical'
                
        return data_types

    def _handle_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Handle categorical variables more carefully"""
        df_processed = df.copy()
        
        for col in categorical_cols:
            # For columns with few unique values, use one-hot encoding
            if df[col].nunique() < 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(columns=[col], inplace=True)
            else:
                # For columns with many unique values, use label encoding
                df_processed[col] = pd.Categorical(df[col]).codes
                
        return df_processed

    def _detect_outliers(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Detect and handle outliers"""
        if len(numerical_cols) < 2:
            return df
            
        df_numerical = df[numerical_cols]
        iso_forest = IsolationForest(random_state=self.random_state)
        outlier_labels = iso_forest.fit_predict(df_numerical)
        
        # Replace outliers with median values
        outlier_indices = np.where(outlier_labels == -1)[0]
        for col in numerical_cols:
            median_val = df[col].median()
            df.loc[outlier_indices, col] = median_val
            
        return df

    def _auto_feature_engineering(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Create new features automatically"""
        df_new = df.copy()
        
        if len(numerical_cols) > 1:
            # Add interaction terms for highly correlated features
            correlations = df[numerical_cols].corr().abs()
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    col1, col2 = numerical_cols[i], numerical_cols[j]
                    if correlations.loc[col1, col2] > 0.5:
                        df_new[f"{col1}_{col2}_mult"] = df[col1] * df[col2]
                        if (df[col2] != 0).all():
                            df_new[f"{col1}_{col2}_ratio"] = df[col1] / df[col2]
            
            # Add aggregate features
            df_new['mean_numerical'] = df[numerical_cols].mean(axis=1)
            df_new['std_numerical'] = df[numerical_cols].std(axis=1)
            
        return df_new

    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """Select the most important features"""
        X = df.drop(columns=[self.target_column])
        
        # Ensure all features are numeric
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X = X[numeric_cols]
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=min(self.max_features, len(X.columns)))
        selector.fit(X, target)
        
        # Get feature scores and names
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        })
        self.feature_importances = feature_scores.sort_values('score', ascending=False)
        
        # Select top features
        selected_features = list(self.feature_importances.nlargest(self.max_features, 'score')['feature'])
        
        # Always include target column
        if self.target_column not in selected_features:
            selected_features.append(self.target_column)
            
        return selected_features

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline...")
        
        # Initial cleaning
        df = df.copy()
        df = df.dropna()
        self.summary_stats = df.describe()
        
        # Detect data types
        data_types = self._detect_data_type(df)
        numerical_cols = [col for col, dtype in data_types.items() if dtype == 'numerical']
        categorical_cols = [col for col, dtype in data_types.items() if dtype == 'categorical']
        
        # Handle outliers in numerical columns
        df = self._detect_outliers(df, numerical_cols)
        
        # Handle categorical variables
        df = self._handle_categorical(df, categorical_cols)
        
        if self.mode == 'auto':
            # Apply automatic feature engineering
            df = self._auto_feature_engineering(df, numerical_cols)
        
        # Select features
        selected_features = self._select_features(df, df[self.target_column])
        self.selected_features = selected_features
        
        # Ensure all selected features exist in the dataframe
        existing_features = [f for f in selected_features if f in df.columns]
        df = df[existing_features]
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        self.logger.info("Preprocessing complete!")
        return train_df, test_df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.feature_importances is None:
            raise ValueError("Feature importance scores not available. Run fit_transform first.")
        return self.feature_importances