# src/preprocessing/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Optional
import re


class CompositeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for composite materials"""
    
    def __init__(
        self,
        composition_features: List[str],
        process_features: List[str],
        add_interactions: bool = True,
        add_polynomials: bool = True,
        add_ratios: bool = True,
    ):
        self.composition_features = composition_features
        self.process_features = process_features
        self.add_interactions = add_interactions
        self.add_polynomials = add_polynomials
        self.add_ratios = add_ratios
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineer"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding engineered features"""
        X_transformed = X.copy()
        
        # Composition-based features
        if self.add_ratios:
            for i, feat1 in enumerate(self.composition_features):
                for feat2 in self.composition_features[i+1:]:
                    col_name = f"{feat1}_{feat2}_ratio"
                    X_transformed[col_name] = (
                        X_transformed[feat1] / (X_transformed[feat2] + 1e-8)
                    )
        
        if self.add_interactions:
            for i, feat1 in enumerate(self.composition_features):
                for feat2 in self.process_features:
                    col_name = f"{feat1}_{feat2}_interaction"
                    X_transformed[col_name] = (
                        X_transformed[feat1] * X_transformed[feat2]
                    )
        
        # Process-based features
        for feat in self.process_features:
            if self.add_polynomials:
                X_transformed[f"{feat}_squared"] = X_transformed[feat] ** 2
                X_transformed[f"{feat}_sqrt"] = np.sqrt(np.abs(X_transformed[feat]))
            
            # Log transform for positive features
            if (X_transformed[feat] > 0).all():
                X_transformed[f"{feat}_log"] = np.log(X_transformed[feat] + 1e-8)
        
        self.feature_names = X_transformed.columns.tolist()
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all features after transformation"""
        return self.feature_names


class MaterialsDataCleaner:
    """Clean and validate composite material data"""
    
    @staticmethod
    def validate_composition(df: pd.DataFrame, comp_cols: List[str]) -> pd.DataFrame:
        """Validate that composition features sum to 1"""
        comp_sum = df[comp_cols].sum(axis=1)
        if not np.allclose(comp_sum, 1.0, rtol=1e-3):
            # Normalize if not summing to 1
            df[comp_cols] = df[comp_cols].div(comp_sum, axis=0)
        return df
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame, 
        columns: List[str], 
        method: str = "iqr", 
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        df_clean = df.copy()
        
        if method == "iqr":
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        elif method == "zscore":
            for col in columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / 
                                 df_clean[col].std())
                df_clean = df_clean[z_scores <= threshold]
                
        return df_clean
    
    @staticmethod
    def impute_missing_values(
        df: pd.DataFrame,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Impute missing values"""
        df_imputed = df.copy()
        
        if columns is None:
            columns = df.columns
            
        for col in columns:
            if strategy == "mean":
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            elif strategy == "median":
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            elif strategy == "mode":
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
            elif strategy == "ffill":
                df_imputed[col] = df_imputed[col].fillna(method="ffill")
                
        return df_imputed
