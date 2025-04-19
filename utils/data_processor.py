import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class DataProcessor:
    """Data preprocessing class"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, List[int]]:
        """Detect outliers in the data
        
        Args:
            df: DataFrame
            columns: Columns to check
            method: Detection method ('iqr' or 'zscore')
            
        Returns:
            Dictionary of outlier indices for each column
        """
        outliers = {}
        
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = df[
                        (df[col] < (Q1 - 1.5 * IQR)) | 
                        (df[col] > (Q3 + 1.5 * IQR))
                    ].index.tolist()
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers[col] = df[z_scores > 3].index.tolist()
        
        return outliers
    
    def handle_missing_values(self, 
                            df: pd.DataFrame,
                            method: str = "drop",
                            fill_value: Optional[Union[str, float]] = None) -> pd.DataFrame:
        """Handle missing values in the data
        
        Args:
            df: DataFrame
            method: Handling method ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_constant')
            fill_value: Value to use when method is 'fill_constant'
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        if method == "drop":
            return df.dropna()
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if method == "fill_mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        elif method == "fill_median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        elif method == "fill_mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "fill_constant":
            df = df.fillna(fill_value)
            
        return df
    
    def normalize_data(self, 
                      df: pd.DataFrame,
                      method: str = "min_max",
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize data
        
        Args:
            df: DataFrame
            method: Normalization method ('min_max', 'standard', 'robust', 'log')
            columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if method == "min_max":
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "standard":
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "robust":
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "log":
            df[columns] = np.log1p(df[columns])
            
        return df
    
    def handle_outliers(self,
                       df: pd.DataFrame,
                       method: str = "iqr",
                       columns: Optional[List[str]] = None,
                       action: str = "remove") -> pd.DataFrame:
        """Handle outliers in the data
        
        Args:
            df: DataFrame
            method: Detection method ('iqr', 'zscore', 'isolation_forest', 'local_outlier_factor')
            columns: Columns to check
            action: Handling method ('remove', 'cap', 'mean', 'median')
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                if action == "cap":
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > 3
                
                if action == "cap":
                    df.loc[z_scores > 3, col] = df[col].mean() + 3 * df[col].std()
                    df.loc[z_scores < -3, col] = df[col].mean() - 3 * df[col].std()
                
            elif method == "isolation_forest":
                iso_forest = IsolationForest(random_state=42)
                col_outliers = iso_forest.fit_predict(df[[col]]) == -1
                
            elif method == "local_outlier_factor":
                lof = LocalOutlierFactor()
                col_outliers = lof.fit_predict(df[[col]]) == -1
            
            if action == "mean":
                df.loc[col_outliers, col] = df[col].mean()
            elif action == "median":
                df.loc[col_outliers, col] = df[col].median()
            
            outlier_mask |= col_outliers
        
        if action == "remove":
            df = df[~outlier_mask]
        
        return df
    
    def encode_categorical(self,
                         df: pd.DataFrame,
                         method: str = "label",
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables
        
        Args:
            df: DataFrame
            method: Encoding method ('label', 'one_hot', 'ordinal')
            columns: Columns to encode
            
        Returns:
            Encoded DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        if method == "label":
            for col in columns:
                df[col] = pd.Categorical(df[col]).codes
        elif method == "one_hot":
            df = pd.get_dummies(df, columns=columns)
        elif method == "ordinal":
            for col in columns:
                df[col] = pd.Categorical(df[col], ordered=True).codes
        
        return df
    
    def detect_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect data quality issues
        
        Returns:
            Data quality report
        """
        report = {}
        
        for col in df.columns:
            stats = {
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100,
            }
            
            if df[col].dtype in ['int64', 'float64']:
                stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                })
            
            report[col] = stats
        
        return report
    
    def suggest_transformations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Suggest data transformations
        
        Returns:
            Dictionary of suggested transformations for each column
        """
        suggestions = {}
        
        for col in df.columns:
            col_suggestions = []
            
            if df[col].dtype in ['int64', 'float64']:
                # Check skewness
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    if skewness > 0:
                        col_suggestions.append('Consider log transformation')
                    else:
                        col_suggestions.append('Consider exponential transformation')
                
                # Check outliers
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                if (z_scores > 3).any():
                    col_suggestions.append('Contains outliers, consider handling')
                
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:
                    col_suggestions.append('High cardinality, consider grouping')
                elif unique_ratio < 0.01:
                    col_suggestions.append('Low cardinality, consider merging categories')
            
            if df[col].isnull().sum() > 0:
                col_suggestions.append('Contains missing values, needs handling')
            
            suggestions[col] = col_suggestions
        
        return suggestions 