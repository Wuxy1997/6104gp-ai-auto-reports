from typing import Dict, List, Optional
import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import json
import numpy as np

class InsightsGenerator:
    """AI insights generation class"""
    
    def __init__(self, host: str = None, model: str = "mistral"):
        """Initialize insights generator
        
        Args:
            host: Ollama server address
            model: Model name to use
        """
        if host is None:
            host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.llm = Ollama(base_url=host, model=model)
        
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
            
    def generate_insights(self, df: pd.DataFrame, query: str = None, quality_report: dict = None) -> str:
        """Generate insights from data
        
        Args:
            df: DataFrame to analyze
            query: Optional query to focus analysis
            quality_report: Optional data quality report
            
        Returns:
            Generated insights text
        """
        try:
            # Prepare data description
            data_description = self._get_data_description(df)
            
            # Convert quality report numpy types to native Python types
            if quality_report:
                quality_report = self._convert_numpy_types(quality_report)
            
            # Build prompt
            prompt = f"""As a data analyst, please analyze the following dataset and provide insights in English.

Data Description:
{data_description}

Data Quality Report:
{json.dumps(quality_report, indent=2) if quality_report else 'Not provided'}

User Query:
{query if query else 'Please provide a comprehensive analysis of this dataset'}

Please provide a detailed analysis including:
1. Key patterns and trends
2. Notable relationships between variables
3. Important insights about the data
4. Potential business implications
5. Any anomalies or points of interest

Format your response in clear sections with bullet points where appropriate.
Keep the analysis professional and data-driven.
IMPORTANT: Provide the analysis in English only.
"""
            
            # Generate insights
            insights = self.llm(prompt)
            return insights
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")  # For debugging
            return f"Error generating insights: {str(e)}"
            
    def _get_data_description(self, df: pd.DataFrame) -> str:
        """Generate data description for LLM prompt"""
        description = []
        
        # Basic information
        description.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Column information
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(numeric_cols) > 0:
            description.append(f"\nNumeric columns: {', '.join(numeric_cols)}")
            # Add basic statistics for numeric columns
            stats = df[numeric_cols].describe()
            description.append("\nNumeric column statistics:")
            for col in numeric_cols:
                description.append(f"{col}:")
                description.append(f"- Range: {stats.loc['min', col]:.2f} to {stats.loc['max', col]:.2f}")
                description.append(f"- Mean: {stats.loc['mean', col]:.2f}")
                description.append(f"- Std: {stats.loc['std', col]:.2f}")
        
        if len(categorical_cols) > 0:
            description.append(f"\nCategorical columns: {', '.join(categorical_cols)}")
            # Add value counts for categorical columns
            description.append("\nCategory distributions:")
            for col in categorical_cols:
                n_unique = df[col].nunique()
                description.append(f"{col}:")
                description.append(f"- Unique values: {n_unique}")
                if n_unique <= 5:  # Only show distribution for columns with few categories
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        description.append(f"- {val}: {count} ({count/len(df)*100:.1f}%)")
        
        if len(datetime_cols) > 0:
            description.append(f"\nDatetime columns: {', '.join(datetime_cols)}")
            # Add date range information
            for col in datetime_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                description.append(f"{col} range: {min_date} to {max_date}")
        
        # Add correlation information for numeric columns
        if len(numeric_cols) >= 2:
            description.append("\nStrong correlations (|r| > 0.7):")
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        description.append(f"- {numeric_cols[i]} vs {numeric_cols[j]}: {corr:.2f}")
        
        return "\n".join(description)
        
    def generate_visualization_analysis(self, df: pd.DataFrame, viz_type: str, columns: List[str]) -> str:
        """Generate analysis for a specific visualization
        
        Args:
            df: DataFrame
            viz_type: Type of visualization
            columns: Columns used in visualization
            
        Returns:
            Analysis text
        """
        try:
            # Prepare visualization-specific description
            viz_description = self._get_visualization_description(df, viz_type, columns)
            
            # Build prompt
            prompt = f"""As a data visualization expert, please analyze the following visualization in English:

Visualization Type: {viz_type}
Columns Used: {', '.join(columns)}

Data Description:
{viz_description}

Please provide a clear and concise analysis including:
1. Key patterns or trends visible in the visualization
2. Notable relationships or comparisons
3. Any outliers or unusual patterns
4. Potential insights or implications

Keep the analysis professional and focused on the data shown in the visualization.
IMPORTANT: Provide the analysis in English only.
"""
            
            # Generate analysis
            analysis = self.llm(prompt)
            return analysis
            
        except Exception as e:
            return f"Error generating visualization analysis: {str(e)}"
            
    def _get_visualization_description(self, df: pd.DataFrame, viz_type: str, columns: List[str]) -> str:
        """Generate visualization-specific data description"""
        description = []
        
        if viz_type == 'line' or viz_type == 'scatter':
            # For time series or relationship plots
            x, y = columns[0], columns[1]
            description.extend([
                f"X-axis: {x}",
                f"Y-axis: {y}",
                f"Number of points: {len(df)}",
                f"Correlation: {df[x].corr(df[y]):.2f}" if df[x].dtype.kind in 'biufc' and df[y].dtype.kind in 'biufc' else ""
            ])
            
        elif viz_type == 'bar':
            # For categorical comparisons
            x, y = columns[0], columns[1]
            agg_data = df.groupby(x)[y].agg(['mean', 'count'])
            description.extend([
                f"Categories: {df[x].nunique()}",
                f"Highest value: {agg_data['mean'].max():.2f}",
                f"Lowest value: {agg_data['mean'].min():.2f}"
            ])
            
        elif viz_type == 'histogram':
            # For distribution analysis
            col = columns[0]
            description.extend([
                f"Column: {col}",
                f"Mean: {df[col].mean():.2f}",
                f"Median: {df[col].median():.2f}",
                f"Std: {df[col].std():.2f}"
            ])
            
        elif viz_type == 'box':
            # For distribution comparison
            x, y = columns[0], columns[1]
            description.extend([
                f"Groups: {df[x].nunique()}",
                f"Overall median: {df[y].median():.2f}",
                f"Overall IQR: {df[y].quantile(0.75) - df[y].quantile(0.25):.2f}"
            ])
            
        return "\n".join(description) 