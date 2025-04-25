import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple, Dict
import numpy as np
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import json
from scipy import stats
import warnings
from datetime import datetime
from io import BytesIO
import base64

class DataVisualizer:
    """Enhanced data visualization class with comprehensive visualization capabilities"""
    
    def __init__(self, host: str = None, model: str = "mistral"):
        """Initialize visualizer with enhanced settings
        
        Args:
            host: Ollama server address
            model: Model name to use
        """
        if host is None:
            host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.llm = Ollama(base_url=host, model=model)
        
        # Set Chinese font and style
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8')
        
        # Set default color theme
        self.color_palette = sns.color_palette("husl", 8)
        sns.set_palette(self.color_palette)
        
        # Supported chart types
        self.supported_charts = {
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'scatter': self._create_scatter_plot,
            'heatmap': self._create_heatmap,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'pie': self._create_pie_chart,
            'histogram': self._create_histogram,
            'area': self._create_area_plot,
            'bubble': self._create_bubble_plot
        }
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data, including type conversion and cleaning"""
        df_processed = df.copy()
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',           # 2023-01-01
            '%Y/%m/%d',           # 2023/01/01
            '%d-%m-%Y',           # 01-01-2023
            '%d/%m/%Y',           # 01/01/2023
            '%Y-%m-%d %H:%M:%S',  # 2023-01-01 12:00:00
            '%Y/%m/%d %H:%M:%S',  # 2023/01/01 12:00:00
            '%Y年%m月%d日',        # January 1, 2023
            '%Y-%m-%d %H:%M',     # 2023-01-01 12:00
        ]
        
        # Process numeric and date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    # Remove currency symbols and thousand separators
                    cleaned = df[col].str.replace(r'[¥$,]', '', regex=True)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    if not numeric.isna().all():
                        df_processed[col] = numeric
                        continue  # Skip date conversion if successfully converted to numeric
                except:
                    pass
                
                # Try using predefined date formats for conversion
                date_converted = False
                for date_format in date_formats:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            date_series = pd.to_datetime(df[col], format=date_format, errors='coerce')
                            if not date_series.isna().all():
                                df_processed[col] = date_series
                                date_converted = True
                                break
                    except:
                        continue
                
                # If all predefined formats fail, try auto-inference (but suppress warnings)
                if not date_converted:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            date_series = pd.to_datetime(df[col], errors='coerce')
                            if not date_series.isna().all():
                                df_processed[col] = date_series
                    except:
                        pass
        
        return df_processed
    
    def _create_line_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create line plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure data is sorted
        df_plot = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_plot[x]):
            df_plot = df_plot.sort_values(x)
        
        # Plot main line
        ax.plot(df_plot[x], df_plot[y], marker='o', linestyle='-', linewidth=2, markersize=6)
        
        # Add trend line
        z = np.polyfit(range(len(df_plot[x])), df_plot[y], 1)
        p = np.poly1d(z)
        ax.plot(df_plot[x], p(range(len(df_plot[x]))), "r--", alpha=0.8, label='Trend')
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} vs {x} Trend')
        
        # Set grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _create_bar_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # If too many data points, aggregate
        if len(df[x].unique()) > 15:
            # Group by x and calculate y mean, take top 15
            df_agg = df.groupby(x)[y].mean().nlargest(15).reset_index()
        else:
            df_agg = df
        
        # Create bar chart
        bars = ax.bar(df_agg[x], df_agg[y])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom')
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} by {x}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Process date column
        df_plot = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_plot[x]):
            df_plot[x] = pd.to_numeric(df_plot[x].astype(np.int64))
        
        # Create scatter plot
        scatter = ax.scatter(df_plot[x], df_plot[y], alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df_plot[x], df_plot[y], 1)
        p = np.poly1d(z)
        ax.plot(df_plot[x], p(df_plot[x]), "r--", alpha=0.8, label=f'Trend (R² = {self._calculate_r2(df_plot[x], df_plot[y]):.2f})')
        
        # Calculate and display correlation coefficient
        corr = df_plot[x].corr(df_plot[y])
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} vs {x} Relationship')
        
        # If x is date, restore date ticks
        if pd.api.types.is_datetime64_any_dtype(df[x]):
            from matplotlib.dates import DateFormatter
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, columns: List[str], title: str = None, **kwargs) -> plt.Figure:
        """Create heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True,
                   ax=ax)
        
        # Set title
        ax.set_title(title or 'Correlation Heatmap')
        
        plt.tight_layout()
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create box plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        
        # Add data points
        sns.stripplot(data=df, x=x, y=y, color='red', alpha=0.3, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} Distribution by {x}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _create_violin_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create violin plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        sns.violinplot(data=df, x=x, y=y, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} Distribution by {x}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str, title: str = None, **kwargs) -> plt.Figure:
        """Create pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate values and labels
        value_counts = df[column].value_counts()
        labels = value_counts.index
        sizes = value_counts.values
        
        # If too many categories, show only top 8, others as "Others"
        if len(labels) > 8:
            sizes_top = value_counts.nlargest(7)
            sizes_others = pd.Series({'Others': value_counts[7:].sum()})
            sizes = pd.concat([sizes_top, sizes_others])
            labels = sizes.index
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        textprops={'fontsize': 8})
        
        # Set title
        ax.set_title(title or f'{column} Distribution')
        
        # Add legend
        ax.legend(wedges, labels,
                 title="Categories",
                 loc="center left",
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, column: str, title: str = None, **kwargs) -> plt.Figure:
        """Create histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        
        # Add statistics
        mean = df[column].mean()
        median = df[column].median()
        std = df[column].std()
        
        stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(title or f'{column} Distribution')
        
        plt.tight_layout()
        return fig
    
    def _create_area_plot(self, df: pd.DataFrame, x: str, y: str, title: str = None, **kwargs) -> plt.Figure:
        """Create area plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure x-axis data is sorted
        if pd.api.types.is_datetime64_any_dtype(df[x]):
            df = df.sort_values(x)
        
        # Create area plot
        ax.fill_between(df[x], df[y], alpha=0.5)
        ax.plot(df[x], df[y], linewidth=2)
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} Trend Over {x}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def _create_bubble_plot(self, df: pd.DataFrame, x: str, y: str, size: str, title: str = None, **kwargs) -> plt.Figure:
        """Create bubble plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bubble plot
        scatter = ax.scatter(df[x], df[y], s=df[size]*100, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df[x], df[y], 1)
        p = np.poly1d(z)
        ax.plot(df[x], p(df[x]), "r--", alpha=0.8, label='Trend')
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} vs {x} (Size: {size})')
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(prop="sizes", alpha=0.6, num=4),
                          loc="upper right", title=size)
        ax.add_artist(legend1)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _calculate_r2(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate R-squared value"""
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0,1]
        return correlation_xy**2
    
    def suggest_visualizations(self, df: pd.DataFrame, query: Optional[str] = None) -> List[Dict]:
        """Generate the most relevant visualization suggestion based on user query and data features"""
        try:
            print("\nStarting visualization suggestion generation...")
            # Prepare data description
            data_description = self._get_data_description(df)
            print("\n1. Data feature analysis completed")
            
            # Build LLM prompt
            prompt = f"""As a data visualization expert, please suggest the SINGLE MOST SUITABLE visualization that best answers the following query.

Data Description:
{data_description}

User Query:
{query or "Please suggest the most important visualization to understand this dataset"}

Please analyze the data and query carefully, then suggest only ONE visualization that would best answer the query or provide the most valuable insight.
The visualization should be one of these types: line, bar, scatter, heatmap, box, pie, histogram.

Return in JSON format with these fields:
- type: The chart type
- columns: The columns to use (x, y, or column depending on chart type)
- title: A descriptive title
- purpose: Why this visualization best answers the query

Example format:
{{
    "type": "bar",
    "x": "Product_Category",
    "y": "Sales_Amount",
    "title": "Total Sales by Product Category",
    "purpose": "This visualization directly shows the sales performance across different product categories, which best answers the query about sales distribution"
}}

IMPORTANT: Return only ONE suggestion in valid JSON format.
"""
            print("2. Generating suggestion using LLM...")
            try:
                response = self.llm(prompt)
                suggestion = json.loads(response)
                print(f"   - LLM returned suggestion for {suggestion.get('type', 'unknown')} chart")
                
                # Validate suggestion
                print("\n3. Validating suggestion:")
                if self._validate_suggestion(suggestion, df):
                    print("   * Validation passed")
                    return [suggestion]  # Return as list for compatibility
                else:
                    print("   * Validation failed, generating default suggestion")
                    return self._generate_default_suggestions(df)[:1]  # Return only first default suggestion
                    
            except Exception as e:
                print(f"\nError generating LLM suggestion: {str(e)}")
                print("Falling back to default suggestion")
                return self._generate_default_suggestions(df)[:1]  # Return only first default suggestion
            
        except Exception as e:
            print(f"\nError in visualization suggestion: {str(e)}")
            return []

    def _generate_default_suggestions(self, df: pd.DataFrame) -> List[Dict]:
        """Generate default visualization suggestion"""
        print("\nGenerating default visualization suggestion...")
        suggestions = []
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns

        # First priority: Time series visualization if datetime column exists
        if len(datetime_columns) > 0 and len(numeric_columns) > 0:
            suggestions.append({
                "type": "line",
                "x": datetime_columns[0],
                "y": numeric_columns[0],
                "title": f"Trend of {numeric_columns[0]} Over Time",
                "purpose": "Show how the main numeric metric changes over time"
            })
            
        # Second priority: Relationship between numeric columns
        elif len(numeric_columns) >= 2:
            suggestions.append({
                "type": "scatter",
                "x": numeric_columns[0],
                "y": numeric_columns[1],
                "title": f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}",
                "purpose": "Analyze correlation between main numeric variables"
            })
            
        # Third priority: Distribution of categorical data
        elif len(categorical_columns) > 0 and len(numeric_columns) > 0:
            suggestions.append({
                "type": "bar",
                "x": categorical_columns[0],
                "y": numeric_columns[0],
                "title": f"{numeric_columns[0]} by {categorical_columns[0]}",
                "purpose": "Compare numeric values across categories"
            })
            
        # Fourth priority: Single numeric distribution
        elif len(numeric_columns) > 0:
            suggestions.append({
                "type": "histogram",
                "column": numeric_columns[0],
                "title": f"Distribution of {numeric_columns[0]}",
                "purpose": "Analyze the distribution pattern of main numeric variable"
            })
            
        # Fifth priority: Categorical distribution
        elif len(categorical_columns) > 0:
            suggestions.append({
                "type": "pie",
                "column": categorical_columns[0],
                "title": f"Distribution of {categorical_columns[0]}",
                "purpose": "Show the proportion distribution of main category"
            })

        print(f"Generated {len(suggestions)} default suggestion")
        return suggestions

    def _get_data_description(self, df: pd.DataFrame) -> str:
        """Generate data description"""
        description = []
        
        # Basic information
        description.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Column type information
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(numeric_cols) > 0:
            description.append(f"\nNumeric columns: {', '.join(numeric_cols)}")
        if len(categorical_cols) > 0:
            description.append(f"\nCategorical columns: {', '.join(categorical_cols)}")
        if len(datetime_cols) > 0:
            description.append(f"\nDatetime columns: {', '.join(datetime_cols)}")
            
        # Numeric column statistics
        if len(numeric_cols) > 0:
            description.append("\nNumeric column statistics:")
            stats = df[numeric_cols].describe()
            for col in numeric_cols:
                description.append(f"{col}:")
                description.append(f"- Range: {stats.loc['min', col]:.2f} to {stats.loc['max', col]:.2f}")
                description.append(f"- Mean: {stats.loc['mean', col]:.2f}")
                description.append(f"- Standard deviation: {stats.loc['std', col]:.2f}")
        
        # Categorical column statistics
        if len(categorical_cols) > 0:
            description.append("\nCategorical column statistics:")
            for col in categorical_cols:
                n_unique = df[col].nunique()
                description.append(f"{col}:")
                description.append(f"- Number of unique values: {n_unique}")
                if n_unique <= 5:  # Only show distribution for small number of categories
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        description.append(f"- {val}: {count} times ({count/len(df)*100:.1f}%)")
        
        return "\n".join(description)

    def _validate_suggestion(self, suggestion: Dict, df: pd.DataFrame) -> bool:
        """Validate visualization suggestion"""
        try:
            # Check required fields
            if "type" not in suggestion:
                return False
            
            chart_type = suggestion["type"]
            if chart_type not in self.supported_charts:
                return False
                
            # Check if required columns exist
            required_columns = []
            if "x" in suggestion:
                required_columns.append(suggestion["x"])
            if "y" in suggestion:
                required_columns.append(suggestion["y"])
            if "column" in suggestion:
                required_columns.append(suggestion["column"])
            if "columns" in suggestion:
                required_columns.extend(suggestion["columns"])
            if "size" in suggestion:
                required_columns.append(suggestion["size"])
                
            for col in required_columns:
                if col not in df.columns:
                    print(f"Column {col} does not exist in the data")
                    return False
                    
            # Check if column types match chart type
            if chart_type in ["line", "scatter", "bubble"]:
                if "x" in suggestion and "y" in suggestion:
                    x_col, y_col = suggestion["x"], suggestion["y"]
                    if not (pd.api.types.is_numeric_dtype(df[y_col]) or 
                           pd.api.types.is_datetime64_dtype(df[x_col])):
                        print(f"Column type mismatch: {chart_type} chart requires numeric or datetime type")
                        return False
                        
            elif chart_type == "bar":
                if "x" in suggestion and "y" in suggestion:
                    y_col = suggestion["y"]
                    if not pd.api.types.is_numeric_dtype(df[y_col]):
                        print(f"Column type mismatch: {chart_type} chart's y-axis requires numeric type")
                        return False
                        
            elif chart_type == "pie":
                if "column" in suggestion:
                    col = suggestion["column"]
                    if df[col].nunique() > 10:  # Limit number of categories
                        print(f"Too many categories: pie chart supports maximum 10 categories")
                        return False
                        
            elif chart_type == "heatmap":
                if "columns" in suggestion:
                    cols = suggestion["columns"]
                    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in cols):
                        print(f"Column type mismatch: heatmap requires numeric columns")
                        return False
            
            # Ensure suggestion contains title
            if "title" not in suggestion:
                suggestion["title"] = self._generate_default_title(suggestion)
                
            return True
            
        except Exception as e:
            print(f"验证建议时出错: {str(e)}")
            return False

    def _generate_default_title(self, suggestion: Dict) -> str:
        """Generate default chart title"""
        chart_type = suggestion["type"]
        
        if chart_type in ["line", "scatter", "bubble", "bar"]:
            x = suggestion.get("x", "")
            y = suggestion.get("y", "")
            return f"{y} vs {x}"
            
        elif chart_type == "pie":
            column = suggestion.get("column", "")
            return f"{column} Distribution"
            
        elif chart_type == "heatmap":
            return "Correlation Heatmap"
            
        elif chart_type == "histogram":
            column = suggestion.get("column", "")
            return f"{column} Distribution"
            
        elif chart_type == "box":
            x = suggestion.get("x", "")
            y = suggestion.get("y", "")
            return f"{y} Distribution by {x}"
            
        return "Data Visualization"

    def create_visualization(self, suggestion: Dict, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Create visualization based on suggestion"""
        try:
            # Preprocess data
            df = self._preprocess_data(df)
            
            chart_type = suggestion.get("type")
            if not chart_type or chart_type not in self.supported_charts:
                print(f"Unsupported chart type: {chart_type}")
                return None

            # Check if required columns exist
            required_columns = []
            if "x" in suggestion:
                required_columns.append(suggestion["x"])
            if "y" in suggestion:
                required_columns.append(suggestion["y"])
            if "column" in suggestion:
                required_columns.append(suggestion["column"])
            if "columns" in suggestion:
                required_columns.extend(suggestion["columns"])
            if "size" in suggestion:
                required_columns.append(suggestion["size"])

            for col in required_columns:
                if col not in df.columns:
                    print(f"Column {col} does not exist in the data")
                    return None

            # Remove purpose field as it's not needed for chart creation
            suggestion_copy = suggestion.copy()
            suggestion_copy.pop('purpose', None)

            # Call corresponding chart creation function
            return self.supported_charts[chart_type](
                df=df,
                **{k: v for k, v in suggestion_copy.items() if k not in ["type"]}
            )

        except Exception as e:
            print(f"创建可视化时出错: {str(e)}")
            return None
    
    def create_interactive_plot(self, df: pd.DataFrame, plot_type: str, **kwargs) -> go.Figure:
        """Create interactive plot using plotly
        
        Args:
            df: DataFrame
            plot_type: Type of plot to create
            **kwargs: Additional arguments for the plot
            
        Returns:
            Plotly figure
        """
        if plot_type == 'scatter':
            fig = px.scatter(df, **kwargs)
        elif plot_type == 'line':
            fig = px.line(df, **kwargs)
        elif plot_type == 'bar':
            fig = px.bar(df, **kwargs)
        elif plot_type == 'box':
            fig = px.box(df, **kwargs)
        elif plot_type == 'violin':
            fig = px.violin(df, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        return fig
    
    def create_anomaly_plot(self, df: pd.DataFrame, column: str, threshold: float = 3) -> plt.Figure:
        """Create anomaly detection plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate Z-scores
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        
        # Plot normal values
        ax.scatter(range(len(df)), df[column], c='blue', alpha=0.5, label='Normal')
        
        # Mark outliers
        ax.scatter(np.where(outliers)[0], df[column][outliers], 
                  c='red', alpha=0.7, label='Outliers')
        
        plt.title(f'{column} Anomaly Detection')
        plt.legend()
        
        return fig
    
    def create_feature_importance_plot(self, importance_scores: dict) -> plt.Figure:
        """Create feature importance plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort and plot
        sorted_items = sorted(importance_scores.items(), key=lambda x: x[1])
        features, scores = zip(*sorted_items)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        
        plt.title('Feature Importance Analysis')
        
        return fig 

    def create_visualization_html(self, df: pd.DataFrame, suggestion: Dict) -> str:
        """Create visualization in HTML format with analysis"""
        try:
            # Create figure
            fig = self.create_visualization(suggestion, df)
            if fig is None:
                return f"""
                <div class="viz-error">
                    <h3>Error Creating Visualization</h3>
                    <p>Could not generate visualization for the suggestion: {suggestion.get('title', 'Unknown')}</p>
                </div>
                """

            # Convert plot to HTML
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

            # Generate analysis based on visualization type
            analysis = self._generate_visualization_analysis(df, suggestion)

            # Create HTML with visualization and analysis
            html = f"""
            <div class="viz-container">
                <div class="viz-title">
                    <h3>{suggestion.get('title', 'Visualization')}</h3>
                    <p class="viz-purpose">{suggestion.get('purpose', '')}</p>
                </div>
                <div class="viz-content">
                    <div class="viz-image">
                        <img src="data:image/png;base64,{img_data}" alt="{suggestion.get('title', 'Visualization')}">
                    </div>
                    <div class="viz-analysis">
                        <h4>Analysis:</h4>
                        {analysis}
                    </div>
                </div>
            </div>
            """

            return html

        except Exception as e:
            return f"""
            <div class="viz-error">
                <h3>Error</h3>
                <p>Error creating visualization: {str(e)}</p>
            </div>
            """

    def _generate_visualization_analysis(self, df: pd.DataFrame, suggestion: Dict) -> str:
        """Generate comprehensive analysis for visualization based on its type and data"""
        try:
            chart_type = suggestion.get('type')
            analysis_points = []
            
            # Add general data information
            used_columns = []
            if 'x' in suggestion:
                used_columns.append(suggestion['x'])
            if 'y' in suggestion:
                used_columns.append(suggestion['y'])
            if 'column' in suggestion:
                used_columns.append(suggestion['column'])
            if 'columns' in suggestion:
                used_columns.extend(suggestion['columns'])
                
            analysis_points.append("<h4>Data Information:</h4>")
            analysis_points.append("<ul class='analysis-points'>")
            
            # Add column information
            for col in used_columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique = df[col].nunique()
                missing_pct = (missing / len(df)) * 100
                
                analysis_points.append(f"<li>Column '{col}':")
                analysis_points.append(f"  - Data type: {dtype}")
                analysis_points.append(f"  - Unique values: {unique}")
                analysis_points.append(f"  - Missing values: {missing} ({missing_pct:.1f}%)")
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    analysis_points.append(f"  - Range: {df[col].min():.2f} to {df[col].max():.2f}")
                    analysis_points.append(f"  - Mean: {df[col].mean():.2f}")
                    analysis_points.append(f"  - Median: {df[col].median():.2f}")
                    analysis_points.append(f"  - Standard deviation: {df[col].std():.2f}")
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    analysis_points.append(f"  - Date range: {df[col].min()} to {df[col].max()}")
                    analysis_points.append(f"  - Time span: {(df[col].max() - df[col].min()).days} days")
                analysis_points.append("</li>")
            
            analysis_points.append("</ul>")
            
            # Add chart-specific analysis
            analysis_points.append("<h4>Visualization Analysis:</h4>")
            analysis_points.append("<ul class='analysis-points'>")
            
            if chart_type == 'bar':
                x, y = suggestion.get('x'), suggestion.get('y')
                if x and y:
                    grouped = df.groupby(x)[y].agg(['mean', 'count', 'std'])
                    max_cat = grouped['mean'].idxmax()
                    min_cat = grouped['mean'].idxmin()
                    std_max = grouped['std'].idxmax()
                    
                    analysis_points.extend([
                        f"<li>Distribution Analysis:",
                        f"  - Highest average {y}: {max_cat} ({grouped['mean'][max_cat]:.2f})",
                        f"  - Lowest average {y}: {min_cat} ({grouped['mean'][min_cat]:.2f})",
                        f"  - Most variable category: {std_max} (std: {grouped['std'][std_max]:.2f})",
                        f"  - Number of categories: {len(grouped)}",
                        "</li>"
                    ])

            elif chart_type == 'scatter':
                x, y = suggestion.get('x'), suggestion.get('y')
                if x and y:
                    corr = df[x].corr(df[y])
                    slope, intercept = np.polyfit(df[x], df[y], 1)
                    
                    analysis_points.extend([
                        f"<li>Correlation Analysis:",
                        f"  - Correlation coefficient: {corr:.3f}",
                        f"  - Relationship strength: {'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'}",
                        f"  - Direction: {'Positive' if corr > 0 else 'Negative'} correlation",
                        f"  - Trend line slope: {slope:.3f}",
                        "</li>"
                    ])

            elif chart_type == 'pie':
                column = suggestion.get('column')
                if column:
                    value_counts = df[column].value_counts()
                    top_category = value_counts.index[0]
                    top_percentage = (value_counts[0] / len(df)) * 100
                    
                    analysis_points.extend([
                        f"<li>Category Distribution:",
                        f"  - Most common category: {top_category} ({top_percentage:.1f}%)",
                        f"  - Number of categories: {len(value_counts)}",
                        f"  - Top 3 categories: {', '.join(value_counts.index[:3])}",
                        f"  - Distribution evenness: {'Even' if value_counts.std() / value_counts.mean() < 0.5 else 'Uneven'}",
                        "</li>"
                    ])

            elif chart_type == 'line':
                x, y = suggestion.get('x'), suggestion.get('y')
                if x and y:
                    # Calculate trend
                    y_values = df[y].values
                    trend = 'increasing' if y_values[-1] > y_values[0] else 'decreasing'
                    volatility = df[y].std() / df[y].mean()
                    
                    # Find peaks and troughs
                    peak_idx = df[y].idxmax()
                    trough_idx = df[y].idxmin()
                    
                    analysis_points.extend([
                        f"<li>Trend Analysis:",
                        f"  - Overall trend: {trend}",
                        f"  - Volatility: {volatility:.3f}",
                        f"  - Peak value: {df[y].max():.2f} at {df[x][peak_idx]}",
                        f"  - Lowest value: {df[y].min():.2f} at {df[x][trough_idx]}",
                        f"  - Average rate of change: {(df[y].iloc[-1] - df[y].iloc[0]) / len(df):.3f} per period",
                        "</li>"
                    ])

            elif chart_type == 'histogram':
                column = suggestion.get('column')
                if column:
                    skewness = df[column].skew()
                    kurtosis = df[column].kurtosis()
                    q1, q3 = df[column].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    
                    analysis_points.extend([
                        f"<li>Distribution Analysis:",
                        f"  - Shape: {'Symmetric' if abs(skewness) < 0.5 else 'Right-skewed' if skewness > 0 else 'Left-skewed'}",
                        f"  - Skewness: {skewness:.3f}",
                        f"  - Kurtosis: {kurtosis:.3f}",
                        f"  - IQR (Interquartile Range): {iqr:.3f}",
                        f"  - Data spread: {'Wide' if iqr > (df[column].max() - df[column].min()) / 4 else 'Narrow'}",
                        "</li>"
                    ])

            elif chart_type == 'box':
                x, y = suggestion.get('x'), suggestion.get('y')
                if x and y:
                    stats = df.groupby(x)[y].describe()
                    outliers = df[df[y] > df[y].quantile(0.75) + 1.5 * iqr].shape[0]
                    
                    analysis_points.extend([
                        f"<li>Distribution Analysis:",
                        f"  - Most variable category: {stats['std'].idxmax()}",
                        f"  - Least variable category: {stats['std'].idxmin()}",
                        f"  - Number of outliers: {outliers}",
                        f"  - Distribution type: {'Similar' if stats['std'].std() / stats['std'].mean() < 0.2 else 'Varied'} across categories",
                        "</li>"
                    ])

            analysis_points.append("</ul>")
            
            # Add recommendations and insights
            analysis_points.append("<h4>Key Insights:</h4>")
            analysis_points.append("<ul class='analysis-points'>")
            analysis_points.append(f"<li>Purpose: {suggestion.get('purpose', 'Not specified')}</li>")
            
            # Add specific recommendations based on chart type
            if chart_type == 'scatter':
                if abs(corr) > 0.7:
                    analysis_points.append("<li>Strong correlation suggests these variables are closely related and might be good predictors of each other.</li>")
            elif chart_type == 'line':
                if volatility > 0.5:
                    analysis_points.append("<li>High volatility suggests significant fluctuations that might need further investigation.</li>")
            elif chart_type == 'histogram':
                if abs(skewness) > 1:
                    analysis_points.append("<li>The significant skewness might indicate underlying patterns or biases in the data collection.</li>")
                    
            analysis_points.append("</ul>")

            return "\n".join(analysis_points)

        except Exception as e:
            return f"<p>Error generating analysis: {str(e)}</p>"

    def generate_visualization_page(self, df: pd.DataFrame, query: str = None) -> str:
        """Generate complete HTML page with visualizations and analysis"""
        # Generate suggestions
        suggestions = self.suggest_visualizations(df, query)
        
        # CSS styles
        css = """
        <style>
            .viz-container {
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: white;
            }
            .viz-title {
                margin-bottom: 15px;
            }
            .viz-title h3 {
                margin: 0;
                color: #2c3e50;
            }
            .viz-purpose {
                color: #666;
                font-style: italic;
                margin: 5px 0;
            }
            .viz-content {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .viz-image {
                flex: 1;
                text-align: center;
            }
            .viz-image img {
                max-width: 100%;
                height: auto;
            }
            .viz-analysis {
                flex: 1;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 4px;
            }
            .viz-analysis h4 {
                margin-top: 0;
                color: #2c3e50;
            }
            .analysis-points {
                margin: 0;
                padding-left: 20px;
            }
            .analysis-points li {
                margin: 5px 0;
            }
            .viz-error {
                padding: 15px;
                background: #fff3cd;
                border: 1px solid #ffeeba;
                border-radius: 4px;
                color: #856404;
            }
            @media (min-width: 768px) {
                .viz-content {
                    flex-direction: row;
                }
            }
        </style>
        """

        # Generate HTML content
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Data Visualizations</title>",
            css,
            "</head>",
            "<body>",
            "<div class='visualizations'>"
        ]

        # Add visualizations
        if suggestions:
            for suggestion in suggestions:
                html_parts.append(self.create_visualization_html(df, suggestion))
        else:
            html_parts.append("""
                <div class="viz-error">
                    <h3>No Visualizations</h3>
                    <p>No valid visualization suggestions were generated.</p>
                </div>
            """)

        html_parts.extend([
            "</div>",
            "</body>",
            "</html>"
        ])

        return "\n".join(html_parts)

    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            if fig is None:
                return ""
                
            # Create a BytesIO buffer
            buf = BytesIO()
            
            # Save the figure to the buffer
            if isinstance(fig, plt.Figure):
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            else:
                # If fig is a numpy array, convert it to a matplotlib figure first
                plt.figure()
                plt.imshow(fig)
                plt.axis('off')
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Reset buffer position
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Close the buffer
            buf.close()
            
            # Close the figure to free memory
            if isinstance(fig, plt.Figure):
                plt.close(fig)
            
            return img_str
        except Exception as e:
            print(f"Error converting figure to base64: {str(e)}")
            return "" 