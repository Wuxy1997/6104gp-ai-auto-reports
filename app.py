import streamlit as st
import pandas as pd
from pathlib import Path
import os
import time
from datetime import datetime
import base64
from utils.data_processor import DataProcessor
from utils.report_generator import ReportGenerator
from utils.visualizer import DataVisualizer
from utils.insights_generator import InsightsGenerator
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Configure page settings
st.set_page_config(
    page_title="Automated Data Report Generator",
    layout="wide"
)

# Hide deploy button and menu, set dark theme
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark theme base styles */
    :root {
        --bg-color: #0e1117;
        --text-color: #fafafa;
        --secondary-bg: #262730;
        --accent-color: #4CAF50;
        --hover-color: #45a049;
    }
    
    /* Streamlit connecting screen */
    .stConnectionStatus {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        border: none !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    
    .stConnectionStatus > div {
        background-color: var(--secondary-bg) !important;
        border: 1px solid var(--accent-color) !important;
        padding: 10px !important;
        border-radius: 5px !important;
        color: var(--text-color) !important;
    }
    
    .stConnectionStatus p {
        color: var(--text-color) !important;
    }
    
    /* Overall background and text */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Title styles */
    h1 {
        color: var(--accent-color);
        font-size: 36px !important;
        font-weight: bold !important;
        padding: 20px 0 !important;
        text-align: center;
        background-color: var(--secondary-bg);
        border-radius: 10px;
        margin-bottom: 30px !important;
    }
    
    /* Subtitle styles */
    h2, h3 {
        color: var(--text-color);
        font-weight: 600 !important;
        margin: 20px 0 10px 0 !important;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--secondary-bg);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-color);
        border: 1px solid var(--secondary-bg);
        border-radius: 5px;
        color: var(--text-color);
        font-weight: 500;
        padding: 10px 20px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--secondary-bg);
        border-color: var(--accent-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: var(--text-color) !important;
        border: none !important;
    }
    
    /* Button styles */
    .stButton > button {
        width: 100%;
        background-color: var(--accent-color);
        color: var(--text-color);
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
        margin: 10px 0;
    }
    
    .stButton > button:hover {
        background-color: var(--hover-color);
        border: none;
    }
    
    /* Input and select box styles */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: var(--secondary-bg);
        color: var(--text-color);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    
    /* DataFrame styles */
    .stDataFrame {
        background-color: var(--secondary-bg);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Progress bar styles */
    .stProgress > div > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* Warning and error message styles */
    .stAlert {
        background-color: rgba(255, 243, 205, 0.1);
        color: #ffd700;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid rgba(255, 243, 205, 0.2);
    }
    
    /* Upload area styles */
    .uploadedFile {
        background-color: var(--secondary-bg);
        border: 2px dashed #4e4e4e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    
    .uploadedFile:hover {
        border-color: var(--accent-color);
    }
    
    /* Download button styles */
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: var(--accent-color);
        color: #ffffff !important;  /* Force white text color */
        text-decoration: none;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        transition: background-color 0.3s;
    }
    
    .download-button:hover {
        background-color: var(--hover-color);
        color: #ffffff !important;  /* Ensure white color on hover */
        text-decoration: none;
    }
    
    /* Code block styles */
    .stCodeBlock {
        background-color: var(--secondary-bg);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        h1 {
            font-size: 28px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

def render_data_upload_tab():
    """Data upload and preview tab"""
    st.header("Data Upload & Preview")

    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            # Rename columns to English
            column_mapping = {
                '日期': 'Date',
                '产品类别': 'Product_Category',
                '产品名称': 'Product_Name',
                '销售数量': 'Sales_Quantity',
                '单价': 'Unit_Price',
                '销售额': 'Sales_Amount',
                '销售地区': 'Sales_Region',
                '销售人员': 'Sales_Person',
                '客户评分': 'Customer_Rating',
                '备注': 'Notes'
            }
            df = df.rename(columns=column_mapping)
            
            # Initialize session state
            st.session_state.original_df = df.copy()  # Save original data
            st.session_state.processed_df = df.copy()  # Create copy for processing
            st.session_state.processor = DataProcessor()
            st.session_state.visualizer = DataVisualizer()
            
            # Initialize data processing state
            st.session_state.data_processing_state = {
                'original_data': df.copy(),  # Save original data backup
                'processing_steps': [],      # Initialize processing steps record
                'last_modified': datetime.now(),  # Record last modification time
                'current_step': None,        # Current processing step
                'current_df_hash': hash(df.to_string())  # Current data hash value
            }

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.processed_df.head())

            # Display data information
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Rows: {st.session_state.processed_df.shape[0]}")
                st.write(f"Columns: {st.session_state.processed_df.shape[1]}")
            with col2:
                st.write("Column Types:")
                st.write(st.session_state.processed_df.dtypes)
                
            # Display data quality report
            st.subheader("Data Quality Report")
            quality_report = st.session_state.processor.detect_data_quality(st.session_state.processed_df)
            
            # Display missing values information
            st.markdown("#### Missing Values")
            missing_values = st.session_state.processed_df.isnull().sum()
            missing_percentages = (missing_values / len(st.session_state.processed_df) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Missing Count': missing_values,
                'Missing Percentage': missing_percentages
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                st.write("Total missing values:", missing_values.sum())
                st.write("Columns with missing values:")
                st.dataframe(missing_df)
            else:
                st.write("Total missing values: 0")
                st.write("No columns contain missing values.")
            
            # Display data type information
            st.markdown("#### Data Types")
            dtype_counts = st.session_state.processed_df.dtypes.value_counts()
            numeric_cols = st.session_state.processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = st.session_state.processed_df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = st.session_state.processed_df.select_dtypes(include=['datetime64']).columns.tolist()
            
            st.write(f"Numeric columns ({len(numeric_cols)}):", ", ".join(numeric_cols) if numeric_cols else "None")
            st.write(f"Categorical columns ({len(categorical_cols)}):", ", ".join(categorical_cols) if categorical_cols else "None")
            st.write(f"DateTime columns ({len(datetime_cols)}):", ", ".join(datetime_cols) if datetime_cols else "None")
            
            # Display basic statistics
            st.markdown("#### Basic Statistics")
            st.write("Numeric Columns Summary:")
            st.dataframe(st.session_state.processed_df.describe())
            
            # Display unique values information
            st.markdown("#### Unique Values")
            unique_counts = st.session_state.processed_df.nunique()
            unique_percentages = (unique_counts / len(st.session_state.processed_df) * 100).round(2)
            
            unique_df = pd.DataFrame({
                'Unique Count': unique_counts,
                'Unique Percentage': unique_percentages
            })
            st.dataframe(unique_df)
            
            # Display duplicate rows information
            st.markdown("#### Duplicate Rows")
            duplicate_count = st.session_state.processed_df.duplicated().sum()
            st.write(f"Number of duplicate rows: {duplicate_count}")
            st.write(f"Percentage of duplicate rows: {(duplicate_count / len(st.session_state.processed_df) * 100):.2f}%")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        # Display prompt if no file is uploaded
        st.info("Please upload a CSV file to begin data analysis.")

def render_data_processing_tab():
    """数据处理标签页"""
    st.header("Data Processing")
    
    # 尝试加载最近处理的数据
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        loaded_df = load_processed_data()
        if loaded_df is not None:
            st.session_state.processed_df = loaded_df
            st.success("Loaded last processed data successfully!")
        else:
            st.warning("Please upload data first!")
            return

    # 显示当前数据预览
    st.markdown("### Current Data Preview")
    st.dataframe(st.session_state.processed_df.head())
    
    # 显示数据信息
    st.markdown("### Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {st.session_state.processed_df.shape[0]}")
        st.write(f"Columns: {st.session_state.processed_df.shape[1]}")
    with col2:
        st.write("Column Types:")
        st.write(st.session_state.processed_df.dtypes)

    # 显示数据处理历史
    if st.session_state.data_processing_state['processing_steps']:
        with st.expander("Processing History", expanded=False):
            st.markdown("### Applied Processing Steps")
            for step in st.session_state.data_processing_state['processing_steps']:
                st.markdown(f"""
                - **{step['type'].replace('_', ' ').title()}**
                  - Method: {step['method']}
                  - Time: {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                  - Saved File: {step.get('saved_file', 'Not saved')}
                {f"- Fill Value: {step['fill_value']}" if 'fill_value' in step and step['fill_value'] is not None else ""}
                """)

    # 创建子标签页
    missing_tab, normalize_tab, outliers_tab, encode_tab = st.tabs([
        "Missing Values",
        "Normalization",
        "Outliers",
        "Categorical Encoding"
    ])
    
    # 处理缺失值标签页
    with missing_tab:
        st.markdown("""
        ### Handle Missing Values
        Process missing values in your dataset using various methods.
        """)
        
        # 使用两列布局
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            # 显示当前缺失值情况
            missing_stats = pd.DataFrame({
                'Missing Values': st.session_state.processed_df.isnull().sum(),
                'Percentage': (st.session_state.processed_df.isnull().sum() / len(st.session_state.processed_df) * 100).round(2)
            })
            
            # 只显示有缺失值的列
            missing_stats = missing_stats[missing_stats['Missing Values'] > 0]
            
            if not missing_stats.empty:
                st.markdown("#### Current Missing Values")
                st.dataframe(missing_stats, use_container_width=True)
            else:
                st.success("No missing values in the dataset!")
            
            with st.form("missing_values_form"):
                missing_method = st.selectbox(
                    "Method",
                    ["drop", "fill_mean", "fill_median", "fill_mode", "fill_constant"],
                    help="""
                    - drop: Remove rows with missing values
                    - fill_mean: Fill missing values with column mean (numeric only)
                    - fill_median: Fill missing values with column median (numeric only)
                    - fill_mode: Fill missing values with column mode
                    - fill_constant: Fill missing values with a specified value
                    """
                )
                
                # 只在选择fill_constant方法时显示填充值输入框
                if missing_method == "fill_constant":
                    fill_value = st.text_input(
                        "Fill Value",
                        value="0",
                        help="Enter the value to fill missing data with"
                    )
                else:
                    fill_value = None
                
                submit_button = st.form_submit_button("Process Missing Values")
                
                if submit_button:
                    try:
                        if missing_method == "fill_constant" and not fill_value:
                            st.error("Please specify a fill value")
                        else:
                            # 处理数据
                            df_processed = handle_missing_values(
                                st.session_state.processed_df,
                                missing_method,
                                fill_value
                            )
                            
                            # 显示处理结果
                            st.success("Missing values processed successfully!")
                            
                            # 显示处理前后的比较
                            st.markdown("#### Processing Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Before Processing:")
                                missing_before = st.session_state.processed_df.isnull().sum()
                                st.write(missing_before[missing_before > 0])
                            with col2:
                                st.write("After Processing:")
                                missing_after = df_processed.isnull().sum()
                                st.write(missing_after[missing_after > 0])
                            
                    except Exception as e:
                        st.error(f"Error handling missing values: {str(e)}")
        
        with right_col:
            st.markdown("#### Processing Results")
            # 计算处理前后的统计信息
            missing_before = st.session_state.processed_df.isnull().sum()
            total_missing_before = missing_before.sum()
            total_cells = st.session_state.processed_df.size
            
            # 创建进度条显示缺失值比例
            missing_percentage = (total_missing_before / total_cells) * 100
            st.metric(
                "Missing Values Percentage",
                f"{missing_percentage:.2f}%",
                delta=f"-{missing_percentage:.2f}%" if missing_percentage > 0 else "0%"
            )
            
            # 显示每列的缺失值分布
            st.markdown("#### Missing Values Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            missing_before.plot(kind='bar')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # 数据标准化标签页
    with normalize_tab:
        st.markdown("""
        ### Normalize Data
        Standardize numeric columns in your dataset.
        """)
        
        # 使用两列布局
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            with st.form("normalization_form"):
                norm_method = st.selectbox(
                    "Method",
                    ["minmax", "standard", "robust", "log"],
                    help="""
                    - minmax: Scale data to 0-1 range
                    - standard: Z-score normalization (mean=0, std=1)
                    - robust: Scale using quartiles (less sensitive to outliers)
                    - log: Logarithmic transformation
                    """
                )
                
                # 选择要标准化的列
                numeric_cols = st.session_state.processed_df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns found for normalization")
                else:
                    norm_columns = st.multiselect(
                        "Select columns to normalize",
                        numeric_cols,
                        default=list(numeric_cols),
                        help="Choose the numeric columns you want to normalize"
                    )
                    
                    submit_button = st.form_submit_button("Normalize Data")
                    
                    if submit_button and norm_columns:
                        try:
                            # 处理数据
                            df_normalized = normalize_data(st.session_state.processed_df, norm_method)
                            
                            # 记录处理步骤
                            step_info = {
                                'type': 'normalization',
                                'method': norm_method,
                                'columns': norm_columns,
                                'timestamp': datetime.now()
                            }
                            st.session_state.data_processing_state['processing_steps'].append(step_info)
                            
                            st.success("Data normalized successfully!")
                            
                            # 显示处理前后的比较
                            st.markdown("#### Processing Results")
                            for col in norm_columns:
                                st.write(f"\n**Column: {col}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Before Normalization:")
                                    st.write(st.session_state.processed_df[col].describe())
                                with col2:
                                    st.write("After Normalization:")
                                    st.write(df_normalized[col].describe())
                            
                        except Exception as e:
                            st.error(f"Error normalizing data: {str(e)}")
        
        with right_col:
            if 'df_normalized' in locals() and df_normalized is not None:
                st.markdown("#### Column Statistics")
                # 为选中的列创建描述性统计图表
                for col in norm_columns:
                    with st.expander(f"{col} Distribution", expanded=False):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                        
                        # 原始数据分布
                        sns.histplot(st.session_state.processed_df[col], ax=ax1)
                        ax1.set_title("Original Distribution")
                        
                        # 标准化后的分布
                        sns.histplot(df_normalized[col], ax=ax2)
                        ax2.set_title("Normalized Distribution")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # 显示基本统计信息
                        st.markdown("**Basic Statistics:**")
                        st.write(df_normalized[col].describe())
            else:
                st.info("Please normalize data first to see the distribution plots.")

    # 处理异常值标签页
    with outliers_tab:
        st.markdown("""
        ### Handle Outliers
        Detect and process outliers in your dataset.
        """)
        
        # 使用两列布局
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            with st.form("outliers_form"):
                outlier_method = st.selectbox(
                    "Detection Method",
                    ["zscore", "iqr", "percentile"],
                    help="""
                    - zscore: Use Z-score method (standard deviations from mean)
                    - iqr: Use Interquartile Range method
                    - percentile: Use percentile method
                    """
                )
                
                outlier_threshold = st.number_input(
                    "Threshold",
                    value=3.0,
                    step=0.1,
                    help="Threshold for outlier detection (Z-score or IQR multiplier)"
                )
                
                # 选择要处理的列
                numeric_cols = st.session_state.processed_df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns found for outlier detection")
                else:
                    outlier_columns = st.multiselect(
                        "Select columns to check for outliers",
                        numeric_cols,
                        default=list(numeric_cols)
                    )
                    
                    submit_button = st.form_submit_button("Handle Outliers")
                    
                    if submit_button and outlier_columns:
                        try:
                            # 处理数据
                            df_processed = handle_outliers(
                                st.session_state.processed_df,
                                outlier_method,
                                outlier_threshold
                            )
                            
                            # 记录处理步骤
                            step_info = {
                                'type': 'outliers',
                                'method': outlier_method,
                                'threshold': outlier_threshold,
                                'columns': outlier_columns,
                                'timestamp': datetime.now()
                            }
                            st.session_state.data_processing_state['processing_steps'].append(step_info)
                            
                            st.success("Outliers handled successfully!")
                            
                            # 显示处理前后的比较
                            st.markdown("#### Processing Results")
                            for col in outlier_columns:
                                st.write(f"\n**Column: {col}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Before Processing:")
                                    st.write(st.session_state.processed_df[col].describe())
                                with col2:
                                    st.write("After Processing:")
                                    st.write(df_processed[col].describe())
                            
                        except Exception as e:
                            st.error(f"Error handling outliers: {str(e)}")
        
        with right_col:
            if 'outlier_columns' in locals() and outlier_columns:
                st.markdown("#### Outlier Analysis")
                for col in outlier_columns:
                    with st.expander(f"{col} Outliers", expanded=False):
                        # 创建箱线图
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.boxplot(data=st.session_state.processed_df[col], ax=ax)
                        plt.title(f"Box Plot for {col}")
                        st.pyplot(fig)
                        plt.close()
                        
                        # 显示异常值统计
                        q1 = st.session_state.processed_df[col].quantile(0.25)
                        q3 = st.session_state.processed_df[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = st.session_state.processed_df[
                            (st.session_state.processed_df[col] < (q1 - 1.5 * iqr)) |
                            (st.session_state.processed_df[col] > (q3 + 1.5 * iqr))
                        ][col]
                        
                        st.markdown(f"""
                        **Outlier Statistics:**
                        - Total values: {len(st.session_state.processed_df[col])}
                        - Outliers found: {len(outliers)}
                        - Outlier percentage: {(len(outliers) / len(st.session_state.processed_df[col]) * 100):.2f}%
                        """)

    # 编码分类变量标签页
    with encode_tab:
        st.markdown("""
        ### Encode Categorical Variables
        Convert categorical variables to numeric format.
        """)
        
        # 使用两列布局
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            with st.form("encoding_form"):
                encode_method = st.selectbox(
                    "Method",
                    ["label", "one_hot", "ordinal"],
                    help="""
                    - label: Convert categories to numeric labels
                    - one_hot: Create binary columns for each category
                    - ordinal: Encode categories based on their order
                    """
                )
                
                # 选择要编码的列
                categorical_cols = st.session_state.processed_df.select_dtypes(include=['object']).columns
                if len(categorical_cols) == 0:
                    st.warning("No categorical columns found for encoding")
                else:
                    encode_columns = st.multiselect(
                        "Select columns to encode",
                        categorical_cols,
                        default=list(categorical_cols),
                        help="Choose the categorical columns you want to encode"
                    )
                    
                    submit_button = st.form_submit_button("Encode Variables")
                    
                    if submit_button and encode_columns:
                        try:
                            # 处理数据
                            df_encoded = encode_categorical(
                                st.session_state.processed_df,
                                encode_method
                            )
                            
                            # 记录处理步骤
                            step_info = {
                                'type': 'categorical_encoding',
                                'method': encode_method,
                                'columns': encode_columns,
                                'timestamp': datetime.now()
                            }
                            st.session_state.data_processing_state['processing_steps'].append(step_info)
                            
                            st.success("Categorical variables encoded successfully!")
                            
                            # 显示处理前后的比较
                            st.markdown("#### Processing Results")
                            for col in encode_columns:
                                st.write(f"\n**Column: {col}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Before Encoding:")
                                    st.write(st.session_state.processed_df[col].value_counts())
                                with col2:
                                    st.write("After Encoding:")
                                    if encode_method == "one_hot":
                                        # 对于one-hot编码，显示新创建的列
                                        new_cols = [c for c in df_encoded.columns if c.startswith(col)]
                                        st.write(df_encoded[new_cols].sum())
                                    else:
                                        st.write(df_encoded[col].value_counts())
                            
                        except Exception as e:
                            st.error(f"Error encoding categorical variables: {str(e)}")
        
        with right_col:
            if 'encode_columns' in locals() and encode_columns:
                st.markdown("#### Encoding Preview")
                for col in encode_columns:
                    with st.expander(f"{col} Encoding", expanded=False):
                        # 显示原始类别分布
                        st.markdown("**Original Categories:**")
                        st.write(st.session_state.processed_df[col].value_counts())
                        
                        # 可视化类别分布
                        fig, ax = plt.subplots(figsize=(8, 4))
                        st.session_state.processed_df[col].value_counts().plot(kind='bar')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

    # 添加导出按钮
    st.markdown("---")
    st.markdown("### Export Processed Data")
    
    # 使用列布局组织导出选项
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON"],
            help="Choose the format for your exported data"
        )
    
    with col2:
        file_name = st.text_input(
            "File Name",
            value="processed_data",
            help="Enter a name for your exported file"
        )
    
    with col3:
        encoding = st.selectbox(
            "File Encoding",
            ["utf-8-sig", "utf-8", "ascii"],
            help="Choose the character encoding for your file"
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)  # 添加空行以对齐按钮
        if st.button("Export", key="export_button"):
            try:
                # 确保输出目录存在
                output_dir = "exported_data"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 根据选择的格式导出数据
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if export_format == "CSV":
                    output_path = os.path.join(output_dir, f"{file_name}_{timestamp}.csv")
                    st.session_state.processed_df.to_csv(output_path, index=False, encoding=encoding)
                elif export_format == "Excel":
                    output_path = os.path.join(output_dir, f"{file_name}_{timestamp}.xlsx")
                    st.session_state.processed_df.to_excel(output_path, index=False)
                else:  # JSON
                    output_path = os.path.join(output_dir, f"{file_name}_{timestamp}.json")
                    st.session_state.processed_df.to_json(output_path, orient='records', force_ascii=False)
                
                # 创建下载链接
                with open(output_path, 'rb') as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode()
                ext = os.path.splitext(output_path)[1]
                mime_types = {
                    '.csv': 'text/csv',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.json': 'application/json'
                }
                mime_type = mime_types.get(ext, 'application/octet-stream')
                
                href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(output_path)}" class="download-button">Download {export_format} File</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"Data exported successfully as {export_format}!")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")

def render_visualization_tab():
    st.header("Data Visualization")
    
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        st.warning("Please upload data first!")
        return
        
    # 创建可视化器实例
    if st.session_state.visualizer is None:
        st.session_state.visualizer = DataVisualizer()
    
    # 显示当前数据预览
    st.markdown("### Current Data Preview")
    st.dataframe(st.session_state.processed_df.head())
    
    # 显示数据信息
    st.markdown("### Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {st.session_state.processed_df.shape[0]}")
        st.write(f"Columns: {st.session_state.processed_df.shape[1]}")
    with col2:
        st.write("Column Types:")
        st.write(st.session_state.processed_df.dtypes)
    
    # 添加用户查询输入
    st.subheader("Visualization Query")
    query_type = st.selectbox(
        "Choose query type",
        ["Use preset questions", "Enter custom query"],
        key="viz_query_type"
    )
    
    if query_type == "Use preset questions":
        preset_questions = [
            "Show the distribution of numeric columns",
            "Compare values across different categories",
            "Show relationships between numeric variables",
            "Show trends over time",
            "Show composition of categorical variables"
        ]
        query = st.selectbox("Select a question", preset_questions)
    else:
        query = st.text_input(
            "Enter your visualization query",
            placeholder="e.g., Show the relationship between sales and profit"
        )
        
    # 添加高级设置
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            chart_style = st.selectbox(
                "Chart Style",
                ["default", "seaborn", "ggplot", "dark_background"],
                help="Choose the visual style for the charts"
            )
        with col2:
            color_theme = st.selectbox(
                "Color Theme",
                ["husl", "deep", "muted", "pastel", "bright", "dark"],
                help="Choose the color palette for the charts"
            )
            
        col3, col4 = st.columns(2)
        with col3:
            chart_width = st.slider("Chart Width", 6, 20, 10, help="Adjust the width of the charts")
        with col4:
            chart_height = st.slider("Chart Height", 4, 15, 6, help="Adjust the height of the charts")
            
        show_grid = st.checkbox("Show Grid", True, help="Toggle grid lines on charts")
        show_stats = st.checkbox("Show Statistics", True, help="Show statistical information on charts")

    # 生成可视化按钮
    if st.button("Generate Visualizations", type="primary"):
        with st.spinner("Generating visualizations..."):
            try:
                # 应用样式设置
                plt.style.use(chart_style)
                sns.set_palette(color_theme)
                plt.rcParams['figure.figsize'] = [chart_width, chart_height]
                plt.rcParams['axes.grid'] = show_grid
                
                # 确保使用最新的processed_df
                df = st.session_state.processed_df.copy()  # 创建副本以避免修改原始数据
                
                # 记录当前数据状态
                current_df_hash = hash(df.to_string())
                
                # 生成可视化建议
                suggestions = st.session_state.visualizer.suggest_visualizations(
                    df,
                    query
                )
                
                # 创建可视化和分析
                figures = []
                analyses = []
                
                # 为每个建议创建独特的可视化
                for i, suggestion in enumerate(suggestions):
                    # 为每个建议添加唯一标识符
                    suggestion['id'] = f"{query}_{i}"
                    
                    # 生成可视化
                    fig = st.session_state.visualizer.create_visualization(
                        suggestion,
                        df
                    )
                    if fig is not None:
                        figures.append(fig)
                        
                        # 生成分析，包含查询上下文
                        analysis = st.session_state.visualizer._generate_visualization_analysis(
                            df,
                            suggestion
                        )
                        analyses.append(analysis)
                
                # 初始化可视化历史记录
                if 'visualization_history' not in st.session_state:
                    st.session_state.visualization_history = []
                
                # 创建新的可视化记录
                new_viz_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': query,
                    'figures_data': [st.session_state.visualizer.fig_to_base64(fig) for fig in figures],  # 将图形转换为base64
                    'analyses': analyses.copy(),
                    'processed_df_hash': current_df_hash,
                    'processed_df': df.copy(),
                    'suggestions': suggestions
                }
                
                # 添加到历史记录
                st.session_state.visualization_history.append(new_viz_record)
                
                # 更新当前可视化结果
                st.session_state.visualization_results = new_viz_record
                
                # 生成HTML可视化页面
                html_content = st.session_state.visualizer.generate_visualization_page(
                    df,
                    query
                )
                
                # 显示可视化结果
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # 添加下载按钮
                st.download_button(
                    "Download Visualization Report",
                    html_content,
                    file_name="visualization_report.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
            finally:
                # 重置样式
                plt.style.use('default')
                
    # 显示历史可视化
    if 'visualization_history' in st.session_state and st.session_state.visualization_history:
        st.markdown("### Visualization History")
        for i, record in enumerate(st.session_state.visualization_history, 1):
            with st.expander(f"Visualization Set {i} - {record['timestamp']} - {record['query']}", expanded=False):
                for j, (fig_data, analysis) in enumerate(zip(record['figures_data'], record['analyses']), 1):
                    st.markdown(f"##### Visualization {j}")
                    # 直接显示base64图像
                    st.markdown(f'<img src="data:image/png;base64,{fig_data}" style="width:100%; max-width:800px; margin:auto; display:block;">', unsafe_allow_html=True)
                    st.markdown("**Analysis:**")
                    st.markdown(analysis, unsafe_allow_html=True)
                st.markdown("---")  # 添加分隔线

def render_ai_analysis_tab():
    """Render AI analysis tab"""
    st.header("AI Analysis")
    
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        st.warning("Please upload data first!")
        return
    
    # 显示当前数据预览
    st.markdown("### Current Data Preview")
    st.dataframe(st.session_state.processed_df.head())
    
    # 显示数据信息
    st.markdown("### Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {st.session_state.processed_df.shape[0]}")
        st.write(f"Columns: {st.session_state.processed_df.shape[1]}")
    with col2:
        st.write("Column Types:")
        st.write(st.session_state.processed_df.dtypes)
    
    # Initialize analysis results in session state if not exists
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Preset questions
    preset_questions = [
        "Select a question...",
        "What are the main patterns in this dataset?",
        "What are the key statistics and their implications?",
        "Are there any notable trends or correlations?",
        "What insights can be drawn about data quality?",
        "What are potential business implications?"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        preset_query = st.selectbox(
            "Choose a preset analysis question:",
            options=preset_questions,
            index=0
        )
    
    with col2:
        custom_query = st.text_input(
            "Or enter your own analysis question:",
            value=st.session_state.analysis_results.get('last_query', '') if st.session_state.analysis_results.get('last_query') and 
                  st.session_state.analysis_results.get('last_query') not in preset_questions else "",
            placeholder="Type your question here..."
        )
    
    if st.button("Start Analysis"):
        analysis_query = custom_query if custom_query else preset_query
        if analysis_query and analysis_query != preset_questions[0]:
            with st.spinner("Generating insights..."):
                try:
                    if st.session_state.insights_generator is None:
                        st.session_state.insights_generator = InsightsGenerator(model="mistral")
                    
                    # 使用具体的查询来生成分析
                    enhanced_query = f"""Analyze the following and provide insights in a structured format:

Query: {analysis_query}

Please provide your analysis in the following sections:
1. Key Patterns and Trends
2. Notable Relationships Between Variables
3. Important Insights About the Data
4. Potential Business Implications
5. Any Anomalies or Points of Interest

Each section should be clearly marked with a heading."""
                    
                    # 使用最新的processed_df
                    df = st.session_state.processed_df.copy()
                    
                    # 记录当前数据状态
                    current_df_hash = hash(df.to_string())
                    
                    insights = st.session_state.insights_generator.generate_insights(
                        df=df,
                        query=enhanced_query,
                        quality_report=st.session_state.processor.detect_data_quality(df)
                    )
                    
                    # 创建新的分析记录
                    new_analysis_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': analysis_query,
                        'insights': insights,
                        'processed_df_hash': current_df_hash,
                        'processed_df': df.copy()  # 保存处理后的数据副本
                    }
                    
                    # 添加到历史记录
                    st.session_state.analysis_history.append(new_analysis_record)
                    
                    # 更新当前分析结果
                    st.session_state.analysis_results = new_analysis_record
                    
                    # 显示分析结果
                    st.markdown("### AI Analysis Results")
                    st.markdown(insights)
                    
                    # 显示数据摘要
                    with st.expander("Data Summary", expanded=False):
                        st.markdown("#### Key Statistics for Numeric Columns")
                        st.dataframe(df.describe())
                        
                        st.markdown("#### Correlation Analysis")
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) > 1:
                            corr_matrix = df[numeric_cols].corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                            st.pyplot(fig)
                            plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.warning("Please make sure Ollama service is running")
        else:
            st.warning("Please select a preset question or enter your own analysis question.")
    
    # 显示历史分析
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.markdown("### Analysis History")
        for i, record in enumerate(st.session_state.analysis_history, 1):
            st.markdown(f"#### Analysis {i} - {record['timestamp']}")
            st.write("Query:", record['query'])
            st.markdown("##### Analysis Results")
            st.markdown(record['insights'])
            st.markdown("##### Data Summary")
            st.dataframe(record['processed_df'].describe())
            st.markdown("---")  # 添加分隔线

def get_binary_file_downloader_html(file_path, file_label='File'):
    """生成文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{os.path.basename(file_path)}" class="download-button">{file_label}</a>'
    return href

def clear_local_cache():
    """清除本地数据缓存"""
    try:
        cache_dir = "processed_data"
        if os.path.exists(cache_dir):
            # 删除目录下的所有文件
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            st.success("Local data cache cleared successfully!")
        else:
            st.info("No local data cache found.")
    except Exception as e:
        st.error(f"Error clearing cache: {str(e)}")

def render_report_tab():
    """Render report generation tab"""
    st.header("Generate Analysis Report")
    
    # Report format selection
    report_format = st.selectbox(
        "Select Report Format",
        options=["HTML", "Markdown"],
        index=0,
        help="HTML format provides better visualization and styling. Markdown is more lightweight and text-focused."
    )
    
    # 添加清除缓存的按钮
    st.markdown("---")
    st.markdown("### Cache Management")
    if st.button("Clear Local Data Cache", help="Delete all processed data files to free up disk space"):
        clear_local_cache()
    
    # 选择要包含在报告中的可视化记录
    st.markdown("### Select Visualizations to Include")
    selected_viz_indices = []
    if 'visualization_history' in st.session_state and st.session_state.visualization_history:
        for i, record in enumerate(st.session_state.visualization_history, 1):
            if st.checkbox(f"Visualization {i} - {record['timestamp']} - {record['query']}", value=True):
                selected_viz_indices.append(i-1)
    
    # 选择要包含在报告中的分析记录
    st.markdown("### Select Analysis to Include")
    selected_analysis_indices = []
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        for i, record in enumerate(st.session_state.analysis_history, 1):
            if st.checkbox(f"Analysis {i} - {record['timestamp']} - {record['query']}", value=True):
                selected_analysis_indices.append(i-1)
    
    # 生成报告按钮
    if st.button("Generate Report"):
        if 'processed_df' not in st.session_state:
            st.error("Please upload data first!")
            return
            
        try:
            with st.spinner("Generating report..."):
                # 使用session state中的处理后的数据
                df = st.session_state.processed_df
                
                # 生成详细的数据质量报告
                quality_report = {
                    'missing_values': {
                        'total': df.isnull().sum().sum(),
                        'by_column': pd.DataFrame({
                            'Missing Count': df.isnull().sum(),
                            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                        })
                    },
                    'data_types': {
                        'numeric': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                        'categorical': df.select_dtypes(include=['object']).columns.tolist(),
                        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
                    },
                    'basic_stats': df.describe(),
                    'unique_values': pd.DataFrame({
                        'Unique Count': df.nunique(),
                        'Unique Percentage': (df.nunique() / len(df) * 100).round(2)
                    }),
                    'duplicates': {
                        'count': df.duplicated().sum(),
                        'percentage': (df.duplicated().sum() / len(df) * 100).round(2)
                    }
                }
                
                # 收集选中的可视化和分析
                figures_data = []
                analyses = []
                analysis_texts = []
                
                # 添加选中的可视化
                if selected_viz_indices:
                    for idx in selected_viz_indices:
                        record = st.session_state.visualization_history[idx]
                        figures_data.extend(record['figures_data'])
                        analyses.extend(record['analyses'])
                
                # 添加选中的分析
                if selected_analysis_indices:
                    for idx in selected_analysis_indices:
                        record = st.session_state.analysis_history[idx]
                        # 创建结构化的分析内容
                        formatted_analysis = f"""
### Analysis Report - {record['timestamp']}

#### Query
{record['query']}

#### Key Findings
{record['insights']}

---
"""
                        analysis_texts.append(formatted_analysis)
                
                # 合并所有分析文本
                analysis_text = "\n\n".join(analysis_texts) if analysis_texts else None
                
                # Generate report content
                report_generator = ReportGenerator()
                
                # Generate report based on format
                if report_format == "HTML":
                    content = report_generator.generate_html_report(
                        df=df,
                        analysis_text=analysis_text,
                        quality_report=quality_report,
                        figures_data=figures_data,
                        analyses=analyses
                    )
                else:
                    content = report_generator.generate_markdown_report(
                        df=df,
                        analysis_text=analysis_text,
                        quality_report=quality_report,
                        figures_data=figures_data,
                        analyses=analyses
                    )
                
                # Get current timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}"
                
                # Save report
                report_path = report_generator.save_report(content, filename, format=report_format)
                
                # Success message
                st.success(f"Report generated successfully!")
                
                # Show report location
                st.code(f"Report saved at: {report_path}", language="bash")
                
                # Add download button for HTML reports
                if report_format == "HTML":
                    st.markdown("### Download Report")
                    st.markdown(get_binary_file_downloader_html(report_path, "Download HTML Report"), unsafe_allow_html=True)
                    
                    # Preview section
                    with st.expander("Preview Report", expanded=True):
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        st.components.v1.html(report_content, height=800, scrolling=True)
                else:
                    st.markdown("### Report Content")
                    with st.expander("Preview Report", expanded=True):
                        with open(report_path, 'r', encoding='utf-8') as f:
                            st.markdown(f.read())
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.error("Please check the data and try again.")

def save_processed_data(df, step_type):
    """Save processed data to local storage"""
    try:
        # Use fixed data directory in Docker environment
        output_dir = "/app/data/processed"
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_type}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save data
        df.to_csv(filepath, index=False)
        
        # Update file path in session state
        st.session_state.data_processing_state['last_saved_file'] = filepath
        
        # Display save information
        st.success(f"Processed data saved to: {filepath}")
        
        # Add download button
        st.markdown("### Download Processed Data")
        st.markdown(get_binary_file_downloader_html(filepath, "Download CSV File"), unsafe_allow_html=True)
        
        return filepath
    except Exception as e:
        st.error(f"Error saving processed data: {str(e)}")
        return None

def load_processed_data():
    """Load recently processed data from local storage"""
    try:
        if 'last_saved_file' in st.session_state.data_processing_state:
            filepath = st.session_state.data_processing_state['last_saved_file']
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
        return None
    except Exception as e:
        st.error(f"Error loading processed data: {str(e)}")
        return None

def handle_missing_values(df, method, fill_value=None):
    """Handle missing values in the dataset"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Processing missing values...")
        
        # Create data copy
        df_processed = df.copy()
        
        # Record pre-processing state
        st.session_state.data_processing_state['current_step'] = {
            'type': 'missing_values',
            'method': method,
            'fill_value': fill_value,
            'timestamp': datetime.now()
        }
        
        if method == "drop":
            # Remove rows with missing values
            df_processed = df_processed.dropna()
        elif method == "fill_mean":
            # Fill with column mean
            df_processed = df_processed.fillna(df_processed.mean())
        elif method == "fill_median":
            # Fill with column median
            df_processed = df_processed.fillna(df_processed.median())
        elif method == "fill_mode":
            # Fill with column mode
            df_processed = df_processed.fillna(df_processed.mode().iloc[0])
        elif method == "fill_constant":
            # Fill with specified value
            if fill_value is not None:
                df_processed = df_processed.fillna(fill_value)
            else:
                st.error("Please specify a fill value")
                return df
        
        progress_bar.progress(100)
        progress_text.text("Missing values processed successfully!")
        
        # Display before and after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before Processing:")
            missing_before = df.isnull().sum()
            st.write(missing_before[missing_before > 0])
        with col2:
            st.write("After Processing:")
            missing_after = df_processed.isnull().sum()
            st.write(missing_after[missing_after > 0])
        
        # Save processed data
        saved_file = save_processed_data(df_processed, "missing_values")
        if saved_file:
            st.success(f"Processed data saved to: {saved_file}")
        
        # Update processed_df in session state
        st.session_state.processed_df = df_processed
        
        # Record processing step
        step_info = {
            'type': 'missing_values',
            'method': method,
            'fill_value': fill_value,
            'timestamp': datetime.now(),
            'data_hash': hash(df_processed.to_string()),
            'saved_file': saved_file
        }
        st.session_state.data_processing_state['processing_steps'].append(step_info)
        
        # Update last modified time and data hash
        st.session_state.data_processing_state['last_modified'] = datetime.now()
        st.session_state.data_processing_state['current_df_hash'] = hash(df_processed.to_string())
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def normalize_data(df, method):
    """Normalize data in the dataset"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Normalizing numeric columns...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for normalization")
            return df
            
        progress_bar.progress(30)
        
        # Create new DataFrame for normalized data
        df_normalized = df.copy()
        
        if method == "minmax":
            # MinMax normalization (0-1 range)
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    st.warning(f"Column {col} has constant value, skipping normalization")
        
        elif method == "standard":
            # Z-score normalization (mean=0, std=1)
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
                else:
                    st.warning(f"Column {col} has constant value, skipping normalization")
        
        elif method == "robust":
            # Quartile-based normalization
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    df_normalized[col] = (df[col] - df[col].median()) / iqr
                else:
                    st.warning(f"Column {col} has too many similar values, skipping normalization")
        
        elif method == "log":
            # Logarithmic transformation
            for col in numeric_cols:
                min_val = df[col].min()
                if min_val <= 0:
                    # Shift data to make all values positive if needed
                    shift = abs(min_val) + 1
                    df_normalized[col] = np.log(df[col] + shift)
                else:
                    df_normalized[col] = np.log(df[col])
        
        progress_bar.progress(100)
        progress_text.text("Data normalization completed!")
        
        # Display statistics before and after normalization
        st.markdown("### Normalization Results")
        for col in numeric_cols:
            st.write(f"\n**Column: {col}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before Normalization:")
                st.write(df[col].describe())
            with col2:
                st.write("After Normalization:")
                st.write(df_normalized[col].describe())
        
        # Save processed data
        saved_file = save_processed_data(df_normalized, "normalization")
        if saved_file:
            st.success(f"Processed data saved to: {saved_file}")
        
        # Update processed_df in session state
        st.session_state.processed_df = df_normalized
        
        # Record processing step
        step_info = {
            'type': 'normalization',
            'method': method,
            'columns': list(numeric_cols),
            'timestamp': datetime.now(),
            'data_hash': hash(df_normalized.to_string()),
            'saved_file': saved_file
        }
        st.session_state.data_processing_state['processing_steps'].append(step_info)
        
        # Update last modified time and data hash
        st.session_state.data_processing_state['last_modified'] = datetime.now()
        st.session_state.data_processing_state['current_df_hash'] = hash(df_normalized.to_string())
        
        return df_normalized
    
    except Exception as e:
        st.error(f"Error normalizing data: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def handle_outliers(df, method, threshold):
    """Handle outliers in the dataset"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Handling outliers...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for outlier handling")
            return df
            
        progress_bar.progress(30)
        
        # Create new DataFrame for processed data
        df_processed = df.copy()
        
        for col in numeric_cols:
            if method == "iqr":
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Replace outliers
                df_processed.loc[df[col] < lower_bound, col] = lower_bound
                df_processed.loc[df[col] > upper_bound, col] = upper_bound
                
            elif method == "zscore":
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_processed.loc[z_scores > threshold, col] = df[col].mean()
                
            elif method == "percentile":
                # Percentile method
                lower_bound = df[col].quantile(threshold/100)
                upper_bound = df[col].quantile(1 - threshold/100)
                df_processed.loc[df[col] < lower_bound, col] = lower_bound
                df_processed.loc[df[col] > upper_bound, col] = upper_bound
        
        progress_bar.progress(100)
        progress_text.text("Outlier handling completed!")
        
        # Display statistics before and after processing
        st.markdown("### Outlier Handling Results")
        for col in numeric_cols:
            st.write(f"\n**Column: {col}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before Processing:")
                st.write(df[col].describe())
            with col2:
                st.write("After Processing:")
                st.write(df_processed[col].describe())
        
        # Save processed data
        saved_file = save_processed_data(df_processed, "outliers")
        if saved_file:
            st.success(f"Processed data saved to: {saved_file}")
        
        # Update processed_df in session state
        st.session_state.processed_df = df_processed
        
        # Record processing step
        step_info = {
            'type': 'outliers',
            'method': method,
            'threshold': threshold,
            'columns': list(numeric_cols),
            'timestamp': datetime.now(),
            'data_hash': hash(df_processed.to_string()),
            'saved_file': saved_file
        }
        st.session_state.data_processing_state['processing_steps'].append(step_info)
        
        # Update last modified time and data hash
        st.session_state.data_processing_state['last_modified'] = datetime.now()
        st.session_state.data_processing_state['current_df_hash'] = hash(df_processed.to_string())
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error handling outliers: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def encode_categorical(df, method):
    """Encode categorical variables in the dataset"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Encoding categorical variables...")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            st.warning("No categorical columns found for encoding")
            return df
            
        progress_bar.progress(30)
        
        # Create new DataFrame for encoded data
        df_encoded = df.copy()
        
        if method == "label":
            # Label encoding
            for col in categorical_cols:
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        elif method == "one_hot":
            # One-hot encoding
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols)
        elif method == "ordinal":
            # Ordinal encoding
            for col in categorical_cols:
                df_encoded[col] = pd.Categorical(df_encoded[col], ordered=True).codes
        
        progress_bar.progress(100)
        progress_text.text("Categorical encoding completed!")
        
        # Display encoding results
        st.markdown("### Encoding Results")
        for col in categorical_cols:
            st.write(f"\n**Column: {col}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before Encoding:")
                st.write(df[col].value_counts())
            with col2:
                st.write("After Encoding:")
                if method == "one_hot":
                    # For one-hot encoding, show new columns
                    new_cols = [c for c in df_encoded.columns if c.startswith(col)]
                    st.write(df_encoded[new_cols].sum())
                else:
                    st.write(df_encoded[col].value_counts())
        
        # Save processed data
        saved_file = save_processed_data(df_encoded, "encoding")
        if saved_file:
            st.success(f"Processed data saved to: {saved_file}")
        
        # Update processed_df in session state
        st.session_state.processed_df = df_encoded
        
        # Record processing step
        step_info = {
            'type': 'categorical_encoding',
            'method': method,
            'columns': list(categorical_cols),
            'timestamp': datetime.now(),
            'data_hash': hash(df_encoded.to_string()),
            'saved_file': saved_file
        }
        st.session_state.data_processing_state['processing_steps'].append(step_info)
        
        # Update last modified time and data hash
        st.session_state.data_processing_state['last_modified'] = datetime.now()
        st.session_state.data_processing_state['current_df_hash'] = hash(df_encoded.to_string())
        
        return df_encoded
    
    except Exception as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def main():
    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'insights_generator' not in st.session_state:
        st.session_state.insights_generator = None
    
    # Maintain visualization and analysis results state
    if 'visualization_history' not in st.session_state:
        st.session_state.visualization_history = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'last_query': '',
            'insights': '',
            'processed_df_hash': None,
            'processed_df': None
        }
    if 'report_status' not in st.session_state:
        st.session_state.report_status = {
            'last_report': None,
            'last_timestamp': None
        }
    # Add data processing state tracking
    if 'data_processing_state' not in st.session_state:
        st.session_state.data_processing_state = {
            'original_data': None,  # Original data backup
            'processing_steps': [],  # Processing steps record
            'last_modified': None,   # Last modification time
            'current_step': None     # Current processing step
        }

    st.title("Automated Data Report Generator")
    
    # Create main tabs
    tabs = st.tabs([
        "Data Upload & Preview",
        "Data Processing",
        "Data Visualization",
        "AI Analysis",
        "Report Generation"
    ])
    
    # Use tab navigation
    with tabs[0]:
        render_data_upload_tab()
    with tabs[1]:
        render_data_processing_tab()
    with tabs[2]:
        render_visualization_tab()
    with tabs[3]:
        render_ai_analysis_tab()
    with tabs[4]:
        render_report_tab()

if __name__ == "__main__":
    main() 