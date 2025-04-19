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

# 设置页面配置
st.set_page_config(
    page_title="Automated Data Report Generator",
    layout="wide"
)

# 隐藏部署按钮和菜单并设置深色主题
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 深色主题基础样式 */
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
    
    /* 整体背景和文字 */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* 标题样式 */
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
    
    /* 子标题样式 */
    h2, h3 {
        color: var(--text-color);
        font-weight: 600 !important;
        margin: 20px 0 10px 0 !important;
    }
    
    /* 标签页样式 */
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
    
    /* 按钮样式 */
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
    
    /* 输入框和选择框样式 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: var(--secondary-bg);
        color: var(--text-color);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    
    /* 数据框样式 */
    .stDataFrame {
        background-color: var(--secondary-bg);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* 警告和错误消息样式 */
    .stAlert {
        background-color: rgba(255, 243, 205, 0.1);
        color: #ffd700;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid rgba(255, 243, 205, 0.2);
    }
    
    /* 上传区域样式 */
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
    
    /* 下载按钮样式 */
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: var(--accent-color);
        color: var(--text-color);
        text-decoration: none;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        transition: background-color 0.3s;
    }
    
    .download-button:hover {
        background-color: var(--hover-color);
        color: var(--text-color);
        text-decoration: none;
    }
    
    /* 代码块样式 */
    .stCodeBlock {
        background-color: var(--secondary-bg);
        border: 1px solid #4e4e4e;
        border-radius: 5px;
    }
    
    /* 响应式调整 */
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
    """数据上传和预览标签页"""
    st.header("Data Upload & Preview")

    # 文件上传
    uploaded_file = st.file_uploader("Upload your data file", type=['csv'])

    if uploaded_file is not None:
        try:
            # 读取数据
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
            
            st.session_state.df = df
            # 保存原始数据备份
            st.session_state.data_processing_state['original_data'] = df.copy()
            st.session_state.data_processing_state['processing_steps'] = []
            st.session_state.data_processing_state['last_modified'] = datetime.now()
            
            st.session_state.processor = DataProcessor()
            st.session_state.visualizer = DataVisualizer()

            # 数据预览
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # 显示数据信息
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
            with col2:
                st.write("Column Types:")
                st.write(df.dtypes)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin.")

def render_data_processing_tab():
    """数据处理标签页"""
    st.header("Data Processing")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload data first!")
        return

    # 显示数据处理历史
    if st.session_state.data_processing_state['processing_steps']:
        with st.expander("Processing History", expanded=False):
            st.markdown("### Applied Processing Steps")
            for step in st.session_state.data_processing_state['processing_steps']:
                st.markdown(f"""
                - **{step['type'].replace('_', ' ').title()}**
                  - Method: {step['method']}
                  - Time: {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
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
            if st.session_state.df is not None:
                missing_stats = pd.DataFrame({
                    'Missing Values': st.session_state.df.isnull().sum(),
                    'Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
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
                    ["drop", "mean", "median", "mode", "constant"],
                    help="""
                    - drop: Remove rows with missing values
                    - mean: Fill missing values with column mean (numeric only)
                    - median: Fill missing values with column median (numeric only)
                    - mode: Fill missing values with column mode
                    - constant: Fill missing values with a specified value
                    """
                )
                
                # 只在选择constant方法时显示填充值输入框
                if missing_method == "constant":
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
                        if missing_method == "constant" and not fill_value:
                            st.error("Please specify a fill value")
                        else:
                            st.session_state.df = handle_missing_values(
                                st.session_state.df,
                                missing_method,
                                fill_value
                            )
                    except Exception as e:
                        st.error(f"Error handling missing values: {str(e)}")
        
        with right_col:
            st.markdown("#### Processing Results")
            if st.session_state.df is not None:
                # 计算处理前后的统计信息
                missing_before = st.session_state.df.isnull().sum()
                total_missing_before = missing_before.sum()
                total_cells = st.session_state.df.size
                
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
                numeric_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
                norm_columns = st.multiselect(
                    "Select columns to normalize",
                    numeric_cols,
                    default=list(numeric_cols),
                    help="Choose the numeric columns you want to normalize"
                )
                
                submit_button = st.form_submit_button("Normalize Data")
                
                if submit_button and norm_columns:
                    try:
                        # 只标准化选中的列
                        df_temp = st.session_state.df.copy()
                        df_temp[norm_columns] = normalize_data(df_temp[norm_columns], norm_method)
                        st.session_state.df = df_temp
                        
                        # 记录处理步骤
                        step_info = {
                            'type': 'normalization',
                            'method': norm_method,
                            'columns': norm_columns,
                            'timestamp': datetime.now()
                        }
                        st.session_state.data_processing_state['processing_steps'].append(step_info)
                        
                        st.success("Data normalized successfully!")
                    except Exception as e:
                        st.error(f"Error normalizing data: {str(e)}")
        
        with right_col:
            if norm_columns:
                st.markdown("#### Column Statistics")
                # 为选中的列创建描述性统计图表
                for col in norm_columns:
                    with st.expander(f"{col} Distribution", expanded=False):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                        
                        # 原始数据分布
                        sns.histplot(st.session_state.df[col], ax=ax1)
                        ax1.set_title("Original Distribution")
                        
                        # 标准化后的分布
                        if st.session_state.df is not None:
                            sns.histplot(st.session_state.df[col], ax=ax2)
                            ax2.set_title("Normalized Distribution")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # 显示基本统计信息
                        st.markdown("**Basic Statistics:**")
                        st.write(st.session_state.df[col].describe())

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
                    ["zscore", "iqr"],
                    help="""
                    - zscore: Use Z-score method (standard deviations from mean)
                    - iqr: Use Interquartile Range method
                    """
                )
                
                outlier_threshold = st.number_input(
                    "Threshold",
                    value=3.0,
                    step=0.1,
                    help="Threshold for outlier detection (Z-score or IQR multiplier)"
                )
                
                # 选择要处理的列
                numeric_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
                outlier_columns = st.multiselect(
                    "Select columns to check for outliers",
                    numeric_cols,
                    default=list(numeric_cols)
                )
                
                submit_button = st.form_submit_button("Handle Outliers")
                
                if submit_button and outlier_columns:
                    try:
                        df_temp = st.session_state.df.copy()
                        df_temp = handle_outliers(
                            df_temp,
                            outlier_method,
                            outlier_threshold
                        )
                        st.session_state.df = df_temp
                        
                        # 记录处理步骤
                        step_info = {
                            'type': 'outliers',
                            'method': outlier_method,
                            'threshold': outlier_threshold,
                            'timestamp': datetime.now()
                        }
                        st.session_state.data_processing_state['processing_steps'].append(step_info)
                        
                        st.success("Outliers handled successfully!")
                    except Exception as e:
                        st.error(f"Error handling outliers: {str(e)}")
        
        with right_col:
            if outlier_columns:
                st.markdown("#### Outlier Analysis")
                for col in outlier_columns:
                    with st.expander(f"{col} Outliers", expanded=False):
                        # 创建箱线图
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.boxplot(data=st.session_state.df[col], ax=ax)
                        plt.title(f"Box Plot for {col}")
                        st.pyplot(fig)
                        plt.close()
                        
                        # 显示异常值统计
                        q1 = st.session_state.df[col].quantile(0.25)
                        q3 = st.session_state.df[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = st.session_state.df[
                            (st.session_state.df[col] < (q1 - 1.5 * iqr)) |
                            (st.session_state.df[col] > (q3 + 1.5 * iqr))
                        ][col]
                        
                        st.markdown(f"""
                        **Outlier Statistics:**
                        - Total values: {len(st.session_state.df[col])}
                        - Outliers found: {len(outliers)}
                        - Outlier percentage: {(len(outliers) / len(st.session_state.df[col]) * 100):.2f}%
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
                categorical_cols = st.session_state.df.select_dtypes(include=['object']).columns
                encode_columns = st.multiselect(
                    "Select columns to encode",
                    categorical_cols,
                    default=list(categorical_cols),
                    help="Choose the categorical columns you want to encode"
                )
                
                submit_button = st.form_submit_button("Encode Variables")
                
                if submit_button and encode_columns:
                    try:
                        df_temp = st.session_state.df.copy()
                        df_temp = encode_categorical(
                            df_temp,
                            encode_method
                        )
                        st.session_state.df = df_temp
                        
                        # 记录处理步骤
                        step_info = {
                            'type': 'categorical_encoding',
                            'method': encode_method,
                            'columns': encode_columns,
                            'timestamp': datetime.now()
                        }
                        st.session_state.data_processing_state['processing_steps'].append(step_info)
                        
                        st.success("Categorical variables encoded successfully!")
                    except Exception as e:
                        st.error(f"Error encoding variables: {str(e)}")
        
        with right_col:
            if encode_columns:
                st.markdown("#### Encoding Preview")
                for col in encode_columns:
                    with st.expander(f"{col} Encoding", expanded=False):
                        # 显示原始类别分布
                        st.markdown("**Original Categories:**")
                        st.write(st.session_state.df[col].value_counts())
                        
                        # 可视化类别分布
                        fig, ax = plt.subplots(figsize=(8, 4))
                        st.session_state.df[col].value_counts().plot(kind='bar')
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
                    st.session_state.df.to_csv(output_path, index=False, encoding=encoding)
                elif export_format == "Excel":
                    output_path = os.path.join(output_dir, f"{file_name}_{timestamp}.xlsx")
                    st.session_state.df.to_excel(output_path, index=False)
                else:  # JSON
                    output_path = os.path.join(output_dir, f"{file_name}_{timestamp}.json")
                    st.session_state.df.to_json(output_path, orient='records', force_ascii=False)
                
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
    
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
        
    # 创建可视化器实例
    if st.session_state.visualizer is None:
        st.session_state.visualizer = DataVisualizer()
    
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
                
                # 生成可视化建议
                suggestions = st.session_state.visualizer.suggest_visualizations(
                    st.session_state.df,
                    query
                )
                
                # 创建可视化和分析
                figures = []
                analyses = []
                
                for suggestion in suggestions:
                    fig = st.session_state.visualizer.create_visualization(
                        suggestion,
                        st.session_state.df
                    )
                    if fig is not None:
                        figures.append(fig)
                        
                        # 生成分析
                        analysis = st.session_state.visualizer._generate_visualization_analysis(
                            st.session_state.df,
                            suggestion
                        )
                        analyses.append(analysis)
                
                # 保存到session state
                st.session_state.visualization_results = {
                    'figures': figures.copy(),  # 创建副本以避免引用问题
                    'analyses': analyses.copy(),
                    'last_query': query
                }
                
                # 生成HTML可视化页面
                html_content = st.session_state.visualizer.generate_visualization_page(
                    st.session_state.df,
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
    if 'visualization_results' in st.session_state and st.session_state.visualization_results.get('figures'):
        with st.expander("Previous Visualizations"):
            st.write("Last query:", st.session_state.visualization_results.get('last_query', 'N/A'))
            st.write(f"Number of visualizations: {len(st.session_state.visualization_results['figures'])}")
            
            # 显示保存的可视化
            for i, (fig, analysis) in enumerate(zip(
                st.session_state.visualization_results['figures'],
                st.session_state.visualization_results['analyses']
            ), 1):
                st.markdown(f"### Visualization {i}")
                st.pyplot(fig)
                st.markdown("**Analysis:**")
                st.markdown(analysis, unsafe_allow_html=True)

def render_ai_analysis_tab():
    """Render AI analysis tab"""
    st.header("AI Analysis")
    
    # Initialize analysis results in session state if not exists
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'insights': '',
            'last_query': ''
        }
    
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
            value=st.session_state.analysis_results['last_query'] if st.session_state.analysis_results['last_query'] and 
                  st.session_state.analysis_results['last_query'] not in preset_questions else "",
            placeholder="Type your question here..."
        )
    
    if st.button("Start Analysis"):
        analysis_query = custom_query if custom_query else preset_query
        if analysis_query and analysis_query != preset_questions[0]:
            with st.spinner("Generating insights..."):
                try:
                    if st.session_state.insights_generator is None:
                        st.session_state.insights_generator = InsightsGenerator(model="mistral")
                    
                    enhanced_query = f"Analyze the following and provide insights in English: {analysis_query}"
                    
                    insights = st.session_state.insights_generator.generate_insights(
                        df=st.session_state.df,
                        query=enhanced_query,
                        quality_report=st.session_state.processor.detect_data_quality(st.session_state.df)
                    )
                    
                    st.session_state.analysis_results['insights'] = insights
                    st.session_state.analysis_results['last_query'] = analysis_query
                    
                    st.markdown("### AI Analysis Results")
                    st.markdown(insights)
                    
                    st.markdown("### Data Summary")
                    st.markdown("Key statistics for numeric columns:")
                    st.dataframe(st.session_state.df.describe())
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.warning("Please make sure Ollama service is running")
        else:
            st.warning("Please select a preset question or enter your own analysis question.")
    
    elif st.session_state.analysis_results['insights']:
        st.markdown("### Previous Analysis Results")
        st.markdown(st.session_state.analysis_results['insights'])
        st.markdown("### Data Summary")
        st.markdown("Key statistics for numeric columns:")
        st.dataframe(st.session_state.df.describe())

def get_binary_file_downloader_html(file_path, file_label='File'):
    """生成文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{os.path.basename(file_path)}" class="download-button">{file_label}</a>'
    return href

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
    
    if st.button("Generate Report"):
        if 'df' not in st.session_state:
            st.error("Please upload data first!")
            return
            
        try:
            with st.spinner("Generating report..."):
                # Get data quality report
                quality_report = st.session_state.processor.detect_data_quality(st.session_state.df)
                
                # Get AI insights
                if st.session_state.insights_generator is None:
                    st.session_state.insights_generator = InsightsGenerator(model="mistral")
                
                analysis_text = st.session_state.insights_generator.generate_insights(
                    df=st.session_state.df,
                    query="Provide a comprehensive analysis of this dataset, including key patterns, trends, and insights.",
                    quality_report=quality_report
                )
                
                # Initialize visualizer if needed
                if st.session_state.visualizer is None:
                    st.session_state.visualizer = DataVisualizer()
                
                # Get or generate visualizations and their analyses
                figures = []
                analyses = []
                
                # First try to get existing visualizations from session state
                if 'visualization_results' in st.session_state and st.session_state.visualization_results:
                    if 'figures' in st.session_state.visualization_results:
                        figures = st.session_state.visualization_results['figures']
                        analyses = st.session_state.visualization_results.get('analyses', [])
                
                # If no visualizations exist, generate default ones
                if not figures:
                    # Get visualization suggestions
                    suggestions = st.session_state.visualizer.suggest_visualizations(
                        st.session_state.df,
                        "Suggest key visualizations to understand the main patterns and relationships in the data"
                    )
                    
                    # Create visualizations from suggestions
                    for suggestion in suggestions:
                        fig = st.session_state.visualizer.create_visualization(
                            suggestion,
                            st.session_state.df
                        )
                        if fig is not None:
                            figures.append(fig)
                            
                            # Generate analysis for this visualization
                            viz_analysis = st.session_state.visualizer._generate_visualization_analysis(
                                st.session_state.df,
                                suggestion
                            )
                            analyses.append(viz_analysis)
                
                # Store the visualizations and analyses in session state
                st.session_state.visualization_results = {
                    'figures': figures,
                    'analyses': analyses
                }
                
                # Generate report
                report_generator = ReportGenerator()
                
                # Get current timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}"
                
                # Generate report based on selected format
                format_ext = 'html' if report_format == "HTML" else 'md'
                if format_ext == 'html':
                    content = report_generator.generate_html_report(
                        st.session_state.df,
                        analysis_text,
                        quality_report,
                        figures,
                        analyses
                    )
                else:
                    content = report_generator.generate_markdown_report(
                        st.session_state.df,
                        analysis_text,
                        quality_report,
                        figures,
                        analyses
                    )
                
                # Save report
                report_path = report_generator.save_report(content, filename, format=format_ext)
                
                # Success message
                st.success(f"Report generated successfully!")
                
                # Show report location
                st.code(f"Report saved at: {report_path}", language="bash")
                
                # Add download button for HTML reports
                if format_ext == 'html':
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

def handle_missing_values(df, method, fill_value=None):
    """处理缺失值"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Processing missing values...")
        progress_bar.progress(30)
        
        # 记录处理步骤
        step_info = {
            'type': 'missing_values',
            'method': method,
            'fill_value': fill_value,
            'timestamp': datetime.now()
        }
        
        if method == "drop":
            df = df.dropna()
            message = "Rows with missing values have been dropped"
        elif method == "mean":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
            message = "Missing values filled with column means"
        elif method == "median":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            message = "Missing values filled with column medians"
        elif method == "mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            message = "Missing values filled with column modes"
        elif method == "constant" and fill_value is not None:
            try:
                numeric_value = float(fill_value)
                df = df.fillna(numeric_value)
                message = f"Missing values filled with constant value: {fill_value}"
            except ValueError:
                df = df.fillna(str(fill_value))
                message = f"Missing values filled with constant value: {fill_value}"
        else:
            raise ValueError("Must specify a valid method and fill value if using constant method")
        
        # 更新处理状态
        st.session_state.data_processing_state['processing_steps'].append(step_info)
        st.session_state.data_processing_state['last_modified'] = datetime.now()
        
        progress_bar.progress(100)
        progress_text.text("Missing values handled successfully!")
        
        # 显示处理结果统计
        st.markdown("### Missing Values Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before processing:")
            st.write(pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            }))
            
        with col2:
            st.write("After processing:")
            st.write(pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            }))
            
        st.success(message)
        return df
        
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def normalize_data(df, method):
    """标准化数据"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Normalizing numeric columns...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for normalization")
            return df
            
        progress_bar.progress(30)
        
        # 创建一个新的DataFrame来存储标准化后的数据
        df_normalized = df.copy()
        
        if method == "minmax":
            # MinMax标准化 (0-1区间)
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    st.warning(f"Column {col} has constant value, skipping normalization")
        
        elif method == "standard":
            # Z-score标准化 (均值0，标准差1)
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
                else:
                    st.warning(f"Column {col} has constant value, skipping normalization")
        
        elif method == "robust":
            # 基于四分位数的标准化
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    df_normalized[col] = (df[col] - df[col].median()) / iqr
                else:
                    st.warning(f"Column {col} has too many similar values, skipping normalization")
        
        elif method == "log":
            # 对数转换
            for col in numeric_cols:
                min_val = df[col].min()
                if min_val <= 0:
                    # 如果有非正值，先平移数据使所有值为正
                    shift = abs(min_val) + 1
                    df_normalized[col] = np.log(df[col] + shift)
                else:
                    df_normalized[col] = np.log(df[col])
        
        progress_bar.progress(100)
        progress_text.text("Data normalization completed!")
        
        # 显示标准化前后的统计信息
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
        
        return df_normalized
    
    except Exception as e:
        st.error(f"Error normalizing data: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def handle_outliers(df, method, threshold=None):
    """处理异常值"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Detecting outliers...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for outlier detection")
            return df
            
        progress_bar.progress(30)
        
        if method == "zscore":
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            df = df[(z_scores < threshold).all(axis=1)]
        elif method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        progress_bar.progress(100)
        progress_text.text("Outlier handling completed!")
        return df
    except Exception as e:
        st.error(f"Error handling outliers: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def encode_categorical(df, method):
    """编码分类变量"""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    try:
        progress_text.text("Encoding categorical columns...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            st.warning("No categorical columns found for encoding")
            return df
            
        progress_bar.progress(30)
        
        if method == "label":
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])
        elif method == "onehot":
            df = pd.get_dummies(df, columns=categorical_cols)
        
        progress_bar.progress(100)
        progress_text.text("Categorical encoding completed!")
        return df
    except Exception as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        return df
    finally:
        progress_bar.empty()
        progress_text.empty()

def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'insights_generator' not in st.session_state:
        st.session_state.insights_generator = None
    
    # 维护可视化和分析结果的状态
    if 'visualization_results' not in st.session_state:
        st.session_state.visualization_results = {
            'suggestions': None,
            'figures': [],
            'analyses': [],
            'last_query': None
        }
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'insights': None,
            'last_query': None
        }
    if 'report_status' not in st.session_state:
        st.session_state.report_status = {
            'last_report': None,
            'last_timestamp': None
        }
    # 添加数据处理状态跟踪
    if 'data_processing_state' not in st.session_state:
        st.session_state.data_processing_state = {
            'original_data': None,  # 原始数据备份
            'processing_steps': [],  # 处理步骤记录
            'last_modified': None    # 最后修改时间
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