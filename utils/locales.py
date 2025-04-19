from typing import Dict

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'zh']

# Default language
DEFAULT_LANGUAGE = 'en'

# Translation dictionary
TRANSLATIONS = {
    'en': {
        # General
        'app_title': 'Data Analysis Tool',
        'upload_file': 'Upload File',
        'process_data': 'Process Data',
        'generate_report': 'Generate Report',
        'download': 'Download',
        'error': 'Error',
        
        # File upload
        'select_file': 'Select a file',
        'file_types': 'Supported file types: CSV, Excel',
        'upload_success': 'File uploaded successfully',
        'upload_error': 'Error uploading file',
        
        # Data processing
        'missing_values': 'Missing Values',
        'outliers': 'Outliers',
        'normalization': 'Normalization',
        'encoding': 'Encoding',
        'processing_complete': 'Processing complete',
        
        # Analysis
        'data_overview': 'Data Overview',
        'quality_analysis': 'Quality Analysis',
        'visualization': 'Visualization',
        'insights': 'Insights',
        
        # Report
        'report_generated': 'Report generated successfully',
        'report_error': 'Error generating report',
        
        # Download
        'download_report': 'Download Report',
        'download_data': 'Download Data',
        
        # Error messages
        'invalid_file': 'Invalid file format',
        'processing_error': 'Error processing data',
        'analysis_error': 'Error generating analysis'
    },
    'zh': {
        # General
        'app_title': '数据分析工具',
        'upload_file': '上传文件',
        'process_data': '处理数据',
        'generate_report': '生成报告',
        'download': '下载',
        'error': '错误',
        
        # File upload
        'select_file': '选择文件',
        'file_types': '支持的文件类型：CSV、Excel',
        'upload_success': '文件上传成功',
        'upload_error': '文件上传失败',
        
        # Data processing
        'missing_values': '缺失值',
        'outliers': '异常值',
        'normalization': '标准化',
        'encoding': '编码',
        'processing_complete': '处理完成',
        
        # Analysis
        'data_overview': '数据概览',
        'quality_analysis': '质量分析',
        'visualization': '可视化',
        'insights': '洞察',
        
        # Report
        'report_generated': '报告生成成功',
        'report_error': '报告生成失败',
        
        # Download
        'download_report': '下载报告',
        'download_data': '下载数据',
        
        # Error messages
        'invalid_file': '无效的文件格式',
        'processing_error': '数据处理失败',
        'analysis_error': '分析生成失败'
    }
} 