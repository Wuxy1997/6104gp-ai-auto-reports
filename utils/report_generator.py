import pandas as pd
from typing import Dict, List, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import numpy as np
from utils.locales import TRANSLATIONS, DEFAULT_LANGUAGE

class ReportGenerator:
    """Report generation class with enhanced visualization support"""
    
    def __init__(self):
        """Initialize report generator"""
        self.output_dir = "reports"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Translation dictionary for report sections
        self.translations = {
            'overview': 'Overview',
            'quality': 'Data Quality Assessment',
            'analysis': 'Analysis Insights',
            'visualizations': 'Data Visualizations'
        }
    
    def save_visualization(self, fig, filename: str) -> str:
        """Save visualization to file and return relative path"""
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            
        filepath = os.path.join(viz_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return os.path.relpath(filepath, self.output_dir)

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

    def generate_markdown_report(self, df: pd.DataFrame, analysis_text: str, quality_report: dict,
                               figures_data: List[str] = None, analyses: List[str] = None) -> str:
        """Generate markdown report with visualizations"""
        report_parts = []
        
        # Title
        report_parts.extend([
            "# Data Analysis Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        ])
        
        # Overview section
        report_parts.extend([
            "## 1. Data Overview",
            f"- Number of records: {len(df)}",
            f"- Number of features: {len(df.columns)}",
            "\nFeature Information:",
        ])
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            report_parts.append(f"- {col} ({dtype})")
            report_parts.append(f"  - Missing values: {missing}")
            report_parts.append(f"  - Unique values: {unique}")
        
        # Quality Assessment section
        if quality_report:
            report_parts.extend([
                "\n## 2. Data Quality Assessment",
                "### Missing Values",
                f"- Total missing values: {quality_report.get('total_missing', 0)}",
                f"- Columns with missing values: {', '.join(quality_report.get('columns_with_missing', []))}",
                "\n### Data Types",
                f"- Numeric columns: {', '.join(quality_report.get('numeric_columns', []))}",
                f"- Categorical columns: {', '.join(quality_report.get('categorical_columns', []))}",
                f"- DateTime columns: {', '.join(quality_report.get('datetime_columns', []))}"
            ])
        
        # Analysis Insights section
        if analysis_text:
            report_parts.extend([
                "\n## 3. Analysis Insights",
                analysis_text
            ])
        
        # Visualizations section
        if figures_data and analyses:
            report_parts.append("\n## 4. Data Visualizations")
            
            for i, (fig_data, analysis) in enumerate(zip(figures_data, analyses), 1):
                # Add visualization and its analysis
                report_parts.extend([
                    f"\n### Visualization {i}",
                    f"\n![Visualization {i}](data:image/png;base64,{fig_data})\n",
                    "\n**Analysis:**\n",
                    analysis.replace('<ul class="analysis-points">', '').replace('</ul>', '')
                    .replace('<li>', '- ').replace('</li>', '')
                ])
        
        return "\n".join(report_parts)

    def generate_html_report(self, df: pd.DataFrame, analysis_text: str, quality_report: dict,
                           figures_data: List[str] = None, analyses: List[str] = None) -> str:
        """Generate HTML report with visualizations"""
        
        # CSS styles
        css = """
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3, h4 {
                color: #2c3e50;
                margin-top: 30px;
            }
            .section {
                margin: 30px 0;
                padding: 20px;
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .visualization {
                margin: 30px 0;
                padding: 20px;
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .visualization img {
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }
            .analysis {
                margin: 20px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 4px;
                line-height: 1.6;
            }
            .analysis-points {
                margin: 10px 0;
                padding-left: 20px;
            }
            .analysis-points li {
                margin: 8px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            th, td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background: #f8f9fa;
            }
            .quality-metric {
                margin: 10px 0;
                padding: 10px;
                background: #e9ecef;
                border-radius: 4px;
            }
        </style>
        """
        
        # Start HTML document
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Data Analysis Report</title>",
            css,
            "</head>",
            "<body>",
            
            # Title
            "<h1>Data Analysis Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            
            # Overview section
            "<div class='section'>",
            "<h2>1. Data Overview</h2>",
            f"<p>Number of records: {len(df)}<br>",
            f"Number of features: {len(df.columns)}</p>",
            "<h3>Feature Information:</h3>",
            "<table>",
            "<tr><th>Column</th><th>Type</th><th>Missing Values</th><th>Unique Values</th></tr>"
        ]
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            html_parts.append(f"<tr><td>{col}</td><td>{dtype}</td><td>{missing}</td><td>{unique}</td></tr>")
        
        html_parts.append("</table></div>")
        
        # Quality Assessment section
        if quality_report:
            html_parts.extend([
                "<div class='section'>",
                "<h2>2. Data Quality Assessment</h2>",
                "<div class='quality-metric'>",
                "<h3>Missing Values</h3>",
                f"<p>Total missing values: {quality_report.get('missing_values', {}).get('total', 0)}</p>",
                "<table>",
                "<tr><th>Column</th><th>Missing Count</th><th>Missing Percentage</th></tr>"
            ])
            
            # Add missing values table
            missing_values = quality_report.get('missing_values', {}).get('by_column', {})
            for col, stats in missing_values.items():
                if isinstance(stats, dict) and stats.get('Missing Count', 0) > 0:
                    html_parts.append(
                        f"<tr><td>{col}</td><td>{stats.get('Missing Count', 0)}</td><td>{stats.get('Missing Percentage', 0):.2f}%</td></tr>"
                    )
            
            html_parts.extend([
                "</table>",
                "</div>",
                "<div class='quality-metric'>",
                "<h3>Data Types</h3>",
                "<table>",
                "<tr><th>Type</th><th>Columns</th></tr>"
            ])
            
            # Add data types table
            data_types = quality_report.get('data_types', {})
            for dtype, cols in data_types.items():
                if isinstance(cols, list) and cols:
                    html_parts.append(f"<tr><td>{dtype}</td><td>{', '.join(cols)}</td></tr>")
            
            html_parts.extend([
                "</table>",
                "</div>",
                "</div>"
            ])
        
        # Analysis Insights section
        if analysis_text:
            html_parts.extend([
                "<div class='section'>",
                "<h2>3. Analysis Insights</h2>",
                "<div class='analysis'>",
                analysis_text.replace("### ", "<h3>").replace(" ###", "</h3>")
                          .replace("## ", "<h2>").replace(" ##", "</h2>")
                          .replace("# ", "<h1>").replace(" #", "</h1>")
                          .replace("\n", "<br>"),
                "</div>",
                "</div>"
            ])
        
        # Visualizations section
        if figures_data and analyses:
            html_parts.extend([
                "<div class='section'>",
                "<h2>4. Data Visualizations</h2>"
            ])
            
            for i, (fig_data, analysis) in enumerate(zip(figures_data, analyses), 1):
                html_parts.extend([
                    "<div class='visualization'>",
                    f"<h3>Visualization {i}</h3>",
                    f"<img src='data:image/png;base64,{fig_data}' alt='Visualization {i}'>",
                    "<div class='analysis'>",
                    analysis,
                    "</div>",
                    "</div>"
                ])
            
            html_parts.append("</div>")
        
        # Close HTML document
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)

    def save_report(self, content: str, filename: str, format: str = 'html') -> str:
        """Save report to file and return file path"""
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return filepath

    def get_translation(self, key: str) -> str:
        """Get translation for a key"""
        return self.translations.get(key, key)
    
    def convert_numpy_types(self, data: Dict) -> Dict:
        """Convert numpy types to Python native types"""
        converted = {}
        for k, v in data.items():
            if isinstance(v, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
                converted[k] = int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                converted[k] = float(v)
            elif isinstance(v, (np.ndarray, list)):
                converted[k] = [self.convert_numpy_types(x) if isinstance(x, dict)
                              else x for x in v]
            elif isinstance(v, dict):
                converted[k] = self.convert_numpy_types(v)
            else:
                converted[k] = v
        return converted

    def _generate_ai_analysis_section(self, insights: List[Dict]) -> str:
        """Generate AI analysis section of the report"""
        if not insights:
            return ""
            
        sections = []
        
        # Direct Response
        if insights[0].get('direct_response'):
            sections.append("1. Direct Response:")
            sections.append(insights[0]['direct_response'])
            sections.append("")
            
        # Key Findings
        if insights[0].get('key_findings'):
            sections.append("2. Key Findings:")
            sections.append(insights[0]['key_findings'])
            sections.append("")
            
        # Trends and Patterns
        if insights[0].get('trends'):
            sections.append("3. Trends and Patterns:")
            sections.append(insights[0]['trends'])
            sections.append("")
            
        # Recommendations
        if insights[0].get('recommendations'):
            sections.append("4. Recommendations:")
            sections.append(insights[0]['recommendations'])
            sections.append("")
            
        return "\n".join(sections)

    def _generate_ai_analysis_section_html(self, insights: List[Dict]) -> str:
        """Generate AI analysis section in HTML format"""
        if not insights:
            return ""
            
        sections = []
        sections.append('<div class="section">')
        sections.append('<h2>AI Analysis Insights</h2>')
        
        # Direct Response
        if insights[0].get('direct_response'):
            sections.append('<div class="subsection">')
            sections.append('<h3>1. Direct Response</h3>')
            sections.append(f'<p>{insights[0]["direct_response"]}</p>')
            sections.append('</div>')
            
        # Key Findings
        if insights[0].get('key_findings'):
            sections.append('<div class="subsection">')
            sections.append('<h3>2. Key Findings</h3>')
            sections.append(f'<p>{insights[0]["key_findings"]}</p>')
            sections.append('</div>')
            
        # Trends and Patterns
        if insights[0].get('trends'):
            sections.append('<div class="subsection">')
            sections.append('<h3>3. Trends and Patterns</h3>')
            sections.append(f'<p>{insights[0]["trends"]}</p>')
            sections.append('</div>')
            
        # Recommendations
        if insights[0].get('recommendations'):
            sections.append('<div class="subsection">')
            sections.append('<h3>4. Recommendations</h3>')
            sections.append(f'<p>{insights[0]["recommendations"]}</p>')
            sections.append('</div>')
            
        sections.append('</div>')
        return "\n".join(sections) 