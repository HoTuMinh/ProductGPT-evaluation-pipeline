"""
Simple Report Generator - Data-focused, minimal styling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional
import io
import os
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimal plot style - grayscale
plt.style.use('default')
plt.rcParams['figure.figsize'] = (7, 3.5)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


class ReportGenerator:
    """Generate simple, data-focused evaluation reports"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.styles = getSampleStyleSheet()
        
        # Minimal styles
        self.title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.black,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.heading_style = ParagraphStyle(
            'Heading',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.black,
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
    
    def generate_simple_metric_comparison(self, all_results: Dict[str, pd.DataFrame]) -> str:
        """Generate simple bar chart comparing metrics"""
        try:
            fig, ax = plt.subplots(figsize=(7, 3))
            
            metrics = list(all_results.keys())
            scores = [results['score'].mean() for results in all_results.values()]
            
            bars = ax.bar(metrics, scores, color='#666666', edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Average Score', fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.set_title('Metric Comparison', fontsize=11, fontweight='bold')
            
            # Capitalize metric names
            ax.set_xticklabels([m.capitalize() for m in metrics])
            
            plt.tight_layout()
            
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'metric_comp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating metric comparison: {e}")
            return None
    
    def generate_simple_distribution(self, results_df: pd.DataFrame, metric: str) -> str:
        """Generate simple histogram of score distribution"""
        try:
            fig, ax = plt.subplots(figsize=(7, 3))
            
            scores = results_df['score'].dropna()
            
            # Simple histogram with 5 bins
            ax.hist(scores, bins=5, range=(0, 1), color='#888888', edgecolor='black', linewidth=0.5)
            
            # Add mean line
            mean_score = scores.mean()
            ax.axvline(mean_score, color='black', linestyle='--', linewidth=1.5, 
                      label=f'Mean: {mean_score:.3f}')
            
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Score Distribution ({metric.capitalize()})', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
            
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'dist_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating distribution: {e}")
            return None
    
    def generate_label_distribution(self, results_df: pd.DataFrame, metric: str) -> str:
        """Generate simple bar chart of label distribution"""
        try:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            
            label_counts = results_df['label'].value_counts()
            
            bars = ax.bar(label_counts.index, label_counts.values, 
                         color='#777777', edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlabel('Label', fontsize=10)
            ax.set_title(f'Label Distribution ({metric.capitalize()})', fontsize=11, fontweight='bold')
            
            # Capitalize labels
            ax.set_xticklabels([str(l).capitalize() for l in label_counts.index])
            
            plt.tight_layout()
            
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'labels_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating label distribution: {e}")
            return None
    
    def generate_pdf_report(
        self,
        output_path: str,
        run_info: Dict,
        all_results: Dict[str, pd.DataFrame],
        api_stats: Dict = None
    ):
        """Generate simple PDF report with essential data only"""
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title
        story.append(Paragraph("ProductGPT Evaluation Report", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Run Information
        story.append(Paragraph("Run Information", self.heading_style))
        
        run_data = [
            ['Field', 'Value'],
            ['Timestamp', run_info.get('timestamp', 'N/A')],
            ['User', run_info.get('user', 'N/A')],
            ['Input File', run_info.get('input_file_name', 'N/A')],
            ['Total Rows', str(run_info.get('total_rows', 0))],
            ['Metrics Evaluated', ', '.join(run_info.get('metrics_evaluated', []))],
            ['Execution Time', f"{run_info.get('execution_time_seconds', 0):.1f}s"],
        ]
        
        if api_stats:
            run_data.append(['Total API Calls', str(api_stats.get('total_calls', 0))])
            run_data.append(['Total Tokens', str(api_stats.get('total_tokens', 0))])
        
        # Add human review info if present
        if 'human_reviews' in run_info:
            run_data.append(['Human Reviews', run_info['human_reviews']])
        
        run_table = Table(run_data, colWidths=[2*inch, 4.5*inch])
        run_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(run_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Results Summary
        story.append(Paragraph("Results Summary", self.heading_style))
        
        summary_data = [['Metric', 'Score', 'Pass Rate', 'Samples']]
        for metric, results_df in all_results.items():
            avg_score = results_df['score'].mean()
            pass_count = (results_df['label'] == 'positive').sum()
            pass_rate = pass_count / len(results_df) * 100
            summary_data.append([
                metric.capitalize(),
                f"{avg_score:.3f}",
                f"{pass_rate:.0f}%",
                str(len(results_df))
            ])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Simple bar chart comparing metrics
        if len(all_results) > 1:
            comp_path = self.generate_simple_metric_comparison(all_results)
            if comp_path and os.path.exists(comp_path):
                img = Image(comp_path, width=5.5*inch, height=2.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
        
        # Detailed Results for Each Metric
        for metric_idx, (metric, results_df) in enumerate(all_results.items()):
            story.append(Paragraph(f"Detailed Results: {metric.capitalize()}", self.heading_style))
            
            # Basic statistics
            pass_count = (results_df['label'] == 'positive').sum()
            pass_rate = pass_count / len(results_df) * 100
            
            stats_text = f"""
            <b>Total Samples:</b> {len(results_df)}<br/>
            <b>Average Score:</b> {results_df['score'].mean():.3f}<br/>
            <b>Pass Rate:</b> {pass_rate:.0f}% ({pass_count}/{len(results_df)})<br/>
            """
            story.append(Paragraph(stats_text, self.styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
            
            # Score distribution
            story.append(Paragraph("Score Distribution:", self.styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Text-based distribution
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            hist, _ = np.histogram(results_df['score'], bins=bins)
            
            dist_data = [['Score Range', 'Count', 'Percentage']]
            for label, count in zip(bin_labels, hist):
                pct = (count / len(results_df)) * 100
                dist_data.append([label, str(count), f"{pct:.0f}%"])
            
            dist_table = Table(dist_data, colWidths=[1.5*inch, 1*inch, 1.5*inch])
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(dist_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Chart
            dist_path = self.generate_simple_distribution(results_df, metric)
            if dist_path and os.path.exists(dist_path):
                img = Image(dist_path, width=5.5*inch, height=2.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            
            # Label distribution
            label_path = self.generate_label_distribution(results_df, metric)
            if label_path and os.path.exists(label_path):
                img = Image(label_path, width=4*inch, height=2*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            
            # Lowest scoring samples (top 3)
            story.append(Paragraph("Lowest Scoring Samples:", self.styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            worst_samples = results_df.nsmallest(min(3, len(results_df)), 'score')
            for idx, (row_idx, row) in enumerate(worst_samples.iterrows(), 1):
                # Get original data if available (assuming they're in the dataframe)
                question = row.get('question', 'N/A')
                response = row.get('response', 'N/A')
                benchmark = row.get('benchmark', row.get('benchmark_answer', 'N/A'))
                reasoning = row.get('reasoning', 'N/A')
                
                # Check for human review
                has_human_review = row.get('human_reviewed', False) or row.get('human_score') is not None
                
                # Build header with score info
                if has_human_review:
                    header = f"<b>Sample #{idx}</b> (LLM Score: {row['score']:.3f} → Human Score: {row.get('human_score', 'N/A'):.3f})<br/>"
                else:
                    header = f"<b>Sample #{idx}</b> (Score: {row['score']:.3f})<br/>"
                
                sample_text = f"""
                {header}
                <br/>
                <b>Question:</b><br/>
                {str(question)[:300]}{'...' if len(str(question)) > 300 else ''}<br/>
                <br/>
                <b>Model Response:</b><br/>
                {str(response)[:300]}{'...' if len(str(response)) > 300 else ''}<br/>
                <br/>
                <b>Expected Answer:</b><br/>
                {str(benchmark)[:300]}{'...' if len(str(benchmark)) > 300 else ''}<br/>
                <br/>
                <b>LLM Evaluation:</b><br/>
                {str(reasoning)[:200]}{'...' if len(str(reasoning)) > 200 else ''}<br/>
                """
                
                # Add human review section if available
                if has_human_review:
                    human_label = row.get('human_label', 'N/A')
                    human_comment = row.get('human_comment', '')
                    sample_text += f"""
                    <br/>
                    <b>Human Review:</b><br/>
                    Label: {human_label}<br/>
                    """
                    if human_comment:
                        sample_text += f"Comment: {str(human_comment)[:200]}{'...' if len(str(human_comment)) > 200 else ''}<br/>"
                
                story.append(Paragraph(sample_text, self.styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
                
                # Add separator line between samples
                if idx < len(worst_samples):
                    story.append(Paragraph("─" * 80, self.styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            if metric_idx < len(all_results) - 1:  # Not last metric
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        logger.info(f"Report generated: {output_path}")
        return output_path
