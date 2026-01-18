"""
Report Generator with PDF export and comprehensive visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import io
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ReportGenerator:
    """Generate comprehensive evaluation reports with visualizations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
    
    def generate_confusion_matrix(
        self,
        results_df: pd.DataFrame,
        metric: str,
        save_path: str = None
    ) -> str:
        """
        Generate confusion matrix comparing human labels vs LLM judge labels
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric name
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Check if human labels exist
        if 'feedback_label' in results_df.columns:
            human_labels = results_df['feedback_label'].fillna('unknown')
            llm_labels = results_df['label']
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            labels = sorted(set(human_labels) | set(llm_labels))
            cm = confusion_matrix(human_labels, llm_labels, labels=labels)
            
            # Plot
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('LLM Judge Labels', fontsize=12)
            ax.set_ylabel('Human Labels', fontsize=12)
            ax.set_title(f'Confusion Matrix: Human vs. LLM ({metric})', fontsize=14, fontweight='bold')
        else:
            # If no human labels, show LLM label distribution
            label_counts = results_df['label'].value_counts()
            label_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Label', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Label Distribution ({metric})', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'confusion_matrix_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_score_distribution(
        self,
        results_df: pd.DataFrame,
        metric: str,
        save_path: str = None
    ) -> str:
        """Generate score distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = results_df['score'].dropna()
        
        # Histogram with KDE
        ax.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        
        # Add KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(0, 1, 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add mean line
        mean_score = scores.mean()
        ax.axvline(mean_score, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        
        # Add threshold line if configured
        threshold = self.config.get('metrics', {}).get(metric, {}).get('threshold', 0.7)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Score Distribution ({metric})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            # Use data/results directory instead of /tmp
            import tempfile
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'score_dist_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_metric_comparison(
        self,
        all_results: Dict[str, pd.DataFrame],
        save_path: str = None
    ) -> str:
        """Generate comparison chart across multiple metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        metrics = list(all_results.keys())
        avg_scores = [all_results[m]['score'].mean() for m in metrics]
        pass_rates = [
            (all_results[m]['label'] == 'positive').sum() / len(all_results[m]) * 100
            for m in metrics
        ]
        
        # Plot 1: Average scores
        ax1 = axes[0]
        bars1 = ax1.bar(metrics, avg_scores, color=['#3498db', '#2ecc71', '#e74c3c'][:len(metrics)])
        ax1.set_ylabel('Average Score', fontsize=12)
        ax1.set_title('Average Scores by Metric', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars1, avg_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Pass rates
        ax2 = axes[1]
        bars2 = ax2.bar(metrics, pass_rates, color=['#3498db', '#2ecc71', '#e74c3c'][:len(metrics)])
        ax2.set_ylabel('Pass Rate (%)', fontsize=12)
        ax2.set_title('Pass Rates by Metric', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, rate in zip(bars2, pass_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'metric_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_error_analysis(
        self,
        results_df: pd.DataFrame,
        metric: str,
        save_path: str = None,
        top_n: int = 10
    ) -> str:
        """Generate error analysis showing lowest scoring examples"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get lowest scoring examples
        lowest = results_df.nsmallest(top_n, 'score')
        
        y_pos = np.arange(len(lowest))
        scores = lowest['score'].values
        
        # Create horizontal bar chart
        bars = ax.barh(y_pos, scores, color='coral')
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Row {idx}" for idx in lowest.index], fontsize=9)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(f'Top {top_n} Lowest Scoring Examples ({metric})', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f'error_analysis_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_pdf_report(
        self,
        output_path: str,
        run_info: Dict,
        all_results: Dict[str, pd.DataFrame],
        api_stats: Dict = None
    ) -> str:
        """
        Generate comprehensive PDF report
        
        Args:
            output_path: Path to save PDF
            run_info: Information about the evaluation run
            all_results: Dictionary of {metric: results_df}
            api_stats: API usage statistics
            
        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph(
            "ProductGPT Evaluation Report",
            self.title_style
        )
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Run Information
        story.append(Paragraph("Evaluation Run Information", self.heading_style))
        
        run_data = [
            ['Field', 'Value'],
            ['Timestamp', run_info.get('timestamp', 'N/A')],
            ['User', run_info.get('user', 'N/A')],
            ['Input File', run_info.get('input_file_name', 'N/A')],
            ['Total Rows', str(run_info.get('total_rows', 'N/A'))],
            ['Metrics Evaluated', ', '.join(run_info.get('metrics_evaluated', []))],
            ['Execution Time', f"{run_info.get('execution_time_seconds', 0):.2f}s"],
        ]
        
        if api_stats:
            run_data.extend([
                ['Total API Calls', str(api_stats.get('total_calls', 'N/A'))],
                ['Total Tokens', str(api_stats.get('total_tokens', 'N/A'))],
            ])
        
        run_table = Table(run_data, colWidths=[2*inch, 4*inch])
        run_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(run_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        summary_data = [['Metric', 'Avg Score', 'Pass Rate', 'Total Samples']]
        for metric, results_df in all_results.items():
            avg_score = results_df['score'].mean()
            pass_count = (results_df['label'] == 'positive').sum()
            pass_rate = pass_count / len(results_df) * 100
            summary_data.append([
                metric.capitalize(),
                f"{avg_score:.3f}",
                f"{pass_rate:.1f}%",
                str(len(results_df))
            ])
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(PageBreak())
        
        # Detailed Results for Each Metric
        for metric, results_df in all_results.items():
            story.append(Paragraph(f"Detailed Analysis: {metric.capitalize()}", self.heading_style))
            
            # Statistics
            stats_text = f"""
            <b>Total Samples:</b> {len(results_df)}<br/>
            <b>Average Score:</b> {results_df['score'].mean():.3f}<br/>
            <b>Std Dev:</b> {results_df['score'].std():.3f}<br/>
            <b>Min Score:</b> {results_df['score'].min():.3f}<br/>
            <b>Max Score:</b> {results_df['score'].max():.3f}<br/>
            <b>Median Score:</b> {results_df['score'].median():.3f}<br/>
            """
            story.append(Paragraph(stats_text, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Generate and add visualizations
            # 1. Score Distribution
            dist_path = self.generate_score_distribution(results_df, metric)
            img = Image(dist_path, width=6*inch, height=3.6*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # 2. Confusion Matrix
            cm_path = self.generate_confusion_matrix(results_df, metric)
            img = Image(cm_path, width=5*inch, height=3.75*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # 3. Error Analysis
            error_path = self.generate_error_analysis(results_df, metric)
            img = Image(error_path, width=6*inch, height=3*inch)
            story.append(img)
            
            story.append(PageBreak())
        
        # Metric Comparison
        if len(all_results) > 1:
            story.append(Paragraph("Cross-Metric Comparison", self.heading_style))
            comp_path = self.generate_metric_comparison(all_results)
            img = Image(comp_path, width=7*inch, height=3*inch)
            story.append(img)
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        
        return output_path
