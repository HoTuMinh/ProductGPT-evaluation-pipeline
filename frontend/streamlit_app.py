"""
Streamlit Frontend for ProductGPT Evaluation Pipeline
"""
import streamlit as st
import pandas as pd
import yaml
import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.database import Database
from app.evaluator import EvaluationPipeline
from app.reports import ReportGenerator

# Page config
st.set_page_config(
    page_title="ProductGPT Evaluation Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CoverGo CSS Theme
def load_css():
    """Load custom CoverGo CSS theme"""
    css_file = Path(__file__).parent / "covergo_theme.css"
    try:
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS theme file not found. Using default styling.")

# Apply CSS theme
load_css()

# Load config
@st.cache_resource
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize database
@st.cache_resource
def get_database():
    config = load_config()
    db_path = Path(__file__).parent.parent / config['database']['path']
    return Database(str(db_path))

# Authentication
def check_authentication():
    """Simple password-based authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown('<div class="main-header">ProductGPT Evaluation Pipeline</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if (username == "123" and password == "123") or password == "covergo2024":
                    st.session_state.authenticated = True
                    st.session_state.username = username if username else "anonymous"
                    st.session_state.step = 'settings'
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        return False
    
    return True


def display_evaluation_results(config, db, eval_data):
    """Display evaluation results with report generation"""
    
    # Check if in review mode
    if 'review_mode' in st.session_state and st.session_state.review_mode:
        if st.session_state.review_mode == 'config':
            show_review_config(config, db, eval_data)
            return
        elif st.session_state.review_mode == 'reviewing':
            show_review_interface(config, db, eval_data)
            return
        elif st.session_state.review_mode == 'summary':
            show_review_summary(config, db, eval_data)
            return
    
    st.markdown("### Evaluation Results")
    
    all_results = st.session_state.last_evaluation['all_results']
    selected_metrics = eval_data['selected_metrics']
    questions = eval_data['questions']
    responses = eval_data['responses']
    benchmarks = eval_data['benchmarks']
    
    review_applied = st.session_state.get('review_applied', False)
    review_metric = st.session_state.get('review_metric_updated', None)
    review_threshold = st.session_state.get('review_threshold_used', 0.7)
    
    # Summary metrics
    cols = st.columns(len(selected_metrics))
    for idx, (metric, results_df) in enumerate(all_results.items()):
        with cols[idx]:
            avg_score = results_df['score'].mean()
            pass_count = sum(1 for score in results_df['score'] if score >= review_threshold)
            pass_rate = pass_count / len(results_df) * 100
            has_reviews = review_applied and metric == review_metric
            
            if has_reviews and 'llm_score' in results_df.columns:
                original_avg = results_df['llm_score'].mean()
                original_pass_rate = (results_df['llm_score'] >= review_threshold).sum() / len(results_df) * 100
                
                st.markdown(f"""
                <div class="metric-highlight-card">
                    <div class="metric-highlight-label">{metric.capitalize()}</div>
                    <div class="metric-highlight-value">{avg_score:.3f}</div>
                    <div class="metric-highlight-delta">{pass_rate:.1f}% pass rate</div>
                    <div class="metric-highlight-help">Original LLM: {original_avg:.3f} avg, {original_pass_rate:.1f}% pass rate</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-highlight-card">
                    <div class="metric-highlight-label">{metric.capitalize()}</div>
                    <div class="metric-highlight-value">{avg_score:.3f}</div>
                    <div class="metric-highlight-delta">{pass_rate:.1f}% pass rate</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Review button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Review Samples", type="primary", use_container_width=True):
            st.session_state.review_mode = 'config'
            st.session_state.review_metric = list(selected_metrics)[0]
            st.rerun()
    
    st.markdown("---")
    
    # Detailed results tabs
    tab_names = [f"{m.capitalize()}" for m in selected_metrics] + ["Generate Report"]
    tabs = st.tabs(tab_names)
    
    for idx, metric in enumerate(selected_metrics):
        with tabs[idx]:
            results_df = all_results[metric].copy()
            results_df['question'] = questions
            results_df['response'] = responses
            results_df['benchmark'] = benchmarks
            
            has_reviews = 'llm_score' in results_df.columns and 'human_reviewed' in results_df.columns
            
            if has_reviews:
                display_cols = ['question', 'llm_score', 'score', 'llm_reasoning', 'human_reasoning']
                display_cols = [col for col in display_cols if col in results_df.columns]
                
                st.info("This metric has been reviewed. 'score' column shows final scores (human overrides where reviewed).")
                
                display_df = results_df[display_cols].copy()
                display_df = display_df.rename(columns={
                    'llm_score': 'LLM Score',
                    'score': 'Final Score',
                    'llm_reasoning': 'LLM Reasoning',
                    'human_reasoning': 'Human Review'
                })
                
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.dataframe(
                    results_df[['question', 'score', 'label', 'reasoning']],
                    use_container_width=True,
                    height=400
                )
            
            csv = results_df.to_csv(index=False)
            download_label = f"Download {metric} Results" + (" (with Human Reviews)" if has_reviews else " (CSV)")
            st.download_button(
                label=download_label,
                data=csv,
                file_name=f"{metric}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_csv_{metric}"
            )
    
    # Report generation tab
    with tabs[-1]:
        st.markdown("""
        <div class="report-section-header">
            <h3>Generate Comprehensive Report</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("Generate a detailed PDF report with visualizations and analysis.")
        
        with st.form("pdf_generation_form"):
            submit_button = st.form_submit_button("Generate PDF Report", type="primary")
            
            if submit_button:
                with st.spinner("Generating report..."):
                    try:
                        report_gen = ReportGenerator(eval_data['config'])
                        
                        api_stats = {
                            'total_calls': eval_data['total_api_calls'],
                            'total_tokens': sum(
                                r.get('input_tokens', 0) + r.get('output_tokens', 0)
                                for results in eval_data['all_results'].values()
                                for r in results.to_dict('records')
                            )
                        }
                        
                        run_info = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'user': st.session_state.username,
                            'input_file_name': eval_data['uploaded_file_name'],
                            'total_rows': eval_data['total_rows'],
                            'metrics_evaluated': eval_data['selected_metrics'],
                            'execution_time_seconds': eval_data['execution_time']
                        }
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_path = f"data/results/evaluation_report_{eval_data['run_id']}_{timestamp}.pdf"
                        
                        os.makedirs("data/results", exist_ok=True)
                        
                        enriched_results = {}
                        for metric, results_df in all_results.items():
                            df_copy = results_df.copy()
                            if 'question' not in df_copy.columns:
                                df_copy['question'] = eval_data['questions']
                            if 'response' not in df_copy.columns:
                                df_copy['response'] = eval_data['responses']
                            if 'benchmark' not in df_copy.columns:
                                df_copy['benchmark'] = eval_data['benchmarks']
                            enriched_results[metric] = df_copy
                        
                        if review_applied and review_metric:
                            run_info['human_reviews'] = f"Human reviews applied to {review_metric} (threshold: {review_threshold})"
                        
                        report_gen.generate_pdf_report(
                            output_path=output_path,
                            run_info=run_info,
                            all_results=enriched_results,
                            api_stats=api_stats
                        )
                        
                        st.success("Report generated successfully!")
                        st.session_state.last_pdf_path = output_path
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        if 'last_pdf_path' in st.session_state and os.path.exists(st.session_state.last_pdf_path):
            st.markdown("---")
            with open(st.session_state.last_pdf_path, 'rb') as f:
                pdf_data = f.read()
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{st.session_state.last_pdf_path}"
                )


def show_review_config(config, db, eval_data):
    """Show review configuration page"""
    
    st.markdown("### Review Configuration")
    st.markdown("Configure your review settings before starting.")
    
    st.markdown("---")
    
    selected_metrics = eval_data['selected_metrics']
    if len(selected_metrics) > 1:
        review_metric = st.selectbox(
            "Select Metric to Review",
            options=selected_metrics,
            format_func=lambda x: x.capitalize(),
            index=selected_metrics.index(st.session_state.review_metric) if st.session_state.review_metric in selected_metrics else 0
        )
        st.session_state.review_metric = review_metric
    else:
        review_metric = selected_metrics[0]
        st.info(f"Reviewing Metric: {review_metric.capitalize()}")
    
    st.markdown("---")
    
    results_df = eval_data['all_results'][review_metric].copy()
    results_df['question'] = eval_data['questions']
    results_df['response'] = eval_data['responses']
    results_df['benchmark'] = eval_data['benchmarks']
    
    st.markdown("#### Set Threshold")
    threshold = st.slider(
        "Pass/Fail Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Samples with score >= threshold are considered Pass"
    )
    
    below_threshold = results_df[results_df['score'] < threshold]
    above_threshold = results_df[results_df['score'] >= threshold]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Below Threshold", f"{len(below_threshold)} samples", delta="Fail", delta_color="inverse")
    with col2:
        st.metric("Above Threshold", f"{len(above_threshold)} samples", delta="Pass", delta_color="normal")
    
    st.markdown("---")
    
    st.markdown("#### Select Samples to Review")
    filter_option = st.radio(
        "Review",
        options=["Below threshold", "All samples"],
        help="Choose which samples you want to review"
    )
    
    if filter_option == "Below threshold":
        samples_to_review = below_threshold
    else:
        samples_to_review = results_df
    
    st.info(f"You will review {len(samples_to_review)} samples")
    
    current_pass_rate = (len(above_threshold) / len(results_df) * 100) if len(results_df) > 0 else 0
    st.markdown(f"**Current Pass Rate:** {current_pass_rate:.1f}% ({len(above_threshold)}/{len(results_df)})")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.review_mode = None
            st.rerun()
    
    with col2:
        if st.button("Start Review", type="primary", use_container_width=True):
            st.session_state.review_threshold = threshold
            st.session_state.review_samples = samples_to_review.reset_index()
            st.session_state.review_current_index = 0
            st.session_state.review_changes = {}
            st.session_state.review_mode = 'reviewing'
            st.rerun()


def show_review_interface(config, db, eval_data):
    """Show interface to review individual samples"""
    
    samples = st.session_state.review_samples
    current_index = st.session_state.review_current_index
    threshold = st.session_state.review_threshold
    metric = st.session_state.review_metric
    
    if current_index >= len(samples):
        st.session_state.review_mode = 'summary'
        st.rerun()
        return
    
    sample = samples.iloc[current_index]
    original_row_index = sample['index'] if 'index' in sample else current_index
    
    st.progress((current_index) / len(samples))
    st.markdown(f"### Review Sample {current_index + 1}/{len(samples)}")
    
    st.markdown("---")
    
    st.markdown("#### LLM Judgment")
    
    llm_score = sample['score']
    llm_label = "Pass" if llm_score >= threshold else "Fail"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if llm_score >= threshold:
            st.success(f"**{llm_label}** (Score: {llm_score:.3f} >= Threshold: {threshold})")
        else:
            st.error(f"**{llm_label}** (Score: {llm_score:.3f} < Threshold: {threshold})")
    with col2:
        st.metric("LLM Score", f"{llm_score:.3f}")
    
    st.markdown("---")
    
    st.markdown("#### Sample Content")
    
    with st.expander("Question", expanded=True):
        st.write(sample['question'])
    
    with st.expander("Response", expanded=True):
        st.write(sample['response'])
    
    with st.expander("Benchmark Answer", expanded=True):
        st.write(sample['benchmark'])
    
    with st.expander("LLM Reasoning", expanded=False):
        st.write(sample.get('reasoning', 'N/A'))
    
    st.markdown("---")
    
    st.markdown("#### Your Review")
    
    existing_review = st.session_state.review_changes.get(original_row_index, {})
    
    st.markdown("**Score:**")
    human_score = st.slider(
        "Your score for this sample",
        min_value=0.0,
        max_value=1.0,
        value=existing_review.get('score', llm_score),
        step=0.05,
        key=f"score_slider_{current_index}",
        label_visibility="collapsed"
    )
    
    st.caption(f"Current: {human_score:.2f} | Threshold: {threshold}")
    
    st.markdown("**Label:**")
    human_label_pass = human_score >= threshold
    human_label = st.radio(
        "Pass or Fail",
        options=["Pass", "Fail"],
        index=0 if human_label_pass else 1,
        horizontal=True,
        key=f"label_radio_{current_index}",
        label_visibility="collapsed"
    )
    
    st.markdown("**Comment (Optional):**")
    human_comment = st.text_area(
        "Add your feedback or notes",
        value=existing_review.get('comment', ''),
        height=100,
        key=f"comment_{current_index}",
        label_visibility="collapsed",
        placeholder="Why did you adjust the score? Any observations?"
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Previous", disabled=(current_index == 0), use_container_width=True):
            st.session_state.review_current_index -= 1
            st.rerun()
    
    with col2:
        if st.button("Skip", use_container_width=True):
            st.session_state.review_current_index += 1
            st.rerun()
    
    with col3:
        if st.button("Save & Next", type="primary", use_container_width=True):
            st.session_state.review_changes[original_row_index] = {
                'score': human_score,
                'label': human_label,
                'comment': human_comment
            }
            
            if 'id' in sample:
                db.update_human_review(
                    result_id=int(sample['id']),
                    human_score=human_score,
                    human_label=human_label.lower(),
                    human_comment=human_comment if human_comment else None
                )
            
            st.session_state.review_current_index += 1
            st.rerun()


def show_review_summary(config, db, eval_data):
    """Show review summary and statistics"""
    
    st.markdown("### Review Complete!")
    st.success("You have finished reviewing the selected samples.")
    
    st.markdown("---")
    
    threshold = st.session_state.review_threshold
    metric = st.session_state.review_metric
    samples = st.session_state.review_samples
    changes = st.session_state.review_changes
    
    total_samples = len(samples)
    reviewed_count = len(changes)
    
    llm_passes = sum(1 for _, row in samples.iterrows() if row['score'] >= threshold)
    llm_pass_rate = (llm_passes / total_samples * 100) if total_samples > 0 else 0
    
    human_passes = 0
    agreements = 0
    
    for idx, row in samples.iterrows():
        row_index = row.get('row_index', idx)
        if row_index in changes:
            if changes[row_index]['score'] >= threshold:
                human_passes += 1
            llm_pass = row['score'] >= threshold
            human_pass = changes[row_index]['score'] >= threshold
            if llm_pass == human_pass:
                agreements += 1
        else:
            if row['score'] >= threshold:
                human_passes += 1
    
    human_pass_rate = (human_passes / total_samples * 100) if total_samples > 0 else 0
    agreement_rate = (agreements / reviewed_count * 100) if reviewed_count > 0 else 0
    pass_rate_change = human_pass_rate - llm_pass_rate
    
    st.markdown("### Review Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples Reviewed", f"{reviewed_count}/{total_samples}")
    with col2:
        st.metric("Changes Made", len([c for c in changes.values() if c['score'] != samples.iloc[0]['score']]))
    with col3:
        st.metric("Agreement Rate", f"{agreement_rate:.1f}%", help="% of reviewed samples where human agrees with LLM")
    
    st.markdown("---")
    
    st.markdown("### Pass Rate Comparison")
    st.markdown(f"**Threshold:** {threshold}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original (LLM)", f"{llm_pass_rate:.1f}%", help=f"{llm_passes}/{total_samples} samples passed")
    with col2:
        st.metric("After Review", f"{human_pass_rate:.1f}%", help=f"{human_passes}/{total_samples} samples passed")
    with col3:
        delta_color = "normal" if pass_rate_change >= 0 else "inverse"
        st.metric("Change", f"{pass_rate_change:+.1f}%", delta=f"{pass_rate_change:+.1f}%", delta_color=delta_color)
    
    st.markdown("---")
    
    if reviewed_count > 0:
        st.markdown("### Review Details")
        
        review_data = []
        for row_index, change in changes.items():
            matching_samples = samples[samples['index'] == row_index] if 'index' in samples.columns else samples[samples.index == row_index]
            
            if not matching_samples.empty:
                original = matching_samples.iloc[0]
                review_data.append({
                    'Sample': row_index + 1,
                    'LLM Score': f"{original['score']:.3f}",
                    'Human Score': f"{change['score']:.3f}",
                    'LLM Label': 'Pass' if original['score'] >= threshold else 'Fail',
                    'Human Label': change['label'],
                    'Comment': change['comment'][:50] + '...' if change['comment'] and len(change['comment']) > 50 else change['comment'] or '-'
                })
        
        if review_data:
            review_df = pd.DataFrame(review_data)
            st.dataframe(review_df, use_container_width=True, height=300)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Discard All Changes", use_container_width=True):
            for row_index in changes.keys():
                matching_samples = samples[samples['index'] == row_index] if 'index' in samples.columns else samples[samples.index == row_index]
                
                if not matching_samples.empty and 'id' in matching_samples.iloc[0]:
                    db.clear_human_review(int(matching_samples.iloc[0]['id']))
            
            st.session_state.review_mode = None
            for key in ['review_threshold', 'review_samples', 'review_current_index', 'review_changes', 'review_metric', 'review_applied']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.warning("All changes discarded")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Apply Changes", type="primary", use_container_width=True):
            st.session_state.review_applied = True
            st.session_state.review_metric_updated = metric
            st.session_state.review_threshold_used = threshold
            
            st.session_state.review_updated_samples = {
                row_index: {
                    'human_score': change['score'],
                    'human_label': change['label'],
                    'human_comment': change['comment']
                }
                for row_index, change in changes.items()
            }
            
            if 'last_evaluation' in st.session_state:
                results_df = st.session_state.last_evaluation['all_results'][metric].copy()
                
                if 'llm_score' not in results_df.columns:
                    results_df['llm_score'] = results_df['score'].copy()
                    results_df['llm_label'] = results_df['label'].copy()
                    results_df['llm_reasoning'] = results_df['reasoning'].copy()
                
                for row_idx, update in st.session_state.review_updated_samples.items():
                    if row_idx < len(results_df):
                        results_df.at[row_idx, 'score'] = update['human_score']
                        results_df.at[row_idx, 'label'] = update['human_label'].lower()
                        results_df.at[row_idx, 'human_reviewed'] = True
                        results_df.at[row_idx, 'human_reasoning'] = update['human_comment'] if update['human_comment'] else 'Reviewed by human'
                
                st.session_state.last_evaluation['all_results'][metric] = results_df
            
            st.session_state.review_mode = None
            
            st.success(f"Applied {reviewed_count} reviews to {metric} results!")
            time.sleep(1.5)
            st.rerun()


def main():
    """Main application"""
    
    if not check_authentication():
        return
    
    config = load_config()
    db = get_database()
    
    # Initialize session state
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'step' not in st.session_state:
        st.session_state.step = 'settings'
    
    # Header with logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">ProductGPT Evaluation Pipeline</div>', unsafe_allow_html=True)
    with col2:
        st.write(f"User: {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    st.markdown("---")
    
    # Step 1: Settings
    show_settings_section(config, db)
    
    # Step 2: New Evaluation (only if settings configured)
    if st.session_state.get('config_saved') or (st.session_state.selected_provider and st.session_state.api_keys.get(st.session_state.selected_provider)):
        st.markdown("---")
        show_evaluation_section(config, db)
    
    # Step 3: Evaluation History (toggle)
    st.markdown("---")
    show_history_section(db)


def show_settings_section(config, db):
    """Show settings section"""
    
    st.markdown("### Step 1: API & Model Configuration")
    
    # Provider selection
    st.markdown("#### Select LLM Provider")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Groq", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "groq" else "secondary"):
            st.session_state.selected_provider = "groq"
            st.rerun()
    
    with col2:
        if st.button("Gemini", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "gemini" else "secondary"):
            st.session_state.selected_provider = "gemini"
            st.rerun()
    
    with col3:
        if st.button("OpenAI", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "openai" else "secondary"):
            st.session_state.selected_provider = "openai"
            st.rerun()
    
    if not st.session_state.get('selected_provider'):
        st.warning("Please select a provider above")
        return
    
    provider = st.session_state.selected_provider
    
    st.markdown(f"#### Configure {provider.upper()}")
    
    # Model selection
    if provider == "groq":
        models = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B - Best quality, balanced speed",
            "llama-3.1-8b-instant": "Llama 3.1 8B - Fastest, good quality",
            "mixtral-8x7b-32768": "Mixtral 8x7B - Large context window",
            "gemma2-9b-it": "Gemma 2 9B - Efficient"
        }
        api_help = "Get your API key at: https://console.groq.com/keys"
        
    elif provider == "gemini":
        models = {
            "gemini-1.5-flash": "Gemini 1.5 Flash - Fast & efficient",
            "gemini-1.5-pro": "Gemini 1.5 Pro - Best quality",
            "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)"
        }
        api_help = "Get your API key at: https://makersuite.google.com/app/apikey"
        
    else:
        models = {
            "gpt-4o": "GPT-4o - Latest, best quality",
            "gpt-4o-mini": "GPT-4o Mini - Fast & affordable",
            "gpt-4-turbo": "GPT-4 Turbo - Previous gen"
        }
        api_help = "Get your API key at: https://platform.openai.com/api-keys"
    
    selected_model = st.selectbox(
        "Model",
        options=list(models.keys()),
        format_func=lambda x: models[x],
        index=0 if not st.session_state.get('selected_model') else (list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
    )
    
    st.caption(api_help)
    
    current_key = st.session_state.api_keys.get(provider, "")
    api_key = st.text_input(
        f"{provider.upper()} API Key",
        value=current_key,
        type="password",
        help="Your API key is stored only in your session"
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative",
            key="settings_temperature"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=2048,
            step=256,
            key="settings_max_tokens"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=20,
            value=5,
            key="settings_batch_size"
        )
        
        max_concurrent = st.slider(
            "Max Concurrent Calls",
            min_value=1,
            max_value=10,
            value=3,
            key="settings_max_concurrent"
        )
    
    # Save button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter an API key")
            else:
                st.session_state.api_keys[provider] = api_key
                st.session_state.selected_provider = provider
                st.session_state.selected_model = selected_model
                st.session_state.temperature = temperature
                st.session_state.max_tokens = max_tokens
                st.session_state.batch_size = batch_size
                st.session_state.max_concurrent = max_concurrent
                st.session_state.config_saved = True
                st.success(f"Configuration saved! Provider: {provider.upper()}, Model: {selected_model}")
    
    with col2:
        if st.button("Test", use_container_width=True):
            if not api_key:
                st.error("Please enter an API key")
            else:
                with st.spinner("Testing..."):
                    try:
                        if provider == "groq":
                            from groq import Groq
                            client = Groq(api_key=api_key)
                            client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "user", "content": "Say 'test'"}],
                                max_tokens=10
                            )
                            st.success("Connection successful!")
                        elif provider == "gemini":
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            model_obj = genai.GenerativeModel(selected_model)
                            model_obj.generate_content("Say 'test'")
                            st.success("Connection successful!")
                        else:
                            st.info("OpenAI test not implemented yet.")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")


def show_evaluation_section(config, db):
    """Show evaluation section"""
    
    # Check if showing results
    if 'last_evaluation' in st.session_state and 'show_results' in st.session_state and st.session_state.show_results:
        display_evaluation_results(config, db, st.session_state.last_evaluation)
        
        if st.button("Start New Evaluation", type="secondary"):
            st.session_state.show_results = False
            st.session_state.pop('last_evaluation', None)
            st.rerun()
        return
    
    st.markdown("### Step 2: Upload & Evaluate")
    
    provider = st.session_state.selected_provider
    model = st.session_state.selected_model
    
    st.info(f"Using: {provider.upper()} - {model}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: question, response, benchmark_answer"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! ({len(df)} rows)")
            
            with st.expander("Preview Data (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("#### Configure Evaluation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                question_col = st.selectbox(
                    "Question Column",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index('question') if 'question' in df.columns else 0
                )
                
                response_col = st.selectbox(
                    "Response Column",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index('response') if 'response' in df.columns else 0
                )
            
            with col2:
                benchmark_col = st.selectbox(
                    "Benchmark Answer Column",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index('benchmark_answer') if 'benchmark_answer' in df.columns else 0
                )
                
                available_metrics = ['accuracy', 'comprehensiveness', 'faithfulness', 'toxicity']
                selected_metrics = st.multiselect(
                    "Select Metrics to Evaluate",
                    options=available_metrics,
                    default=['accuracy']
                )
            
            with st.expander("Advanced Settings"):
                batch_size = st.slider("Batch Size", 1, 20, config['batch']['size'], key="eval_batch_size")
                max_concurrent = st.slider("Max Concurrent API Calls", 1, 10, config['batch']['max_concurrent'], key="eval_max_concurrent")
            
            if st.button("Run Evaluation", type="primary", use_container_width=True):
                if not selected_metrics:
                    st.error("Please select at least one metric to evaluate.")
                    return
                
                run_evaluation(
                    df=df,
                    api_key=st.session_state.api_keys[st.session_state.selected_provider],
                    config=config,
                    db=db,
                    question_col=question_col,
                    response_col=response_col,
                    benchmark_col=benchmark_col,
                    selected_metrics=selected_metrics,
                    batch_size=st.session_state.get('batch_size', batch_size),
                    max_concurrent=st.session_state.get('max_concurrent', max_concurrent),
                    uploaded_file=uploaded_file,
                    provider=st.session_state.selected_provider,
                    model=st.session_state.selected_model
                )
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def run_evaluation(df, api_key, config, db, question_col, response_col, benchmark_col, 
                   selected_metrics, batch_size, max_concurrent, uploaded_file, provider=None, model=None):
    """Run the evaluation pipeline"""
    
    start_time = time.time()
    
    config['batch']['size'] = batch_size
    config['batch']['max_concurrent'] = max_concurrent
    
    if provider:
        config['llm']['provider'] = provider
    if model:
        config['llm']['model'] = model
    if 'temperature' in st.session_state:
        config['llm']['temperature'] = st.session_state.temperature
    if 'max_tokens' in st.session_state:
        config['llm']['max_tokens'] = st.session_state.max_tokens
    
    run_id = db.create_evaluation_run(
        user=st.session_state.username,
        input_file_name=uploaded_file.name,
        input_file_size=uploaded_file.size,
        metrics_evaluated=selected_metrics,
        total_rows=len(df),
        llm_model=config['llm']['model'],
        batch_size=batch_size
    )
    
    pipeline = EvaluationPipeline(
        api_key=api_key,
        config=config,
        database=db
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    
    try:
        questions = df[question_col].fillna("").tolist()
        responses = df[response_col].fillna("").tolist()
        benchmarks = df[benchmark_col].fillna("").tolist()
        
        for idx, metric in enumerate(selected_metrics):
            status_text.markdown(f"**Evaluating {metric}... ({idx+1}/{len(selected_metrics)})**")
            
            def update_progress(progress, message):
                overall_progress = (idx + progress) / len(selected_metrics)
                progress_bar.progress(overall_progress)
                status_text.markdown(f"**{message}**")
            
            results = asyncio.run(
                pipeline.evaluate_dataset(
                    metric=metric,
                    questions=questions,
                    responses=responses,
                    benchmarks=benchmarks,
                    run_id=run_id,
                    progress_callback=update_progress
                )
            )
            
            results_df = pd.DataFrame(results)
            all_results[metric] = results_df
        
        execution_time = time.time() - start_time
        
        avg_scores = {metric: results['score'].mean() for metric, results in all_results.items()}
        total_api_calls = sum(len(results) for results in all_results.values())
        
        db.update_evaluation_run(
            run_id=run_id,
            status="completed",
            average_scores=avg_scores,
            total_api_calls=total_api_calls,
            execution_time_seconds=execution_time
        )
        
        st.session_state.last_evaluation = {
            'all_results': all_results,
            'questions': questions,
            'responses': responses,
            'benchmarks': benchmarks,
            'selected_metrics': selected_metrics,
            'run_id': run_id,
            'execution_time': execution_time,
            'total_api_calls': total_api_calls,
            'uploaded_file_name': uploaded_file.name,
            'total_rows': len(df),
            'config': config
        }
        
        st.session_state.show_results = True
        
        progress_bar.progress(1.0)
        status_text.markdown("**Evaluation completed!**")
        
        st.rerun()
    
    except Exception as e:
        db.update_evaluation_run(
            run_id=run_id,
            status="failed",
            error_message=str(e)
        )
        st.error(f"Evaluation failed: {str(e)}")
        raise


def show_history_section(db):
    """Show evaluation history as toggle"""
    
    with st.expander("Evaluation History", expanded=False):
        runs = db.get_evaluation_runs(user=st.session_state.username)
        
        if not runs:
            st.info("No evaluation history yet. Run your first evaluation to see results here.")
            return
        
        runs_df = pd.DataFrame(runs)
        
        st.dataframe(
            runs_df[[
                'id', 'timestamp', 'input_file_name', 'metrics_evaluated', 
                'total_rows', 'status', 'average_scores', 'execution_time_seconds'
            ]],
            use_container_width=True,
            height=300
        )
        
        st.markdown("#### View Run Details")
        
        run_id = st.selectbox(
            "Select a run to view details",
            options=runs_df['id'].tolist(),
            format_func=lambda x: f"Run #{x} - {runs_df[runs_df['id']==x]['timestamp'].values[0]}"
        )
        
        if run_id:
            results = db.get_evaluation_results(run_id)
            if results:
                results_df = pd.DataFrame(results)
                st.markdown(f"#### Results for Run #{run_id}")
                st.dataframe(results_df, use_container_width=True, height=300)


if __name__ == "__main__":
    main()
