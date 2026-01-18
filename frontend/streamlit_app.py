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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        st.markdown('<div class="main-header">🔐 ProductGPT Evaluation Pipeline</div>', unsafe_allow_html=True)
        st.markdown("### Please Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                # Simple auth - in production, use proper authentication
                if password == "covergo2024":  # Change this!
                    st.session_state.authenticated = True
                    st.session_state.username = username if username else "anonymous"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.info("💡 Default password: covergo2024 (Please change this in production!)")
        return False
    
    return True

def display_evaluation_results(config, db, eval_data):
    """Display evaluation results with report generation"""
    
    st.markdown("### 📊 Evaluation Results")
    
    all_results = eval_data['all_results']
    selected_metrics = eval_data['selected_metrics']
    questions = eval_data['questions']
    responses = eval_data['responses']
    benchmarks = eval_data['benchmarks']
    
    # Summary metrics
    cols = st.columns(len(selected_metrics))
    for idx, (metric, results_df) in enumerate(all_results.items()):
        with cols[idx]:
            avg_score = results_df['score'].mean()
            pass_count = (results_df['label'] == 'positive').sum()
            pass_rate = pass_count / len(results_df) * 100
            
            st.metric(
                label=f"{metric.capitalize()}",
                value=f"{avg_score:.3f}",
                delta=f"{pass_rate:.1f}% pass rate"
            )
    
    # Detailed results tabs
    tabs = st.tabs([f"📋 {m.capitalize()}" for m in selected_metrics] + ["📄 Generate Report"])
    
    for idx, metric in enumerate(selected_metrics):
        with tabs[idx]:
            results_df = all_results[metric].copy()
            
            # Add original data for context
            results_df['question'] = questions
            results_df['response'] = responses
            results_df['benchmark'] = benchmarks
            
            st.dataframe(
                results_df[['question', 'score', 'label', 'reasoning']],
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {metric} Results (CSV)",
                data=csv,
                file_name=f"{metric}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_csv_{metric}"
            )
    
    # Report generation tab
    with tabs[-1]:
        st.markdown("### 📄 Generate Comprehensive Report")
        st.info("Generate a detailed PDF report with visualizations and analysis.")
        
        # Use form to prevent rerun
        with st.form("pdf_generation_form"):
            submit_button = st.form_submit_button("📑 Generate PDF Report", type="primary")
            
            if submit_button:
                with st.spinner("Generating report..."):
                    try:
                        # Generate report
                        report_gen = ReportGenerator(eval_data['config'])
                        
                        # Get API stats
                        api_stats = {
                            'total_calls': eval_data['total_api_calls'],
                            'total_tokens': sum(
                                r.get('input_tokens', 0) + r.get('output_tokens', 0)
                                for results in eval_data['all_results'].values()
                                for r in results.to_dict('records')
                            )
                        }
                        
                        # Run info
                        run_info = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'user': st.session_state.username,
                            'input_file_name': eval_data['uploaded_file_name'],
                            'total_rows': eval_data['total_rows'],
                            'metrics_evaluated': eval_data['selected_metrics'],
                            'execution_time_seconds': eval_data['execution_time']
                        }
                        
                        # Generate PDF
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_path = f"data/results/evaluation_report_{eval_data['run_id']}_{timestamp}.pdf"
                        
                        # Ensure directory exists
                        os.makedirs("data/results", exist_ok=True)
                        
                        report_gen.generate_pdf_report(
                            output_path=output_path,
                            run_info=run_info,
                            all_results=eval_data['all_results'],
                            api_stats=api_stats
                        )
                        
                        st.success("✅ Report generated successfully!")
                        
                        # Store PDF path in session state
                        st.session_state.last_pdf_path = output_path
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Show download button outside form if PDF was generated
        if 'last_pdf_path' in st.session_state and os.path.exists(st.session_state.last_pdf_path):
            st.markdown("---")
            with open(st.session_state.last_pdf_path, 'rb') as f:
                pdf_data = f.read()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{st.session_state.last_pdf_path}"
                )


def main():
    """Main application"""
    
    # Check authentication
    if not check_authentication():
        return
    
    # Initialize
    config = load_config()
    db = get_database()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Information")
        st.write(f"**Logged in as:** {st.session_state.username}")
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select Page",
            ["🔍 New Evaluation", "📈 Evaluation History", "⚙️ Settings"],
            label_visibility="collapsed"
        )
    
    # Main content
    st.markdown('<div class="main-header">🤖 ProductGPT Evaluation Pipeline</div>', unsafe_allow_html=True)
    
    if page == "🔍 New Evaluation":
        show_evaluation_page(config, db)
    elif page == "📈 Evaluation History":
        show_history_page(db)
    elif page == "⚙️ Settings":
        show_settings_page(config)

def show_evaluation_page(config, db):
    """Show the main evaluation page"""
    
    # Check if we have results to display
    if 'last_evaluation' in st.session_state and 'show_results' in st.session_state and st.session_state.show_results:
        display_evaluation_results(config, db, st.session_state.last_evaluation)
        
        # Button to start new evaluation
        st.markdown("---")
        if st.button("🔄 Start New Evaluation", type="secondary"):
            st.session_state.show_results = False
            st.session_state.pop('last_evaluation', None)
            st.rerun()
        return
    
    # Otherwise show upload form
    st.markdown("### Upload CSV Data File")
    
    # API Key input
    config = load_config()
    provider = config.get('llm', {}).get('provider', 'gemini')
    model = config.get('llm', {}).get('model', 'unknown')
    
    if provider == "groq":
        api_label = "🔑 Enter your Groq API Key"
        api_help = "Your API key will not be stored. Get one at https://console.groq.com/keys"
    else:
        api_label = "🔑 Enter your Gemini API Key"
        api_help = "Your API key will not be stored. Get one at https://makersuite.google.com/app/apikey"
    
    st.info(f"**Current Configuration**: {provider.upper()} - {model}")
    
    api_key = st.text_input(
        api_label,
        type="password",
        help=api_help
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your API key to continue.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: question, response, benchmark_answer"
    )
    
    if uploaded_file is not None:
        # Preview data
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
            
            with st.expander("👀 Preview Data (first 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Detect columns
            st.markdown("### Configure Evaluation")
            
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
                
                # Metric selection
                available_metrics = ['accuracy', 'comprehensiveness', 'faithfulness']
                selected_metrics = st.multiselect(
                    "Select Metrics to Evaluate",
                    options=available_metrics,
                    default=['accuracy'],
                    help="Choose which metrics to evaluate"
                )
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                batch_size = st.slider("Batch Size", 1, 20, config['batch']['size'], 
                                      help="Number of rows to process in parallel")
                max_concurrent = st.slider("Max Concurrent API Calls", 1, 10, config['batch']['max_concurrent'],
                                          help="Maximum number of simultaneous API calls")
            
            # Run evaluation
            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                if not selected_metrics:
                    st.error("Please select at least one metric to evaluate.")
                    return
                
                run_evaluation(
                    df=df,
                    api_key=api_key,
                    config=config,
                    db=db,
                    question_col=question_col,
                    response_col=response_col,
                    benchmark_col=benchmark_col,
                    selected_metrics=selected_metrics,
                    batch_size=batch_size,
                    max_concurrent=max_concurrent,
                    uploaded_file=uploaded_file
                )
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def run_evaluation(df, api_key, config, db, question_col, response_col, benchmark_col, 
                   selected_metrics, batch_size, max_concurrent, uploaded_file):
    """Run the evaluation pipeline"""
    
    start_time = time.time()
    
    # Update config with user settings
    config['batch']['size'] = batch_size
    config['batch']['max_concurrent'] = max_concurrent
    
    # Create evaluation run in database
    run_id = db.create_evaluation_run(
        user=st.session_state.username,
        input_file_name=uploaded_file.name,
        input_file_size=uploaded_file.size,
        metrics_evaluated=selected_metrics,
        total_rows=len(df),
        llm_model=config['llm']['model'],
        batch_size=batch_size
    )
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(
        api_key=api_key,
        config=config,
        database=db
    )
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    
    try:
        # Extract data
        questions = df[question_col].fillna("").tolist()
        responses = df[response_col].fillna("").tolist()
        benchmarks = df[benchmark_col].fillna("").tolist()
        
        # Evaluate each metric
        for idx, metric in enumerate(selected_metrics):
            status_text.markdown(f"**Evaluating {metric}... ({idx+1}/{len(selected_metrics)})**")
            
            # Progress callback
            def update_progress(progress, message):
                overall_progress = (idx + progress) / len(selected_metrics)
                progress_bar.progress(overall_progress)
                status_text.markdown(f"**{message}**")
            
            # Run evaluation
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
            
            # Store results
            results_df = pd.DataFrame(results)
            all_results[metric] = results_df
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate statistics
        avg_scores = {metric: results['score'].mean() for metric, results in all_results.items()}
        total_api_calls = sum(len(results) for results in all_results.values())
        
        # Update database
        db.update_evaluation_run(
            run_id=run_id,
            status="completed",
            average_scores=avg_scores,
            total_api_calls=total_api_calls,
            execution_time_seconds=execution_time
        )
        
        # Store results in session state to persist across reruns
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
        
        # Set flag to show results
        st.session_state.show_results = True
        
        progress_bar.progress(1.0)
        status_text.markdown("**✅ Evaluation completed!**")
        
        # Rerun to show results page
        st.rerun()
    
    except Exception as e:
        db.update_evaluation_run(
            run_id=run_id,
            status="failed",
            error_message=str(e)
        )
        st.error(f"❌ Evaluation failed: {str(e)}")
        raise

def show_history_page(db):
    """Show evaluation history"""
    
    st.markdown("### 📈 Evaluation History")
    
    # Get runs
    runs = db.get_evaluation_runs(user=st.session_state.username)
    
    if not runs:
        st.info("No evaluation history yet. Run your first evaluation to see results here.")
        return
    
    # Convert to DataFrame
    runs_df = pd.DataFrame(runs)
    
    # Display as table
    st.dataframe(
        runs_df[[
            'id', 'timestamp', 'input_file_name', 'metrics_evaluated', 
            'total_rows', 'status', 'average_scores', 'execution_time_seconds'
        ]],
        use_container_width=True,
        height=400
    )
    
    # Select run to view details
    st.markdown("---")
    st.markdown("### 🔍 View Run Details")
    
    run_id = st.selectbox(
        "Select a run to view details",
        options=runs_df['id'].tolist(),
        format_func=lambda x: f"Run #{x} - {runs_df[runs_df['id']==x]['timestamp'].values[0]}"
    )
    
    if run_id:
        # Get results
        results = db.get_evaluation_results(run_id)
        if results:
            results_df = pd.DataFrame(results)
            
            st.markdown(f"#### Results for Run #{run_id}")
            st.dataframe(results_df, use_container_width=True, height=400)

def show_settings_page(config):
    """Show settings page"""
    
    st.markdown("### ⚙️ Application Settings")
    
    st.info("Settings are configured in `config.yaml`. Changes here are for current session only.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### LLM Configuration")
        st.json(config.get('llm', {}))
        
        st.markdown("#### Batch Processing")
        st.json(config.get('batch', {}))
    
    with col2:
        st.markdown("#### Metrics Configuration")
        st.json(config.get('metrics', {}))
        
        st.markdown("#### Report Configuration")
        st.json(config.get('report', {}))

if __name__ == "__main__":
    main()
