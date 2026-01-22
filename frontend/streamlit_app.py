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
                # Simple auth with test account
                if (username == "123" and password == "123") or password == "covergo2024":
                    st.session_state.authenticated = True
                    st.session_state.username = username if username else "anonymous"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
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
                        
                        # Add original data to results for PDF
                        enriched_results = {}
                        for metric, results_df in eval_data['all_results'].items():
                            df_copy = results_df.copy()
                            df_copy['question'] = eval_data['questions']
                            df_copy['response'] = eval_data['responses']
                            df_copy['benchmark'] = eval_data['benchmarks']
                            enriched_results[metric] = df_copy
                        
                        report_gen.generate_pdf_report(
                            output_path=output_path,
                            run_info=run_info,
                            all_results=enriched_results,
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
    
    # Initialize settings in session state if not exists
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Information")
        st.write(f"**Logged in as:** {st.session_state.username}")
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        
        # Show API status
        if st.session_state.selected_provider:
            st.markdown("### 🔑 API Configuration")
            provider_icon = {"groq": "🚀", "gemini": "🔮", "openai": "🤖"}
            st.success(f"{provider_icon.get(st.session_state.selected_provider, '✅')} {st.session_state.selected_provider.upper()} - {st.session_state.selected_model}")
        else:
            st.markdown("### ⚠️ API Not Configured")
            st.warning("Please configure API in Settings")
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select Page",
            ["📖 Getting Started", "⚙️ Settings", "🔍 New Evaluation", "📈 Evaluation History"],
            label_visibility="collapsed"
        )
    
    # Main content
    st.markdown('<div class="main-header">🤖 ProductGPT Evaluation Pipeline</div>', unsafe_allow_html=True)
    
    if page == "📖 Getting Started":
        show_getting_started_page(config, db)
    elif page == "⚙️ Settings":
        show_settings_page(config, db)
    elif page == "🔍 New Evaluation":
        show_evaluation_page(config, db)
    elif page == "📈 Evaluation History":
        show_history_page(db)

def show_getting_started_page(config, db):
    """Show getting started guide for first-time users"""
    
    st.markdown("### 📖 Getting Started Guide")
    st.markdown("Welcome! This guide will help you get started with the ProductGPT Evaluation Pipeline.")
    
    # Create tabs for different sections
    tabs = st.tabs([
        "🚀 Quick Start",
        "📄 CSV Format", 
        "📊 Understanding Metrics",
        "📑 Reading Reports",
        "💾 Sample Dataset"
    ])
    
    # Tab 1: Quick Start
    with tabs[0]:
        st.markdown("## 🚀 Quick Start")
        st.markdown("""
        Follow these 5 simple steps to run your first evaluation:
        """)
        
        st.markdown("#### Step 1: Configure API")
        st.markdown("""
        1. Go to **⚙️ Settings** tab
        2. Click on a provider button (currently only Groq is available)
        3. Select a model
        4. Enter your API key
        5. Click **Save Configuration**
        
        💡 **Getting API Key:**
        - **Groq**: https://console.groq.com/keys (Free tier available)
        """)
        
        st.markdown("---")
        
        st.markdown("#### Step 2: Prepare Your Data")
        st.markdown("""
        Your CSV file must have these 3 columns:
        - `question` - The question asked
        - `response` - The AI's response
        - `benchmark_answer` - The expected/correct answer
        
        📥 Download our sample CSV below to see the format!
        """)
        
        st.markdown("---")
        
        st.markdown("#### Step 3: Upload & Configure")
        st.markdown("""
        1. Go to **🔍 New Evaluation** tab
        2. Upload your CSV file
        3. Verify column mappings are correct
        4. Select which metrics to evaluate (start with **Accuracy**)
        """)
        
        st.markdown("---")
        
        st.markdown("#### Step 4: Run Evaluation")
        st.markdown("""
        1. Click **🚀 Run Evaluation**
        2. Wait for processing (typically 1-2 seconds per sample)
        3. View results in the tabs
        """)
        
        st.markdown("---")
        
        st.markdown("#### Step 5: Review & Export")
        st.markdown("""
        1. Check the **Summary** metrics
        2. Review individual results in each metric tab
        3. Generate a **PDF Report** for sharing
        4. Download results as **CSV** for further analysis
        """)
    
    # Tab 2: CSV Format
    with tabs[1]:
        st.markdown("## 📄 CSV Format Requirements")
        
        st.markdown("### Required Columns")
        st.markdown("""
        Your CSV file **must** contain these 3 columns:
        
        | Column Name | Description | Example |
        |------------|-------------|---------|
        | `question` | The question asked to the AI | "What is the premium plan price?" |
        | `response` | The AI's actual response | "$99 per month" |
        | `benchmark_answer` | The correct/expected answer | "$149 per month" |
        """)
        
        st.markdown("---")
        
        st.markdown("### File Requirements")
        st.markdown("""
        - **Format**: CSV (comma-separated values)
        - **Encoding**: UTF-8
        - **Size**: Up to 10,000 rows recommended
        - **File size**: No hard limit, but larger files take longer to process
        """)
        
        st.markdown("---")
        
        st.markdown("### Example CSV Structure")
        
        # Show sample data preview
        sample_path = Path(__file__).parent.parent / "data" / "uploads" / "productgpt_accuracy.csv"
        
        if sample_path.exists():
            try:
                df_sample = pd.read_csv(sample_path)
                st.markdown("**Preview of sample file:**")
                st.dataframe(df_sample.head(3), use_container_width=True)
                
                st.info(f"📊 This sample contains {len(df_sample)} rows")
            except Exception as e:
                st.warning(f"Could not load sample file: {e}")
        
        st.markdown("---")
        
        st.markdown("### Common Issues")
        st.markdown("""
        ❌ **Column names don't match**
        - Make sure columns are named exactly: `question`, `response`, `benchmark_answer`
        - Column names are case-sensitive
        
        ❌ **Wrong file format**
        - Save as CSV, not Excel (.xlsx) or Google Sheets
        - Use "Save As" → "CSV (Comma delimited)"
        
        ❌ **Encoding issues**
        - Save with UTF-8 encoding
        - Avoid special characters or use Unicode properly
        """)
    
    # Tab 3: Understanding Metrics
    with tabs[2]:
        st.markdown("## 📊 Understanding Metrics")
        
        st.markdown("### Available Metrics")
        
        st.markdown("#### 1. Accuracy")
        st.markdown("""
        **What it measures:** Does the response correctly answer the question compared to the benchmark?
        
        **Scoring:**
        - `1.0` = Perfect match, completely correct
        - `0.7-0.9` = Mostly correct, minor differences
        - `0.4-0.6` = Partially correct
        - `0.0-0.3` = Incorrect or missing key information
        """)
        
        st.markdown("---")
        
        st.markdown("#### 2. Comprehensiveness")
        st.markdown("""
        **What it measures:** Does the response cover all important points from the benchmark?
        
        **Scoring:**
        - `1.0` = Covers all key points thoroughly
        - `0.7-0.9` = Covers most key points
        - `0.4-0.6` = Missing some important details
        - `0.0-0.3` = Incomplete or superficial
        """)
        
        st.markdown("---")
        
        st.markdown("#### 3. Faithfulness")
        st.markdown("""
        **What it measures:** Is the response consistent with the benchmark (no added/wrong info)?
        
        **Scoring:**
        - `1.0` = Fully faithful, no incorrect additions
        - `0.7-0.9` = Mostly faithful, minor extra details
        - `0.4-0.6` = Some contradictions or wrong info
        - `0.0-0.3` = Contains significant misinformation
        """)
    
    # Tab 4: Reading Reports
    with tabs[3]:
        st.markdown("## 📑 Reading Reports")
        
        st.markdown("### Summary Metrics")
        st.markdown("""
        At the top of your results, you'll see summary cards:
        
        ```
        Accuracy          Comprehensiveness
        0.550            0.780
        ▲ 60% pass rate  ▲ 80% pass rate
        ```
        
        **How to interpret:**
        - **Score (0-1)**: Average across all samples
        - **Pass Rate**: % of samples scoring ≥ 0.7
        - Higher is better for both
        """)
        
        st.markdown("---")
        
        st.markdown("### Detailed Results Tables")
        st.markdown("""
        Each metric tab shows a table with:
        - **Question**: Original question
        - **Score**: Individual score (0-1)
        - **Label**: Positive (≥0.7) or Negative (<0.7)
        - **Reasoning**: Why the LLM gave this score
        """)
        
        st.markdown("---")
        
        st.markdown("### PDF Reports")
        st.markdown("""
        Generated reports include:
        
        **Page 1: Overview**
        - Run information (date, user, file)
        - Results summary table
        - Metric comparison chart
        
        **Page 2+: Per Metric**
        - Score distribution
        - Label distribution (positive/negative)
        - Top 3 lowest scoring samples with full details
        """)
    
    # Tab 5: Sample Dataset
    with tabs[4]:
        st.markdown("## 💾 Sample Dataset")
        
        st.markdown("""
        Download our sample CSV to see the correct format and try your first evaluation!
        
        This sample contains 5 questions about a product (ProductGPT) with responses and benchmark answers.
        """)
        
        # Load and show sample
        sample_path = Path(__file__).parent.parent / "data" / "uploads" / "productgpt_accuracy.csv"
        
        if sample_path.exists():
            try:
                df_sample = pd.read_csv(sample_path)
                
                # Preview
                st.markdown("### Preview")
                st.dataframe(df_sample, use_container_width=True)
                
                st.markdown("---")
                
                # Download button
                st.markdown("### Download")
                
                csv_data = df_sample.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=csv_data,
                    file_name="sample_evaluation.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                
                st.success("✅ Click the button above to download the sample file")
                
                st.markdown("---")
                
                st.markdown("### Next Steps")
                st.markdown("""
                1. Download the sample CSV
                2. Go to **⚙️ Settings** and configure your API
                3. Go to **🔍 New Evaluation** and upload the sample
                4. Select **Accuracy** metric
                5. Click **Run Evaluation**
                6. Review your first results!
                """)
                
            except Exception as e:
                st.error(f"Error loading sample file: {e}")
                st.info("Please make sure the file exists at: data/uploads/productgpt_accuracy.csv")
        else:
            st.warning("⚠️ Sample file not found")
            st.info(f"Expected location: {sample_path}")
            st.markdown("""
            To add the sample file:
            1. Place your CSV at: `data/uploads/productgpt_accuracy.csv`
            2. Commit to GitHub
            3. Redeploy the app
            """)

def show_settings_page(config, db):
    """Show settings page for API configuration"""
    
    st.markdown("### ⚙️ API & Model Configuration")
    
    st.info("💡 Configure your API keys and model preferences here. Settings will be saved for your session.")
    
    # Provider selection
    st.markdown("#### 1️⃣ Select LLM Provider")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Groq", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "groq" else "secondary"):
            st.session_state.selected_provider = "groq"
            st.rerun()
    
    with col2:
        if st.button("🔮 Gemini", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "gemini" else "secondary"):
            st.session_state.selected_provider = "gemini"
            st.rerun()
    
    with col3:
        if st.button("🤖 OpenAI", use_container_width=True, type="primary" if st.session_state.get('selected_provider') == "openai" else "secondary"):
            st.session_state.selected_provider = "openai"
            st.rerun()
    
    if not st.session_state.get('selected_provider'):
        st.warning("⬆️ Please select a provider above")
        return
    
    st.markdown("---")
    
    # Provider-specific configuration
    provider = st.session_state.selected_provider
    
    st.markdown(f"#### 2️⃣ Configure {provider.upper()}")
    
    # Model selection based on provider
    if provider == "groq":
        st.markdown("**Available Models:**")
        models = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B - Best quality, balanced speed",
            "llama-3.1-8b-instant": "Llama 3.1 8B - Fastest, good quality",
            "mixtral-8x7b-32768": "Mixtral 8x7B - Large context window",
            "gemma2-9b-it": "Gemma 2 9B - Efficient"
        }
        selected_model = st.selectbox(
            "Model",
            options=list(models.keys()),
            format_func=lambda x: models[x],
            index=0 if not st.session_state.get('selected_model') else (list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        )
        
        api_help = "Get your API key at: https://console.groq.com/keys"
        
    elif provider == "gemini":
        st.markdown("**Available Models:**")
        models = {
            "gemini-1.5-flash": "Gemini 1.5 Flash - Fast & efficient",
            "gemini-1.5-pro": "Gemini 1.5 Pro - Best quality",
            "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)"
        }
        selected_model = st.selectbox(
            "Model",
            options=list(models.keys()),
            format_func=lambda x: models[x],
            index=0 if not st.session_state.get('selected_model') else (list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        )
        
        api_help = "Get your API key at: https://makersuite.google.com/app/apikey"
        
    else:  # openai
        st.markdown("**Available Models:**")
        models = {
            "gpt-4o": "GPT-4o - Latest, best quality",
            "gpt-4o-mini": "GPT-4o Mini - Fast & affordable",
            "gpt-4-turbo": "GPT-4 Turbo - Previous gen"
        }
        selected_model = st.selectbox(
            "Model",
            options=list(models.keys()),
            format_func=lambda x: models[x],
            index=0 if not st.session_state.get('selected_model') else (list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        )
        
        api_help = "Get your API key at: https://platform.openai.com/api-keys"
    
    st.caption(f"ℹ️ {api_help}")
    
    # API Key input
    current_key = st.session_state.api_keys.get(provider, "")
    
    api_key = st.text_input(
        f"🔑 {provider.upper()} API Key",
        value=current_key,
        type="password",
        help="Your API key is stored only in your session and never saved to disk"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=2048,
            step=256,
            help="Maximum length of model response"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of rows to process in parallel"
        )
        
        max_concurrent = st.slider(
            "Max Concurrent Calls",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum simultaneous API calls"
        )
    
    # Save button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please enter an API key")
            else:
                # Save to session state
                st.session_state.api_keys[provider] = api_key
                st.session_state.selected_provider = provider
                st.session_state.selected_model = selected_model
                st.session_state.temperature = temperature
                st.session_state.max_tokens = max_tokens
                st.session_state.batch_size = batch_size
                st.session_state.max_concurrent = max_concurrent
                
                # FIX: Changed provider.UPPER() to provider.upper()
                st.success(f"✅ Configuration saved! Provider: {provider.upper()}, Model: {selected_model}")
                st.balloons()
    
    with col2:
        if st.button("🧪 Test", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please enter an API key")
            else:
                with st.spinner("Testing connection..."):
                    try:
                        # Simple test
                        if provider == "groq":
                            try:
                                from groq import Groq
                                client = Groq(api_key=api_key)
                                response = client.chat.completions.create(
                                    model=selected_model,
                                    messages=[{"role": "user", "content": "Say 'test'"}],
                                    max_tokens=10
                                )
                                st.success("✅ Connection successful!")
                            except ImportError:
                                st.warning("⚠️ Groq package not installed. Connection will be tested during evaluation.")
                        elif provider == "gemini":
                            try:
                                import google.generativeai as genai
                                genai.configure(api_key=api_key)
                                model_obj = genai.GenerativeModel(selected_model)
                                response = model_obj.generate_content("Say 'test'")
                                st.success("✅ Connection successful!")
                            except ImportError:
                                st.warning("⚠️ Gemini package not installed. Connection will be tested during evaluation.")
                        else:
                            st.info("ℹ️ OpenAI test not implemented yet. Connection will be tested during evaluation.")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
    
    # Show current configuration
    if st.session_state.api_keys.get(provider):
        st.markdown("---")
        st.markdown("#### ✅ Current Configuration")
        
        config_data = {
            "Provider": provider.upper(),
            "Model": st.session_state.get('selected_model', selected_model),
            "API Key": "•" * 20 + st.session_state.api_keys[provider][-4:] if len(st.session_state.api_keys[provider]) > 4 else "Set",
            "Temperature": st.session_state.get('temperature', 0.2),
            "Max Tokens": st.session_state.get('max_tokens', 2048),
            "Batch Size": st.session_state.get('batch_size', 5),
            "Max Concurrent": st.session_state.get('max_concurrent', 3)
        }
        
        for key, value in config_data.items():
            st.text(f"{key}: {value}")
    
    # Tips
    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown("""
    - **Groq**: Best for speed & cost (free tier is generous)
    - **Gemini**: Good balance of quality & speed
    - **OpenAI**: Highest quality but more expensive
    - **Temperature**: 0.1-0.3 recommended for evaluation tasks
    - **Batch Size**: Increase for faster processing (watch rate limits)
    """)


def show_evaluation_page(config, db):
    """Show the main evaluation page"""
    
    # Check if API is configured
    if not st.session_state.selected_provider or not st.session_state.api_keys.get(st.session_state.selected_provider):
        st.warning("⚠️ API not configured. Please go to Settings first.")
        if st.button("⚙️ Go to Settings"):
            st.session_state.page = "⚙️ Settings"
            st.rerun()
        return
    
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
    
    # Show current API configuration
    provider = st.session_state.selected_provider
    model = st.session_state.selected_model
    
    st.info(f"**Using**: {provider.upper()} - {model}")
    
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
            st.error(f"❌ Error reading file: {str(e)}")

def run_evaluation(df, api_key, config, db, question_col, response_col, benchmark_col, 
                   selected_metrics, batch_size, max_concurrent, uploaded_file, provider=None, model=None):
    """Run the evaluation pipeline"""
    
    start_time = time.time()
    
    # Update config with user settings and session state
    config['batch']['size'] = batch_size
    config['batch']['max_concurrent'] = max_concurrent
    
    # Update LLM config from session state
    if provider:
        config['llm']['provider'] = provider
    if model:
        config['llm']['model'] = model
    if 'temperature' in st.session_state:
        config['llm']['temperature'] = st.session_state.temperature
    if 'max_tokens' in st.session_state:
        config['llm']['max_tokens'] = st.session_state.max_tokens
    
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

if __name__ == "__main__":
    main()
