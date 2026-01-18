# ProductGPT Evaluation Pipeline

A comprehensive LLM-as-a-Judge evaluation tool for assessing chatbot responses with automated scoring, visualization, and PDF reporting.

## 🎯 Features

- **Multi-Metric Evaluation**: Accuracy, Comprehensiveness, and Faithfulness
- **LLM Judge**: Uses Google Gemini 2.0 Flash for intelligent evaluation
- **Batch Processing**: Efficient parallel processing with rate limiting
- **Real-time Progress**: Live progress tracking during evaluation
- **Comprehensive Reports**: Automated PDF generation with visualizations
- **Interactive UI**: User-friendly Streamlit interface
- **Full Logging**: Complete audit trail of all evaluations and API usage
- **Human-in-the-loop**: Review and correct evaluations manually
- **On-Premise Deployment**: Docker-based deployment for security

## 📋 Requirements

- Docker & Docker Compose (recommended)
- OR Python 3.11+ with pip
- Google Gemini API key

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone and navigate to the project**:
```bash
cd evaluation-tool
```

2. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

3. **Access the application**:
Open your browser and go to `http://localhost:8501`

4. **Login**:
- Default password: `covergo2024` (⚠️ Change this in production!)

### Option 2: Local Development

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run frontend/streamlit_app.py
```

## 📊 Usage Guide

### 1. Prepare Your Data

Create a CSV file with the following columns:
- `question`: User's question
- `response`: Chatbot's response to evaluate
- `benchmark_answer`: Ground truth/expected answer

Example CSV structure:
```csv
question,response,benchmark_answer
"What is the coverage?","Coverage includes X, Y, Z","The coverage includes X, Y, Z with limits..."
```

### 2. Run Evaluation

1. **Login** to the application
2. **Enter your Gemini API Key** (get one at https://makersuite.google.com/app/apikey)
3. **Upload your CSV file**
4. **Select metrics** to evaluate (accuracy, comprehensiveness, faithfulness)
5. **Configure settings** (optional):
   - Batch size (default: 5)
   - Max concurrent API calls (default: 3)
6. **Click "Run Evaluation"**

### 3. View Results

- **Real-time progress**: Watch the progress bar as evaluation runs
- **Summary metrics**: View average scores and pass rates
- **Detailed results**: Explore per-row scores and reasoning
- **Generate PDF report**: Create comprehensive PDF with visualizations

### 4. Review History

- Navigate to "Evaluation History" to view past runs
- Review detailed results and statistics
- Compare different evaluation runs

## 📁 Project Structure

```
evaluation-tool/
├── backend/
│   └── app/
│       ├── database.py       # Database models and management
│       ├── evaluator.py      # LLM judge and evaluation pipeline
│       └── reports.py        # Report generation with visualizations
├── frontend/
│   └── streamlit_app.py      # Streamlit UI application
├── data/
│   ├── uploads/              # User uploaded files
│   ├── results/              # Generated reports
│   └── logs.db               # SQLite database
├── config.yaml               # Application configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### LLM Settings
```yaml
llm:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  temperature: 0.2
  max_tokens: 2048
```

### Batch Processing
```yaml
batch:
  size: 5                    # Rows per batch
  max_concurrent: 3          # Max parallel API calls
  retry_attempts: 3
  retry_delay: 2
```

### Metric Thresholds
```yaml
metrics:
  accuracy:
    threshold: 0.7           # Minimum score for "pass"
  comprehensiveness:
    threshold: 0.6
  faithfulness:
    threshold: 0.7
```

## 📈 Metrics Explained

### 1. Accuracy
Evaluates factual correctness of the response compared to benchmark:
- **1.0**: Perfectly accurate
- **0.7-0.9**: Mostly accurate with minor discrepancies
- **0.4-0.6**: Partially accurate
- **0.0-0.3**: Inaccurate

### 2. Comprehensiveness
Assesses completeness of the response:
- **1.0**: Fully comprehensive, covers all key points
- **0.7-0.9**: Mostly comprehensive
- **0.4-0.6**: Missing some important points
- **0.0-0.3**: Incomplete

### 3. Faithfulness
Checks if response is faithful to source material without hallucination:
- **1.0**: Perfectly faithful, no unsupported claims
- **0.7-0.9**: Mostly faithful
- **0.4-0.6**: Some unsupported claims
- **0.0-0.3**: Significant hallucinations

## 🔒 Security Considerations

### For Production Deployment:

1. **Change default password**: Edit authentication in `streamlit_app.py`
2. **Use environment variables**: Store API keys securely
3. **Enable HTTPS**: Use reverse proxy (nginx) with SSL
4. **Restrict network access**: Use firewall rules
5. **Regular backups**: Backup `data/logs.db` regularly
6. **Update dependencies**: Keep packages up-to-date

### Environment Variables (Recommended)
```bash
export GEMINI_API_KEY="your-api-key-here"
export AUTH_PASSWORD="your-secure-password"
```

## 🐛 Troubleshooting

### Common Issues

**1. API Rate Limits**
- Reduce `batch.max_concurrent` in config.yaml
- Increase `batch.retry_delay`

**2. Out of Memory**
- Reduce `batch.size` for large datasets
- Process in smaller chunks

**3. Database Locked**
- Only one evaluation at a time
- Wait for current evaluation to complete

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 Sample Data

Sample CSV files are available in the examples:
- `productgpt_accuracy.csv`: Accuracy evaluation example
- `productgpt_comprehensiveness.csv`: Comprehensiveness example
- `promotracker_faithfulness.csv`: Faithfulness example

## 🔄 Upgrade Guide

To update the application:

```bash
# Pull latest changes
git pull

# Rebuild Docker image
docker-compose down
docker-compose build
docker-compose up -d
```

## 📝 API Usage Tracking

All API calls are logged in the database with:
- Timestamp
- Model used
- Token counts (input/output)
- Latency
- Success/failure status

View API usage in the "Evaluation History" page.

## 🤝 Contributing

This is an internal tool. For issues or feature requests, contact the development team.

## 📄 License

Internal use only - CoverGo

## 🆘 Support

For support, please contact:
- Email: [your-email@covergo.com]
- Slack: #ai-team

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Author**: AI Engineering Team
