# Technical Architecture - ProductGPT Evaluation Pipeline

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend Layer                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Streamlit Web Application                      │ │
│  │  • File Upload Interface                               │ │
│  │  • Configuration Panel                                 │ │
│  │  • Real-time Progress Tracking                         │ │
│  │  • Results Visualization                               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Evaluation Pipeline Orchestrator               │ │
│  │  • Request Routing                                     │ │
│  │  • Batch Processing Coordination                       │ │
│  │  • Progress Management                                 │ │
│  │  • Error Handling & Retry Logic                        │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         LLM Judge Evaluator                            │ │
│  │  • Prompt Engineering                                  │ │
│  │  • API Communication                                   │ │
│  │  • Response Parsing                                    │ │
│  │  • Score Calculation                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌──────────────────────────┐  ┌─────────────────────────────┐
│    External Services      │  │      Data Layer             │
│  ┌────────────────────┐  │  │  ┌───────────────────────┐ │
│  │  Google Gemini API │  │  │  │   SQLite Database     │ │
│  │  • Text Generation │  │  │  │  • Evaluation Runs    │ │
│  │  • Model: Gemini   │  │  │  │  • Results Storage    │ │
│  │    2.0 Flash       │  │  │  │  • API Usage Logs     │ │
│  └────────────────────┘  │  │  └───────────────────────┘ │
└──────────────────────────┘  │  ┌───────────────────────┐ │
                              │  │   File System         │ │
                              │  │  • Uploads            │ │
                              │  │  • Reports (PDF)      │ │
                              │  │  • Visualizations     │ │
                              │  └───────────────────────┘ │
                              └─────────────────────────────┘
```

---

## Component Details

### 1. Frontend Layer (Streamlit)

**File**: `frontend/streamlit_app.py`

**Responsibilities**:
- User authentication and session management
- File upload and validation
- Configuration interface
- Real-time progress display
- Results presentation
- Report download

**Key Features**:
- Single-page application (SPA) design
- Responsive layout with Bootstrap-inspired styling
- Interactive data tables with pandas DataFrames
- Real-time progress bars during evaluation
- Tabbed interface for multi-metric results

**Technology Stack**:
- Streamlit 1.30.0
- Pandas for data display
- Custom CSS for styling

---

### 2. Evaluation Pipeline

**File**: `backend/app/evaluator.py`

#### 2.1 LLMJudge Class

**Purpose**: Core evaluation engine using LLM-as-a-Judge methodology

**Key Methods**:

```python
class LLMJudge:
    def __init__(self, api_key, model, temperature, max_tokens)
    def _build_accuracy_prompt(question, response, benchmark) -> str
    def _build_comprehensiveness_prompt(...) -> str
    def _build_faithfulness_prompt(...) -> str
    def evaluate_single(metric, question, response, benchmark) -> Dict
    def evaluate_batch(metric, data, max_concurrent) -> List[Dict]
```

**Prompt Engineering Strategy**:

Each metric uses a specialized prompt template with:
1. **Role Definition**: "You are an expert evaluator..."
2. **Task Description**: Clear evaluation objective
3. **Input Data**: Question, Response, Benchmark
4. **Evaluation Criteria**: Specific rubric
5. **Scoring Instructions**: Scale and thresholds
6. **Output Format**: Structured JSON response

**Example Prompt Structure**:
```
You are an expert evaluator assessing [METRIC].

Task: [Description]

Question: [User question]
Response: [Chatbot response]
Benchmark: [Ground truth]

Evaluation Criteria:
- [Criterion 1]
- [Criterion 2]
...

Instructions:
1. [Step 1]
2. [Step 2]
...

Output Format:
{
    "score": 0.0-1.0,
    "label": "positive/negative",
    "reasoning": "Explanation..."
}
```

#### 2.2 EvaluationPipeline Class

**Purpose**: Orchestrate batch evaluation with logging

**Workflow**:
```
1. Initialize LLMJudge with API key and config
2. Split dataset into batches
3. For each batch:
   a. Create concurrent tasks (respecting max_concurrent limit)
   b. Execute evaluations in parallel
   c. Log results to database
   d. Track API usage
   e. Update progress callback
4. Aggregate and return results
```

**Concurrency Control**:
- Uses `asyncio.Semaphore` for rate limiting
- Configurable batch size and max concurrent requests
- Automatic retry with exponential backoff

---

### 3. Database Layer

**File**: `backend/app/database.py`

#### 3.1 Schema Design

**Table: evaluation_runs**
```sql
CREATE TABLE evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    user VARCHAR(100) NOT NULL,
    input_file_name VARCHAR(255) NOT NULL,
    input_file_size INTEGER,
    metrics_evaluated JSON,
    total_rows INTEGER,
    status VARCHAR(50),
    error_message TEXT,
    average_scores JSON,
    total_api_calls INTEGER,
    execution_time_seconds FLOAT,
    llm_model VARCHAR(100),
    batch_size INTEGER
);
```

**Table: evaluation_results**
```sql
CREATE TABLE evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    question TEXT,
    response TEXT,
    benchmark_answer TEXT,
    metric_name VARCHAR(50),
    score FLOAT,
    label VARCHAR(50),
    reasoning TEXT,
    human_reviewed INTEGER DEFAULT 0,
    human_score FLOAT,
    human_label VARCHAR(50),
    human_comment TEXT,
    review_timestamp DATETIME,
    api_call_duration_ms FLOAT,
    confidence FLOAT,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
);
```

**Table: api_usage_logs**
```sql
CREATE TABLE api_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    run_id INTEGER,
    provider VARCHAR(50),
    model VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    estimated_cost_usd FLOAT,
    latency_ms FLOAT,
    success INTEGER,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
);
```

#### 3.2 Database Operations

**Write Operations**:
- Create evaluation run (returns run_id)
- Add evaluation results (batch insert for performance)
- Log API usage (for cost tracking)
- Update run status (on completion/failure)

**Read Operations**:
- Get recent runs (with filtering by user)
- Get detailed results for specific run
- Query API usage statistics
- Generate analytics reports

---

### 4. Report Generator

**File**: `backend/app/reports.py`

#### 4.1 Visualization Components

**Confusion Matrix**:
- Compares human labels vs LLM judge labels
- Uses seaborn heatmap
- Shows agreement/disagreement patterns

**Score Distribution**:
- Histogram with KDE overlay
- Mean line and threshold line
- Identifies score patterns

**Metric Comparison**:
- Side-by-side bar charts
- Average scores and pass rates
- Cross-metric analysis

**Error Analysis**:
- Horizontal bar chart of lowest scoring examples
- Highlights areas for improvement
- Detailed per-row analysis

#### 4.2 PDF Generation

**Technology**: ReportLab

**Report Structure**:
1. **Title Page**
   - Application name and logo
   - Generation timestamp

2. **Executive Summary**
   - Run information table
   - High-level metrics summary
   - Key statistics

3. **Detailed Analysis (per metric)**
   - Statistical summary
   - Score distribution chart
   - Confusion matrix
   - Error analysis chart

4. **Cross-Metric Comparison**
   - Comparative charts
   - Overall performance analysis

**Styling**:
- Professional layout with consistent spacing
- Color-coded metrics (accuracy: blue, comprehensiveness: green, faithfulness: red)
- High-resolution charts (300 DPI)
- Table formatting with alternating row colors

---

## Data Flow

### Evaluation Request Flow

```
User Upload → CSV Validation → Column Mapping → Metric Selection
                                                        │
                                                        ▼
                                              Create DB Run Record
                                                        │
                                                        ▼
                                              Split into Batches
                                                        │
                                                        ▼
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                    Process Batch 1...N                               Process Batch N+1...M
                              │                                                   │
                              ▼                                                   ▼
                    Concurrent API Calls                              Concurrent API Calls
                    (max_concurrent limit)                            (max_concurrent limit)
                              │                                                   │
                              ▼                                                   ▼
                    Parse & Validate Responses                        Parse & Validate Responses
                              │                                                   │
                              ▼                                                   ▼
                    Log to Database                                   Log to Database
                              │                                                   │
                              └─────────────────────────┬─────────────────────────┘
                                                        ▼
                                              Aggregate Results
                                                        │
                                                        ▼
                                              Update Run Status
                                                        │
                                                        ▼
                                              Return to Frontend
                                                        │
                                                        ▼
                                              Display Results
                                                        │
                                                        ▼
                                              Generate PDF Report
```

---

## API Integration

### Gemini API Specification

**Endpoint**: `https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`

**Request Format**:
```python
{
    "contents": [
        {
            "role": "user",
            "parts": [{"text": prompt}]
        }
    ],
    "generationConfig": {
        "temperature": 0.2,
        "maxOutputTokens": 2048,
        "topP": 0.95,
        "topK": 40
    }
}
```

**Response Format**:
```python
{
    "candidates": [
        {
            "content": {
                "parts": [{"text": "..."}],
                "role": "model"
            },
            "finishReason": "STOP"
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 123,
        "candidatesTokenCount": 456,
        "totalTokenCount": 579
    }
}
```

**Error Handling**:
- 429: Rate limit → Retry with exponential backoff
- 500: Server error → Retry up to max_attempts
- 400: Bad request → Log and mark as failed
- Authentication errors → Alert user immediately

**Rate Limits** (as of Jan 2025):
- Free tier: 60 requests per minute
- Paid tier: Variable based on quota

---

## Performance Considerations

### Optimization Strategies

#### 1. Batch Processing
- **Small datasets (<100)**: Larger batches (10-20 rows)
- **Large datasets (>500)**: Smaller batches (3-5 rows) to prevent timeouts

#### 2. Concurrency Control
```python
# Conservative (avoid rate limits)
batch_size = 3
max_concurrent = 1

# Moderate (balanced)
batch_size = 5
max_concurrent = 3

# Aggressive (faster, may hit limits)
batch_size = 10
max_concurrent = 5
```

#### 3. Caching Strategy
- Cache parsed responses
- Reuse connections with keep-alive
- Database connection pooling (for future PostgreSQL migration)

#### 4. Memory Management
- Stream large CSV files instead of loading entirely
- Clear matplotlib figures after saving
- Limit in-memory result storage

### Benchmarks

**Test Setup**: 100 rows, 3 metrics, default settings

| Configuration | Time | API Calls | Notes |
|--------------|------|-----------|-------|
| batch=5, concurrent=3 | ~8 min | 300 | Recommended |
| batch=10, concurrent=5 | ~5 min | 300 | May hit rate limits |
| batch=3, concurrent=1 | ~15 min | 300 | Most conservative |

---

## Security Architecture

### 1. Authentication Layer
- Session-based authentication
- Password hashing (SHA-256 minimum, recommend bcrypt)
- Session timeout (24 hours default)
- CSRF protection (Streamlit built-in)

### 2. Data Security
- API keys stored in environment variables
- Database encryption at rest (optional)
- No sensitive data in logs
- Secure file upload validation

### 3. Network Security
- Reverse proxy with rate limiting (Nginx)
- HTTPS/TLS encryption
- Firewall rules restricting access
- IP whitelisting (optional)

### 4. Input Validation
```python
# File validation
allowed_extensions = ['.csv']
max_file_size = 50 * 1024 * 1024  # 50MB

# CSV validation
required_columns = ['question', 'response', 'benchmark_answer']
max_rows = 10000  # Prevent DOS

# API key validation
def validate_api_key(key):
    return len(key) > 20 and key.startswith('AI')
```

---

## Error Handling

### Error Categories

#### 1. User Errors
- Invalid file format → Clear error message
- Missing required columns → Suggest column names
- Empty data → Validation before processing

#### 2. System Errors
- Database connection failure → Retry logic
- Disk space full → Alert and cleanup
- Memory overflow → Reduce batch size

#### 3. API Errors
- Rate limiting → Exponential backoff
- Authentication failure → User notification
- Network timeout → Retry with longer timeout

### Logging Strategy

```python
# Log levels
DEBUG: Detailed debugging information
INFO: General operational messages
WARNING: Unexpected but recoverable issues
ERROR: Errors that prevent operation
CRITICAL: System failure

# Log format
timestamp - module - level - message
2026-01-17 10:30:45 - evaluator - INFO - Starting evaluation run #42
2026-01-17 10:30:46 - evaluator - ERROR - API call failed: Rate limit exceeded
```

---

## Future Enhancements

### Phase 2 Features
1. **Multi-user Management**
   - Role-based access control (admin, user, viewer)
   - Team collaboration features
   - Shared evaluation runs

2. **Advanced Analytics**
   - Time-series analysis of model performance
   - A/B testing between different prompts
   - Statistical significance testing

3. **Model Fine-tuning**
   - Collect human feedback for judge model
   - Few-shot learning with corrections
   - Custom metric definitions

4. **Integration APIs**
   - REST API for programmatic access
   - Webhook notifications on completion
   - Integration with CI/CD pipelines

### Phase 3 Features
1. **Distributed Processing**
   - Kubernetes deployment
   - Horizontal scaling
   - Load balancing

2. **Database Migration**
   - PostgreSQL for production
   - Redis for caching
   - Time-series database for metrics

3. **Advanced Visualizations**
   - Interactive dashboards (Plotly Dash)
   - Real-time monitoring
   - Anomaly detection

---

## Maintenance Procedures

### Daily
- Check application health
- Monitor disk usage
- Review error logs

### Weekly
- Database backup
- Clean old results (>30 days)
- Review API usage costs

### Monthly
- Security updates
- Performance optimization
- User feedback review

### Quarterly
- Full system audit
- Capacity planning
- Feature prioritization

---

## Glossary

**LLM Judge**: Large Language Model used to evaluate other model outputs
**CAG**: Cached Augmented Generation
**Batch Processing**: Processing multiple items together for efficiency
**Human-in-the-loop**: Manual review process integrated into automated workflow
**Pass Rate**: Percentage of evaluations meeting threshold
**Faithfulness**: Measure of output being grounded in source material
**Comprehensiveness**: Measure of output completeness
**Accuracy**: Measure of output factual correctness

---

**Document Version**: 1.0.0  
**Last Updated**: January 2026  
**Author**: AI Engineering Team, CoverGo
