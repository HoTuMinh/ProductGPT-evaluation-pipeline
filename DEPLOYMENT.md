# Deployment Guide - ProductGPT Evaluation Pipeline

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Security Setup](#security-setup)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+) or macOS
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.11 or higher
- **Gemini API Key**: Get from https://makersuite.google.com/app/apikey

### Optional
- **Docker**: 20.10+ and Docker Compose 2.0+ (for containerized deployment)
- **Nginx**: For reverse proxy and SSL (production)

---

## Installation

### Method 1: Docker Deployment (Recommended for Production)

#### Step 1: Install Docker & Docker Compose
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

#### Step 2: Clone and Setup
```bash
# Clone repository
git clone <repository-url>
cd evaluation-tool

# Verify structure
ls -la
```

#### Step 3: Configure Environment
```bash
# Copy environment template
cp .env.template .env

# Edit configuration
nano .env

# Set your API key:
# GEMINI_API_KEY=your_actual_api_key_here
# AUTH_PASSWORD=your_secure_password_here
```

#### Step 4: Deploy
```bash
# Build and start containers
docker-compose up -d

# Verify deployment
docker-compose ps

# Check logs
docker-compose logs -f
```

#### Step 5: Access Application
- Open browser: `http://localhost:8501`
- Default password: Check your `.env` file

---

### Method 2: Local Python Deployment

#### Step 1: Install Python Dependencies
```bash
# Navigate to project directory
cd evaluation-tool

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configure
```bash
# Copy environment template
cp .env.template .env

# Edit with your settings
nano .env
```

#### Step 3: Create Data Directories
```bash
mkdir -p data/uploads data/results
```

#### Step 4: Run Application
```bash
# Using startup script
./start.sh

# OR manually
streamlit run frontend/streamlit_app.py
```

---

## Configuration

### Application Configuration (`config.yaml`)

#### LLM Settings
```yaml
llm:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"  # or "gemini-1.5-pro"
  temperature: 0.2               # 0.0-1.0 (lower = more deterministic)
  max_tokens: 2048               # Max response length
  timeout: 60                    # API timeout in seconds
```

#### Batch Processing
```yaml
batch:
  size: 5                        # Rows processed in parallel
  max_concurrent: 3              # Max simultaneous API calls
  retry_attempts: 3              # Retry failed requests
  retry_delay: 2                 # Seconds between retries
```

**Performance Tuning**:
- **Small datasets (<100 rows)**: `size: 10`, `max_concurrent: 5`
- **Large datasets (>500 rows)**: `size: 5`, `max_concurrent: 2`
- **Rate limiting**: Reduce `max_concurrent` if hitting API limits

#### Metrics Configuration
```yaml
metrics:
  accuracy:
    enabled: true
    threshold: 0.7              # Minimum pass score
  comprehensiveness:
    enabled: true
    threshold: 0.6
  faithfulness:
    enabled: true
    threshold: 0.7
```

---

## Deployment Options

### Option A: Standalone Server (Small Team)

Best for: 5-10 users, low-moderate usage

```bash
# Start on specific port
streamlit run frontend/streamlit_app.py --server.port=8501

# Keep running in background
nohup streamlit run frontend/streamlit_app.py &
```

**Access**: `http://server-ip:8501`

---

### Option B: Docker with Nginx Reverse Proxy (Production)

Best for: Production environment, HTTPS required

#### Step 1: Setup Nginx
```bash
sudo apt-get install nginx
```

#### Step 2: Configure Nginx
Create `/etc/nginx/sites-available/evaluation-tool`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long-running evaluations
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

#### Step 3: Enable Site
```bash
sudo ln -s /etc/nginx/sites-available/evaluation-tool /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Step 4: Setup SSL (Let's Encrypt)
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### Option C: Docker with Custom Network (Isolated)

Best for: High security, isolated network

```yaml
# docker-compose.yml with custom network
version: '3.8'

services:
  evaluation-tool:
    build: .
    networks:
      - secure-net
    ports:
      - "127.0.0.1:8501:8501"  # Only localhost access

networks:
  secure-net:
    driver: bridge
    internal: true  # No external internet access
```

---

## Security Setup

### 1. Change Default Password

Edit `frontend/streamlit_app.py`:
```python
# Replace simple auth with proper authentication
if password == "YOUR_SECURE_PASSWORD":
    st.session_state.authenticated = True
```

### 2. Implement Multi-User Authentication

```python
# Example: Hash-based authentication
import hashlib

USERS = {
    "admin": hashlib.sha256("password123".encode()).hexdigest(),
    "user1": hashlib.sha256("userpass".encode()).hexdigest(),
}

def authenticate(username, password):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == pwd_hash
```

### 3. Secure API Keys

**Method 1: Environment Variables**
```bash
export GEMINI_API_KEY="your-key-here"
```

**Method 2: Secrets File** (Streamlit)
Create `.streamlit/secrets.toml`:
```toml
[api_keys]
gemini = "your-api-key-here"
```

Access in code:
```python
api_key = st.secrets["api_keys"]["gemini"]
```

### 4. Firewall Configuration
```bash
# Allow only specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 8501
sudo ufw enable
```

### 5. File Upload Restrictions

In `config.yaml`:
```yaml
upload:
  max_size_mb: 50
  allowed_extensions: [".csv"]
  scan_for_malware: true  # If antivirus available
```

---

## Monitoring & Maintenance

### 1. Application Logs

**Docker**:
```bash
docker-compose logs -f evaluation-tool
```

**Local**:
```bash
tail -f data/app.log
```

### 2. Database Management

**Backup**:
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/evaluation-tool"
DATE=$(date +%Y%m%d_%H%M%S)
cp data/logs.db "$BACKUP_DIR/logs_$DATE.db"
```

**Restore**:
```bash
cp /backups/evaluation-tool/logs_20260117_120000.db data/logs.db
```

### 3. Disk Usage Monitoring

```bash
# Check data directory size
du -sh data/

# Clean old results (older than 30 days)
find data/results/ -name "*.pdf" -mtime +30 -delete
```

### 4. API Usage Tracking

Query database for API costs:
```sql
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_calls,
    SUM(total_tokens) as total_tokens,
    SUM(estimated_cost_usd) as estimated_cost
FROM api_usage_logs
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### 5. Health Check Script

```bash
#!/bin/bash
# health_check.sh

# Check if application is running
if curl -s http://localhost:8501 > /dev/null; then
    echo "✓ Application is running"
else
    echo "✗ Application is down - restarting..."
    docker-compose restart
fi

# Check disk space
USAGE=$(df -h /data | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $USAGE -gt 80 ]; then
    echo "⚠ Disk usage high: ${USAGE}%"
fi

# Check database size
DB_SIZE=$(du -m data/logs.db | cut -f1)
if [ $DB_SIZE -gt 1000 ]; then
    echo "⚠ Database size large: ${DB_SIZE}MB"
fi
```

---

## Troubleshooting

### Issue: Application Won't Start

**Symptom**: Docker container exits immediately

**Solutions**:
```bash
# Check logs
docker-compose logs evaluation-tool

# Verify config
python test_setup.py

# Reset and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

### Issue: API Rate Limit Errors

**Symptom**: "429 Too Many Requests" errors

**Solutions**:
1. Reduce `batch.max_concurrent` in `config.yaml`
2. Increase `batch.retry_delay`
3. Check API quotas at https://console.cloud.google.com/

```yaml
# Conservative settings
batch:
  size: 3
  max_concurrent: 1
  retry_delay: 5
```

---

### Issue: Slow Evaluation Performance

**Causes**:
- Large batch size
- High API latency
- Network issues

**Solutions**:
```yaml
# Optimize batch processing
batch:
  size: 10              # Increase for faster processing
  max_concurrent: 5     # More parallel calls
```

Monitor performance:
```sql
SELECT 
    metric_name,
    AVG(api_call_duration_ms) as avg_latency_ms
FROM evaluation_results
GROUP BY metric_name;
```

---

### Issue: Database Locked

**Symptom**: "database is locked" error

**Cause**: Multiple simultaneous evaluations

**Solution**:
- Only run one evaluation at a time
- Use PostgreSQL for production if needed

---

### Issue: Memory Issues

**Symptom**: Out of memory errors

**Solutions**:
1. Process smaller batches
2. Increase Docker memory limit:
```yaml
# docker-compose.yml
services:
  evaluation-tool:
    mem_limit: 4g
```

---

## Production Checklist

Before deploying to production:

- [ ] Change default password
- [ ] Secure API keys (environment variables)
- [ ] Setup HTTPS with SSL certificate
- [ ] Configure firewall rules
- [ ] Setup automated backups
- [ ] Configure log rotation
- [ ] Test disaster recovery
- [ ] Document user credentials
- [ ] Setup monitoring/alerting
- [ ] Test with production data
- [ ] Load testing (if high usage expected)
- [ ] Security audit

---

## Support & Contact

For deployment issues:
- Email: [support@covergo.com]
- Documentation: [internal-wiki]
- Emergency: [on-call contact]

---

**Version**: 1.0.0  
**Last Updated**: January 2026
