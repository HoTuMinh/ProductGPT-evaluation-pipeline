# Changelog

All notable changes to the ProductGPT Evaluation Pipeline will be documented in this file.

## [1.0.0] - 2026-01-17

### Initial Release

#### Added
- **Core Evaluation Engine**
  - LLM-as-a-Judge evaluation using Google Gemini 2.0 Flash
  - Support for three metrics: Accuracy, Comprehensiveness, Faithfulness
  - Batch processing with configurable concurrency
  - Automatic retry logic with exponential backoff
  - Real-time progress tracking

- **Web Interface**
  - Streamlit-based user interface
  - File upload with CSV validation
  - Interactive configuration panel
  - Real-time progress monitoring
  - Results visualization with tabs
  - Evaluation history browser

- **Database System**
  - SQLite database for logging
  - Three main tables: evaluation_runs, evaluation_results, api_usage_logs
  - Full audit trail of all evaluations
  - API usage tracking for cost management

- **Report Generation**
  - Automated PDF report creation
  - Multiple visualization types:
    - Confusion matrices
    - Score distributions
    - Metric comparisons
    - Error analysis charts
  - Comprehensive statistical summaries

- **Security Features**
  - Password-based authentication
  - Session management
  - API key protection (environment variables)
  - File upload validation

- **Deployment Options**
  - Docker containerization
  - Docker Compose setup
  - Local Python deployment
  - Nginx reverse proxy configuration

- **Documentation**
  - README with quick start guide
  - Comprehensive deployment guide
  - Technical architecture documentation
  - API usage examples
  - Troubleshooting guide

#### Features in Detail

**Evaluation Metrics**
- Accuracy: Measures factual correctness against benchmark
- Comprehensiveness: Assesses completeness of response
- Faithfulness: Checks for hallucinations and unsupported claims

**Batch Processing**
- Configurable batch size (default: 5 rows)
- Maximum concurrent API calls (default: 3)
- Intelligent rate limiting
- Automatic error recovery

**Visualization**
- Score distribution histograms with KDE curves
- Confusion matrices for label agreement
- Cross-metric comparison charts
- Lowest scoring examples analysis

**Configuration**
- YAML-based configuration
- Environment variable support
- Per-metric threshold settings
- Customizable batch processing parameters

#### Technical Specifications

**Technology Stack**
- Python 3.11+
- Streamlit 1.30.0
- Google Gemini API
- SQLite database
- ReportLab for PDF generation
- Matplotlib & Seaborn for visualizations
- Pandas for data processing

**System Requirements**
- Minimum 2GB RAM (4GB recommended)
- 10GB free disk space
- Python 3.11 or higher
- Gemini API key

**Performance**
- Processes ~100 rows in 8-10 minutes (default settings)
- Handles CSV files up to 50MB
- Supports up to 10,000 rows per evaluation

#### Known Limitations

1. **Database**
   - SQLite may have concurrency issues with multiple simultaneous users
   - Recommend PostgreSQL for production (future enhancement)

2. **API Rate Limits**
   - Gemini free tier: 60 requests/minute
   - May need to reduce concurrent calls to avoid limits

3. **File Processing**
   - Large files (>1000 rows) may take significant time
   - No pause/resume functionality (future enhancement)

4. **Authentication**
   - Simple password-based auth only
   - No user management system (future enhancement)

#### Security Considerations

- Default password should be changed in production
- API keys should be stored in environment variables
- HTTPS recommended for production deployment
- Regular security updates required

---

## [Unreleased]

### Planned Features

#### Version 1.1.0 (Q2 2026)
- [ ] Multi-user authentication system
- [ ] Role-based access control
- [ ] Improved error messages
- [ ] Export results to Excel
- [ ] Email notifications on completion

#### Version 1.2.0 (Q3 2026)
- [ ] REST API for programmatic access
- [ ] Webhook support
- [ ] Custom metric definitions
- [ ] Advanced analytics dashboard
- [ ] PostgreSQL support

#### Version 2.0.0 (Q4 2026)
- [ ] Distributed processing with Kubernetes
- [ ] Real-time collaborative features
- [ ] Model fine-tuning from feedback
- [ ] A/B testing framework
- [ ] Advanced visualization dashboard

### Under Consideration
- Integration with CI/CD pipelines
- Support for other LLM providers (OpenAI, Claude)
- Automated benchmark generation
- Natural language query interface
- Mobile app for result viewing

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-01-17 | Initial release with core functionality |

---

## Migration Guides

### Upgrading to Future Versions

*(Will be populated as new versions are released)*

---

## Bug Fixes

### Version 1.0.0
*Initial release - no previous bugs to fix*

---

## Contributors

- AI Engineering Team, CoverGo
- Intern: Hồ Tú Minh
- Supervisor: Nguyễn Hoàng Anh

---

## Support

For issues, questions, or feature requests:
- Email: ai-team@covergo.com
- Internal Slack: #evaluation-tool
- Documentation: /docs

---

**Format**: This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
**Versioning**: This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
