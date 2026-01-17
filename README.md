# HenryFood Health Analytics Pipeline

A secure, privacy-focused health data analytics pipeline for tracking correlations between food intake, sleep, stress, and symptoms.

## Overview

This pipeline helps identify potential trigger foods and health patterns by:
- Tracking meals, symptoms, sleep, and stress in a structured timeline
- Generating lag features to identify delayed reactions (4h, 8h, 24h windows)
- Training simple interpretable models to identify correlations
- Generating weekly reports with top signals

## Architecture

```
henryfood/
├── data/
│   ├── raw/              # CSV input files (meals, symptoms, sleep, stress)
│   └── curated/          # DuckDB database (henry.duckdb)
├── reports/              # Generated weekly reports
├── logs/                 # Pipeline execution logs
└── scripts/
    ├── requirements.txt  # Python dependencies
    ├── Makefile         # Pipeline commands
    └── src/
        ├── utils/       # Database and path utilities
        ├── features/    # Timeline and feature generation
        ├── models/      # Model training
        └── reports/     # Report generation
```

## Quick Start

### 1. Install Dependencies

```bash
cd scripts
make install
```

### 2. Add Your Data

Edit the CSV files in `data/raw/`:
- `meals.csv` - Timestamped meal logs with tags
- `symptoms.csv` - Pain/symptom logs with severity
- `sleep.csv` - Daily sleep metrics
- `stress.csv` - Daily stress levels

### 3. Run the Pipeline

Daily (process new data):
```bash
make daily
```

Weekly (train model and generate report):
```bash
make weekly
```

View the report:
```bash
cat ../reports/weekly.md
```

## Data Format

### meals.csv
```csv
ts,meal_id,items,tags,notes
2026-01-05T07:30:00Z,m1,"toast with butter","wheat|dairy",""
```

### symptoms.csv
```csv
ts,pain,location,notes
2026-01-05T20:30:00Z,7,"stomach","evening pain"
```

### sleep.csv
```csv
date,sleep_hours,wake_ups,sleep_quality
2026-01-05,7.5,1,7
```

### stress.csv
```csv
date,stress,notes
2026-01-05,3,"normal day"
```

## Security Features

This pipeline implements multiple security hardening measures:

### 1. Input Validation
- CSV file size limits (100MB max)
- String length limits (1000 chars per field)
- Numeric range validation (pain 0-10, sleep 0-24h, etc.)
- Safe timestamp parsing with error handling

### 2. SQL Injection Prevention
- Parameterized queries throughout
- Tag sanitization (alphanumeric only)
- Safe string construction for dynamic SQL
- No user input directly in SQL strings

### 3. Path Traversal Protection
- Path validation against base directory
- All paths resolved and checked
- Uses pathlib for safe path manipulation

### 4. Data Privacy
- Local-only storage (no external transmission)
- Read-only database access for reporting
- Secure file permissions (0600 for reports)
- No external API calls

### 5. Resource Limits
- Thread limits on database (max 4)
- File size limits on CSV input
- String length limits on all fields
- Data volume checks in training

### 6. Code Quality
- Type hints throughout
- Comprehensive error handling
- Logging for debugging
- Clean separation of concerns

## Security Best Practices for Users

1. **Keep data local**: Never commit sensitive health data to public repositories
2. **Backup regularly**: Use encrypted backups for the database
3. **Limit access**: Use proper file permissions (chmod 600 for sensitive files)
4. **Review inputs**: Validate any data before ingesting
5. **Monitor resources**: Check disk space and memory usage
6. **Update dependencies**: Keep Python packages up to date for security patches

## Automated Security Scanning

This repository includes automated security scanning:
- CodeQL analysis for vulnerability detection
- Dependency scanning for known CVEs
- Code review for security issues

## Privacy Considerations

This is a **personal health tracking system** designed for:
- Single-user, local operation
- No cloud storage or transmission
- No third-party integrations by default
- Complete user control over data

### HIPAA Compliance Notes

If using for clinical purposes:
- This tool does NOT provide HIPAA compliance out-of-box
- Requires additional safeguards (encryption at rest, audit logging, access controls)
- Consult with compliance experts before clinical use

## Future Hardening Roadmap

Potential enhancements for increased security:

1. **Encryption**
   - Database encryption at rest
   - Encrypted backups
   - Memory encryption for sensitive data

2. **Audit Logging**
   - Track all data access
   - Log all modifications
   - Tamper-evident logging

3. **Access Control**
   - User authentication
   - Role-based access
   - Session management

4. **Data Anonymization**
   - PII detection and removal
   - Differential privacy for reports
   - Secure multi-party computation for group analysis

5. **Secure Development**
   - Pre-commit hooks for security checks
   - Automated dependency updates
   - Continuous security monitoring

## Contributing

When contributing, please:
1. Run security scanners before submitting PRs
2. Follow secure coding practices
3. Add tests for new features
4. Document security implications of changes

## License

[Add your license here]

## Disclaimer

This tool is for personal health tracking and research purposes only. It is not intended as medical advice or diagnosis. Always consult healthcare professionals for medical decisions.
