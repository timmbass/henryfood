# Security Hardening Assessment

## Executive Summary

This document provides a comprehensive security assessment of the HenryFood health analytics pipeline and recommendations for hardening the codebase.

**Current Security Posture: MODERATE**
- ✅ Strong input validation and sanitization
- ✅ SQL injection prevention measures in place
- ✅ Path traversal protection implemented
- ✅ Resource limits configured
- ⚠️ No encryption at rest
- ⚠️ No audit logging
- ⚠️ No authentication/authorization (single-user system)

## Threat Model

### Assets to Protect
1. **Personal Health Information (PHI)**
   - Meal logs with food sensitivities
   - Symptom data including pain levels
   - Sleep and stress patterns
   - Correlation patterns revealing health conditions

2. **System Integrity**
   - Database consistency
   - Model training integrity
   - Report accuracy

3. **System Availability**
   - Pipeline processing capability
   - Data access for analysis

### Threat Actors
1. **Malicious local user** - Access to the file system
2. **Malware** - Running on the same system
3. **Accidental data corruption** - User error or bugs
4. **Data breach** - If system is compromised

### Attack Vectors
1. Malicious CSV input files
2. SQL injection through user data
3. Path traversal attacks
4. Resource exhaustion (DoS)
5. Model poisoning through crafted data
6. Data exfiltration if system compromised

## Implemented Security Controls

### 1. Input Validation & Sanitization

**Location**: `scripts/src/features/timeline.py`

```python
# File size limits
if file_size > 100 * 1024 * 1024:
    raise ValueError(f"File too large ({file_size} bytes). Max 100MB.")

# String length limits
df[col] = df[col].astype(str).str.strip().str[:1000]

# Numeric range validation
symptoms.loc[symptoms["pain"] < 0, "pain"] = 0
symptoms.loc[symptoms["pain"] > 10, "pain"] = 10
```

**Effectiveness**: HIGH
- Prevents DoS through large files
- Limits memory usage
- Validates data ranges
- Protects against malformed input

**Recommendations**:
- ✅ Add checksum validation for data integrity
- ✅ Implement schema validation (e.g., using pydantic)
- ✅ Add rate limiting for automated ingestion

### 2. SQL Injection Prevention

**Location**: `scripts/src/features/lag_features.py`, `scripts/src/utils/db.py`

```python
# Parameterized queries
q = """
SELECT COUNT(*)::INT
FROM information_schema.tables
WHERE table_name = ?
"""
return con.execute(q, [name]).fetchone()[0] > 0

# Tag sanitization
def _sanitize_tag(tag: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', tag)
    return sanitized[:50]
```

**Effectiveness**: HIGH
- All queries use parameterization where user input involved
- Tag-based searches sanitized
- No dynamic SQL with unsanitized input

**Recommendations**:
- ✅ Consider using an ORM for additional abstraction
- ✅ Add SQL query logging for auditing
- ✅ Implement prepared statement caching

### 3. Path Traversal Protection

**Location**: `scripts/src/utils/paths.py`

```python
def validate_path(path: Path) -> Path:
    """Validate that a path is within the BASE directory."""
    resolved = path.resolve()
    try:
        resolved.relative_to(BASE)
        return resolved
    except ValueError:
        raise ValueError(f"Path {path} is outside base directory")
```

**Effectiveness**: HIGH
- All file operations validate paths
- Uses pathlib for safe manipulation
- Prevents directory traversal attacks

**Recommendations**:
- ✅ Add file type validation (whitelist extensions)
- ✅ Implement file integrity checks
- ✅ Add inode/handle verification for race conditions

### 4. Resource Limits

**Location**: Multiple modules

```python
# Database thread limits
con.execute("PRAGMA threads=4;")

# File size limits
if file_size > 100 * 1024 * 1024:
    raise ValueError("File too large")

# String length limits
sanitized[:50]  # Tag limit
str[:1000]  # Field limit
```

**Effectiveness**: MODERATE
- Prevents basic DoS attacks
- Limits memory usage
- Controls CPU usage

**Recommendations**:
- ✅ Add memory usage monitoring
- ✅ Implement disk space checks before operations
- ✅ Add timeout limits on long-running operations
- ✅ Consider container resource limits (cgroups)

### 5. Error Handling & Logging

**Location**: All modules

```python
try:
    # Operation
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    return
```

**Effectiveness**: MODERATE
- Prevents crashes from bad input
- Logs errors for debugging
- Fails safely

**Recommendations**:
- ⚠️ **CRITICAL**: Implement structured logging
- ⚠️ **HIGH**: Add log rotation and retention policies
- ✅ Separate error types (validation vs system errors)
- ✅ Add security event logging

## Security Gaps & Remediation

### CRITICAL Priority

#### 1. No Encryption at Rest
**Risk**: If system is compromised, all PHI is exposed in plaintext

**Remediation**:
```python
# Option 1: DuckDB with encryption (when available)
# Option 2: File system encryption (LUKS, FileVault, BitLocker)
# Option 3: Application-level encryption

import cryptography.fernet as fernet

def encrypt_database(db_path: Path, key: bytes):
    """Encrypt database file with Fernet symmetric encryption."""
    f = fernet.Fernet(key)
    with open(db_path, 'rb') as file:
        encrypted = f.encrypt(file.read())
    with open(db_path, 'wb') as file:
        file.write(encrypted)
```

**Effort**: MEDIUM
**Impact**: HIGH

#### 2. No Audit Logging
**Risk**: Cannot detect unauthorized access or track data usage

**Remediation**:
```python
# Add audit log module
import logging
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_path: Path):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(self, table: str, operation: str, user: str):
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'table': table,
            'operation': operation,
            'user': user
        }
        self.logger.info(json.dumps(event))
```

**Effort**: LOW
**Impact**: HIGH

### HIGH Priority

#### 3. No Authentication/Authorization
**Risk**: Anyone with file system access can read/modify data

**Remediation**:
```python
# Add basic authentication wrapper
import getpass
import hashlib
import os

def authenticate():
    """Simple password protection for the pipeline."""
    stored_hash = os.getenv('HENRYFOOD_PASSWORD_HASH')
    if not stored_hash:
        return True  # No password set
    
    password = getpass.getpass('Enter password: ')
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if password_hash != stored_hash:
        raise PermissionError('Invalid password')
    
    return True
```

**Effort**: LOW
**Impact**: MEDIUM (for multi-user systems)

#### 4. No Data Integrity Verification
**Risk**: Cannot detect data tampering or corruption

**Remediation**:
```python
# Add checksums for data files
import hashlib

def compute_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_integrity(file_path: Path, expected_checksum: str) -> bool:
    """Verify file hasn't been tampered with."""
    actual = compute_checksum(file_path)
    return actual == expected_checksum
```

**Effort**: LOW
**Impact**: MEDIUM

### MEDIUM Priority

#### 5. Model Poisoning Risk
**Risk**: Malicious input data could bias the model

**Current Mitigation**: Input validation, range checks
**Additional Remediation**:
```python
# Add anomaly detection for training data
from sklearn.ensemble import IsolationForest

def detect_anomalies(X: pd.DataFrame) -> pd.DataFrame:
    """Detect and flag anomalous training samples."""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(X)
    return X[predictions == 1]  # Keep only normal samples
```

**Effort**: MEDIUM
**Impact**: MEDIUM

#### 6. No Secure Deletion
**Risk**: Deleted data may remain on disk

**Remediation**:
```python
import os

def secure_delete(file_path: Path):
    """Securely delete a file by overwriting before removal."""
    if not file_path.exists():
        return
    
    file_size = file_path.stat().st_size
    with open(file_path, 'wb') as f:
        # Overwrite with random data
        f.write(os.urandom(file_size))
    
    # Then delete
    file_path.unlink()
```

**Effort**: LOW
**Impact**: LOW (depends on threat model)

## Dependency Security

### Current Dependencies
```
duckdb>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Recommendations

1. **Pin exact versions** for reproducibility:
```
duckdb==0.9.2
pandas==2.1.4
numpy==1.26.2
python-dateutil==2.8.2
scikit-learn==1.3.2
tqdm==4.66.1
```

2. **Add security scanning**:
```bash
# Add to CI/CD
pip install safety bandit
safety check
bandit -r scripts/src/
```

3. **Regular updates**:
```bash
# Weekly dependency updates
pip list --outdated
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

4. **Add requirements-dev.txt**:
```
# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
bandit>=1.7.5
safety>=2.3.5
mypy>=1.7.0
```

## Operational Security

### Secure Deployment Checklist

- [ ] Install on encrypted file system
- [ ] Set proper file permissions (chmod 600 for data files)
- [ ] Use dedicated user account with limited privileges
- [ ] Enable automatic updates for OS security patches
- [ ] Configure firewall (no network access needed)
- [ ] Disable unnecessary services
- [ ] Regular backups to encrypted storage
- [ ] Test restore procedures
- [ ] Document incident response plan
- [ ] Regular security audits

### Backup Strategy

```bash
#!/bin/bash
# Secure backup script

BACKUP_DIR="/secure/backup/henryfood"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_PATH="/home/user/henryfood/data/curated/henry.duckdb"

# Create encrypted backup
tar -czf - henryfood/ | \
    openssl enc -aes-256-cbc -salt -pbkdf2 \
    -out "$BACKUP_DIR/henryfood_$TIMESTAMP.tar.gz.enc"

# Keep only last 30 days
find "$BACKUP_DIR" -name "henryfood_*.tar.gz.enc" -mtime +30 -delete
```

### Monitoring

```python
# Add monitoring module
import psutil
import logging

class ResourceMonitor:
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger('monitor')
    
    def check_resources(self):
        """Check system resources and alert if exceeded."""
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        if memory > self.max_memory_mb:
            self.logger.warning(f'Memory usage high: {memory:.1f}MB')
            raise MemoryError(f'Memory limit exceeded: {memory:.1f}MB')
```

## Code Review Checklist

For future changes, review against these security criteria:

### Input Handling
- [ ] All user input validated
- [ ] File size limits enforced
- [ ] String lengths bounded
- [ ] Numeric ranges checked
- [ ] Timestamps parsed safely
- [ ] Encoding validated (UTF-8)

### Data Access
- [ ] SQL queries parameterized
- [ ] Paths validated against base directory
- [ ] Read-only access where possible
- [ ] Resource limits enforced
- [ ] Errors handled gracefully

### Output Generation
- [ ] HTML/Markdown escaped
- [ ] File permissions set correctly
- [ ] Sensitive data not logged
- [ ] Reports sanitized

### Dependencies
- [ ] Version pinned
- [ ] Security scanned
- [ ] Regularly updated
- [ ] Minimal necessary packages

## Testing for Security

### Recommended Test Cases

```python
# tests/test_security.py

def test_sql_injection_protection():
    """Test that SQL injection attempts are blocked."""
    malicious_tag = "'; DROP TABLE events_meals; --"
    sanitized = _sanitize_tag(malicious_tag)
    assert ";" not in sanitized
    assert "DROP" not in sanitized

def test_path_traversal_protection():
    """Test that path traversal is blocked."""
    malicious_path = Path("../../etc/passwd")
    with pytest.raises(ValueError):
        validate_path(malicious_path)

def test_file_size_limits():
    """Test that oversized files are rejected."""
    # Create 200MB file
    large_file = Path("/tmp/large.csv")
    with open(large_file, 'wb') as f:
        f.write(b'0' * 200 * 1024 * 1024)
    
    with pytest.raises(ValueError, match="too large"):
        _read_csv_safe(large_file)

def test_xss_protection():
    """Test that XSS in reports is blocked."""
    malicious_note = "<script>alert('xss')</script>"
    escaped = _escape_html(malicious_note)
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
```

## Compliance Considerations

### GDPR (if applicable)
- Right to access ✅ (data in local database)
- Right to rectification ✅ (can modify database)
- Right to erasure ✅ (can delete data)
- Data portability ✅ (CSV export)
- Privacy by design ⚠️ (needs encryption)

### HIPAA (if applicable)
- Access controls ❌ (needs implementation)
- Audit logging ❌ (needs implementation)
- Encryption at rest ❌ (needs implementation)
- Encryption in transit ✅ (no network transmission)
- Minimum necessary ✅ (single user, minimal data)

## Conclusion

The current implementation provides a **solid foundation** with good input validation and SQL injection prevention. However, for use with sensitive health data, the following are **strongly recommended**:

1. **Implement encryption at rest** (CRITICAL)
2. **Add audit logging** (CRITICAL)
3. **Enable data integrity checks** (HIGH)
4. **Set up automated security scanning** (HIGH)
5. **Regular dependency updates** (HIGH)

The codebase demonstrates security awareness and follows many best practices. With the recommended enhancements, it would be suitable for production use with sensitive personal health information.

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
