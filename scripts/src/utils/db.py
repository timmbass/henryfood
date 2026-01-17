"""Database connection utilities for DuckDB.

Security hardening:
- Read-only mode available for queries
- Connection pooling disabled (single-user application)
- Progress bar enabled for long-running queries
- Explicit thread control
- Path validation to prevent injection
"""
from __future__ import annotations

import duckdb
from pathlib import Path
from .paths import DB_PATH, CURATED_DIR, validate_path

def connect(db_path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection with security-hardened settings.
    
    Args:
        db_path: Path to the database file
        read_only: If True, open in read-only mode
    
    Returns:
        DuckDB connection object
    
    Security:
        - Validates path is within expected directory
        - Supports read-only mode for queries
        - Limits threads for resource control
    """
    # Validate path to prevent directory traversal
    db_path = validate_path(db_path)
    
    # Ensure parent directory exists
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Connect with appropriate mode
    con = duckdb.connect(str(db_path), read_only=read_only)
    
    # Quality-of-life and security settings
    con.execute("PRAGMA threads=4;")  # Limit resource usage
    con.execute("PRAGMA enable_progress_bar=true;")
    
    return con

def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        con: Database connection
        name: Table name to check
    
    Returns:
        True if table exists, False otherwise
    
    Security:
        - Uses parameterized query to prevent SQL injection
    """
    # Use parameterized query to prevent SQL injection
    q = """
    SELECT COUNT(*)::INT
    FROM information_schema.tables
    WHERE table_name = ?
    """
    return con.execute(q, [name]).fetchone()[0] > 0
