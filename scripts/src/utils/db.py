"""DuckDB helper utilities"""
from __future__ import annotations

from pathlib import Path
import duckdb
from .paths import DB_PATH, CURATED_DIR


def connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads=4;")
    return con


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = """
    SELECT COUNT(*)::INT
    FROM information_schema.tables
    WHERE table_name = ?
    """
    return con.execute(q, [name]).fetchone()[0] > 0
