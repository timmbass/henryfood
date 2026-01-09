from pathlib import Path

BASE = Path.cwd()
RAW_DIR = BASE / "data" / "raw"
CURATED_DIR = BASE / "data" / "curated"
REPORTS_DIR = BASE / "reports"
DB_PATH = CURATED_DIR / "henry.duckdb"
