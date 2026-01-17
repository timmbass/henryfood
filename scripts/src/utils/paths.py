"""Path configuration for the henryfood pipeline.

Security considerations:
- Uses pathlib for safe path manipulation
- All paths are resolved to prevent directory traversal
- Base directory is explicitly defined
"""
from pathlib import Path

# Base directory - explicitly defined to prevent confusion
BASE = Path(__file__).resolve().parent.parent.parent.parent

RAW_DIR = BASE / "data" / "raw"
CURATED_DIR = BASE / "data" / "curated"
REPORTS_DIR = BASE / "reports"
LOGS_DIR = BASE / "logs"

DB_PATH = CURATED_DIR / "henry.duckdb"

# Ensure all paths are within BASE to prevent directory traversal
def validate_path(path: Path) -> Path:
    """Validate that a path is within the BASE directory."""
    resolved = path.resolve()
    try:
        resolved.relative_to(BASE)
        return resolved
    except ValueError:
        raise ValueError(f"Path {path} is outside the base directory {BASE}")
