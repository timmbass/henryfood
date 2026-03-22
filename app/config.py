"""Application configuration."""
from __future__ import annotations

import os
from pathlib import Path

# Base directory of the app package
APP_DIR = Path(__file__).parent
# Project root
ROOT_DIR = APP_DIR.parent
# Default database path
DEFAULT_DB_PATH = ROOT_DIR / "data" / "food_diary.db"
# Data directory
DATA_DIR = ROOT_DIR / "data"

def get_db_path() -> Path:
    """Return database path, using env override if set."""
    env_path = os.environ.get("FOOD_DIARY_DB")
    if env_path:
        return Path(env_path)
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_PATH
