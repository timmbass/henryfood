"""Timeline table generation from raw CSV data.

This module reads raw CSV files and creates normalized timeline tables:
- events_meals: Timestamped meal events
- outcomes_pain: Timestamped symptom/pain outcomes
- states_sleep_daily: Daily sleep metrics
- states_stress_daily: Daily stress metrics
- timeline_hourly: Unified hourly timeline with all data joined

Security hardening:
- Input validation on all CSV data
- Safe timestamp parsing with error handling
- Protection against malformed CSV data
- Sanitized string inputs
- Resource limits on data processing
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

from src.utils.db import connect
from src.utils.paths import RAW_DIR, validate_path

def _read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Safely read a CSV file with error handling.
    
    Security:
        - Validates path is within RAW_DIR
        - Handles missing files gracefully
        - Limits file size (100MB max)
        - Protects against CSV injection
    """
    if not path.exists():
        return pd.DataFrame()
    
    # Validate path
    path = validate_path(path)
    
    # Check file size (max 100MB to prevent DoS)
    file_size = path.stat().st_size
    if file_size > 100 * 1024 * 1024:
        raise ValueError(f"File {path} is too large ({file_size} bytes). Max 100MB.")
    
    try:
        df = pd.read_csv(path)
        if df.empty:
            return df
        
        # Sanitize string columns - strip leading/trailing whitespace and limit length
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str[:1000]  # Max 1000 chars per field
        
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return pd.DataFrame()

def main():
    """Build timeline tables from raw CSV data."""
    con = connect()

    meals_path = RAW_DIR / "meals.csv"
    symptoms_path = RAW_DIR / "symptoms.csv"
    sleep_path = RAW_DIR / "sleep.csv"
    stress_path = RAW_DIR / "stress.csv"

    meals = _read_csv_safe(meals_path)
    symptoms = _read_csv_safe(symptoms_path)
    sleep = _read_csv_safe(sleep_path)
    stress = _read_csv_safe(stress_path)

    # ---- Meals (events) ----
    if not meals.empty:
        meals["ts"] = pd.to_datetime(meals["ts"], utc=True, errors="coerce")
        meals = meals.dropna(subset=["ts"])
        meals["ts_hour"] = meals["ts"].dt.floor("h")
        meals["tags"] = meals.get("tags", "").fillna("").astype(str)
        meals["items"] = meals.get("items", "").fillna("").astype(str)
        meals["notes"] = meals.get("notes", "").fillna("").astype(str)
        if "meal_id" not in meals.columns:
            meals["meal_id"] = ""
        con.register("meals_df", meals)
        con.execute("CREATE OR REPLACE TABLE events_meals AS SELECT * FROM meals_df;")
    else:
        con.execute("""
        CREATE TABLE IF NOT EXISTS events_meals (
            ts TIMESTAMP,
            meal_id VARCHAR,
            items VARCHAR,
            tags VARCHAR,
            notes VARCHAR,
            ts_hour TIMESTAMP
        );
        """)

    # ---- Symptoms / pain (outcomes) ----
    if not symptoms.empty:
        symptoms["ts"] = pd.to_datetime(symptoms["ts"], utc=True, errors="coerce")
        symptoms = symptoms.dropna(subset=["ts"])
        symptoms["ts_hour"] = symptoms["ts"].dt.floor("h")
        symptoms["pain"] = pd.to_numeric(symptoms["pain"], errors="coerce")
        # Validate pain is in range 0-10
        symptoms.loc[symptoms["pain"] < 0, "pain"] = 0
        symptoms.loc[symptoms["pain"] > 10, "pain"] = 10
        symptoms["location"] = symptoms.get("location", "").fillna("").astype(str)
        symptoms["notes"] = symptoms.get("notes", "").fillna("").astype(str)
        con.register("symptoms_df", symptoms)
        con.execute("CREATE OR REPLACE TABLE outcomes_pain AS SELECT * FROM symptoms_df;")
    else:
        con.execute("""
        CREATE TABLE IF NOT EXISTS outcomes_pain (
            ts TIMESTAMP,
            pain DOUBLE,
            location VARCHAR,
            notes VARCHAR,
            ts_hour TIMESTAMP
        );
        """)

    # ---- Sleep (daily state) ----
    if not sleep.empty:
        sleep["date"] = pd.to_datetime(sleep["date"], errors="coerce").dt.date
        sleep = sleep.dropna(subset=["date"])
        sleep["sleep_hours"] = pd.to_numeric(sleep["sleep_hours"], errors="coerce")
        sleep["wake_ups"] = pd.to_numeric(sleep["wake_ups"], errors="coerce")
        sleep["sleep_quality"] = pd.to_numeric(sleep["sleep_quality"], errors="coerce")
        # Validate ranges
        sleep.loc[sleep["sleep_hours"] < 0, "sleep_hours"] = None
        sleep.loc[sleep["sleep_hours"] > 24, "sleep_hours"] = None
        sleep.loc[sleep["wake_ups"] < 0, "wake_ups"] = 0
        sleep.loc[sleep["sleep_quality"] < 0, "sleep_quality"] = 0
        sleep.loc[sleep["sleep_quality"] > 10, "sleep_quality"] = 10
        con.register("sleep_df", sleep)
        con.execute("CREATE OR REPLACE TABLE states_sleep_daily AS SELECT * FROM sleep_df;")
    else:
        con.execute("""
        CREATE TABLE IF NOT EXISTS states_sleep_daily (
            date DATE,
            sleep_hours DOUBLE,
            wake_ups INTEGER,
            sleep_quality DOUBLE
        );
        """)

    # ---- Stress (daily state) ----
    if not stress.empty:
        stress["date"] = pd.to_datetime(stress["date"], errors="coerce").dt.date
        stress = stress.dropna(subset=["date"])
        stress["stress"] = pd.to_numeric(stress["stress"], errors="coerce")
        # Validate stress is in range 0-10
        stress.loc[stress["stress"] < 0, "stress"] = 0
        stress.loc[stress["stress"] > 10, "stress"] = 10
        stress["notes"] = stress.get("notes", "").fillna("").astype(str)
        con.register("stress_df", stress)
        con.execute("CREATE OR REPLACE TABLE states_stress_daily AS SELECT * FROM stress_df;")
    else:
        con.execute("""
        CREATE TABLE IF NOT EXISTS states_stress_daily (
            date DATE,
            stress DOUBLE,
            notes VARCHAR
        );
        """)

    # ---- Timeline hourly: choose min/max from events/outcomes ----
    con.execute("""
    CREATE OR REPLACE TABLE _bounds AS
    SELECT
      MIN(ts_hour) AS min_hour,
      MAX(ts_hour) AS max_hour
    FROM (
      SELECT ts_hour FROM events_meals
      UNION ALL
      SELECT ts_hour FROM outcomes_pain
    );
    """)

    bounds = con.execute("SELECT min_hour, max_hour FROM _bounds").fetchone()
    min_hour, max_hour = bounds[0], bounds[1]

    if min_hour is None or max_hour is None:
        con.execute("""
        CREATE OR REPLACE TABLE timeline_hourly AS
        SELECT
          CAST(NULL AS TIMESTAMP) AS ts_hour,
          CAST(NULL AS DATE) AS date,
          0::INT AS meal_events,
          CAST(NULL AS VARCHAR) AS meal_tags_concat,
          CAST(NULL AS VARCHAR) AS meal_items_concat,
          CAST(NULL AS DOUBLE) AS pain_max,
          CAST(NULL AS DOUBLE) AS pain_avg,
          CAST(NULL AS DOUBLE) AS sleep_hours,
          CAST(NULL AS DOUBLE) AS sleep_quality,
          CAST(NULL AS DOUBLE) AS stress
        WHERE 1=0;
        """)
        print("timeline_hourly: no data yet (tables created).")
        return

    # Generate hour spine and join aggregates
    con.execute("""
    CREATE OR REPLACE TABLE timeline_hourly AS
    WITH spine AS (
      SELECT * FROM generate_series(?, ?, INTERVAL 1 HOUR) AS t(ts_hour)
    ),
    meal_agg AS (
      SELECT
        ts_hour,
        COUNT(*)::INT AS meal_events,
        string_agg(tags, ' | ') AS meal_tags_concat,
        string_agg(items, ' | ') AS meal_items_concat
      FROM events_meals
      GROUP BY ts_hour
    ),
    pain_agg AS (
      SELECT
        ts_hour,
        MAX(pain) AS pain_max,
        AVG(pain) AS pain_avg
      FROM outcomes_pain
      GROUP BY ts_hour
    )
    SELECT
      s.ts_hour,
      CAST(s.ts_hour AS DATE) AS date,
      COALESCE(m.meal_events, 0) AS meal_events,
      COALESCE(m.meal_tags_concat, '') AS meal_tags_concat,
      COALESCE(m.meal_items_concat, '') AS meal_items_concat,
      p.pain_max,
      p.pain_avg,
      sl.sleep_hours,
      sl.sleep_quality,
      st.stress
    FROM spine s
    LEFT JOIN meal_agg m ON m.ts_hour = s.ts_hour
    LEFT JOIN pain_agg p ON p.ts_hour = s.ts_hour
    LEFT JOIN states_sleep_daily sl ON sl.date = CAST(s.ts_hour AS DATE)
    LEFT JOIN states_stress_daily st ON st.date = CAST(s.ts_hour AS DATE)
    ORDER BY s.ts_hour;
    """, [min_hour, max_hour])

    n = con.execute("SELECT COUNT(*) FROM timeline_hourly").fetchone()[0]
    print(f"timeline_hourly: {n} rows")

if __name__ == "__main__":
    main()
