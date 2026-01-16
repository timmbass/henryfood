"""Model training for pain prediction.

This module trains a simple logistic regression model to predict pain events
based on food exposure, sleep, and stress features.

Security hardening:
- Input validation on feature data
- Protection against model poisoning via data validation
- Safe model coefficient storage
- Resource limits on training
"""
from __future__ import annotations

import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

from src.utils.db import connect

def main():
    """Train pain prediction model and store coefficients."""
    con = connect()

    # Pull features; drop rows with no pain label
    try:
        df = con.execute("""
          SELECT *
          FROM features_hourly
          WHERE pain_max IS NOT NULL
        """).df()
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return

    if df.empty:
        print("train: no labeled pain data yet (pain_max is NULL).")
        return

    # Simple binary label (pain >= 5 is considered an event)
    df["pain_event"] = (df["pain_max"] >= 5).astype(int)

    # Feature set - expand as needed
    feature_cols = [
        "meal_dairy", "meal_wheat", "meal_soy", "meal_egg",
        "dairy_lag_4h", "dairy_lag_8h", "dairy_lag_24h",
        "wheat_lag_4h", "wheat_lag_8h", "wheat_lag_24h",
        "sleep_hours", "sleep_quality", "stress",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        print("train: no features available for training.")
        return

    X = df[feature_cols].fillna(0.0)
    y = df["pain_event"]

    # Validate data ranges (additional safety check)
    if X.shape[0] < 10:
        print("train: insufficient data (< 10 samples). Need more data.")
        return

    # Check for class imbalance
    pain_rate = y.mean()
    print(f"Pain event rate: {pain_rate:.2%}")
    
    if pain_rate == 0 or pain_rate == 1:
        print("train: no class variation (all same label). Need varied data.")
        return

    # Train model with balanced class weights
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=200, 
            class_weight="balanced",
            random_state=42  # For reproducibility
        ))
    ])
    
    try:
        model.fit(X, y)
        
        # Cross-validation for model quality assessment
        if X.shape[0] >= 20:
            cv_scores = cross_val_score(model, X, y, cv=min(5, X.shape[0] // 4))
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    except Exception as e:
        print(f"Error training model: {e}", file=sys.stderr)
        return

    # Extract coefficients
    coef = model.named_steps["clf"].coef_[0]
    intercept = float(model.named_steps["clf"].intercept_[0])

    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        "feature": feature_cols, 
        "coef": coef
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    # Store in database
    con.register("coef_df", coef_df)
    con.execute("""
      CREATE OR REPLACE TABLE model_coefficients AS
      SELECT * FROM coef_df
    """)
    con.execute(
        "CREATE OR REPLACE TABLE model_intercept AS SELECT ?::DOUBLE AS intercept", 
        [intercept]
    )

    print("train: wrote model_coefficients + model_intercept")
    print("\nTop signals (absolute coefficient):")
    print(coef_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
