"""Minimal weekly training script (logistic regression baseline)"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.db import connect


def main():
    con = connect()

    # Pull features; drop rows with no pain label
    df = con.execute("""
      SELECT *
      FROM features_hourly
      WHERE pain_max IS NOT NULL
    """).df()

    if df.empty:
        print("train: no labeled pain data yet (pain_max is NULL).")
        return

    # Simple binary label
    df["pain_event"] = (df["pain_max"] >= 5).astype(int)

    # Minimal feature set to start; expand later
    feature_cols = [
        "meal_dairy", "meal_wheat", "meal_soy", "meal_egg",
        "dairy_lag_4h", "dairy_lag_8h", "dairy_lag_24h",
        "wheat_lag_4h", "wheat_lag_8h", "wheat_lag_24h",
        "sleep_hours", "sleep_quality", "stress",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0.0)
    y = df["pain_event"]

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    model.fit(X, y)

    coef = model.named_steps["clf"].coef_[0]
    intercept = float(model.named_steps["clf"].intercept_[0])

    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    con.register("coef_df", coef_df)
    con.execute("""
      CREATE OR REPLACE TABLE model_coefficients AS
      SELECT * FROM coef_df
    """)
    con.execute("CREATE OR REPLACE TABLE model_intercept AS SELECT ?::DOUBLE AS intercept", [intercept])

    print("train: wrote model_coefficients + model_intercept")
    print("Top signals:")
    print(coef_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
