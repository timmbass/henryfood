#!/usr/bin/env python3
"""Train a Random Forest on meal-level data (meal-level unit).

Labeling: derives meal-level labels from a symptoms/events file when
--symptoms is provided. Meal datetimes are inferred from a datetime-like
column or from 'date' + 'meal' columns using anchors (Breakfast 08:00,
Lunch 12:00, Dinner 18:00).

Split: time-based holdout; last 20% (configurable) reserved for test.

Metrics: PR-AUC primary for binary; MAE primary for regression.

RF params (small-data defaults): n_estimators=500, min_samples_leaf=2,
max_features='sqrt' (classifier) / default (regressor), class_weight='balanced'
for classifier.

Outputs (written to --out-dir): model.pkl, metrics.json, features.json,
feature_importances.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MEAL_ANCHORS = {"breakfast": "08:00:00", "lunch": "12:00:00", "dinner": "18:00:00"}
DEFAULT_NUTRIENT_COLS = [
    "calories",
    "protein_g",
    "fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
    "saturated_fat_g",
]


def _parse_datetime_cols(df: pd.DataFrame) -> pd.Series:
    # prefer explicit datetime-like columns
    for c in ("meal_datetime", "datetime", "timestamp", "ts", "time"):
        if c in df.columns:
            return pd.to_datetime(df[c])
    # fallback: date + meal anchors
    if "date" in df.columns and "meal" in df.columns:
        def _anchor(row):
            try:
                date = str(row["date"]).split(" ")[0]
            except Exception:
                return pd.NaT
            meal = str(row.get("meal", "")).lower()
            anchor = MEAL_ANCHORS.get(meal)
            if anchor:
                return pd.to_datetime(f"{date} {anchor}")
            return pd.to_datetime(f"{date} 12:00:00")
        return df.apply(_anchor, axis=1)
    # date-only -> set to midday
    if "date" in df.columns:
        return pd.to_datetime(df["date"]).dt.normalize() + pd.to_timedelta("12:00:00")
    raise ValueError("No datetime-like column found in meals; provide meal_datetime or date+meal")


def build_labels_from_symptoms(meals_df: pd.DataFrame, symptoms_df: pd.DataFrame, window_hours: float = 4.0, severity_col: str = "severity") -> pd.DataFrame:
    meals = meals_df.copy()
    meals["meal_datetime"] = _parse_datetime_cols(meals)

    syms = symptoms_df.copy()
    # detect timestamp column in symptoms
    time_col = None
    for c in ("timestamp", "time", "datetime"):
        if c in syms.columns:
            time_col = c
            break
    if time_col is None:
        time_col = syms.columns[0]
    syms["_sym_dt"] = pd.to_datetime(syms[time_col])
    if severity_col in syms.columns:
        syms["_severity"] = pd.to_numeric(syms[severity_col], errors="coerce").fillna(0.0)
    else:
        syms["_severity"] = 1.0

    sym_times = syms["_sym_dt"].values
    sym_sev = syms["_severity"].values
    wh = pd.to_timedelta(window_hours, unit="h")

    bins = []
    regs = []
    for md in meals["meal_datetime"].values:
        if pd.isnull(md):
            bins.append(0)
            regs.append(0.0)
            continue
        deltas = sym_times - md
        mask = (deltas >= np.timedelta64(0, "s")) & (deltas <= wh)
        if mask.any():
            bins.append(1)
            regs.append(float(np.nanmax(sym_sev[mask])))
        else:
            bins.append(0)
            regs.append(0.0)
    meals["label_bin"] = bins
    meals["label_reg"] = regs
    return meals


def time_holdout_split(df: pd.DataFrame, time_col: str = "meal_datetime", test_frac: float = 0.2):
    df2 = df.sort_values(time_col).reset_index(drop=True)
    n = len(df2)
    cut = max(1, int(np.floor((1.0 - test_frac) * n)))
    train = df2.iloc[:cut].reset_index(drop=True)
    test = df2.iloc[cut:].reset_index(drop=True)
    return train, test


def build_pipeline(is_classifier: bool, seed: int = 42):
    imputer = SimpleImputer(strategy="median")
    if is_classifier:
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
        )
    else:
        # use default max_features (None) for regressor to avoid sklearn warnings
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=None, random_state=seed)
    pipe = Pipeline([("imputer", imputer), ("model", model)])
    return pipe


def evaluate_classifier(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    # get probability for positive class (1). Handle edge case where the
    # classifier was trained on a single class: predict_proba then returns
    # shape (n_samples, 1). Inspect the trained estimator's classes_ to map
    # columns correctly or synthesize a vector when only the negative class
    # is present.
    proba = pipe.predict_proba(X_test)
    # model object (sklearn estimator) lives in pipeline's 'model' step
    model = pipe.named_steps.get('model')
    try:
        classes = getattr(model, 'classes_', None)
    except Exception:
        classes = None

    if proba.ndim == 2 and proba.shape[1] == 1:
        # Only one class seen during training.
        if classes is not None and len(classes) == 1 and classes[0] == 1:
            probs = proba[:, 0]
        else:
            # Only negative class (0) seen during training — positive probs are 0.
            probs = np.zeros(proba.shape[0])
    else:
        # Normal case: two (or more) columns. Find index for class==1 if possible.
        if classes is not None and 1 in classes:
            pos_idx = int(np.where(classes == 1)[0][0])
        else:
            # fallback to last column (common in binary classifiers)
            pos_idx = proba.shape[1] - 1
        probs = proba[:, pos_idx]

    pr = float(average_precision_score(y_test, probs))
    try:
        roc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc = None
    return {"pr_auc": pr, "roc_auc": roc}


def evaluate_regressor(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"mae": mae, "r2": r2}


def _add_lag_features(df: pd.DataFrame, nutrient_cols: list, lag_days: list[int]):
    """For each row, compute sum of nutrients in prior N days (exclusive of current meal).
    Adds columns like 'lag1d_calories', 'lag3d_protein_g', etc.
    This is O(N^2) but acceptable for small datasets.
    """
    df = df.sort_values('meal_datetime').reset_index(drop=True)
    for d in (lag_days or []):
        col_prefix = f'lag{d}d_'
        vals = []
        delta = pd.to_timedelta(d, unit='d')
        times = df['meal_datetime'].values
        for i, t in enumerate(times):
            if pd.isnull(t):
                # produce NaNs for missing time
                vals.append({c: np.nan for c in nutrient_cols})
                continue
            start = t - delta
            # select prior rows strictly before t and >= start
            mask = (times >= np.datetime64(start)) & (times < np.datetime64(t))
            # sum nutrients for masked rows
            if mask.any():
                subset = df.loc[mask, nutrient_cols]
                s = subset.sum(skipna=True)
            else:
                s = pd.Series({c: 0.0 for c in nutrient_cols})
            vals.append(s.to_dict())
        # append columns
        for c in nutrient_cols:
            df[f'{col_prefix}{c}'] = [v.get(c, np.nan) for v in vals]
    return df


def _maybe_compute_shap(pipe, X_test, feat_cols, out_dir: Path, is_clf: bool):
    try:
        import shap
    except Exception:
        logger.info('shap not installed; skipping SHAP importance')
        return
    try:
        # get trained estimator and imputer
        imputer = pipe.named_steps['imputer']
        model = pipe.named_steps['model']
        X_imp = imputer.transform(X_test)
        # for tree models, TreeExplainer is appropriate
        expl = shap.TreeExplainer(model)
        shap_vals = expl.shap_values(X_imp)
        # shap_vals for classifier is list per class; take positive class if present
        if is_clf and isinstance(shap_vals, list) and len(shap_vals) > 1:
            arr = np.abs(shap_vals[1]).mean(axis=0)
        else:
            arr = np.abs(shap_vals).mean(axis=0)
        import pandas as _pd
        _pd.DataFrame({'feature': feat_cols, 'shap_mean_abs': arr}).to_csv(out_dir / 'shap_importances.csv', index=False)
        logger.info('Wrote SHAP importances to %s', out_dir / 'shap_importances.csv')
    except Exception as e:
        logger.warning('Failed to compute SHAP values: %s', e)


def run_train(args) -> dict:
    # load meals
    meals_path = Path(args.meals)
    meals = pd.read_parquet(meals_path) if meals_path.suffix.lower() in ('.parquet', '.pq') else pd.read_csv(meals_path)

    # symptoms required for labeling in this script
    if not args.symptoms:
        raise ValueError('Provide --symptoms (CSV/Parquet of symptom events) for meal-level labeling')
    syms_path = Path(args.symptoms)
    syms = pd.read_parquet(syms_path) if syms_path.suffix.lower() in ('.parquet', '.pq') else pd.read_csv(syms_path)

    # parse windows and lag-days
    windows = [float(x) for x in (args.windows.split(',') if isinstance(args.windows, str) else args.windows)]
    lag_days = [int(x) for x in (args.lag_days.split(',') if isinstance(args.lag_days, str) else args.lag_days or [])]

    ml_path = Path(args.ml_parquet)
    if not ml_path.exists():
        raise FileNotFoundError(f'ML parquet not found: {ml_path}')
    ml_df = pd.read_parquet(ml_path)
    if 'canonical' not in ml_df.columns or 'canonical' not in meals.columns:
        raise ValueError("Both meals and ML parquet must have 'canonical' column to join")

    results = {}
    for w in windows:
        label_window = float(w)
        logger.info('Running training for window=%.1fh', label_window)
        meals_labeled = build_labels_from_symptoms(meals, syms, window_hours=label_window)
        merged = meals_labeled.merge(ml_df, on='canonical', how='left')
        if 'meal_datetime' not in merged.columns:
            merged['meal_datetime'] = _parse_datetime_cols(merged)

        # select nutrient feature cols that exist
        feat_cols = [c for c in DEFAULT_NUTRIENT_COLS if c in merged.columns]
        if not feat_cols:
            raise ValueError('No nutrient feature columns available in merged data')

        # add lag features
        merged = _add_lag_features(merged, feat_cols, lag_days)
        # extend feat_cols with lag columns
        for d in lag_days:
            for c in feat_cols[:]:
                merged_col = f'lag{d}d_{c}'
                feat_cols.append(merged_col)

        # time split
        train_df, test_df = time_holdout_split(merged, time_col='meal_datetime', test_frac=args.test_frac)
        X_train = train_df[feat_cols].copy()
        X_test = test_df[feat_cols].copy()
        target_col = 'label_bin' if args.target == 'binary' else 'label_reg'
        y_train = train_df[target_col].astype(float)
        y_test = test_df[target_col].astype(float)

        pipe = build_pipeline(is_classifier=(args.target == 'binary'), seed=args.seed)
        pipe.fit(X_train, y_train)

        # evaluate
        if args.target == 'binary':
            metrics = evaluate_classifier(pipe, X_test.fillna(0), y_test)
        else:
            metrics = evaluate_regressor(pipe, X_test.fillna(0), y_test)

        # feature importances
        model = pipe.named_steps['model']
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = list(model.feature_importances_)

        # write outputs per-window
        out_dir = Path(args.out_dir) / f'window_{int(label_window)}h'
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / 'model.pkl')
        meta = {'metrics': metrics, 'n_train': len(X_train), 'n_test': len(X_test), 'seed': args.seed, 'window_hours': label_window, 'lag_days': lag_days}
        with open(out_dir / 'metrics.json', 'w', encoding='utf8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        with open(out_dir / 'features.json', 'w', encoding='utf8') as fh:
            json.dump(feat_cols, fh, indent=2, ensure_ascii=False)
        if importances is not None:
            pd.DataFrame({'feature': feat_cols, 'importance': importances}).to_csv(out_dir / 'feature_importances.csv', index=False)

        # attempt SHAP
        if getattr(args, 'compute_shap', False):
            _maybe_compute_shap(pipe, X_test.fillna(0), feat_cols, out_dir, is_clf=(args.target == 'binary'))
        else:
            logger.info('SHAP computation disabled for window %dh (use --compute-shap to enable)', int(label_window))

        results[f'{int(label_window)}h'] = metrics

    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--meals', default='scripts/data/raw/meals.parquet')
    p.add_argument('--ml-parquet', default='scripts/data/curated/foods_ml.parquet')
    p.add_argument('--symptoms', help='CSV/Parquet of symptom events (timestamp[,severity])', default=None)
    p.add_argument('--target', choices=('binary', 'regression'), default='binary')
    p.add_argument('--windows', help='Comma-separated label windows in hours (e.g. "4,24")', default='4,24')
    p.add_argument('--lag-days', help='Comma-separated integer lag windows in days (e.g. "1,3")', default='1,3')
    p.add_argument('--window-hours', type=float, default=4.0)
    p.add_argument('--compute-shap', help='Compute SHAP importances for each window (disabled by default)', action='store_true')
    p.add_argument('--out-dir', default='models/rf_out')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-frac', type=float, default=0.2)
    args = p.parse_args(argv)
    results = run_train(args)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
