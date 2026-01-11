#!/usr/bin/env python3
"""Run training, predictions, and plotting for multiple symptoms in order.

This script runs `src/models/train_rf.py` sequentially for the list of
symptoms (defaults: pain, sleep, stress), then aggregates metrics and
produces top-food predictions and PR/ROC plots per symptom/window.

Usage:
  python scripts/tools/run_all_models.py
  python scripts/tools/run_all_models.py --symptoms pain,sleep --windows 4,24

It uses the same defaults as `train_rf.py` for ml-parquet, meals file,
lag days, and test fraction. Runs in the current virtualenv by invoking
`sys.executable`.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# defaults to match the rest of the repo
TRAIN_SCRIPT = Path("src/models/train_rf.py")
MEALS = Path("scripts/data/raw/meals_with_symptoms.parquet")
ML_PARQUET = Path("scripts/data/curated/foods_ml.parquet")
OUT_BASE = Path("models")


def run_training(symptom: str, out_dir: Path, windows: list[str], lag_days: list[str], compute_shap: bool = False):
    args = [
        sys.executable, str(TRAIN_SCRIPT),
        "--meals", str(MEALS),
        "--ml-parquet", str(ML_PARQUET),
        "--infer-from-meals",
        "--symptom-col", symptom,
        "--target", "binary",
        "--windows", ",".join(windows),
        "--lag-days", ",".join(lag_days),
        "--out-dir", str(out_dir),
    ]
    if compute_shap:
        args.append("--compute-shap")
    logger.info("Running training for symptom=%s -> %s", symptom, out_dir)
    subprocess.run(args, check=True)


def _robust_pos_probs(pipe, X):
    proba = pipe.predict_proba(X)
    # handle single-column predict_proba
    if proba.ndim == 2 and proba.shape[1] == 1:
        model = pipe.named_steps.get("model")
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1 and classes[0] == 1:
            return proba[:, 0]
        return np.zeros(proba.shape[0])
    model = pipe.named_steps.get("model")
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = proba.shape[1] - 1
    return proba[:, pos_idx]


def produce_predictions_and_plots(symptom: str, out_dir: Path, window_h: int, test_frac: float = 0.2):
    window_dir = out_dir / f"window_{int(window_h)}h"
    model_pkl = window_dir / "model.pkl"
    features_json = window_dir / "features.json"
    if not model_pkl.exists() or not features_json.exists():
        logger.warning("Missing artifacts for %s %sh at %s; skipping", symptom, window_h, window_dir)
        return

    pipe = joblib.load(model_pkl)
    feat = json.load(open(features_json))

    ml = pd.read_parquet(ML_PARQUET)
    meals = pd.read_parquet(MEALS)
    merged = meals.merge(ml, on="canonical", how="left")

    # ensure datetime and sort for consistent test split
    if "meal_datetime" not in merged.columns:
        merged["meal_datetime"] = pd.to_datetime(merged.get("meal_datetime") if "meal_datetime" in merged else merged.get("datetime", merged.get("date")))
    merged = merged.sort_values("meal_datetime").reset_index(drop=True)

    # reconstruct time holdout split
    n = len(merged)
    cut = max(1, int((1.0 - test_frac) * n))
    test_df = merged.iloc[cut:].reset_index(drop=True)

    # ensure feature columns
    for c in feat:
        if c not in test_df.columns:
            test_df[c] = 0.0
    X_test = test_df[feat].fillna(0.0)
    y_test = test_df.get("label_bin")
    if y_test is None:
        logger.warning("No label_bin in meals for %s; skipping plots", symptom)
        return
    y_test = y_test.astype(float)

    probs = _robust_pos_probs(pipe, X_test)

    # save top foods by mean predicted risk
    test_df["pred_prob"] = probs
    agg = (
        test_df.groupby("canonical")
        .agg(n_meals=("pred_prob", "size"), mean_pred=("pred_prob", "mean"))
        .reset_index()
        .sort_values("mean_pred", ascending=False)
    )
    agg_path = window_dir / f"top_foods_by_pred_{symptom}_{int(window_h)}h.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Wrote top foods by pred to %s", agg_path)

    # PR curve
    try:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision-Recall ({symptom} {int(window_h)}h)"); plt.legend()
        pr_path = window_dir / "pr_curve.png"
        plt.savefig(pr_path, bbox_inches="tight")
        plt.close()
        logger.info("Wrote PR curve to %s", pr_path)
    except Exception as e:
        logger.warning("Failed to write PR curve for %s: %s", symptom, e)

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({symptom} {int(window_h)}h)"); plt.legend()
        roc_path = window_dir / "roc_curve.png"
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        logger.info("Wrote ROC curve to %s", roc_path)
    except Exception as e:
        logger.warning("Failed to write ROC curve for %s: %s", symptom, e)


def aggregate_metrics(models_base: Path, out_csv: Path):
    rows = []
    for f in models_base.rglob("*/window_*/metrics.json"):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        row = {"path": str(f.parent), "window_hours": m.get("window_hours"), "n_train": m.get("n_train"), "n_test": m.get("n_test")}
        row.update(m.get("metrics", {}))
        rows.append(row)
    if not rows:
        logger.warning("No metrics files found under %s", models_base)
        return
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote aggregated metrics to %s", out_csv)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--symptoms", default="pain,sleep,stress", help="Comma-separated symptom columns to run")
    p.add_argument("--windows", default="4,24")
    p.add_argument("--lag-days", default="1,3")
    p.add_argument("--out-base", default=str(OUT_BASE))
    p.add_argument("--compute-shap", action="store_true")
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args(argv)

    symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    lag_days = [l.strip() for l in args.lag_days.split(",") if l.strip()]
    out_base = Path(args.out_base)

    for s in symptoms:
        out_dir = out_base / f"rf_out_{s}"
        run_training(s, out_dir, windows, lag_days, compute_shap=args.compute_shap)
        # produce predictions & plots for each window
        for w in windows:
            produce_predictions_and_plots(s, out_dir, int(float(w)), test_frac=args.test_frac)

    # aggregate metrics across all model outputs
    aggregate_metrics(out_base, out_base / "aggregate_metrics.csv")


if __name__ == "__main__":
    raise SystemExit(main())
