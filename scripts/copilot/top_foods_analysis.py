#!/usr/bin/env python3
"""Headless top-foods analysis script (runnable with `python`).

This is the standalone, executable version of the notebook so it can be run
on a headless Ubuntu server. It forces a non-interactive matplotlib backend
and writes plots to disk.

Outputs written to the model window directory:
 - top_foods.csv
 - top_foods_bar.png
 - nutrient_scatter.png
 - pr_curve.png, roc_curve.png (if labels available)
 - feature_importances.png
 - distributions_{col}.png for selected nutrient columns

Usage:
  python scripts/copilot/top_foods_analysis.py
  python scripts/copilot/top_foods_analysis.py --model-window models/rf_out_pain/window_4h

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Force headless backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

sns.set(style='whitegrid')


def robust_pos_probs(pipe, X):
    proba = pipe.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == 1:
        model = pipe.named_steps.get('model')
        classes = getattr(model, 'classes_', None)
        if classes is not None and len(classes) == 1 and classes[0] == 1:
            return proba[:, 0]
        return np.zeros(proba.shape[0])
    model = pipe.named_steps.get('model')
    classes = getattr(model, 'classes_', None)
    if classes is not None and 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = proba.shape[1] - 1
    return proba[:, pos_idx]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model-window', default='models/rf_out_pain/window_4h', help='Path to model window dir (contains model.pkl and features.json)')
    p.add_argument('--ml-parquet', default='scripts/data/curated/foods_ml.parquet')
    p.add_argument('--meals', default='scripts/data/raw/meals_with_symptoms.parquet')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--test-frac', type=float, default=0.2)
    args = p.parse_args(argv)

    model_dir = Path(args.model_window)
    if not model_dir.exists():
        print('Model window directory not found:', model_dir, file=sys.stderr)
        raise SystemExit(2)

    # artifacts
    model_pkl = model_dir / 'model.pkl'
    features_json = model_dir / 'features.json'
    fi_csv = model_dir / 'feature_importances.csv'

    if not model_pkl.exists() or not features_json.exists():
        print('Missing model.pkl or features.json in', model_dir, file=sys.stderr)
        raise SystemExit(2)

    # load
    pipe = joblib.load(model_pkl)
    feat_cols = json.load(open(features_json))
    ml_df = pd.read_parquet(args.ml_parquet)
    meals_df = pd.read_parquet(args.meals)

    # merge and ensure datetime
    merged = meals_df.merge(ml_df, on='canonical', how='left')
    if 'meal_datetime' not in merged.columns:
        merged['meal_datetime'] = pd.to_datetime(merged.get('meal_datetime') if 'meal_datetime' in merged else merged.get('datetime', merged.get('date')))
    merged = merged.sort_values('meal_datetime').reset_index(drop=True)

    # ensure features present
    for c in feat_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    X = merged[feat_cols].fillna(0.0)

    # predict
    probs = robust_pos_probs(pipe, X)
    merged['pred_prob'] = probs

    # aggregate top foods
    agg = merged.groupby('canonical').agg(n_meals=('pred_prob', 'size'), mean_pred=('pred_prob', 'mean')).reset_index()
    agg = agg.sort_values('mean_pred', ascending=False)
    out_csv = model_dir / 'top_foods.csv'
    agg.to_csv(out_csv, index=False)
    print('Wrote', out_csv)

    # top foods bar chart
    topN = agg.head(args.top_n).merge(ml_df, on='canonical', how='left')
    fig, ax = plt.subplots(figsize=(8, max(3, args.top_n * 0.5)))
    sns.barplot(x='mean_pred', y='canonical', data=topN, palette='magma', ax=ax)
    ax.set_xlabel('Mean predicted risk')
    ax.set_ylabel('Canonical food')
    ax.set_title(f'Top {args.top_n} foods by predicted risk')
    fig.tight_layout()
    bar_path = model_dir / 'top_foods_bar.png'
    fig.savefig(bar_path, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', bar_path)

    # nutrient scatter (calories vs protein) for top foods, colored by mean_pred
    if 'calories' in topN.columns and 'protein_g' in topN.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(topN['calories'], topN['protein_g'], c=topN['mean_pred'], cmap='coolwarm', s=80)
        for i, txt in enumerate(topN['canonical']):
            ax.annotate(txt, (topN['calories'].iat[i], topN['protein_g'].iat[i]), textcoords='offset points', xytext=(3,3), fontsize=8)
        fig.colorbar(sc, ax=ax, label='mean_pred')
        ax.set_xlabel('Calories')
        ax.set_ylabel('Protein (g)')
        ax.set_title('Calories vs Protein for Top Foods (colored by mean_pred)')
        fig.tight_layout()
        sc_path = model_dir / 'nutrient_scatter.png'
        fig.savefig(sc_path, bbox_inches='tight')
        plt.close(fig)
        print('Wrote', sc_path)
    else:
        print('Skipping nutrient scatter; calories/protein not found in ML parquet')

    # feature importances plot
    if fi_csv.exists():
        fi = pd.read_csv(fi_csv)
    else:
        model = pipe.named_steps.get('model')
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_})
        else:
            fi = pd.DataFrame({'feature': feat_cols, 'importance': [0.0] * len(feat_cols)})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    topfi = fi.head(30)
    fig, ax = plt.subplots(figsize=(8, max(3, len(topfi) * 0.25)))
    sns.barplot(x='importance', y='feature', data=topfi, palette='viridis', ax=ax)
    ax.set_title('Top feature importances')
    fig.tight_layout()
    fi_path = model_dir / 'feature_importances.png'
    fig.savefig(fi_path, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', fi_path)

    # PR / ROC curves if label present; reconstruct time-based test split
    if 'label_bin' in merged.columns:
        n = len(merged)
        cut = max(1, int((1.0 - args.test_frac) * n))
        test_df = merged.iloc[cut:].reset_index(drop=True)
        X_test = test_df[feat_cols].fillna(0.0)
        y_test = test_df['label_bin'].astype(float)
        probs_test = robust_pos_probs(pipe, X_test)
        # PR
        try:
            prec, rec, _ = precision_recall_curve(y_test, probs_test)
            ap = average_precision_score(y_test, probs_test)
            fig, ax = plt.subplots()
            ax.plot(rec, prec, label=f'AP={ap:.3f}')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall')
            ax.legend()
            pr_path = model_dir / 'pr_curve.png'
            fig.tight_layout()
            fig.savefig(pr_path, bbox_inches='tight')
            plt.close(fig)
            print('Wrote', pr_path)
        except Exception as e:
            print('PR curve failed:', e)
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_test, probs_test)
            auc = roc_auc_score(y_test, probs_test)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC'); ax.legend()
            roc_path = model_dir / 'roc_curve.png'
            fig.tight_layout()
            fig.savefig(roc_path, bbox_inches='tight')
            plt.close(fig)
            print('Wrote', roc_path)
        except Exception as e:
            print('ROC curve failed:', e)
    else:
        print('No label_bin column in meals; skipping PR/ROC')

    # distributions for top nutrient-like features
    nutrient_candidates = [c for c in fi['feature'].tolist() if c in ml_df.columns]
    if not nutrient_candidates:
        nutrient_candidates = [c for c in ['calories','protein_g','fat_g','carbs_g','fiber_g','sugar_g'] if c in ml_df.columns]

    for col in nutrient_candidates[:10]:
        try:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.boxplot(x=merged[col].dropna(), ax=ax)
            ax.set_title(f'Distribution of {col} across meals')
            fig.tight_layout()
            path = model_dir / f'distribution_{col}.png'
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print('Wrote', path)

            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(merged[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Histogram of {col}')
            fig.tight_layout()
            path = model_dir / f'hist_{col}.png'
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print('Wrote', path)
        except Exception as e:
            print('Failed to plot distribution for', col, e)

    print('Done')


if __name__ == '__main__':
    raise SystemExit(main())
