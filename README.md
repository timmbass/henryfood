HenryFood — Diet→Pain pipeline

Modeling pipeline (meal-level random forest)

- Location: `src/models/train_rf.py`.
- Purpose: trains meal-level random forest models to predict downstream labels (e.g. pain) using engineered features (lags, aggregation windows). The script supports multi-window labeling (for example 4h and 24h outcome windows), time-based train/holdout splits, lag-day features, and outputs per-window artifacts (models, metrics CSVs, and diagnostics).
- Metrics produced: PR-AUC for classification-style labels and MAE for continuous outcomes (depending on label), plus other per-window summaries.

SHAP/feature importance

- SHAP computation is optional and disabled by default. Use the `--compute-shap` flag to enable SHAP importance calculation. Note: SHAP can be slow and memory-intensive; enable it only for debugging or final model interpretation.

Example usage

- Basic training (defaults, SHAP disabled):

  python src/models/train_rf.py --meals scripts/data/curated/meals.parquet --out-dir outputs/models

- Enable SHAP (may be slow):

  python src/models/train_rf.py --meals scripts/data/curated/meals.parquet --out-dir outputs/models --compute-shap

- Notes:
  - Replace `--meals` with the path to your prepared meal-level dataset (the pipeline produces `scripts/data/curated/foods_ml.parquet` and the meal-aggregation scripts typically write meal-level files to `scripts/data/curated/`).
  - Check `src/models/train_rf.py --help` for available flags (windows, holdout fraction, feature options, etc.).

Running the unit test (smoke test)

- To run the modeling smoke test included with the project (uses synthetic data and runs the training loop for 4h and 24h windows), run:

  pytest -q tests/test_train_rf.py

This will execute the test suite for the RF training pipeline. The tests are configured by `pytest.ini` and `tests/conftest.py` to ensure a reproducible environment.

Troubleshooting

- If tests fail due to missing dependencies, install them from `requirements.txt`:

  pip install -r requirements.txt

- If you want CI to run a quick smoke test, add a job that runs the pytest command above; gate SHAP runs behind an explicit flag to avoid long CI jobs.

