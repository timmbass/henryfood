# scripts — Health Analytics Pipeline

DuckDB-based pipeline that turns raw CSV data into lag features, trains a logistic regression model, and generates weekly markdown reports.

## Usage

```bash
cd scripts
pip install -r requirements.txt
make daily    # ingest + feature engineering
make weekly   # train + weekly report
```

## Data ingestion

There is no ingestion script in this directory. New food-diary entries are appended directly to `data/raw/meals.csv` (and the other CSVs) by the `app/` capture component on the Raspberry Pi. The pipeline reads those CSVs at run time.

## Source layout

```
src/
├── features/
│   ├── timeline.py       # CSV → hourly timeline (DuckDB)
│   └── lag_features.py   # Lag/rolling features (FODMAP, histamine, …)
├── models/
│   └── train.py          # Logistic regression with cross-validation
├── reports/
│   └── weekly_report.py  # Markdown weekly report
└── utils/
    ├── db.py             # DuckDB connection helpers
    └── paths.py          # Path validation
```
