Diet→Pain project - starter

Setup

1. Create a virtualenv: python -m venv .venv
2. Activate: source .venv/bin/activate
3. Install: python -m pip install -r requirements.txt

Run example

python scripts/run_feature_pipeline.py --input data/processed/diary.csv

Next steps
- Populate ingest scripts to download and parse external datasets.
- Implement feature engineering and modeling scripts.
