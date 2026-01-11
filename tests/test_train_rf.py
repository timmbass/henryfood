import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from src.models import train_rf


def _make_synthetic_meals(n=100):
    dates = pd.date_range('2023-01-01', periods=n, freq='6H')
    rows = []
    foods = ['apple','bread','cheese','chicken']
    for i, dt in enumerate(dates):
        rows.append({'canonical': foods[i % len(foods)], 'date': dt.date().isoformat(), 'meal': 'breakfast' if dt.hour<10 else 'lunch' if dt.hour<15 else 'dinner'})
    return pd.DataFrame(rows)


def _make_synthetic_ml():
    foods = ['apple','bread','cheese','chicken']
    rows = []
    import random
    for f in foods:
        rows.append({'canonical': f, 'calories': random.uniform(50,500), 'protein_g': random.uniform(0,30), 'fat_g': random.uniform(0,30), 'carbs_g': random.uniform(0,80), 'fiber_g': random.uniform(0,10), 'sugar_g': random.uniform(0,30), 'sodium_mg': random.uniform(0,500), 'saturated_fat_g': random.uniform(0,10)})
    return pd.DataFrame(rows)


def _make_synthetic_symptoms(meals_df):
    # create symptom events that follow some meals more often (e.g., chicken)
    events = []
    for i, r in meals_df.iterrows():
        if r['canonical'] == 'chicken' and np.random.rand() < 0.4:
            events.append({'timestamp': str(pd.to_datetime(r['date']) + pd.Timedelta(hours=3)), 'severity': float(np.random.uniform(1,10))})
        elif r['canonical']=='cheese' and np.random.rand() < 0.2:
            events.append({'timestamp': str(pd.to_datetime(r['date']) + pd.Timedelta(hours=2)), 'severity': float(np.random.uniform(1,6))})
    if not events:
        # fallback: create one event
        events.append({'timestamp': str(pd.Timestamp('2023-01-02')), 'severity': 5.0})
    return pd.DataFrame(events)


def test_train_rf_smoke(tmp_path):
    meals = _make_synthetic_meals(80)
    ml = _make_synthetic_ml()
    syms = _make_synthetic_symptoms(meals)

    meals_path = tmp_path / 'meals.parquet'
    ml_path = tmp_path / 'foods_ml.parquet'
    syms_path = tmp_path / 'symptoms.csv'
    meals.to_parquet(meals_path, index=False)
    ml.to_parquet(ml_path, index=False)
    syms.to_csv(syms_path, index=False)

    out_dir = tmp_path / 'out'
    args = argparse = None
    # call run_train with constructed args-like object
    class A: pass
    a = A()
    a.meals = str(meals_path)
    a.ml_parquet = str(ml_path)
    a.symptoms = str(syms_path)
    a.target = 'binary'
    a.windows = '4,24'
    a.lag_days = '1'
    a.window_hours = 4.0
    a.out_dir = str(out_dir)
    a.seed = 42
    a.test_frac = 0.2

    results = train_rf.run_train(a)
    assert '4h' in results and '24h' in results
    for k, v in results.items():
        assert isinstance(v, dict)
        assert 'pr_auc' in v or 'mae' in v

