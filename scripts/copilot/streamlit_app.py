#!/usr/bin/env python3
"""Simple Streamlit dashboard to browse model window outputs.

Usage:
  pip install streamlit pandas joblib
  streamlit run scripts/copilot/streamlit_app.py

The app scans for model window directories under `models/` (folders named
`window_*`) and lets you pick one. It displays:
 - Top foods table (from top_foods.csv)
 - Feature importances (CSV or model-derived)
 - Plots (top_foods_bar.png, nutrient_scatter.png, pr/roc, distributions)
 - Metrics (metrics.json)

If files are missing the app suggests running the analysis script.
"""
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import os
import joblib
import subprocess
import sys

ROOT = Path('.').resolve()
MODELS_ROOT = ROOT / 'models'
ANALYSIS_SCRIPT = Path('scripts/copilot/top_foods_analysis.py')

st.set_page_config(page_title='Diet→Pain: Model Outputs', layout='wide')
st.title('Diet→Pain — Model Outputs Browser')

# discover model window directories
window_dirs = sorted([p for p in MODELS_ROOT.rglob('window_*') if p.is_dir()])
options = [str(p.relative_to(ROOT)) for p in window_dirs]
window_map = {opt: p for opt, p in zip(options, window_dirs)}

st.sidebar.write(f'Found {len(options)} model window(s)')
if not options:
    st.warning('No model window directories found under `models/`. Run training first or place model outputs under models/.')
    st.stop()

# selectbox with explicit index/key and graceful fallback
sel = st.sidebar.selectbox('Select model window', options=options, index=0, key='model_window_select')
model_dir = window_map.get(sel)
if model_dir is None and window_dirs:
    model_dir = window_dirs[0]
st.sidebar.write(model_dir)

# show basic artifacts
st.header('Artifacts')
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Top foods')
    top_csv = model_dir / 'top_foods.csv'
    if top_csv.exists():
        df_top = pd.read_csv(top_csv)
        n = st.slider('Show top N rows', min_value=5, max_value=200, value=20)
        st.dataframe(df_top.head(n))
        st.download_button('Download top_foods.csv', data=top_csv.read_bytes(), file_name=f'{model_dir.name}_top_foods.csv')
    else:
        st.info('top_foods.csv not found. Run the analysis script to generate outputs.')
        if st.button('Run analysis script now'):
            st.write('Running analysis script — this may take a moment...')
            try:
                subprocess.run([sys.executable, str(ANALYSIS_SCRIPT), '--model-window', str(model_dir)], check=True)
                st.success('Analysis script completed. Reload the page to see updated outputs.')
                # Attempt to programmatically rerun if the Streamlit runtime supports it
                if hasattr(st, 'experimental_rerun'):
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f'Analysis script failed: {e}')

    # metrics
    metrics_json = model_dir / 'metrics.json'
    if metrics_json.exists():
        st.subheader('Metrics')
        try:
            m = json.loads(metrics_json.read_text())
            st.json(m)
        except Exception as e:
            st.write('Failed to read metrics.json:', e)

with col2:
    st.subheader('Plots')
    def show_if_exists(p: Path, caption: str = None):
        if p.exists():
            st.image(str(p), caption=caption or p.name)
            return True
        return False

    shown = []
    for fname in ['top_foods_bar.png', 'nutrient_scatter.png', 'pr_curve.png', 'roc_curve.png']:
        p = model_dir / fname
        if show_if_exists(p, caption=fname):
            shown.append(fname)
    if not shown:
        st.info('No standard plot images found in window dir. Run analysis script to generate them.')

st.markdown('---')

# Feature importances
st.header('Feature importances')
fi_csv = model_dir / 'feature_importances.csv'
features_json = model_dir / 'features.json'
if fi_csv.exists():
    fi_df = pd.read_csv(fi_csv)
    st.dataframe(fi_df.head(200))
    st.download_button('Download feature_importances.csv', data=fi_csv.read_bytes(), file_name=f'{model_dir.name}_feature_importances.csv')
else:
    # attempt to build from model + features.json
    if (model_dir / 'model.pkl').exists() and features_json.exists():
        if st.button('Load feature importances from model'):
            pipe = joblib.load(str(model_dir / 'model.pkl'))
            feat_cols = json.loads(open(features_json).read())
            model = pipe.named_steps.get('model')
            if hasattr(model, 'feature_importances_'):
                fi_df = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                st.dataframe(fi_df.head(200))
            else:
                st.write('Model does not expose feature_importances_.')
    else:
        st.info('feature_importances.csv not found and model or features.json missing.')

st.markdown('---')

# Distribution images
st.header('Distribution plots')
dist_files = sorted(model_dir.glob('distribution_*.png'))
if dist_files:
    for p in dist_files:
        st.image(str(p), caption=p.name)
else:
    st.info('No distribution plots found (distribution_*.png).')

st.markdown('---')

st.sidebar.markdown('---')
if st.sidebar.button('Open model directory'): 
    st.sidebar.write(f"{model_dir}")

st.sidebar.markdown('Tips:')
st.sidebar.write(' - If outputs are missing, run `python scripts/copilot/top_foods_analysis.py --model-window <model_dir>` on the server.')
st.sidebar.write(' - Run `streamlit run scripts/copilot/streamlit_app.py` and open the URL shown in the terminal.')
