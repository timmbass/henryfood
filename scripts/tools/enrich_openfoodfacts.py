#!/usr/bin/env python3
"""Enrich candidate foods using OpenFoodFacts search API (PoC).

- Reads canonical names from scripts/data/raw/foods_candidates.parquet (or foods.duckdb -> foods table)
- Queries OpenFoodFacts search.pl for each canonical name (cached)
- Extracts common nutrients (per 100g) and writes a cache JSON and parquet
- Produces a merged ML parquet that fills missing ML nutrients where possible

Usage:
  python scripts/tools/enrich_openfoodfacts.py
  python scripts/tools/enrich_openfoodfacts.py --cache-json scripts/data/raw/of_cache.json --out-parquet scripts/data/curated/openfoodfacts.parquet
"""
from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import requests
import pandas as pd
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def search_off(name: str, timeout: int = 15) -> Optional[Dict]:
    """Search OpenFoodFacts for a product matching name and return top product dict or None."""
    if not name:
        return None
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {"search_terms": name, "search_simple": 1, "action": "process", "json": 1, "page_size": 5}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("OFF search failed for %s: %s", name, e)
        return None
    products = data.get("products") or []
    if not products:
        return None
    # choose best product (first) -- keep raw nutriments
    p = products[0]
    return {"code": p.get("code"), "product_name": p.get("product_name"), "brands": p.get("brands"), "nutriments": p.get("nutriments"), "url": p.get("url")}


def extract_nutrients_from_off(nutriments: Dict) -> Dict:
    """Given OpenFoodFacts `nutriments` dict, extract canonical nutrients normalized to per-100g units.

    Returns dict with keys matching foods_ml.parquet: calories, protein_g, fat_g, carbs_g, fiber_g, sugar_g, sodium_mg, saturated_fat_g
    All values are floats or None. calories in kcal, sodium in mg.
    """
    out = {"calories": None, "protein_g": None, "fat_g": None, "carbs_g": None, "fiber_g": None, "sugar_g": None, "sodium_mg": None, "saturated_fat_g": None}
    if not nutriments or not isinstance(nutriments, dict):
        return out

    def get_key(k):
        # OF often uses keys like 'proteins_100g' or 'proteins_value'
        if k + "_100g" in nutriments:
            return nutriments.get(k + "_100g")
        # fallback to direct key
        return nutriments.get(k)

    def _as_float(v):
        try:
            return float(v)
        except Exception:
            return None

    # calories: energy-kcal_100g or energy_100g (kcal)
    c = nutriments.get("energy-kcal_100g") or nutriments.get("energy_100g") or nutriments.get("energy-kcal")
    out["calories"] = _as_float(c)

    out["protein_g"] = _as_float(get_key("proteins"))
    out["fat_g"] = _as_float(get_key("fat")) or _as_float(get_key("lipids"))
    out["carbs_g"] = _as_float(get_key("carbohydrates")) or _as_float(get_key("carbohydrates_100g"))
    out["fiber_g"] = _as_float(get_key("fiber")) or _as_float(get_key("fiber_100g"))
    out["sugar_g"] = _as_float(get_key("sugars")) or _as_float(get_key("sugars_100g"))
    # sodium: OF may provide 'sodium_100g' in mg OR 'salt_100g' in g. Handle both.
    s = None
    if "sodium_100g" in nutriments:
        s = _as_float(nutriments.get("sodium_100g"))
    elif "salt_100g" in nutriments:
        salt_g = _as_float(nutriments.get("salt_100g"))
        if salt_g is not None:
            # convert salt (g) to sodium (mg): sodium(mg) = salt(g) * 1000 * 0.393
            s = salt_g * 1000.0 * 0.393
    # sometimes sodium is provided as 'sodium' (per 100g) without suffix
    elif nutriments.get("sodium") is not None:
        s = _as_float(nutriments.get("sodium"))
    if s is not None:
        out["sodium_mg"] = s

    out["saturated_fat_g"] = _as_float(get_key("saturated-fat")) or _as_float(get_key("saturated_fat"))

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", help="Foods candidates parquet", default="scripts/data/raw/foods_candidates.parquet")
    p.add_argument("--cache-json", help="OpenFoodFacts cache JSON", default="scripts/data/raw/openfoodfacts_cache.json")
    p.add_argument("--out-parquet", help="Write OpenFoodFacts enrichment parquet", default="scripts/data/curated/openfoodfacts.parquet")
    p.add_argument("--merged-out", help="Write merged ML enriched parquet", default="scripts/data/curated/foods_ml_enriched.parquet")
    p.add_argument("--delay", help="Delay between OFF requests (seconds)", type=float, default=0.2)
    args = p.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        logger.error('Candidates parquet not found: %s', cand_path)
        raise SystemExit(2)

    df = pd.read_parquet(cand_path)
    if df.empty:
        logger.error('No candidates loaded')
        return 1

    names = sorted(df["canonical"].dropna().unique())
    cache_path = Path(args.cache_json)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf8') as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    new = 0
    for name in names:
        if not name:
            continue
        if name in cache:
            continue
        logger.info('Querying OFF for %s', name)
        res = search_off(name)
        cache[name] = res or {}
        new += 1
        time.sleep(args.delay)

    if new:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf8') as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
        logger.info('Wrote OFF cache to %s (%d new)', cache_path, new)
    else:
        logger.info('OFF cache up-to-date: %s', cache_path)

    # Build enrichment rows
    rows = []
    matched = 0
    for name in names:
        entry = cache.get(name) or {}
        if entry:
            nutr = extract_nutrients_from_off(entry.get('nutriments') or {})
            row = {**nutr, 'canonical': name, 'of_product_name': entry.get('product_name'), 'of_brands': entry.get('brands'), 'of_code': entry.get('code'), 'of_url': entry.get('url')}
            matched += 1
        else:
            row = {'canonical': name}
        rows.append(row)

    of_df = pd.DataFrame(rows)
    outp = Path(args.out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    of_df.to_parquet(outp, index=False)
    logger.info('Wrote OpenFoodFacts enrichments to %s (%d matched)', outp, matched)

    # Merge with existing ML parquet
    ml_path = Path('scripts/data/curated/foods_ml.parquet')
    if not ml_path.exists():
        logger.error('ML parquet not found at %s; run normalize_and_cache first', ml_path)
        return 1
    ml_df = pd.read_parquet(ml_path)

    merged = ml_df.merge(of_df, on='canonical', how='left', suffixes=('', '_of'))

    # Fill missing numeric ML fields from OFF when available
    num_keys = ['calories','protein_g','fat_g','carbs_g','fiber_g','sugar_g','sodium_mg','saturated_fat_g']
    fill_count = 0
    for k in num_keys:
        ofk = k
        mask = merged[k].isnull() & merged[ofk].notnull()
        c = int(mask.sum())
        if c:
            merged.loc[mask, k] = merged.loc[mask, ofk]
            fill_count += c
    merged_out = Path(args.merged_out)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_out, index=False)
    logger.info('Wrote merged ML enriched parquet to %s (filled %d values)', merged_out, fill_count)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
