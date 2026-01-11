#!/usr/bin/env python3
"""Normalize meal items, deduplicate, and build a cached foods table.

Writes:
- Parquet of candidate foods: scripts/data/raw/foods_candidates.parquet (default)
- DuckDB table of foods with optional USDA lookup: scripts/data/curated/foods.duckdb (default)
- Optional JSON cache of USDA responses

Usage examples:
  python scripts/tools/normalize_and_cache.py --meals scripts/data/raw/meals.parquet
  python scripts/tools/normalize_and_cache.py --meals scripts/data/raw/meals.json --mapping config/food_map.yaml --out-duckdb scripts/data/curated/foods.duckdb --usda-key $FDC_API_KEY

Note: USDA lookup requires an API key. This script will cache results to avoid repeat queries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
import random

# configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def normalize_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.lower()
    # remove parentheticals
    s = re.sub(r"\([^)]*\)", "", s)
    # remove common measurements and leading quantities like '1 cup', '2-3 slices', '100g'
    s = re.sub(r"^\s*[\d\./]+\s*(oz|g|kg|ml|l|cup|cups|tbsp|tbsps|tsp|slice|slices|serving|servings|piece|pieces)?\b", "", s)
    # remove fractions like 1/2 or unicode fractions
    s = re.sub(r"\b\d+/\d+\b", "", s)
    # strip units inside string (simple)
    units = r"\b(oz|g|kg|ml|l|cup|cups|tbsp|tbsps|tbsp.|tsp|slice|slices|serving|servings|piece|pieces|can|packet)\b"
    s = re.sub(units, "", s)
    # replace punctuation with space
    s = re.sub(r"[^\w\s]", " ", s)
    # remove extra whitespace
    s = re.sub(r"\s+", " ", s).strip()
    if s == "":
        return None
    return s


def load_meals(path: Path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(path)
    # fallback: read as csv
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_foods_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_duckdb_table(df, duckdb_path: Path, table: str = "foods"):
    import duckdb
    con = duckdb.connect(database=str(duckdb_path), read_only=False)
    try:
        con.register("__foods_df", df)
        con.execute(f"DROP TABLE IF EXISTS \"{table}\";")
        con.execute(f"CREATE TABLE \"{table}\" AS SELECT * FROM __foods_df;")
        con.unregister("__foods_df")
    finally:
        con.close()


def usda_search_and_choose(api_key: str, name: str) -> Optional[Dict]:
    """Search USDA FoodData Central and return a compact chosen result or None.
    Note: this is a best-effort heuristic and caches results externally.
    """
    import requests
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    payload = {"query": name, "pageSize": 5}
    try:
        r = requests.post(url, params={"api_key": api_key}, json=payload, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logger.warning("USDA lookup failed for %s: %s", name, e)
        return None
    data = r.json()
    foods = data.get("foods") or []
    if not foods:
        logger.debug("USDA: no foods found for %s", name)
        return None
    # pick first; include nutrients dictionary
    best = foods[0]
    res = {
        "fdcId": best.get("fdcId"),
        "description": best.get("description"),
        "dataType": best.get("dataType"),
        "brandOwner": best.get("brandOwner"),
        "nutrients": best.get("foodNutrients"),
    }
    return res


def enrich_foods_df_with_cache(foods_df, cache: Dict[str, Optional[Dict]]):
    """Enrich a foods DataFrame with USDA cache values (fdcId and description).

    This function is defensive: it sanitizes cache entries that are None and
    always uses a dict fallback to avoid AttributeError when a cache value is null.
    Returns the enriched DataFrame (modified in place) for convenience.
    """
    # sanitize cache: replace explicit None values with empty dicts
    for k, v in list(cache.items()):
        if v is None:
            cache[k] = {}

    # safe access using (cache.get(x) or {})
    foods_df["fdc_id"] = foods_df["canonical"].apply(lambda x: (cache.get(x) or {}).get("fdcId") if x else None)
    foods_df["usda_description"] = foods_df["canonical"].apply(lambda x: (cache.get(x) or {}).get("description") if x else None)
    return foods_df


def _find_nutrient(nutrients, candidates):
    """Return tuple(value, unit) for the first nutrient whose name matches any candidate (case-insensitive substring)."""
    if not nutrients:
        return None, None
    for n in nutrients:
        name = n.get("nutrientName") or n.get("name") or ""
        if not name:
            continue
        lname = name.lower()
        for c in candidates:
            if c.lower() in lname:
                return n.get("value"), (n.get("unitName") or n.get("unit") or "")
    return None, None


def extract_nutrients_from_cache_entry(entry: Dict) -> Dict:
    """Given a USDA cache entry (res), extract common nutrient fields with normalized units.

    Returns dict with keys: calories (kcal), protein_g, fat_g, carbs_g, fiber_g, sugar_g, sodium_mg, saturated_fat_g
    """
    # New version: prefer using a provided nutrient_map; else fall back to heuristics
    # This function will accept an optional nutrient_map when called (injection below).
    return extract_nutrients_from_cache_entry.__wrapped__(entry) if hasattr(extract_nutrients_from_cache_entry, '__wrapped__') else _extract_nutrients_default(entry)


def _extract_nutrients_default(entry: Dict) -> Dict:
    # original heuristic implementation preserved here
    out = {"calories": None, "protein_g": None, "fat_g": None, "carbs_g": None, "fiber_g": None, "sugar_g": None, "sodium_mg": None, "saturated_fat_g": None}
    nutrients = entry.get("nutrients") if isinstance(entry, dict) else None
    if not nutrients:
        return out

    def _as_float(v):
        try:
            return float(v)
        except Exception:
            return None

    # calories
    val, unit = _find_nutrient(nutrients, ["energy", "calorie", "kcal"])
    if val is not None:
        out["calories"] = _as_float(val)

    # protein (g)
    val, unit = _find_nutrient(nutrients, ["protein"])
    if val is not None:
        out["protein_g"] = _as_float(val)

    # total fat (g)
    val, unit = _find_nutrient(nutrients, ["total lipid", "fatty acids, total", "total fat", "total lipid (fat)"])
    if val is not None:
        out["fat_g"] = _as_float(val)

    # carbs (g)
    val, unit = _find_nutrient(nutrients, ["carbohydrate, by difference", "carbohydrate"]) 
    if val is not None:
        out["carbs_g"] = _as_float(val)

    # fiber (g)
    val, unit = _find_nutrient(nutrients, ["fiber", "dietary fiber", "total dietary fiber"]) 
    if val is not None:
        out["fiber_g"] = _as_float(val)

    # sugars (g)
    val, unit = _find_nutrient(nutrients, ["sugars", "sugar"]) 
    if val is not None:
        out["sugar_g"] = _as_float(val)

    # sodium (mg)
    val, unit = _find_nutrient(nutrients, ["sodium", "sodium, na"]) 
    if val is not None:
        v = _as_float(val)
        if v is not None:
            if (unit or "").lower().startswith("mg"):
                out["sodium_mg"] = v
            else:
                # assume grams -> convert to mg
                out["sodium_mg"] = v * 1000

    # saturated fat (g)
    val, unit = _find_nutrient(nutrients, ["saturated", "saturated fatty acids"]) 
    if val is not None:
        out["saturated_fat_g"] = _as_float(val)

    return out


def extract_nutrients_from_cache_entry_with_map(entry: Dict, nutrient_map: Optional[Dict] = None) -> Dict:
    """Extract nutrient values using a provided nutrient_name -> canonical mapping.

    The mapping should map observed nutrient names (lowercased) -> canonical keys (e.g. 'protein_g').
    Multiple observed nutrients mapping to the same canonical key will be summed.
    """
    # start with zero accumulators and seen flags
    keys = ["calories", "protein_g", "fat_g", "carbs_g", "fiber_g", "sugar_g", "sodium_mg", "saturated_fat_g"]
    acc = {k: 0.0 for k in keys}
    seen = {k: False for k in keys}

    nutrients = entry.get("nutrients") if isinstance(entry, dict) else None
    def _as_float(v):
        try:
            return float(v)
        except Exception:
            return None

    if nutrient_map and nutrients:
        # build lowercased mapping for quick lookup
        nm = {k.lower(): v for k, v in nutrient_map.items() if v}
        for n in nutrients:
            name = (n.get("nutrientName") or n.get("name") or "").strip()
            if not name:
                continue
            key = nm.get(name.lower())
            if not key:
                continue
            val = _as_float(n.get("value"))
            if val is None:
                continue
            unit = (n.get("unitName") or n.get("unit") or "").lower()
            # handle unit conversions for sodium and calories
            if key == "sodium_mg":
                if unit.startswith("mg"):
                    acc[key] += val
                elif unit.startswith("g"):
                    acc[key] += val * 1000.0
                elif unit.startswith("ug"):
                    acc[key] += val / 1000.0
                else:
                    acc[key] += val
                seen[key] = True
            elif key == "calories":
                if "kcal" in unit or "kcal" == unit:
                    acc[key] += val
                elif "kj" in unit:
                    acc[key] += val * 0.239006
                else:
                    acc[key] += val
                seen[key] = True
            else:
                acc[key] += val
                seen[key] = True

    # if mapping produced values, return them (convert zeros with seen flag)
    out = {k: (acc[k] if seen[k] else None) for k in keys}

    # fallback: for any None, run default heuristic on the nutrient list
    defaults = _extract_nutrients_default(entry)
    for k in keys:
        if out.get(k) is None:
            out[k] = defaults.get(k)

    return out


# replace the original function used elsewhere with a wrapper that can accept a nutrient_map
def extract_nutrients_from_cache_entry(entry: Dict, nutrient_map: Optional[Dict] = None) -> Dict:
    if nutrient_map:
        return extract_nutrients_from_cache_entry_with_map(entry, nutrient_map)
    return _extract_nutrients_default(entry)


def suggest_nutrient_name_map(cache: Dict[str, Dict], cutoff: float = 0.6):
    """Scan cache nutrient names and suggest a mapping from observed nutrient names -> canonical keys.

    Uses simple token substring matching first, then difflib fallback.
    Returns dict observed_name -> canonical_key.
    """
    import difflib
    # canonical tokens for common targets
    tokens_by_key = {
        "calories": ["energy", "calorie", "kcal"],
        "protein_g": ["protein"],
        "fat_g": ["total lipid", "total fat", "fatty acids", "total lipid (fat)", "total lipid (fat)"],
        "carbs_g": ["carbohydrate", "carbohydrate, by difference", "carbohydrate, by difference"],
        "fiber_g": ["fiber", "dietary fiber", "total dietary fiber"],
        "sugar_g": ["sugars", "total sugars", "sugar"],
        "sodium_mg": ["sodium", "sodium, na"],
        "saturated_fat_g": ["saturated", "fatty acids, total saturated", "sfa", "saturated fatty acids"],
    }
    flat_tokens = []
    token_to_key = {}
    for k, toks in tokens_by_key.items():
        for t in toks:
            flat_tokens.append(t)
            token_to_key[t] = k

    observed = set()
    for v in cache.values():
        if not v or not isinstance(v, dict):
            continue
        for n in v.get("nutrients") or []:
            nm = (n.get("nutrientName") or n.get("name") or "").strip()
            if nm:
                observed.add(nm)

    suggestions = {}
    for nm in sorted(observed):
        lname = nm.lower()
        mapped = None
        # token substring matching
        for t in flat_tokens:
            if t in lname:
                mapped = token_to_key[t]
                break
        if not mapped:
            # difflib match against tokens
            best = difflib.get_close_matches(lname, flat_tokens, n=1, cutoff=cutoff)
            if best:
                mapped = token_to_key[best[0]]
        if mapped:
            suggestions[nm] = mapped
        else:
            suggestions[nm] = None

    return suggestions


def usda_search_with_retries(api_key: str, name: str, max_attempts: int = 4, base_delay: float = 0.5, backoff_factor: float = 2.0, max_delay: float = 10.0, jitter: float = 0.5, show_progress: bool = True) -> Optional[Dict]:
    """Attempt USDA lookup with exponential backoff and jitter.

    Writes progress to the terminal so a user running the script sees live updates.
    Returns the USDA result dict or None if no result after retries.
    """
    for attempt in range(1, max_attempts + 1):
        if show_progress:
            print(f"USDA lookup [{attempt}/{max_attempts}] for: '{name}'", flush=True)
        try:
            res = usda_search_and_choose(api_key, name)
        except Exception as e:
            # usda_search_and_choose should catch exceptions, but be defensive
            logger.warning("USDA lookup raised exception for %s: %s", name, e)
            res = None

        if res is not None:
            if show_progress:
                print(f"USDA lookup successful for: '{name}' -> fdcId={res.get('fdcId')}", flush=True)
            return res

        # No result; decide whether to retry
        if attempt < max_attempts:
            delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))
            delay = delay + random.uniform(0, jitter)
            if show_progress:
                print(f"No USDA result for '{name}' (attempt {attempt}). Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
        else:
            if show_progress:
                print(f"USDA lookup exhausted for '{name}' after {max_attempts} attempts", flush=True)
    return None


def validate_nutrient_map(nm: Dict[str, Optional[str]]):
    """Validate a nutrient name mapping and return a list of suspicious mapping issues.

    Returns list of tuples (observed_name, canonical, message)
    """
    issues = []
    amino_acids = {"alanine","arginine","aspartic","cystine","glutamic","glycine","histidine","isoleucine","leucine","lysine","methionine","phenylalanine","proline","serine","threonine","tryptophan","tyrosine","valine"}
    vitamins = {"vitamin","biotin","niacin","riboflavin","folate","retinol","tocopherol","thiamin","pantothenic","choline"}
    minerals = {"calcium","iron","magnesium","potassium","sodium","zinc","selenium","phosphorus","copper","manganese"}
    for obs, canon in (nm or {}).items():
        if not canon:
            continue
        lname = obs.lower()
        tokens = set(re.split(r"[\s,()\-:]+", lname))
        # rule: amino acids or vitamins or minerals mapped to calories is suspicious
        if canon == 'calories' and (tokens & amino_acids or tokens & vitamins or tokens & minerals):
            issues.append((obs, canon, 'implausible -> calories (amino acid / vitamin / mineral)'))
        # rule: vitamins mapped to protein/fat/carbs suspicious
        if canon in ('protein_g','fat_g','carbs_g') and (tokens & vitamins):
            issues.append((obs, canon, 'vitamin token mapped to macronutrient'))
        # rule: element/mineral names mapped to calories/protein suspicious
        if canon in ('calories','protein_g') and (tokens & minerals):
            issues.append((obs, canon, 'mineral token mapped to kcal/protein'))
        # rule: single-letter or very short names mapped to macronutrient
        if len(lname) <= 3 and canon in ('calories','protein_g','fat_g','carbs_g'):
            issues.append((obs, canon, 'very short observed name mapped to macronutrient'))
    return issues


def conservative_fix_nutrient_map(nm: Dict[str, Optional[str]]):
    """Produce a conservative fixed nutrient map by nulling suspicious mappings.

    Returns (fixed_map, changes) where changes is a list of (obs, old, new, reason).
    """
    fixed = {k: (v if v else None) for k, v in (nm or {}).items()}
    issues = validate_nutrient_map(nm)
    changes = []
    for obs, canon, msg in issues:
        old = fixed.get(obs)
        if old is None:
            continue
        # conservative action: remove mapping (set to None) rather than guessing
        fixed[obs] = None
        changes.append((obs, old, None, msg))
    return fixed, changes


def propose_nutrient_map_fixes(nm: Dict[str, Optional[str]]):
    """Propose less-conservative fixes for suspicious nutrient-name mappings.

    For each suspicious observed name (per validate_nutrient_map) try to propose
    a canonical key by token-substring matching against known canonical tokens
    and difflib fallback. Returns (suggested_map, changes) where changes is
    list of (obs, old, new, reason).
    """
    import difflib
    # reuse token sets similar to suggest_nutrient_name_map
    tokens_by_key = {
        "calories": ["energy", "calorie", "kcal"],
        "protein_g": ["protein"],
        "fat_g": ["total lipid", "total fat", "fatty acids", "total lipid (fat)"],
        "carbs_g": ["carbohydrate", "carbohydrate, by difference"],
        "fiber_g": ["fiber", "dietary fiber", "total dietary fiber"],
        "sugar_g": ["sugars", "total sugars", "sugar"],
        "sodium_mg": ["sodium", "sodium, na"],
        "saturated_fat_g": ["saturated", "saturated fatty acids", "sfa"],
    }
    flat_tokens = []
    token_to_key = {}
    for k, toks in tokens_by_key.items():
        for t in toks:
            flat_tokens.append(t)
            token_to_key[t] = k

    suggested = {k: (v if v else None) for k, v in (nm or {}).items()}
    issues = validate_nutrient_map(nm)
    changes = []
    for obs, old_canon, reason in issues:
        lname = obs.lower()
        mapped = None
        # token substring matching
        for t in flat_tokens:
            if t in lname:
                mapped = token_to_key[t]
                break
        # difflib fallback
        if not mapped:
            best = difflib.get_close_matches(lname, flat_tokens, n=1, cutoff=0.6)
            if best:
                mapped = token_to_key[best[0]]
        # only propose if we found a candidate different from current
        if mapped and mapped != old_canon:
            suggested[obs] = mapped
            changes.append((obs, old_canon, mapped, f'proposed via token/difflib: {reason}'))
    return suggested, changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meals", help="Path to meals parquet/json produced by sync", default="scripts/data/raw/meals.parquet")
    p.add_argument("--out-parquet", help="Write candidate foods parquet", default="scripts/data/raw/foods_candidates.parquet")
    p.add_argument("--out-duckdb", help="DuckDB path to write foods table", default="scripts/data/curated/foods.duckdb")
    p.add_argument("--mapping", help="Optional YAML mapping file path (norm->canonical)", default=None)
    p.add_argument("--usda-key", help="USDA FDC API key (optional)", default=None)
    p.add_argument("--usda-max-retries", help="USDA lookup max retry attempts", type=int, default=4)
    p.add_argument("--usda-base-delay", help="USDA lookup base backoff delay in seconds", type=float, default=0.5)
    p.add_argument("--usda-request-delay", help="Delay to wait between USDA requests (politeness), seconds", type=float, default=0.2)
    p.add_argument("--no-stdout-progress", help="Disable stdout progress prints for USDA lookups (use logs only)", action='store_true')
    p.add_argument("--usda-missing-out", help="Write missing canonical names to this file (one per line)", default="scripts/data/raw/usda_missing.txt")
    p.add_argument("--cache-json", help="JSON file to cache USDA results", default="scripts/data/raw/usda_cache.json")
    p.add_argument("--min-count", help="Only keep items seen at least this many times", type=int, default=1)
    p.add_argument("--dump-nutrient-map", help="Dump suggested nutrient name mapping to file and exit", action='store_true')
    p.add_argument("--nutrient-map", help="Optional JSON file for nutrient name mapping", default=None)
    p.add_argument("--validate-nutrient-map", help="Validate nutrient-name suggestions and report suspicious mappings", action='store_true')
    # new options: suggest or apply conservative fixes to nutrient name map
    p.add_argument("--auto-fix-nutrient-map", help="Suggest conservative fixes for suspicious nutrient-name mappings and write a .fixed.json file", action='store_true')
    p.add_argument("--apply-nutrient-map-fixes", help="Apply conservative fixes in-place to the nutrient map file (creates a .bak copy) and continue", action='store_true')
    p.add_argument("--propose-nutrient-map-fixes", help="Propose less-conservative fixes for suspicious nutrient mappings and write a .proposed.json file", action='store_true')
    p.add_argument("--apply-proposed-nutrient-map-fixes", help="Apply proposed fixes (creates .bak) and continue", action='store_true')
    args = p.parse_args()
    # validation path: check nutrient map and exit early if requested
    if args.validate_nutrient_map or args.auto_fix_nutrient_map or args.apply_nutrient_map_fixes or args.propose_nutrient_map_fixes or args.apply_proposed_nutrient_map_fixes:
        # load map
        path = Path(args.nutrient_map) if args.nutrient_map else Path('scripts/data/raw/nutrient_name_suggestions.json')
        if not path.exists():
            logger.error('Nutrient map not found at %s', path)
            raise SystemExit(2)
        try:
            with open(path, 'r', encoding='utf8') as fh:
                nm = json.load(fh) or {}
        except Exception as e:
            logger.error('Failed to load nutrient map: %s', e)
            raise SystemExit(2)

        if args.propose_nutrient_map_fixes:
            suggested, changes = propose_nutrient_map_fixes(nm)
            out_path = path.with_suffix(path.suffix + '.proposed.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf8') as of:
                json.dump(suggested, of, indent=2, ensure_ascii=False)
            logger.info('Wrote proposed nutrient map fixes to %s (%d proposals)', out_path, len(changes))
            for ob, old, new, msg in changes:
                print(f"PROPOSE: {ob!r}: {old} -> {new} : {msg}")
            raise SystemExit(0)

        if args.apply_proposed_nutrient_map_fixes:
            suggested, changes = propose_nutrient_map_fixes(nm)
            if not changes:
                logger.info('No proposed fixes found; nothing to apply')
            else:
                # backup original
                bak = path.with_suffix(path.suffix + '.bak')
                try:
                    path.rename(bak)
                except Exception:
                    import shutil
                    shutil.copy2(path, bak)
                with open(path, 'w', encoding='utf8') as of:
                    json.dump(suggested, of, indent=2, ensure_ascii=False)
                logger.info('Applied %d proposed fixes to %s (original backed up to %s)', len(changes), path, bak)
                for ob, old, new, msg in changes:
                    print(f"APPLY-PROP: {ob!r}: {old} -> {new} : {msg}")
            # continue execution

        if args.auto_fix_nutrient_map:
            fixed, changes = conservative_fix_nutrient_map(nm)
            out_path = path.with_suffix(path.suffix + '.fixed.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf8') as of:
                json.dump(fixed, of, indent=2, ensure_ascii=False)
            logger.info('Wrote conservative fixed nutrient map to %s (%d changes)', out_path, len(changes))
            for ob, old, new, msg in changes:
                print(f"FIX: {ob!r}: {old} -> {new} : {msg}")
            raise SystemExit(0)

        if args.apply_nutrient_map_fixes:
            fixed, changes = conservative_fix_nutrient_map(nm)
            if not changes:
                logger.info('No suspicious mappings found; nothing to apply')
                # continue normally
            else:
                # backup original
                bak = path.with_suffix(path.suffix + '.bak')
                try:
                    path.rename(bak)
                except Exception:
                    # fallback to copy
                    import shutil
                    shutil.copy2(path, bak)
                # write fixed map in-place
                with open(path, 'w', encoding='utf8') as of:
                    json.dump(fixed, of, indent=2, ensure_ascii=False)
                logger.info('Applied %d conservative fixes to %s (original backed up to %s)', len(changes), path, bak)
                for ob, old, new, msg in changes:
                    print(f"APPLY: {ob!r}: {old} -> {new} : {msg}")
            # continue execution after applying fixes

        if args.validate_nutrient_map:
            issues = validate_nutrient_map(nm)
            if not issues:
                logger.info('Nutrient map validation: no issues found in %s', path)
                raise SystemExit(0)
            logger.warning('Nutrient map validation found %d suspicious mappings:', len(issues))
            for o,c,msg in issues:
                print(f"- {o!r} -> {c} : {msg}")
            logger.info('Consider editing %s to correct or remove these mappings', path)
            raise SystemExit(2)

    # Resolve USDA API key: CLI > USDA_API_KEY env > common key files
    if not args.usda_key:
        cand_paths = [Path('scripts/copilot/usda_key.txt'), Path.home() / '.config/henryfood/usda_key.txt']
        for pth in cand_paths:
            if pth.exists():
                try:
                    txt = pth.read_text(encoding='utf8')
                except Exception:
                    continue
                # try quoted string first
                m = re.search(r'"([^\"]+)"', txt)
                if m:
                    args.usda_key = m.group(1)
                    break
                # try key = value or key=value (unquoted)
                m2 = re.search(r'key\s*=\s*"?([^"\'\s]+)"?', txt)
                if m2:
                    args.usda_key = m2.group(1)
                    break
    # finally check environment variable
    if not args.usda_key:
        args.usda_key = os.environ.get('USDA_API_KEY')

    meals_path = Path(args.meals)
    out_parquet = Path(args.out_parquet)
    out_duckdb = Path(args.out_duckdb)
    cache_path = Path(args.cache_json)

    try:
        import pandas as pd
    except Exception:
        logger.error("pandas required; install with pip install pandas")
        return 2

    # load meals
    if not meals_path.exists():
        # try JSON diary
        if Path("scripts/data/raw/meals.json").exists():
            meals_path = Path("scripts/data/raw/meals.json")
        else:
            raise SystemExit(f"Meals file not found: {meals_path}")

    df = load_meals(meals_path)
    if df.empty:
        logger.info("No meals data loaded")
        return 1

    # assume df has 'item' or 'raw_text'
    if "item" not in df.columns:
        # try to build from raw JSON diary structure with meal columns like Breakfast/Lunch/Dinner/Snack
        import pandas as _pd
        meal_cols = [c for c in df.columns if c.lower() in ('breakfast','lunch','dinner','snack','brunch','supper','meal','meals')]
        if meal_cols:
            rows = []
            for _, r in df.iterrows():
                # try to preserve a date column if present
                date = None
                if 'Date' in df.columns:
                    date = r.get('Date')
                elif 'date' in df.columns:
                    date = r.get('date')
                for mc in meal_cols:
                    txt = r.get(mc)
                    if txt is None:
                        continue
                    # split on commas and newlines
                    parts = re.split(r"[,\n]+", str(txt))
                    for p in parts:
                        it = p.strip()
                        if it:
                            rows.append({'item': it, 'date': date, 'meal': mc})
            if not rows:
                logger.info("No items extracted from diary-style columns")
                return 1
            df = _pd.DataFrame(rows)
            logger.info("Flattened diary-style meals into %d rows (columns: %s)", len(df), meal_cols)
        else:
            # try to build from raw JSON structure
            logger.info("'item' column missing in meals; expecting rows with 'item' or use build_meals_df first")
            return 1

    # normalize
    df["item_norm"] = df["item"].apply(normalize_name)
    counts = Counter([v for v in df["item_norm"].tolist() if v])

    items = [(it, counts[it]) for it in sorted(counts.keys(), key=lambda x: -counts[x]) if counts[it] >= args.min_count]

    # load mapping
    mapping: Dict[str, str] = {}
    if args.mapping and Path(args.mapping).exists():
        try:
            import yaml
            with open(args.mapping, "r", encoding="utf8") as fh:
                loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    mapping = {k.lower(): v for k, v in loaded.items()}
        except Exception:
            logger.warning("Failed to load mapping YAML; continuing without it")

    rows = []
    for it, cnt in items:
        canonical = mapping.get(it, it)
        rows.append({"item_norm": it, "count": cnt, "canonical": canonical})

    foods_df = pd.DataFrame(rows)
    if foods_df.empty:
        logger.info("No candidate foods after filtering")
        return 0

    # provenance: mark these candidate rows as originating from USDA lookups by default
    if 'source' not in foods_df.columns:
        foods_df['source'] = 'usda'

    # save parquet candidates
    save_foods_parquet(foods_df, out_parquet)
    logger.info("Wrote %d food candidates to %s", len(foods_df), out_parquet)

    # USDA lookups (optional)
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf8") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    if args.usda_key:
        api_key = args.usda_key
        for idx, row in foods_df.iterrows():
            key = row["canonical"]
            if not key:
                continue
            if key in cache:
                logger.debug("Skipping USDA lookup for %s; cached", key)
                continue
            logger.info("Querying USDA for %s", key)
            res = usda_search_with_retries(
                api_key,
                key,
                max_attempts=args.usda_max_retries,
                base_delay=args.usda_base_delay,
                show_progress=(not args.no_stdout_progress),
            )
            if res is None:
                logger.info("No USDA match for %s; recording empty entry", key)
            cache[key] = res or {}
            # be polite: wait configured request delay between lookups
            time.sleep(args.usda_request_delay)
        # write cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
        print(f"Wrote USDA cache to {cache_path}")

    # enrich foods_df with cache where available (use helper to be testable)
    enrich_foods_df_with_cache(foods_df, cache)

    # load optional nutrient name mapping (used to extract nutrients reliably)
    nutrient_name_map: Dict[str, str] = {}
    # prefer CLI-specified map, else default suggestions file
    nm_path = Path(args.nutrient_map) if args.nutrient_map else Path('scripts/data/raw/nutrient_name_suggestions.json')
    if nm_path.exists():
        try:
            with open(nm_path, 'r', encoding='utf8') as fh:
                loaded = json.load(fh) or {}
                # keep only non-null mappings
                nutrient_name_map = {k.lower(): v for k, v in loaded.items() if v}
                logger.info('Loaded nutrient name map with %d entries from %s', len(nutrient_name_map), nm_path)
        except Exception:
            logger.warning('Failed to load nutrient name map from %s; continuing without it', nm_path)
    else:
        logger.debug('No nutrient name map at %s; continuing with heuristics', nm_path)

    # extract numeric nutrient columns for ML, then merge OpenFoodFacts (if available) into the same ML file with provenance
    try:
        ml_rows = []
        for idx, r in foods_df.iterrows():
            key = r.get("canonical")
            entry = cache.get(key) or {}
            nutrients = extract_nutrients_from_cache_entry(entry, nutrient_name_map if nutrient_name_map else None)
            ml_rows.append(nutrients)
        ml_df = pd.DataFrame(ml_rows)
        # attach canonical and count to ML df
        ml_df["canonical"] = foods_df["canonical"].values
        ml_df["count"] = foods_df["count"].values

        # attempt to merge OpenFoodFacts nutrient columns into the ML DataFrame (keep off_ prefixed provenance)
        off_path = Path("scripts/data/curated/openfoodfacts.parquet")
        nutrient_keys = ['calories','protein_g','fat_g','carbs_g','fiber_g','sugar_g','sodium_mg','saturated_fat_g']
        if off_path.exists():
            try:
                off_df = pd.read_parquet(off_path)
                # canonical join key: prefer explicit 'canonical', then 'query', then 'product_name'
                if 'canonical' in off_df.columns:
                    off_df['canonical_join'] = off_df['canonical'].fillna('').astype(str)
                elif 'query' in off_df.columns:
                    off_df['canonical_join'] = off_df['query'].fillna('').astype(str)
                elif 'product_name' in off_df.columns:
                    off_df['canonical_join'] = off_df['product_name'].fillna('').astype(str)
                else:
                    off_df['canonical_join'] = off_df.index.astype(str)

                off_pref = off_df.copy()
                # create off_ prefixed nutrient columns
                for k in nutrient_keys:
                    off_pref[f'off_{k}'] = off_pref[k] if k in off_pref.columns else None

                # provenance columns
                # prefer explicit 'id' column, then 'code'; avoid using `or` on Series which can lead to surprising truthiness
                # if 'id' in off_pref.columns:
                #     off_pref['off_id'] = off_pref['id']
                # elif 'code' in off_pref.columns:
                #     off_pref['off_id'] = off_pref['code']
                # else:
                #     off_pref['off_id'] = None
                # off_pref['off_product_name'] = off_pref['product_name'] if 'product_name' in off_pref.columns else None
                # provenance columns: prefer common id/code columns and several alternate product-name columns
                id_candidates = ['id', 'code', 'codes', 'off_id', 'product_code', 'barcode']
                name_candidates = ['product_name', 'of_product_name', 'of_name', 'name', 'product', 'of_product']
                sel_id = next((c for c in id_candidates if c in off_pref.columns), None)
                if sel_id:
                    try:
                        # coerce to str, strip whitespace and convert empty strings to None
                        col = off_pref[sel_id].fillna('').astype(str).str.strip()
                        col = col.replace('', None)
                        off_pref['off_id'] = col
                    except Exception:
                        off_pref['off_id'] = None
                else:
                    off_pref['off_id'] = None

                sel_name = next((c for c in name_candidates if c in off_pref.columns), None)
                if sel_name:
                    try:
                        coln = off_pref[sel_name].fillna('').astype(str).str.strip()
                        coln = coln.replace('', None)
                        off_pref['off_product_name'] = coln
                    except Exception:
                        off_pref['off_product_name'] = None
                else:
                    off_pref['off_product_name'] = None

                keep_cols = ['canonical_join'] + [f'off_{k}' for k in nutrient_keys] + ['off_id','off_product_name']
                off_keep = off_pref[keep_cols].rename(columns={'canonical_join':'canonical'})

                # drop rows with empty/blank canonical join keys to avoid false-positive merges
                # convert empty strings to NA and drop them before merging
                import pandas as _pd
                off_keep['canonical'] = off_keep['canonical'].astype(str).str.strip()
                off_keep.loc[off_keep['canonical'] == '', 'canonical'] = _pd.NA
                off_keep = off_keep.dropna(subset=['canonical'])

                # left-merge so ML rows remain 1:1
                ml_df = ml_df.merge(off_keep, on='canonical', how='left')

                # add fdc_id presence from foods_df for source logic
                ml_df['fdc_id'] = ml_df['canonical'].map(foods_df.set_index('canonical')['fdc_id'])

                # flag presence and prefer OFF values when present
                ml_df['off_present'] = ml_df[[f'off_{k}' for k in nutrient_keys] + ['off_id']].notnull().any(axis=1)
                for k in nutrient_keys:
                    # prefer OFF value if available, else keep USDA-derived
                    ml_df[k] = ml_df[f'off_{k}'].combine_first(ml_df.get(k))

                # unified source column
                def _src_row(row):
                    usda = pd.notnull(row.get('fdc_id'))
                    off = bool(row.get('off_present'))
                    if usda and off:
                        return 'both'
                    if usda:
                        return 'usda'
                    if off:
                        return 'off'
                    return 'unknown'
                ml_df['source'] = ml_df.apply(_src_row, axis=1)
                logger.info('Merged OpenFoodFacts into ML DataFrame (off_ provenance kept)')
            except Exception as e:
                logger.warning('Failed to merge OpenFoodFacts into ML DataFrame: %s', e)

        ml_out = Path('scripts/data/curated/foods_ml.parquet')
        ml_out.parent.mkdir(parents=True, exist_ok=True)
        ml_df.to_parquet(ml_out, index=False)
        logger.info('Wrote ML-ready foods parquet (merged with OFF provenance) to %s', ml_out)
    except Exception as e:
        logger.warning("Failed to extract/merge nutrients for ML: %s", e)

    # summary of USDA enrichment
    try:
        total = len(foods_df)
        matched = int(foods_df["fdc_id"].notnull().sum()) if "fdc_id" in foods_df.columns else 0
        missing_count = total - matched
        logger.info("USDA enrichment summary: %d total candidates, %d matched, %d missing", total, matched, missing_count)
        # show some missing examples for quick inspection
        if missing_count > 0:
            missing_list = list(foods_df[foods_df["fdc_id"].isnull()]["canonical"].dropna().unique())
            logger.info("Missing examples (first 20): %s", missing_list[:20])
            # optionally write missing canonical names to a file for review
            if args.usda_missing_out:
                miss_path = Path(args.usda_missing_out)
                miss_path.parent.mkdir(parents=True, exist_ok=True)
                with open(miss_path, "w", encoding="utf8") as mf:
                    for k in missing_list:
                        mf.write(k + "\n")
                logger.info("Wrote %d missing canonical names to %s", len(missing_list), miss_path)
    except Exception as e:
        logger.warning("Failed to summarize USDA enrichment: %s", e)

    # write to duckdb
    out_duckdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        # make a working copy and add a provenance/source column for original rows
        combined = foods_df.copy()
        if 'source' not in combined.columns:
            combined['source'] = 'usda'

        # Try to merge OpenFoodFacts enrichments (if available) into the same table
        off_path = Path("scripts/data/curated/openfoodfacts.parquet")
        if off_path.exists():
            try:
                off_df = pd.read_parquet(off_path)
                # build a canonical key in OFF frame for joining: prefer 'canonical', then 'query', then 'product_name'
                if 'canonical' in off_df.columns:
                    off_df['canonical_join'] = off_df['canonical'].fillna('').astype(str)
                elif 'query' in off_df.columns:
                    off_df['canonical_join'] = off_df['query'].fillna('').astype(str)
                elif 'product_name' in off_df.columns:
                    off_df['canonical_join'] = off_df['product_name'].fillna('').astype(str)
                else:
                    off_df['canonical_join'] = off_df.index.astype(str)

                # prefix OFF nutrient/provenance columns so we can keep both sources in one row
                nutrient_keys = ['calories','protein_g','fat_g','carbs_g','fiber_g','sugar_g','sodium_mg','saturated_fat_g']
                off_pref = off_df.copy()
                for k in nutrient_keys:
                    if k in off_pref.columns:
                        off_pref[f'off_{k}'] = off_pref[k]
                    else:
                        off_pref[f'off_{k}'] = None

                # provenance columns
                # prefer explicit 'id' column, then 'code'; avoid using `or` on Series which can lead to surprising truthiness
                # if 'id' in off_pref.columns:
                #     off_pref['off_id'] = off_pref['id']
                # elif 'code' in off_pref.columns:
                #     off_pref['off_id'] = off_pref['code']
                # else:
                #     off_pref['off_id'] = None
                # off_pref['off_product_name'] = off_pref['product_name'] if 'product_name' in off_pref.columns else None
                # provenance columns: prefer common id/code columns and several alternate product-name columns
                id_candidates = ['id', 'code', 'codes', 'off_id', 'product_code', 'barcode']
                name_candidates = ['product_name', 'of_product_name', 'of_name', 'name', 'product', 'of_product']
                sel_id = next((c for c in id_candidates if c in off_pref.columns), None)
                if sel_id:
                    try:
                        # coerce to str, strip whitespace and convert empty strings to None
                        col = off_pref[sel_id].fillna('').astype(str).str.strip()
                        col = col.replace('', None)
                        off_pref['off_id'] = col
                    except Exception:
                        off_pref['off_id'] = None
                else:
                    off_pref['off_id'] = None

                sel_name = next((c for c in name_candidates if c in off_pref.columns), None)
                if sel_name:
                    try:
                        coln = off_pref[sel_name].fillna('').astype(str).str.strip()
                        coln = coln.replace('', None)
                        off_pref['off_product_name'] = coln
                    except Exception:
                        off_pref['off_product_name'] = None
                else:
                    off_pref['off_product_name'] = None

                # keep only join key and off_ columns
                keep_cols = ['canonical_join'] + [f'off_{k}' for k in nutrient_keys] + ['off_id','off_product_name','off_raw']
                off_keep = off_pref[keep_cols].rename(columns={'canonical_join':'canonical'})

                # drop empty/blank canonical join keys from OFF keep-frame to avoid spurious matches
                import pandas as _pd
                off_keep['canonical'] = off_keep['canonical'].astype(str).str.strip()
                off_keep.loc[off_keep['canonical'] == '', 'canonical'] = _pd.NA
                off_keep = off_keep.dropna(subset=['canonical'])

                # perform left merge so original foods_df rows remain 1:1
                combined = combined.merge(off_keep, on='canonical', how='left')

                # flag presence and update unified 'source' column ('usda','off','both')
                combined['off_present'] = combined[[f'off_{k}' for k in nutrient_keys] + ['off_id']].notnull().any(axis=1)
                def _src(row):
                    usda = pd.notnull(row.get('fdc_id'))
                    off = bool(row.get('off_present'))
                    if usda and off:
                        return 'both'
                    if usda:
                        return 'usda'
                    if off:
                        return 'off'
                    return row.get('source') or 'unknown'
                combined['source'] = combined.apply(_src, axis=1)
                logger.info("Merged OpenFoodFacts enrichments into foods table (kept as off_ prefixed columns)")
            except Exception as e:
                logger.warning("Failed to load/merge OpenFoodFacts parquet: %s", e)

        # write combined table to DuckDB
        write_duckdb_table(combined, out_duckdb, table="foods")
        logger.info("Wrote foods table to %s:foods (including OFF enrichments merged if available)", out_duckdb)

        # Quick provenance / enrichment summary: ML parquet nutrient coverage, OFF parquet overview, DuckDB source counts
        try:
            # ML parquet
            ml_out = Path("scripts/data/curated/foods_ml.parquet")
            if ml_out.exists():
                ml_df = pd.read_parquet(ml_out)
                total_ml = len(ml_df)
                print(f"ML parquet: {total_ml} rows. Nutrient non-null counts:")
                nutrient_cols = [c for c in ml_df.columns if c not in ('canonical','count')]
                for c in nutrient_cols:
                    nonnull = int(ml_df[c].notnull().sum())
                    print(f" - {c}: {nonnull}/{total_ml} non-null ({nonnull/total_ml:.1%})")
            else:
                print("ML parquet not found at", ml_out)

            # OpenFoodFacts parquet overview
            if off_path.exists():
                off_df = pd.read_parquet(off_path)
                print(f"OpenFoodFacts parquet: {len(off_df)} rows. Columns: {list(off_df.columns)[:10]}{'...' if len(off_df.columns)>10 else ''}")
                # check common nutrient-like columns
                possible_nut_cols = [c for c in off_df.columns if 'nutri' in c.lower() or 'energy' in c.lower() or 'calor' in c.lower()]
                for c in possible_nut_cols:
                    nonnull = int(off_df[c].notnull().sum())
                    print(f" - OFF {c}: {nonnull}/{len(off_df)} non-null ({nonnull/len(off_df):.1%})")
            else:
                print("OpenFoodFacts parquet not found at", off_path)

            # DuckDB foods source counts (if source column exists)
            try:
                import duckdb
                con = duckdb.connect(database=str(out_duckdb), read_only=True)
                try:
                    src_counts = con.execute('SELECT source, COUNT(*) AS cnt FROM foods GROUP BY source ORDER BY cnt DESC').fetchdf()
                    print("Foods table source counts:")
                    for row in src_counts.itertuples(index=False):
                        print(f" - {row[0]}: {row[1]}")

                    # Per-column null counts for key nutrient/provenance fields in the DuckDB foods table
                    try:
                        # get existing table columns
                        tbl_info = con.execute("PRAGMA table_info('foods')").fetchdf()
                        existing_cols = set(tbl_info['name'].tolist())
                        check_cols = ['calories','protein_g','fat_g','carbs_g','fiber_g','sugar_g','sodium_mg','saturated_fat_g','fdc_id','off_id','canonical']
                        cols_to_check = [c for c in check_cols if c in existing_cols]
                        if cols_to_check:
                            total = int(con.execute('SELECT COUNT(*) FROM foods').fetchone()[0])
                            print('Foods table null counts:')
                            for c in cols_to_check:
                                # count non-null values for the column
                                nonnull = int(con.execute(f'SELECT COUNT("{c}") FROM foods').fetchone()[0])
                                nulls = total - nonnull
                                pct = (nulls / total) if total > 0 else 0.0
                                print(f" - {c}: {nulls} null / {total} total ({pct:.1%} null)")
                    except Exception as e:
                        logger.debug('DuckDB per-column null count summary failed: %s', e)
                finally:
                    con.close()
            except Exception as e:
                logger.debug('DuckDB source summary failed: %s', e)
        except Exception as e:
            logger.warning('Failed to produce quick summary: %s', e)
    except Exception as e:
        logger.warning("Failed to write combined foods table: %s", e)
        # fallback: write original foods_df
        write_duckdb_table(foods_df, out_duckdb, table="foods")
        logger.info("Wrote foods table to %s:foods", out_duckdb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
