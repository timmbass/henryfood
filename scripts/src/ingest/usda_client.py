"""USDA FoodData Central client (starter)

Provides simple search and detail lookup using the FoodData Central API.
"""

import argparse
import os
import json
import re
from typing import Any, Dict, Optional
from pathlib import Path
import datetime

import requests

API_SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search"
API_FOOD = "https://api.nal.usda.gov/fdc/v1/food/{}"


def _session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.params = {"api_key": api_key}
    return s


def search_foods(query: str, api_key: str, page_size: int = 25) -> Dict[str, Any]:
    """Search foods by text query. Returns parsed JSON response."""
    s = _session(api_key)
    payload = {"query": query, "pageSize": page_size}
    resp = s.post(API_SEARCH, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_food_details(fdc_id: int, api_key: str) -> Dict[str, Any]:
    """Get full food details by FDC ID."""
    s = _session(api_key)
    url = API_FOOD.format(fdc_id)
    resp = s.get(url, timeout=30, params={})
    resp.raise_for_status()
    return resp.json()


def summarize_food(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary including core IDs, ingredients, nutrient subtypes, flags and derived placeholders."""
    out: Dict[str, Any] = {}

    # 1) Core Identification & Provenance
    out.update({
        "fdcId": detail.get("fdcId"),
        "dataType": detail.get("dataType"),
        "description": detail.get("description") or detail.get("lowercaseDescription"),
        "brandOwner": detail.get("brandOwner"),
        "brandName": detail.get("brandName"),
        "subbrandName": detail.get("subbrandName"),
        "gtinUpc": detail.get("gtinUpc"),
        "foodCategory": detail.get("foodCategory"),
        "publicationDate": detail.get("publicationDate"),
    })

    # 2) Ingredient & Processing Signals
    out["ingredients"] = detail.get("ingredients")
    out["foodAttributes"] = detail.get("foodAttributes")
    out["foodClass"] = detail.get("foodClass")
    out["foodPortions"] = detail.get("foodPortions")

    # 3) Allergens & Sensitivity Proxies (try multiple locations)
    allergen_flags = {
        "containsMilk": False,
        "containsEgg": False,
        "containsWheat": False,
        "containsSoy": False,
        "containsPeanut": False,
        "containsTreeNut": False,
        "containsShellfish": False,
        "containsFish": False,
        "containsSesame": False,
    }
    text_sources = " ".join(filter(None, [
        str(out.get("ingredients") or ""),
        str(out.get("description") or ""),
        json.dumps(out.get("foodAttributes") or {})
    ])).lower()
    # simple keyword heuristics
    heuristics = {
        "containsMilk": ["milk", "dairy", "casein", "whey"],
        "containsEgg": ["egg", "albumin"],
        "containsWheat": ["wheat", "gluten", "farina"],
        "containsSoy": ["soy", "soybean", "soya"],
        "containsPeanut": ["peanut", "groundnut"],
        "containsTreeNut": ["almond", "walnut", "hazelnut", "cashew", "pecan", "pistachio"],
        "containsShellfish": ["shrimp", "crab", "lobster", "shellfish"],
        "containsFish": ["fish", "tuna", "salmon", "cod"],
        "containsSesame": ["sesame"],
    }
    for key, toks in heuristics.items():
        for t in toks:
            if t in text_sources:
                allergen_flags[key] = True
                break
    out.update(allergen_flags)

    # 4) Carbohydrate substructure / 5) Fat subtypes / 6) Protein detail / 8) Micronutrients
    # Map available nutrients by name (build normalized lookup for robust matching)
    nutrients = {}
    nutrients_norm = {}
    def _normalize_name(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, dict):
            # stringify nested nutrient objects
            try:
                s = x.get("name") or x.get("nutrientName") or x.get("description") or json.dumps(x)
            except Exception:
                s = json.dumps(x)
        else:
            s = str(x)
        s = s.strip().lower()
        # collapse whitespace and remove punctuation except plus/minus signs and letters/numbers
        s = re.sub(r"[\(\),:\-\/]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    for n in detail.get("foodNutrients", []) or []:
        raw_name = n.get("nutrientName") or n.get("name") or n.get("nutrient")
        if not raw_name:
            continue
        # Some API responses use nested objects for nutrient names; coerce to a stable string key
        if isinstance(raw_name, dict):
            # try common nested fields, fall back to JSON string
            name = raw_name.get("name") or raw_name.get("nutrientName") or raw_name.get("description") or json.dumps(raw_name)
        else:
            name = str(raw_name)
        name = name.strip()
        norm = _normalize_name(name)
        value = {"value": n.get("value"), "unit": n.get("unitName") or n.get("nutrientUnitName")}
        nutrients[name] = value
        # keep first-seen mapping for normalized key
        if norm and norm not in nutrients_norm:
            nutrients_norm[norm] = value

    def pick(keys):
        result = {}
        for k in keys:
            nk = _normalize_name(k)
            v = nutrients_norm.get(nk)
            if v is not None:
                result[k] = v
        return result

    out["carbohydrate_substructure"] = pick([
        "Sugars, total including NLEA",
        "Sugars, added",
        "Starch",
        "Fiber, total dietary",
        "Fiber, soluble",
        "Fiber, insoluble",
        "Sugar alcohols",
    ])

    out["fat_subtypes"] = pick([
        "Total lipid (fat)",
        "Fatty acids, total saturated",
        "Fatty acids, total monounsaturated",
        "Fatty acids, total polyunsaturated",
        "Fatty acids, total trans",
        "Cholesterol",
    ])

    out["protein_detail"] = pick([
        "Protein",
        "Amino acids, total",
        "Glutamic acid",
        "Histidine",
    ])

    out["micronutrients"] = pick([
        "Sodium, Na",
        "Potassium, K",
        "Calcium, Ca",
        "Magnesium, Mg",
        "Iron, Fe",
        "Zinc, Zn",
    ])

    # 7) Histamine & Biogenic Amine Proxies: derive simple flags from text
    ferment_keywords = ["ferment", "fermented", "aged", "cured", "pickled", "sour" ]
    out["fermented_flag"] = any(t in text_sources for t in ferment_keywords)
    out["aged_flag"] = "aged" in text_sources
    out["cured_flag"] = "cured" in text_sources
    out["processed_meat_flag"] = any(t in text_sources for t in ["salami", "bacon", "ham", "sausage"]) 
    out["leftover_likelihood_flag"] = any(t in text_sources for t in ["leftover", "left-overs", "day-old"]) 

    # 9) Confidence / quality signals
    out.update({
        "nutrientDerivation": detail.get("nutrientDerivation"),
        "nutrientAcquisitionDetails": detail.get("nutrientAcquisitionDetails"),
        "confidenceCode": detail.get("confidenceCode"),
        "dataSource": detail.get("dataSource"),
    })

    # 10) Derived fields (placeholders)
    out["processing_score"] = None
    out["histamine_risk_score"] = None
    out["fodmap_risk_score"] = None
    out["allergen_load_score"] = None
    out["evening_consumption_flag"] = None
    out["lag_hours_to_symptom"] = None

    return out


def fetch_usda(item_name: str, api_key: Optional[str] = None, out: Optional[str] = None, details: int = 0) -> Dict[str, Any]:
    """Search for an item_name and optionally save the top results to `out`.

    If details>0, fetch detailed nutrient info for the top `details` results and print summaries.
    """
    if not api_key:
        api_key = os.environ.get("USDA_API_KEY")
    if not api_key:
        raise RuntimeError("USDA API key required via --api-key or USDA_API_KEY env var")

    data = search_foods(item_name, api_key)
    print(f"Found {len(data.get('foods', []))} results for '{item_name}'")
    top = data.get("foods", [])[:5]
    for i, f in enumerate(top, 1):
        name = f.get("description") or f.get("lowercaseDescription") or f.get("description", "")
        print(f"{i}. FDC ID={f.get('fdcId')} - {name}")

    # Save search results
    if out:
        out_path = Path(out)
        if out_path.is_dir() or str(out).endswith("/"):
            # save into directory
            out_dir = out_path if out_path.is_dir() else out_path
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"search_{item_name.replace(' ', '_')}.json"
        else:
            out_file = out_path
        with open(out_file, "w", encoding="utf8") as fh:
            json.dump(data, fh, indent=2)
        print(f"Saved search results to {out_file}")

    # Optionally fetch details for top N and print compact summary
    if details and len(top) > 0:
        for i, f in enumerate(top[:details], 1):
            fdc = f.get("fdcId")
            try:
                det = get_food_details(fdc, api_key)
            except Exception as e:
                print(f"Failed to fetch details for FDC {fdc}: {e}")
                continue
            summary = summarize_food(det)
            print("---")
            print(json.dumps(summary, indent=2))
            # save each detailed JSON next to search output if out provided
            if out:
                det_path = out_file.parent / f"food_{fdc}.json"
                with open(det_path, "w", encoding="utf8") as fh:
                    json.dump(det, fh, indent=2)
                print(f"Saved details to {det_path}")

    return data


def build_summary_dataframe(summaries: list) -> "pd.DataFrame":
    """Convert list of summary dicts into a flattened pandas DataFrame."""
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to build summary DataFrame: install pandas") from e

    if not summaries:
        return pd.DataFrame()
    # Use json_normalize to flatten nested structures
    df = pd.json_normalize(summaries)
    # add collection timestamp (use timezone-aware UTC)
    df["collected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return df


def save_summaries(item_name: str, api_key: str, out_parquet: Optional[str] = None, duckdb_path: Optional[str] = None, details: int = 5) -> None:
    """Fetch top `details` food details, summarize them and save to Parquet and/or DuckDB."""
    # fetch search results
    search = search_foods(item_name, api_key, page_size=25)
    top = search.get("foods", [])[:details]

    summaries = []
    detailed = []
    for f in top:
        fdc = f.get("fdcId")
        try:
            det = get_food_details(fdc, api_key)
        except Exception as e:
            print(f"Warning: failed to fetch details for FDC {fdc}: {e}")
            continue
        detailed.append(det)
        summaries.append(summarize_food(det))

    if out_parquet:
        df = build_summary_dataframe(summaries)
        if df.empty:
            print("No summaries to save.")
        else:
            out_parquet_path = Path(out_parquet)
            out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(out_parquet_path, index=False)
                print(f"Saved summaries to {out_parquet_path}")
            except Exception as e:
                print(f"Failed to write Parquet: {e}")

    if duckdb_path:
        # write into a DuckDB database file as table usda_summaries
        try:
            import duckdb
            import pandas as pd
        except Exception as e:
            print("duckdb and pandas are required to save to DuckDB. Install duckdb and pandas.")
            return
        df = build_summary_dataframe(summaries)
        if df.empty:
            print("No summaries to write to DuckDB.")
            return
        db_file = Path(duckdb_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(database=str(db_file), read_only=False)
        try:
            con.register("tmp_df", df)
            con.execute("CREATE OR REPLACE TABLE usda_summaries AS SELECT * FROM tmp_df")
            print(f"Wrote usda_summaries table to {db_file}")
        finally:
            con.close()


def main():
    parser = argparse.ArgumentParser(description="USDA FoodData Central client")
    parser.add_argument("--api-key", help="USDA API key (or set USDA_API_KEY env)")
    parser.add_argument("--item", help="Text query to search for")
    parser.add_argument("--fdc-id", type=int, help="Fetch details for a specific FDC ID")
    parser.add_argument("--out", help="Optional path to save JSON output (file or dir)")
    parser.add_argument("--details", type=int, default=0, help="Fetch detailed nutrient info for top N search results")
    parser.add_argument("--parquet", help="Optional path to save summary output as Parquet file")
    parser.add_argument("--duckdb", help="Optional path to DuckDB database file to save summaries")
    args = parser.parse_args()

    if args.fdc_id:
        if not args.api_key and not os.environ.get("USDA_API_KEY"):
            parser.error("FDC detail lookup requires --api-key or USDA_API_KEY env var")
        details = get_food_details(args.fdc_id, args.api_key or os.environ.get("USDA_API_KEY"))
        print(json.dumps(details, indent=2)[:1000])
        if args.out:
            with open(args.out, "w", encoding="utf8") as fh:
                json.dump(details, fh, indent=2)
            print(f"Saved details to {args.out}")
        return

    if args.item:
        fetch_usda(args.item, api_key=args.api_key, out=args.out, details=args.details)
        if args.parquet or args.duckdb:
            save_summaries(args.item, api_key=args.api_key, out_parquet=args.parquet, duckdb_path=args.duckdb, details=args.details)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
